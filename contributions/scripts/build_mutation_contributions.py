#!/usr/bin/env python3
"""Build verified APRIL-format contributions using registered proof mutators."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    from .april_formatter import format_april_record
    from .proof_annotator import (
        ANNOTATION_PROMPT_VERSION,
        ANNOTATION_SYSTEM_PROMPT,
        annotation_user_prompt,
        parse_annotation,
    )
    from .lean_verifier import LeanInteractVerifier, VerifierPool, verify_parallel
    from .llm_client import (
        DEFAULT_HUGGINGFACE_MODEL,
        DEFAULT_OPENAI_MODEL,
        PAPER_MODEL,
        HuggingFaceLocalModel,
        OpenAIModel,
        sanitize_endpoint,
    )
    from .mutation_adapters import (
        DEFAULT_MUTATORS,
        MUTATOR_DESCRIPTORS,
        build_adapter,
        manifest_metadata,
    )
    from .mutation_orchestrator import (
        SEED_DERIVATION_VERSION,
        MutationAdapter,
        make_jobs,
        mutation_job_id,
    )
    from .mutation_utils import JsonlJournal, atomic_json, atomic_jsonl, sha256_bytes, sha256_text
    from .proof_datasets import OpenAIMathDataset, default_openai_math_input
except ImportError:  # Support direct execution from contributions/scripts.
    from april_formatter import format_april_record
    from proof_annotator import (
        ANNOTATION_PROMPT_VERSION,
        ANNOTATION_SYSTEM_PROMPT,
        annotation_user_prompt,
        parse_annotation,
    )
    from lean_verifier import LeanInteractVerifier, VerifierPool, verify_parallel
    from llm_client import (
        DEFAULT_HUGGINGFACE_MODEL,
        DEFAULT_OPENAI_MODEL,
        PAPER_MODEL,
        HuggingFaceLocalModel,
        OpenAIModel,
        sanitize_endpoint,
    )
    from mutation_adapters import (
        DEFAULT_MUTATORS,
        MUTATOR_DESCRIPTORS,
        build_adapter,
        manifest_metadata,
    )
    from mutation_orchestrator import (
        SEED_DERIVATION_VERSION,
        MutationAdapter,
        make_jobs,
        mutation_job_id,
    )
    from mutation_utils import JsonlJournal, atomic_json, atomic_jsonl, sha256_bytes, sha256_text
    from proof_datasets import OpenAIMathDataset, default_openai_math_input


REPO_ROOT = Path(__file__).resolve().parents[2]


def record_rejection(journal: JsonlJournal, key: str, phase: str, reason: str) -> None:
    if journal.get(key) is None:
        journal.append({"job_id": key, "phase": phase, "reason": reason})


def endpoint_from_args(args: argparse.Namespace) -> str | None:
    return args.base_url or os.environ.get(args.base_url_env)


def model_endpoint(args: argparse.Namespace) -> str:
    return "local" if args.model_provider == "huggingface" else sanitize_endpoint(endpoint_from_args(args))


def model_metadata(args: argparse.Namespace) -> dict[str, Any]:
    path = args.work_dir / "model_metadata.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    return {
        "provider": args.model_provider,
        "model": args.model,
        "requested_revision": args.hf_revision if args.model_provider == "huggingface" else None,
        "resolved_revision": None,
        "device": None,
    }


def expected_job_ids(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[str]:
    return [
        mutation_job_id(mutator_id, str(row["src_hash"]), attempt)
        for row in rows
        for mutator_id in args.mutators
        for attempt in range(args.attempts_per_mutator)
    ]


def ensure_run_config(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    config = {
        "protocol": "openai-math-generic-mutations-v2",
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "selected_sources": sorted(str(row["src_hash"]) for row in rows),
        "selected_theorems": sorted(str(row["theorem"]) for row in rows),
        "project_dir": str(args.project_dir),
        "verified_project": getattr(args, "verified_project", None),
        "seed": args.seed,
        "seed_derivation": SEED_DERIVATION_VERSION,
        "mutators": list(args.mutators),
        "attempts_per_mutator": args.attempts_per_mutator,
        "model_provider": args.model_provider,
        "model": args.model,
        "endpoint": model_endpoint(args),
        "hf_device": args.hf_device if args.model_provider == "huggingface" else None,
        "hf_revision": args.hf_revision if args.model_provider == "huggingface" else None,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "annotation_max_tokens": args.annotation_max_tokens,
        "annotation_retries": args.annotation_retries,
        "annotation_prompt_version": ANNOTATION_PROMPT_VERSION,
    }
    path = args.work_dir / "run_config.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"work directory has incompatible settings: {path}; choose a new --work-dir")
        return
    journal_names = ("baseline.jsonl", "generations.jsonl", "verifications.jsonl", "annotations.jsonl")
    if any((args.work_dir / name).exists() for name in journal_names):
        raise ValueError(f"journals exist without {path}; choose a new --work-dir")
    atomic_json(path, config)


def project_fingerprint(project_dir: Path) -> str:
    fingerprint = hashlib.sha256()
    for name in ("lean-toolchain", "lakefile.lean", "lakefile.toml", "lake-manifest.json"):
        path = project_dir / name
        if path.is_file():
            fingerprint.update(name.encode() + b"\0" + path.read_bytes())
    return fingerprint.hexdigest()


def verify_project_provenance(rows: Sequence[Mapping[str, Any]], project_dir: Path) -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(project_dir), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"OpenAI Math project is not a readable Git checkout: {project_dir}") from error
    if dirty:
        raise ValueError("OpenAI Math checkout is dirty; verification must use the extracted source commit")
    if {str(row.get("source_commit")) for row in rows} != {commit}:
        raise ValueError("input source commit does not match the OpenAI Math checkout")
    if any(bool(row.get("source_dirty")) for row in rows):
        raise ValueError("input was extracted from a dirty source checkout")
    fingerprint = project_fingerprint(project_dir)
    if {str(row.get("verifier_fingerprint")) for row in rows} != {fingerprint}:
        raise ValueError("OpenAI Math verifier fingerprint does not match the extracted input")
    toolchain_path = project_dir / "lean-toolchain"
    if not toolchain_path.is_file():
        raise ValueError(f"missing OpenAI Math lean-toolchain: {toolchain_path}")
    toolchain = toolchain_path.read_text(encoding="utf-8").strip()
    if {str(row.get("lean_toolchain")) for row in rows} != {toolchain}:
        raise ValueError("OpenAI Math lean-toolchain does not match the extracted input")
    return {"commit": commit, "fingerprint": fingerprint, "toolchain": toolchain}


def baseline_phase(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    verifier: VerifierPool,
    journal: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    jobs = [(row["src_hash"], row["correct_proof"]) for row in rows if journal.get(row["src_hash"]) is None]
    for src_hash, result in verify_parallel(jobs, verifier.verify, args.workers):
        if result.valid:
            journal.append({"src_hash": src_hash, "status": "valid"})
            continue
        reason = "baseline_failed" if result.diagnostic else "baseline_failed_without_structured_state"
        journal.append({
            "src_hash": src_hash,
            "status": "rejected",
            "reason": reason,
            "diagnostic": asdict(result.diagnostic) if result.diagnostic else None,
        })
        record_rejection(rejections, "baseline:" + src_hash, "baseline", reason)


def generation_phase(
    rows: Sequence[dict[str, Any]],
    adapters: Sequence[MutationAdapter],
    args: argparse.Namespace,
    baseline: JsonlJournal,
    generations: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    for job in make_jobs(
        rows,
        adapters,
        attempts_per_mutator=args.attempts_per_mutator,
        global_seed=args.seed,
    ):
        if generations.get(job.job_id) is not None:
            continue
        baseline_record = baseline.get(job.src_hash)
        if not baseline_record or baseline_record.get("status") != "valid":
            continue
        try:
            pair = job.run()
            if pair is None:
                raise ValueError("no_eligible_mutation")
        except ValueError as error:
            rejected: dict[str, Any] = {
                "job_id": job.job_id,
                "src_hash": job.src_hash,
                "mutator_id": job.adapter.mutator_id,
                "error_type": job.adapter.april_error_type,
                "attempt_index": job.attempt_index,
                "status": "rejected",
                "reason": str(error),
            }
            raw_response = getattr(error, "raw_response", None)
            if isinstance(raw_response, str):
                rejected["raw_response"] = raw_response
                rejected["response_sha256"] = sha256_text(raw_response)
            generations.append(rejected)
            record_rejection(rejections, job.job_id, "generation", str(error))
            continue
        generations.append({
            "job_id": job.job_id,
            "src_hash": job.src_hash,
            "mutator_id": job.adapter.mutator_id,
            "error_type": job.adapter.april_error_type,
            "attempt_index": job.attempt_index,
            "status": "generated",
            "pair": pair,
            "incorrect_sha256": sha256_text(pair["incorrect_proof"]),
        })


def verification_phase(
    args: argparse.Namespace,
    verifier: VerifierPool,
    generations: JsonlJournal,
    verifications: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    generated = sorted(
        (record for record in generations.records.values() if record.get("status") == "generated"),
        key=lambda record: record["job_id"],
    )
    owner_by_hash: dict[str, str] = {}
    jobs: list[tuple[str, str]] = []
    for record in generated:
        key = record["job_id"]
        owner = owner_by_hash.setdefault(record["incorrect_sha256"], key)
        if verifications.get(key) is not None:
            continue
        if owner != key:
            verifications.append({
                "job_id": key,
                "status": "rejected",
                "reason": "duplicate_incorrect_proof",
                "duplicate_of": owner,
            })
            record_rejection(rejections, key, "verification", "duplicate_incorrect_proof")
            continue
        jobs.append((key, record["pair"]["incorrect_proof"]))
    for key, result in verify_parallel(jobs, verifier.verify, args.workers):
        if result.valid:
            verifications.append({"job_id": key, "status": "rejected", "reason": "mutation_compiles"})
            record_rejection(rejections, key, "verification", "mutation_compiles")
        elif result.diagnostic is None:
            verifications.append({"job_id": key, "status": "rejected", "reason": "missing_structured_proof_state"})
            record_rejection(rejections, key, "verification", "missing_structured_proof_state")
        else:
            verifications.append({
                "job_id": key,
                "status": "accepted_failure",
                "diagnostic": asdict(result.diagnostic),
            })


def messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def annotation_phase(
    args: argparse.Namespace,
    model: Any,
    generations: JsonlJournal,
    verifications: JsonlJournal,
    annotations: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    for key, verified in sorted(verifications.records.items()):
        existing = annotations.get(key)
        if verified.get("status") != "accepted_failure" or (
            existing is not None and existing.get("status") in {"annotated", "rejected"}
        ):
            continue
        generated = generations.get(key)
        assert generated is not None
        diagnostic = verified["diagnostic"]
        user = annotation_user_prompt(
            generated["pair"]["incorrect_proof"],
            diagnostic["state_at_error"],
            diagnostic["line_at_error"],
            diagnostic["error"],
        )
        raw_responses = list(existing.get("raw_responses", [])) if existing else []
        parsed: dict[str, str] | None = None
        parse_error = str(existing.get("reason", "")) if existing else ""
        while len(raw_responses) < args.annotation_retries + 1:
            response = model.complete(messages(ANNOTATION_SYSTEM_PROMPT, user), args.annotation_max_tokens)
            raw_responses.append(response)
            try:
                parsed = parse_annotation(response)
                break
            except ValueError as error:
                parse_error = str(error)
                annotations.append({
                    "job_id": key,
                    "status": "retrying",
                    "reason": parse_error,
                    "raw_responses": raw_responses,
                })
        if parsed is None:
            annotations.append({
                "job_id": key,
                "status": "rejected",
                "reason": parse_error,
                "raw_responses": raw_responses,
            })
            record_rejection(rejections, key, "annotation", parse_error)
            continue
        annotations.append({
            "job_id": key,
            "status": "annotated",
            **parsed,
            "raw_responses": raw_responses,
        })


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def mutator_manifest(mutator_id: str) -> dict[str, Any]:
    descriptor = MUTATOR_DESCRIPTORS[mutator_id]
    values = dict(manifest_metadata(mutator_id))
    template = values.pop("prompt_template", None)
    system = values.pop("system_prompt", None)
    if isinstance(template, str) and isinstance(system, str):
        values["prompt_sha256"] = sha256_text(system + "\0" + template)
    return {
        "april_error_type": descriptor["april_error_type"],
        "requires_model": descriptor["requires_model"],
        **values,
    }


def export_outputs(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    generations: JsonlJournal,
    verifications: JsonlJournal,
    annotations: JsonlJournal,
    baseline: JsonlJournal,
    rejections: JsonlJournal,
) -> dict[str, Any]:
    originals = {row["src_hash"]: row for row in rows}
    configured_types = sorted({MUTATOR_DESCRIPTORS[name]["april_error_type"] for name in args.mutators})
    outputs: dict[str, list[dict[str, Any]]] = {error_type: [] for error_type in configured_types}
    accepted_by_mutator = {name: 0 for name in args.mutators}
    for key, generated in generations.records.items():
        verified, annotated = verifications.get(key), annotations.get(key)
        if (
            generated.get("status") != "generated"
            or not verified or verified.get("status") != "accepted_failure"
            or not annotated or annotated.get("status") != "annotated"
        ):
            continue
        original = originals.get(generated["src_hash"])
        if original is None:
            continue
        error_type = generated["error_type"]
        outputs.setdefault(error_type, []).append(format_april_record(
            original,
            generated["pair"],
            error_type=error_type,
            verification=verified,
            annotation=annotated,
        ))
        accepted_by_mutator[generated["mutator_id"]] += 1
    for values in outputs.values():
        values.sort(key=lambda value: (value["src_hash"], sha256_text(value["incorrect_proof"])))
    output_hashes = {
        error_type: atomic_jsonl(args.output_dir / f"{error_type}.jsonl", values)
        for error_type, values in outputs.items()
    }
    counts: dict[str, Any] = {
        "input_records": len(rows),
        "baseline_valid": sum(record.get("status") == "valid" for record in baseline.records.values()),
        "baseline_rejected": sum(record.get("status") == "rejected" for record in baseline.records.values()),
        "generated": sum(record.get("status") == "generated" for record in generations.records.values()),
        "generation_rejected": sum(record.get("status") == "rejected" for record in generations.records.values()),
        "accepted_failures": sum(record.get("status") == "accepted_failure" for record in verifications.records.values()),
        "verification_rejected": sum(record.get("status") == "rejected" for record in verifications.records.values()),
        "annotated": sum(record.get("status") == "annotated" for record in annotations.records.values()),
        "annotation_rejected": sum(record.get("status") == "rejected" for record in annotations.records.values()),
        "accepted_by_mutator": accepted_by_mutator,
        "accepted_by_error_type": {key: len(value) for key, value in outputs.items()},
    }
    counts["by_mutator"] = {}
    for mutator_id in args.mutators:
        generated_for_mutator = [
            record
            for record in generations.records.values()
            if record.get("mutator_id") == mutator_id
        ]
        job_ids = {record["job_id"] for record in generated_for_mutator}
        verified_for_mutator = [
            verifications.get(key) for key in job_ids if verifications.get(key) is not None
        ]
        annotated_for_mutator = [
            annotations.get(key) for key in job_ids if annotations.get(key) is not None
        ]
        counts["by_mutator"][mutator_id] = {
            "generated": sum(record.get("status") == "generated" for record in generated_for_mutator),
            "generation_rejected": sum(
                record.get("status") == "rejected" for record in generated_for_mutator
            ),
            "accepted_failures": sum(
                record and record.get("status") == "accepted_failure"
                for record in verified_for_mutator
            ),
            "verification_rejected": sum(
                record and record.get("status") == "rejected"
                for record in verified_for_mutator
            ),
            "annotated": sum(
                record and record.get("status") == "annotated"
                for record in annotated_for_mutator
            ),
            "annotation_rejected": sum(
                record and record.get("status") == "rejected"
                for record in annotated_for_mutator
            ),
            "exported": accepted_by_mutator[mutator_id],
        }
    rejection_counts: dict[str, int] = {}
    for rejection in rejections.records.values():
        label = f"{rejection.get('phase')}:{rejection.get('reason')}"
        rejection_counts[label] = rejection_counts.get(label, 0) + 1
    loaded_model_metadata = model_metadata(args)
    annotation_template = annotation_user_prompt(
        "{incorrect_proof}",
        "{state_at_error}",
        "{line_at_error}",
        "{error}",
    )
    manifest = {
        "protocol": "openai-math-generic-mutations-v2",
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "input_records": len(rows),
        "selected_theorems": sorted(str(row["theorem"]) for row in rows),
        "source_commits": sorted({str(row.get("source_commit")) for row in rows}),
        "source_file_sha256": sorted({str(row.get("file_sha256")) for row in rows}),
        "lean_toolchains": sorted({str(row.get("lean_toolchain")) for row in rows}),
        "verifier_fingerprints": sorted({str(row.get("verifier_fingerprint")) for row in rows}),
        "project_dir": str(args.project_dir),
        "verified_project": getattr(args, "verified_project", None),
        "lean_interact_version": package_version("lean-interact"),
        "openai_python_version": package_version("openai"),
        "transformers_version": package_version("transformers"),
        "torch_version": package_version("torch"),
        "model": loaded_model_metadata,
        "paper_faithful_model": args.model == PAPER_MODEL,
        "seed": args.seed,
        "seed_derivation": SEED_DERIVATION_VERSION,
        "attempts_per_mutator": args.attempts_per_mutator,
        "mutators": {name: mutator_manifest(name) for name in args.mutators},
        "annotation_prompt": {
            "version": ANNOTATION_PROMPT_VERSION,
            "sha256": sha256_text(ANNOTATION_SYSTEM_PROMPT + "\0" + annotation_template),
        },
        "sampling": {
            "temperature": None,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "annotation_max_tokens": args.annotation_max_tokens,
            "annotation_retries": args.annotation_retries,
        },
        "verification": {"workers": args.workers, "timeout_seconds": args.lean_timeout},
        "output_sha256": output_hashes,
        "counts": counts,
        "rejections_by_phase_and_reason": dict(sorted(rejection_counts.items())),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_openai_math_input(REPO_ROOT))
    parser.add_argument("--project-dir", type=Path, default=Path("external/openai-math/lean"))
    parser.add_argument("--output-dir", type=Path, default=Path("contributions/openai-math-line-mutations"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/openai_math_mutations"))
    parser.add_argument(
        "--mutators",
        nargs="+",
        choices=tuple(MUTATOR_DESCRIPTORS),
        default=DEFAULT_MUTATORS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts-per-mutator", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Pilot limit applied before verification or model calls")
    parser.add_argument("--theorem", action="append", default=[], help="Only process this theorem; repeatable")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Lean-Interact servers")
    parser.add_argument("--lean-timeout", type=float, default=120.0)
    parser.add_argument("--model-provider", choices=("openai", "huggingface"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--hf-device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint; overrides --base-url-env")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--annotation-max-tokens", type=int, default=512)
    parser.add_argument("--annotation-retries", type=int, default=2)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.model is None:
        args.model = DEFAULT_HUGGINGFACE_MODEL if args.model_provider == "huggingface" else DEFAULT_OPENAI_MODEL
    if args.attempts_per_mutator < 1:
        raise ValueError("--attempts-per-mutator must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if len(set(args.mutators)) != len(args.mutators):
        raise ValueError("--mutators may not contain duplicates")
    if not args.input.is_file():
        raise ValueError(f"input does not exist: {args.input}")
    if not args.project_dir.is_dir():
        raise ValueError(f"OpenAI Math Lean project does not exist: {args.project_dir}")


def run_pipeline(
    args: argparse.Namespace,
    *,
    model: Any | None = None,
    verifier_pool: VerifierPool | None = None,
    adapters: Sequence[MutationAdapter] | None = None,
) -> dict[str, Any]:
    validate_args(args)
    args.input = args.input.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    rows = OpenAIMathDataset(args.input).load(limit=args.limit, theorems=args.theorem)
    using_real_verifier = verifier_pool is None
    args.verified_project = verify_project_provenance(rows, args.project_dir) if using_real_verifier else None
    ensure_run_config(args, rows)

    def make_model() -> Any:
        if args.model_provider == "huggingface":
            local_model = HuggingFaceLocalModel(
                model=args.model,
                device=args.hf_device,
                revision=args.hf_revision,
            )
            atomic_json(args.work_dir / "model_metadata.json", {
                "provider": args.model_provider,
                "model": args.model,
                "endpoint": "local",
                "requested_revision": args.hf_revision,
                "resolved_revision": local_model.revision,
                "device": local_model.device,
            })
            return local_model
        atomic_json(args.work_dir / "model_metadata.json", {
            "provider": args.model_provider,
            "model": args.model,
            "endpoint": model_endpoint(args),
            "requested_revision": None,
            "resolved_revision": None,
            "device": None,
        })
        return OpenAIModel(
            model=args.model,
            base_url=endpoint_from_args(args),
            api_key_env=args.api_key_env,
            reasoning_effort=args.reasoning_effort,
            timeout=args.api_timeout,
            retries=args.api_retries,
        )

    if using_real_verifier:
        verifier_pool = VerifierPool(lambda: LeanInteractVerifier(args.project_dir, args.lean_timeout))
    assert verifier_pool is not None
    baseline = JsonlJournal(args.work_dir / "baseline.jsonl", "src_hash")
    generations = JsonlJournal(args.work_dir / "generations.jsonl", "job_id")
    verifications = JsonlJournal(args.work_dir / "verifications.jsonl", "job_id")
    annotations = JsonlJournal(args.work_dir / "annotations.jsonl", "job_id")
    rejections = JsonlJournal(args.work_dir / "rejections.jsonl", "job_id")

    baseline_phase(rows, args, verifier_pool, baseline, rejections)
    pending_ids = [key for key in expected_job_ids(rows, args) if generations.get(key) is None]
    generation_pending = bool(pending_ids) and any(
        baseline.get(row["src_hash"]) and baseline.get(row["src_hash"])["status"] == "valid"
        for row in rows
    )
    if generation_pending:
        if adapters is None:
            needs_model = any(MUTATOR_DESCRIPTORS[name]["requires_model"] for name in args.mutators)
            if needs_model:
                model = model or make_model()
            adapters = [build_adapter(name, model=model, max_tokens=args.max_tokens) for name in args.mutators]
        elif tuple(adapter.mutator_id for adapter in adapters) != tuple(args.mutators):
            raise ValueError("injected adapters must match --mutators in the same order")
        generation_phase(rows, adapters, args, baseline, generations, rejections)
    verification_phase(args, verifier_pool, generations, verifications, rejections)
    annotation_pending = any(
        verified.get("status") == "accepted_failure"
        and (annotations.get(key) is None or annotations.get(key).get("status") not in {"annotated", "rejected"})
        for key, verified in verifications.records.items()
    )
    if annotation_pending:
        model = model or make_model()
        annotation_phase(args, model, generations, verifications, annotations, rejections)
    return export_outputs(rows, args, generations, verifications, annotations, baseline, rejections)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        counts = run_pipeline(args)
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
