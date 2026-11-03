#!/usr/bin/env python3
"""Build verified APRIL-format line-mutation contributions from OpenAI Math."""
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
    from .april_formatter import (
        ANNOTATION_PROMPT_VERSION,
        ANNOTATION_SYSTEM_PROMPT,
        AprilProvenance,
        annotation_user_prompt,
        format_april_record,
        parse_annotation,
    )
    from .llm_client import DEFAULT_OPENAI_MODEL, PAPER_MODEL, OpenAIModel, sanitize_endpoint
    from .lean_verifier import LeanInteractVerifier, VerifierPool, verify_parallel
    from .mutate_line import (
        LME_PROMPT_VERSION,
        MLME_PROMPT_VERSION,
        SYSTEM_PROMPT,
        MutationError,
        lme_user_prompt,
        messages,
        mlme_user_prompt,
        mutate_proof,
    )
    from .mutation_utils import (
        JsonlJournal,
        atomic_json,
        atomic_jsonl,
        sha256_bytes,
        sha256_text,
    )
    from .proof_datasets import OpenAIMathDataset, default_openai_math_input
except ImportError:  # Support direct execution from contributions/scripts.
    from april_formatter import (
        ANNOTATION_PROMPT_VERSION,
        ANNOTATION_SYSTEM_PROMPT,
        AprilProvenance,
        annotation_user_prompt,
        format_april_record,
        parse_annotation,
    )
    from llm_client import DEFAULT_OPENAI_MODEL, PAPER_MODEL, OpenAIModel, sanitize_endpoint
    from lean_verifier import LeanInteractVerifier, VerifierPool, verify_parallel
    from mutate_line import (
        LME_PROMPT_VERSION,
        MLME_PROMPT_VERSION,
        SYSTEM_PROMPT,
        MutationError,
        lme_user_prompt,
        messages,
        mlme_user_prompt,
        mutate_proof,
    )
    from mutation_utils import (
        JsonlJournal,
        atomic_json,
        atomic_jsonl,
        sha256_bytes,
        sha256_text,
    )
    from proof_datasets import OpenAIMathDataset, default_openai_math_input


REPO_ROOT = Path(__file__).resolve().parents[2]


def mutation_job_id(src_hash: str, error_type: str, attempt_index: int) -> str:
    return f"{error_type}:{src_hash}:{attempt_index}"


def record_rejection(journal: JsonlJournal, key: str, phase: str, reason: str) -> None:
    if journal.get(key) is None:
        journal.append({"job_id": key, "phase": phase, "reason": reason})


def endpoint_from_args(args: argparse.Namespace) -> str | None:
    return args.base_url or os.environ.get(args.base_url_env)


def ensure_run_config(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse to combine journals produced by incompatible experiments."""
    config = {
        "protocol": "openai-math-line-mutations-v1",
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "selected_sources": sorted(row["src_hash"] for row in rows),
        "selected_theorems": sorted(row["theorem"] for row in rows),
        "project_dir": str(args.project_dir),
        "verified_project": getattr(args, "verified_project", None),
        "seed": args.seed,
        "mutation_types": sorted(args.mutation_types),
        "attempts_per_proof": args.attempts_per_proof,
        "model": args.model,
        "endpoint": sanitize_endpoint(endpoint_from_args(args)),
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "annotation_max_tokens": args.annotation_max_tokens,
        "annotation_retries": args.annotation_retries,
        "prompt_versions": [LME_PROMPT_VERSION, MLME_PROMPT_VERSION, ANNOTATION_PROMPT_VERSION],
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
    source_commits = {str(row.get("source_commit")) for row in rows}
    if source_commits != {commit}:
        raise ValueError(f"input source commit {sorted(source_commits)} does not match checkout commit {commit}")
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
    args: argparse.Namespace,
    model: Any,
    baseline: JsonlJournal,
    generations: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    for row in rows:
        src_hash = row["src_hash"]
        baseline_record = baseline.get(src_hash)
        if not baseline_record or baseline_record["status"] != "valid":
            continue
        for error_type in args.mutation_types:
            for attempt in range(args.attempts_per_proof):
                key = mutation_job_id(src_hash, error_type, attempt)
                if generations.get(key) is not None:
                    continue
                try:
                    pair = mutate_proof(
                        row,
                        model=model,
                        error_type=error_type,
                        attempt_index=attempt,
                        global_seed=args.seed,
                        max_tokens=args.max_tokens,
                    )
                except ValueError as error:
                    rejected: dict[str, Any] = {"job_id": key, "status": "rejected", "reason": str(error)}
                    if isinstance(error, MutationError) and error.raw_response is not None:
                        rejected.update({
                            "raw_response": error.raw_response,
                            "response_sha256": sha256_text(error.raw_response),
                            "prompt_version": error.prompt_version,
                            "user_prompt_sha256": error.prompt_sha256,
                        })
                    generations.append(rejected)
                    record_rejection(rejections, key, "generation", str(error))
                    continue
                generated = pair.to_dict()
                generated.update({
                    "job_id": key,
                    "status": "generated",
                    "incorrect_sha256": sha256_text(pair.incorrect_proof),
                    "user_prompt_sha256": generated.pop("prompt_sha256"),
                })
                generations.append(generated)


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
                "job_id": key, "status": "rejected", "reason": "duplicate_incorrect_proof", "duplicate_of": owner,
            })
            record_rejection(rejections, key, "verification", "duplicate_incorrect_proof")
            continue
        jobs.append((key, record["incorrect_proof"]))
    for key, result in verify_parallel(jobs, verifier.verify, args.workers):
        if result.valid:
            verifications.append({"job_id": key, "status": "rejected", "reason": "mutation_compiles"})
            record_rejection(rejections, key, "verification", "mutation_compiles")
        elif result.diagnostic is None:
            verifications.append({"job_id": key, "status": "rejected", "reason": "missing_structured_proof_state"})
            record_rejection(rejections, key, "verification", "missing_structured_proof_state")
        else:
            verifications.append({"job_id": key, "status": "accepted_failure", "diagnostic": asdict(result.diagnostic)})


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
            generated["incorrect_proof"], diagnostic["state_at_error"], diagnostic["error"]
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
                    "job_id": key, "status": "retrying", "reason": parse_error, "raw_responses": raw_responses,
                })
        if parsed is None:
            annotations.append({
                "job_id": key, "status": "rejected", "reason": parse_error, "raw_responses": raw_responses,
            })
            record_rejection(rejections, key, "annotation", parse_error)
            continue
        annotations.append({
            "job_id": key,
            "status": "annotated",
            **parsed,
            "raw_responses": raw_responses,
            "response_sha256": sha256_text(raw_responses[-1]),
            "prompt_version": ANNOTATION_PROMPT_VERSION,
            "system_prompt": ANNOTATION_SYSTEM_PROMPT,
            "user_prompt_sha256": sha256_text(user),
            "attempts": len(raw_responses),
        })


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def export_outputs(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    generations: JsonlJournal,
    verifications: JsonlJournal,
    annotations: JsonlJournal,
    baseline: JsonlJournal,
    rejections: JsonlJournal,
) -> dict[str, int]:
    originals = {row["src_hash"]: row for row in rows}
    provenance = AprilProvenance(
        model=args.model,
        endpoint=sanitize_endpoint(endpoint_from_args(args)),
        attempts_per_proof=args.attempts_per_proof,
        max_tokens=args.max_tokens,
        annotation_max_tokens=args.annotation_max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    outputs: dict[str, list[dict[str, Any]]] = {"lme": [], "mlme": []}
    for key, generated in generations.records.items():
        verified, annotated = verifications.get(key), annotations.get(key)
        if (
            generated.get("status") != "generated"
            or not verified or verified.get("status") != "accepted_failure"
            or not annotated or annotated.get("status") != "annotated"
        ):
            continue
        original = originals.get(generated["src_hash"])
        if original is not None:
            outputs[generated["error_type"]].append(
                format_april_record(original, generated, verified, annotated, provenance)
            )
    for values in outputs.values():
        values.sort(key=lambda value: value["selected_id"])
    output_hashes = {
        error_type: atomic_jsonl(args.output_dir / f"{error_type}.jsonl", outputs[error_type])
        for error_type in ("lme", "mlme")
    }
    counts = {
        "input_records": len(rows),
        "baseline_valid": sum(record.get("status") == "valid" for record in baseline.records.values()),
        "baseline_rejected": sum(record.get("status") == "rejected" for record in baseline.records.values()),
        "lme_accepted": len(outputs["lme"]),
        "mlme_accepted": len(outputs["mlme"]),
        "generated": sum(record.get("status") == "generated" for record in generations.records.values()),
        "generation_rejected": sum(record.get("status") == "rejected" for record in generations.records.values()),
        "accepted_failures": sum(record.get("status") == "accepted_failure" for record in verifications.records.values()),
        "verification_rejected": sum(record.get("status") == "rejected" for record in verifications.records.values()),
        "annotated": sum(record.get("status") == "annotated" for record in annotations.records.values()),
        "annotation_rejected": sum(record.get("status") == "rejected" for record in annotations.records.values()),
    }
    rejection_counts: dict[str, int] = {}
    for rejection in rejections.records.values():
        label = f"{rejection.get('phase')}:{rejection.get('reason')}"
        rejection_counts[label] = rejection_counts.get(label, 0) + 1
    lme_template = lme_user_prompt("{broken_proof}")
    mlme_template = mlme_user_prompt("{broken_proof}")
    annotation_template = annotation_user_prompt("{incorrect_proof}", "{state_at_error}", "{error}")
    manifest = {
        "protocol": "openai-math-line-mutations-v1",
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "input_records": len(rows),
        "selected_theorems": sorted(row["theorem"] for row in rows),
        "source_commits": sorted({str(row.get("source_commit")) for row in rows}),
        "source_file_sha256": sorted({str(row.get("file_sha256")) for row in rows}),
        "source_dirty_values": sorted({bool(row.get("source_dirty")) for row in rows}),
        "lean_toolchains": sorted({str(row.get("lean_toolchain")) for row in rows}),
        "verifier_fingerprints": sorted({str(row.get("verifier_fingerprint")) for row in rows}),
        "project_dir": str(args.project_dir),
        "verified_project": getattr(args, "verified_project", None),
        "lean_interact_version": package_version("lean-interact"),
        "openai_python_version": package_version("openai"),
        "model": args.model,
        "paper_faithful_model": args.model == PAPER_MODEL,
        "endpoint": sanitize_endpoint(endpoint_from_args(args)),
        "seed": args.seed,
        "attempts_per_proof": args.attempts_per_proof,
        "mutation_types": list(args.mutation_types),
        "protocol_deviations": provenance.protocol_deviations,
        "sampling": {
            "temperature": None,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "annotation_max_tokens": args.annotation_max_tokens,
            "annotation_retries": args.annotation_retries,
        },
        "verification": {"workers": args.workers, "timeout_seconds": args.lean_timeout},
        "prompts": {
            "lme": {
                "version": LME_PROMPT_VERSION, "system": SYSTEM_PROMPT, "template": lme_template,
                "sha256": sha256_text(SYSTEM_PROMPT + "\0" + lme_template),
            },
            "mlme": {
                "version": MLME_PROMPT_VERSION, "system": SYSTEM_PROMPT, "template": mlme_template,
                "sha256": sha256_text(SYSTEM_PROMPT + "\0" + mlme_template),
            },
            "annotation": {
                "version": ANNOTATION_PROMPT_VERSION, "system": ANNOTATION_SYSTEM_PROMPT,
                "template": annotation_template,
                "sha256": sha256_text(ANNOTATION_SYSTEM_PROMPT + "\0" + annotation_template),
            },
        },
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
    parser.add_argument("--work-dir", type=Path, default=Path("runs/openai_math_line_mutations"))
    parser.add_argument("--mutation-types", nargs="+", choices=("lme", "mlme"), default=("lme", "mlme"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts-per-proof", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Pilot limit applied before verification or API calls")
    parser.add_argument("--theorem", action="append", default=[], help="Only process this theorem; repeatable")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Lean-Interact servers")
    parser.add_argument("--lean-timeout", type=float, default=120.0)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint; overrides --base-url-env")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="medium"
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--annotation-max-tokens", type=int, default=512)
    parser.add_argument("--annotation-retries", type=int, default=2)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.attempts_per_proof < 1:
        raise ValueError("--attempts-per-proof must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.model.strip():
        raise ValueError("--model must be nonempty")
    if not args.input.is_file():
        raise ValueError(f"input does not exist: {args.input}")
    if not args.project_dir.is_dir():
        raise ValueError(f"OpenAI Math Lean project does not exist: {args.project_dir}")


def run_pipeline(
    args: argparse.Namespace,
    *,
    model: Any | None = None,
    verifier_pool: VerifierPool | None = None,
) -> dict[str, int]:
    validate_args(args)
    args.input = args.input.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    rows = OpenAIMathDataset(args.input).load(limit=args.limit, theorems=args.theorem)
    using_real_verifier = verifier_pool is None
    args.verified_project = verify_project_provenance(rows, args.project_dir) if using_real_verifier else None
    ensure_run_config(args, rows)

    def make_model() -> OpenAIModel:
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

    generation_requested = any(
        generations.get(mutation_job_id(row["src_hash"], error_type, attempt)) is None
        for row in rows
        for error_type in args.mutation_types
        for attempt in range(args.attempts_per_proof)
    )
    if generation_requested and model is None:
        model = make_model()  # Fail on missing credentials before expensive Lean startup.
    baseline_phase(rows, args, verifier_pool, baseline, rejections)
    generation_pending = any(
        baseline.get(row["src_hash"]) is not None
        and baseline.get(row["src_hash"])["status"] == "valid"
        and generations.get(mutation_job_id(row["src_hash"], error_type, attempt)) is None
        for row in rows
        for error_type in args.mutation_types
        for attempt in range(args.attempts_per_proof)
    )
    if generation_pending:
        assert model is not None
        generation_phase(rows, args, model, baseline, generations, rejections)
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
