#!/usr/bin/env python3
"""Build verified APRIL-format theorem-mutation contributions from OpenAI Math.

Phases per proof: baseline, occurrences, mutations, verification, annotation, export.
Each is journaled in --work-dir, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

try:
    from .april_formatter import (
        THEOREM_ANNOTATION_PROMPT_VERSION,
        THEOREM_ANNOTATION_SCHEMA,
        THEOREM_ANNOTATION_SYSTEM_PROMPT,
        TheoremProvenance,
        format_lean_errors,
        format_theorem_record,
        parse_theorem_annotation,
        theorem_annotation_user_prompt,
    )
    from .build_mutation_contributions import (
        endpoint_from_args,
        package_version,
        record_rejection,
        verify_project_provenance,
    )
    from .lean_verifier import CompileResult, LeanEnvSession, PrefixError, TacticSpan
    from .lemma_retrieval import (
        DEFAULT_PACKAGES,
        QUERY_STRATEGIES,
        CachedRetriever,
        LeanExploreLocalRetriever,
        LeanExploreRetriever,
        NameOverlapRetriever,
    )
    from .llm_client import DEFAULT_OPENAI_MODEL, OpenAIModel, sanitize_endpoint
    from .mutate_line import messages
    from .mutate_theorem import (
        ERROR_TYPE,
        OCCURRENCE_POLICIES,
        LeanConstant,
        LemmaOccurrence,
        NeighborFilter,
        TargetSpans,
        build_mutation,
        choose_occurrences,
        choose_substitute,
        classify_compile,
        declaration_text,
        filter_neighbors,
        identifier_tokens,
        lemma_card,
        normalize_proof,
        resolution_queries,
        select_occurrences,
        target_spans,
    )
    from .mutation_utils import JsonlJournal, atomic_json, atomic_jsonl, sha256_bytes, sha256_text
    from .proof_datasets import OpenAIMathDataset, default_openai_math_input
except ImportError:  # Support direct execution from contributions/scripts.
    from april_formatter import (
        THEOREM_ANNOTATION_PROMPT_VERSION,
        THEOREM_ANNOTATION_SCHEMA,
        THEOREM_ANNOTATION_SYSTEM_PROMPT,
        TheoremProvenance,
        format_lean_errors,
        format_theorem_record,
        parse_theorem_annotation,
        theorem_annotation_user_prompt,
    )
    from build_mutation_contributions import (
        endpoint_from_args,
        package_version,
        record_rejection,
        verify_project_provenance,
    )
    from lean_verifier import CompileResult, LeanEnvSession, PrefixError, TacticSpan
    from lemma_retrieval import (
        DEFAULT_PACKAGES,
        QUERY_STRATEGIES,
        CachedRetriever,
        LeanExploreLocalRetriever,
        LeanExploreRetriever,
        NameOverlapRetriever,
    )
    from llm_client import DEFAULT_OPENAI_MODEL, OpenAIModel, sanitize_endpoint
    from mutate_line import messages
    from mutate_theorem import (
        ERROR_TYPE,
        OCCURRENCE_POLICIES,
        LeanConstant,
        LemmaOccurrence,
        NeighborFilter,
        TargetSpans,
        build_mutation,
        choose_occurrences,
        choose_substitute,
        classify_compile,
        declaration_text,
        filter_neighbors,
        identifier_tokens,
        lemma_card,
        normalize_proof,
        resolution_queries,
        select_occurrences,
        target_spans,
    )
    from mutation_utils import JsonlJournal, atomic_json, atomic_jsonl, sha256_bytes, sha256_text
    from proof_datasets import OpenAIMathDataset, default_openai_math_input


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "openai-math-theorem-mutations-v1"
JOURNAL_FILES = (
    "baseline.jsonl", "occurrences.jsonl", "mutations.jsonl", "verifications.jsonl",
    "annotations.jsonl", "rejections.jsonl", "neighbors.jsonl",
)


class LockedJournal(JsonlJournal):
    def __init__(self, path: Path, key: str) -> None:
        super().__init__(path, key)
        self._lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return super().get(key)

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            super().append(record)


@dataclass
class Journals:
    baseline: LockedJournal
    occurrences: LockedJournal
    mutations: LockedJournal
    verifications: LockedJournal
    annotations: LockedJournal
    rejections: LockedJournal
    neighbors: LockedJournal


def open_journals(work_dir: Path) -> Journals:
    return Journals(
        baseline=LockedJournal(work_dir / "baseline.jsonl", "src_hash"),
        occurrences=LockedJournal(work_dir / "occurrences.jsonl", "src_hash"),
        mutations=LockedJournal(work_dir / "mutations.jsonl", "job_id"),
        verifications=LockedJournal(work_dir / "verifications.jsonl", "job_id"),
        annotations=LockedJournal(work_dir / "annotations.jsonl", "job_id"),
        rejections=LockedJournal(work_dir / "rejections.jsonl", "job_id"),
        neighbors=LockedJournal(work_dir / "neighbors.jsonl", "key"),
    )


class ThreadLocalSessions:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.local = threading.local()
        self.created: list[Any] = []
        self.lock = threading.Lock()

    def get(self) -> Any:
        session = getattr(self.local, "session", None)
        if session is None:
            session = self.factory()
            self.local.session = session
            with self.lock:
                self.created.append(session)
        return session

    def close(self) -> None:
        for session in self.created:
            restart = getattr(session, "restart", None)
            if callable(restart):
                restart()


def mutation_job_id(src_hash: str, occurrence_index: int, attempt: int) -> str:
    return f"{ERROR_TYPE}:{src_hash}:{occurrence_index}:{attempt}"


def lake_package_rev(project_dir: Path, package: str = "mathlib") -> str | None:
    path = project_dir / "lake-manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for entry in manifest.get("packages", []):
        if str(entry.get("name", "")).lower() == package:
            return entry.get("rev")
    return None


def read_toolchain(project_dir: Path) -> str | None:
    path = project_dir / "lean-toolchain"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


# ----------------------------------------------------------------------- Lean


def compile_variant(
    correct: str, variant: str, spans: TargetSpans, mode: str, session: Any, key: str
) -> CompileResult:
    """Compile the original or a mutation in the chosen mode."""
    if mode == "file":
        return session.compile_file(variant)
    delta = len(variant) - len(correct)
    return session.compile_declaration(
        key, correct[:spans.split], declaration_text(variant, spans, delta), spans.line_offset, variant
    )


def compile_summary(result: CompileResult) -> dict[str, Any]:
    value = result.to_dict()
    value["errors"] = value["errors"][:5]
    value["ok"] = result.ok
    return value


def resolve_names(
    row: Mapping[str, Any], spans: TargetSpans, mode: str, names: Sequence[str], session: Any
) -> dict[str, list[LeanConstant]]:
    source = row["correct_proof"]
    if mode == "env":
        raw = session.resolve(row["src_hash"], source[:spans.split], names)
    else:
        raw = session.resolve(row["src_hash"], source[:spans.declaration_end], names, inline=True)
    return {query: [LeanConstant.from_dict(match) for match in matches] for query, matches in raw.items()}


def baseline_check(row: Mapping[str, Any], spans: TargetSpans, args: argparse.Namespace, session: Any) -> dict[str, Any]:
    src_hash = row["src_hash"]
    if row.get("contains_sorry_or_admit"):
        return {"src_hash": src_hash, "status": "rejected", "reason": "target_contains_sorry_or_admit"}
    modes = ("env", "file") if args.compile_mode == "env" else ("file",)
    attempts: dict[str, Any] = {}
    result = CompileResult()
    for mode in modes:
        result = compile_variant(row["correct_proof"], row["correct_proof"], spans, mode, session, src_hash)
        attempts[mode] = compile_summary(result)
        if result.ok and not result.uses_sorry:
            return {
                "src_hash": src_hash,
                "status": "valid",
                "compile_mode": mode,
                "attempts": attempts,
                "tactics": [asdict(tactic) for tactic in result.tactics],
            }
    if result.timed_out:
        reason = "baseline_timeout"
    elif result.ok and result.uses_sorry:
        reason = "baseline_uses_sorry"
    else:
        reason = "baseline_failed"
    return {"src_hash": src_hash, "status": "rejected", "reason": reason, "attempts": attempts}


def occurrence_phase(
    row: Mapping[str, Any], spans: TargetSpans, baseline: Mapping[str, Any], args: argparse.Namespace, session: Any
) -> dict[str, Any]:
    src_hash = row["src_hash"]
    tokens = identifier_tokens(row, spans)
    try:
        resolutions = resolve_names(row, spans, baseline["compile_mode"], resolution_queries(tokens), session)
    except (TimeoutError, PrefixError) as error:
        session.restart()
        reason = "timeout" if isinstance(error, TimeoutError) else "repl_crash"
        return {"src_hash": src_hash, "status": "rejected", "reason": reason, "detail": str(error)[:300]}
    tactics = [
        TacticSpan(**{**tactic, "used_constants": tuple(tactic.get("used_constants", ()))})
        for tactic in baseline.get("tactics", [])
    ]
    occurrences, skipped = select_occurrences(
        row["correct_proof"], tokens, resolutions, tactics, packages=args.packages
    )
    return {
        "src_hash": src_hash,
        "status": "extracted",
        "tokens": len(tokens),
        "lemma_occurrences": len(occurrences),
        "skipped_tokens": skipped,
        "occurrences": [occurrence.to_dict() for occurrence in occurrences],
    }


def neighbor_phase(
    row: Mapping[str, Any],
    spans: TargetSpans,
    baseline: Mapping[str, Any],
    selected: Sequence[LemmaOccurrence],
    args: argparse.Namespace,
    session: Any,
    retriever: Any,
    journals: Journals,
) -> dict[int, NeighborFilter | str]:
    """Retrieve and filter neighbours; returns a NeighborFilter or a rejection reason per occurrence."""
    src_hash = row["src_hash"]
    pending = [o for o in selected if journals.mutations.get(mutation_job_id(src_hash, o.occurrence_index, 0)) is None]
    if not pending:
        return {}
    candidates: dict[int, list[str]] = {}
    for occurrence in pending:
        if args.force_substitute:
            candidates[occurrence.occurrence_index] = [args.force_substitute]
        else:
            found = retriever.neighbors(occurrence.lemma.name, occurrence.lemma.statement, args.neighbors_k)
            candidates[occurrence.occurrence_index] = [candidate.name for candidate in found]
    names = sorted({name for values in candidates.values() for name in values})
    resolutions: dict[str, list[LeanConstant]] = {}
    if names:
        try:
            resolutions = resolve_names(row, spans, baseline["compile_mode"], names, session)
        except (TimeoutError, PrefixError) as error:
            session.restart()
            reason = "timeout" if isinstance(error, TimeoutError) else "repl_crash"
            return {occurrence.occurrence_index: reason for occurrence in pending}
    outcome: dict[int, NeighborFilter | str] = {}
    for occurrence in pending:
        names_for = candidates[occurrence.occurrence_index]
        if not names_for:
            outcome[occurrence.occurrence_index] = "no_neighbors"
            continue
        outcome[occurrence.occurrence_index] = filter_neighbors(
            occurrence.lemma,
            names_for,
            resolutions,
            drop_namespace_variants=not args.keep_namespace_variants,
            drop_deprecated_aliases=not args.keep_deprecated_aliases,
            forced=bool(args.force_substitute),
        )
    return outcome


def rejected_generation(
    job_id: str,
    row: Mapping[str, Any],
    occurrence: LemmaOccurrence,
    attempt: int,
    reason: str,
    neighbor_filter: NeighborFilter | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "job_id": job_id,
        "src_hash": row["src_hash"],
        "status": "rejected",
        "reason": reason,
        "occurrence_index": occurrence.occurrence_index,
        "attempt_index": attempt,
        "original_lemma": lemma_card(occurrence.lemma, occurrence.token),
        "substitute_lemma": None,
    }
    if neighbor_filter is not None:
        record["neighbors_considered"] = list(neighbor_filter.considered)
        record["neighbors_dropped"] = [list(item) for item in neighbor_filter.dropped]
    return record


def generate_job(
    row: Mapping[str, Any],
    occurrence: LemmaOccurrence,
    attempt: int,
    filters: Mapping[int, NeighborFilter | str],
    tried: Sequence[str],
    args: argparse.Namespace,
    journals: Journals,
) -> dict[str, Any]:
    src_hash = row["src_hash"]
    job_id = mutation_job_id(src_hash, occurrence.occurrence_index, attempt)
    if attempt == 0:
        outcome = filters[occurrence.occurrence_index]
        if isinstance(outcome, str):
            return rejected_generation(job_id, row, occurrence, attempt, outcome)
        neighbor_filter = outcome
    else:
        first = journals.mutations.get(mutation_job_id(src_hash, occurrence.occurrence_index, 0))
        assert first is not None and "neighbor_filter" in first
        neighbor_filter = NeighborFilter.from_dict(first["neighbor_filter"])
    choice = choose_substitute(
        neighbor_filter.kept,
        global_seed=args.seed,
        src_hash=src_hash,
        occurrence_index=occurrence.occurrence_index,
        attempt=attempt,
        exclude=tried,
    )
    if choice is None:
        return rejected_generation(job_id, row, occurrence, attempt, "no_neighbors_after_filter", neighbor_filter)
    replacement, derived = choice
    mutation = build_mutation(
        row, occurrence, replacement, neighbor_filter, attempt=attempt, global_seed=args.seed, derived=derived
    )
    mutation.update({
        "job_id": job_id,
        "status": "generated",
        "neighbor_filter": neighbor_filter.to_dict(),
        "incorrect_sha256": sha256_text(mutation["incorrect_proof"]),
    })
    return mutation


def verify_job(
    row: Mapping[str, Any],
    spans: TargetSpans,
    mode: str,
    generated: Mapping[str, Any],
    accepted_texts: set[str],
    args: argparse.Namespace,
    session: Any,
) -> dict[str, Any]:
    base = {"job_id": generated["job_id"], "src_hash": row["src_hash"], "compile_mode": mode}
    if normalize_proof(generated["incorrect_proof"]) in accepted_texts:
        return {**base, "status": "rejected", "reason": "duplicate"}
    result = compile_variant(row["correct_proof"], generated["incorrect_proof"], spans, mode, session, row["src_hash"])
    reason = classify_compile(result, reject_unknown_identifier=not args.keep_unknown_identifier)
    if reason is not None:
        return {**base, "status": "rejected", "reason": reason, "compile": compile_summary(result)}
    diagnostic = result.diagnostic()
    assert diagnostic is not None and result.first_error is not None
    return {
        **base,
        "status": "accepted_failure",
        "errors": [asdict(error) for error in result.errors],
        "first_error": asdict(result.first_error),
        "diagnostic": asdict(diagnostic),
    }


def mutation_phase(
    row: Mapping[str, Any],
    spans: TargetSpans,
    baseline: Mapping[str, Any],
    selected: Sequence[LemmaOccurrence],
    filters: Mapping[int, NeighborFilter | str],
    args: argparse.Namespace,
    session: Any,
    journals: Journals,
) -> None:
    """Generate and verify each occurrence; retry while it still compiles."""
    src_hash = row["src_hash"]
    mode = baseline["compile_mode"]
    accepted_texts: set[str] = set()
    for occurrence in selected:
        tried: list[str] = []
        for attempt in range(args.retry_if_compiles + 1):
            job_id = mutation_job_id(src_hash, occurrence.occurrence_index, attempt)
            generated = journals.mutations.get(job_id)
            if generated is None:
                generated = generate_job(row, occurrence, attempt, filters, tried, args, journals)
                journals.mutations.append(generated)
                if generated["status"] != "generated":
                    record_rejection(journals.rejections, job_id, "generation", generated["reason"])
            if generated["status"] != "generated":
                break
            tried.append(generated["substitute_lemma"]["resolved_name"])
            verified = journals.verifications.get(job_id)
            if verified is None:
                verified = verify_job(row, spans, mode, generated, accepted_texts, args, session)
                journals.verifications.append(verified)
                if verified["status"] != "accepted_failure":
                    record_rejection(journals.rejections, job_id, "verification", verified["reason"])
            if verified["status"] == "accepted_failure":
                accepted_texts.add(normalize_proof(generated["incorrect_proof"]))
                break
            if verified.get("reason") != "still_compiles":
                break


def process_proof(
    row: Mapping[str, Any], args: argparse.Namespace, session: Any, retriever: Any, journals: Journals
) -> None:
    src_hash = row["src_hash"]
    session.begin_proof(src_hash)
    try:
        spans = target_spans(row)
    except ValueError as error:
        if journals.baseline.get(src_hash) is None:
            journals.baseline.append({
                "src_hash": src_hash, "status": "rejected", "reason": "invalid_record", "detail": str(error),
            })
            record_rejection(journals.rejections, "baseline:" + src_hash, "baseline", "invalid_record")
        return
    baseline = journals.baseline.get(src_hash)
    if baseline is None:
        baseline = baseline_check(row, spans, args, session)
        journals.baseline.append(baseline)
        if baseline["status"] != "valid":
            record_rejection(journals.rejections, "baseline:" + src_hash, "baseline", baseline["reason"])
    if baseline["status"] != "valid":
        return
    found = journals.occurrences.get(src_hash)
    if found is None:
        found = occurrence_phase(row, spans, baseline, args, session)
        journals.occurrences.append(found)
        if found["status"] != "extracted":
            record_rejection(journals.rejections, "occurrences:" + src_hash, "occurrences", found["reason"])
    if found["status"] != "extracted":
        return
    occurrences = [LemmaOccurrence.from_dict(value) for value in found["occurrences"]]
    selected = choose_occurrences(
        occurrences,
        src_hash=src_hash,
        global_seed=args.seed,
        policy=args.occurrence_policy,
        max_candidates=args.max_candidates_per_proof,
        lemma=args.lemma,
    )
    filters = neighbor_phase(row, spans, baseline, selected, args, session, retriever, journals)
    mutation_phase(row, spans, baseline, selected, filters, args, session, journals)


# ----------------------------------------------------------------- annotation


def request_annotation(model: Any, prompt: list[dict[str, str]], args: argparse.Namespace, state: dict[str, bool]) -> str:
    if state["structured"]:
        try:
            return model.complete_json(prompt, THEOREM_ANNOTATION_SCHEMA, args.annotation_max_tokens, name="thm_annotation")
        except RuntimeError as error:
            if "status=400" not in str(error):
                raise
            state["structured"] = False
            print("warning: structured output rejected (HTTP 400); using plain completions", file=sys.stderr)
    return model.complete(prompt, args.annotation_max_tokens)


def annotation_phase(args: argparse.Namespace, model: Any, journals: Journals) -> None:
    state = {"structured": hasattr(model, "complete_json") and not args.no_structured_output}
    for key, verified in sorted(journals.verifications.records.items()):
        existing = journals.annotations.get(key)
        if verified.get("status") != "accepted_failure" or (
            existing is not None and existing.get("status") in {"annotated", "rejected"}
        ):
            continue
        generated = journals.mutations.get(key)
        assert generated is not None
        original = generated["original_lemma"]
        replacement = generated["substitute_lemma"]
        user = theorem_annotation_user_prompt(
            mutated_code=generated["incorrect_proof"],
            errors_formatted=format_lean_errors(verified["errors"]),
            goal_state=verified["diagnostic"]["state_at_error"],
            substitute_name=replacement["name"],
            substitute_statement=replacement["statement"],
            original_name=original["name"],
            original_statement=original["statement"],
            line=generated["mutation_site"]["line"],
        )
        prompt = messages(THEOREM_ANNOTATION_SYSTEM_PROMPT, user)
        raw_responses = list(existing.get("raw_responses", [])) if existing else []
        parsed: dict[str, str] | None = None
        parse_error = str(existing.get("reason", "")) if existing else ""
        while len(raw_responses) < args.annotation_retries + 1:
            response = request_annotation(model, prompt, args, state)
            raw_responses.append(response)
            try:
                parsed = parse_theorem_annotation(response, replacement["name"], original["name"])
                break
            except ValueError as error:
                parse_error = str(error)
                journals.annotations.append({
                    "job_id": key, "status": "retrying", "reason": parse_error, "raw_responses": raw_responses,
                })
        if parsed is None:
            journals.annotations.append({
                "job_id": key, "status": "rejected", "reason": "explanation_invalid", "detail": parse_error,
                "raw_responses": raw_responses,
            })
            record_rejection(journals.rejections, key, "annotation", "explanation_invalid")
            continue
        journals.annotations.append({
            "job_id": key,
            "status": "annotated",
            **parsed,
            "raw_responses": raw_responses,
            "response_sha256": sha256_text(raw_responses[-1]),
            "prompt_version": THEOREM_ANNOTATION_PROMPT_VERSION,
            "system_prompt": THEOREM_ANNOTATION_SYSTEM_PROMPT,
            "user_prompt_sha256": sha256_text(user),
            "structured_output": state["structured"],
            "attempts": len(raw_responses),
        })


# --------------------------------------------------------------------- export


def provenance_for(args: argparse.Namespace, retriever: Any) -> TheoremProvenance:
    explain = not args.no_explain
    return TheoremProvenance(
        annotation_model=args.model if explain else None,
        endpoint=sanitize_endpoint(endpoint_from_args(args)) if explain else None,
        reasoning_effort=args.reasoning_effort if explain else None,
        annotation_max_tokens=args.annotation_max_tokens if explain else None,
        retriever_backend=getattr(retriever, "backend_id", args.retriever),
        retriever_version=getattr(retriever, "version", None),
        query_strategy=args.query_strategy,
        neighbors_k=args.neighbors_k,
        retry_if_compiles=args.retry_if_compiles,
        lean_toolchain=read_toolchain(args.project_dir),
        mathlib_rev=lake_package_rev(args.project_dir),
    )


def status_count(journal: JsonlJournal, status: str, *, reason: str | None = None) -> int:
    return sum(
        record.get("status") == status and (reason is None or record.get("reason") == reason)
        for record in journal.records.values()
    )


def export_outputs(
    rows: Sequence[Mapping[str, Any]], args: argparse.Namespace, journals: Journals, retriever: Any
) -> dict[str, int]:
    originals = {row["src_hash"]: row for row in rows}
    provenance = provenance_for(args, retriever)
    output: list[dict[str, Any]] = []
    for key, generated in journals.mutations.records.items():
        verified = journals.verifications.get(key)
        if generated.get("status") != "generated" or not verified or verified.get("status") != "accepted_failure":
            continue
        annotated = None
        if not args.no_explain:
            annotated = journals.annotations.get(key)
            if not annotated or annotated.get("status") != "annotated":
                continue
        original = originals.get(generated["src_hash"])
        if original is not None:
            output.append(format_theorem_record(original, generated, verified, annotated, provenance))
    output.sort(key=lambda record: record["selected_id"])
    output_sha256 = atomic_jsonl(args.output_dir / "thm.jsonl", output)

    occurrence_records = list(journals.occurrences.records.values())
    counts = {
        "input_records": len(rows),
        "baseline_valid": status_count(journals.baseline, "valid"),
        "baseline_rejected": status_count(journals.baseline, "rejected"),
        "baseline_file_mode": sum(
            record.get("compile_mode") == "file" for record in journals.baseline.records.values()
        ),
        "lemma_occurrences": sum(int(record.get("lemma_occurrences", 0)) for record in occurrence_records),
        "proofs_without_lemmas": sum(
            record.get("status") == "extracted" and not record.get("lemma_occurrences")
            for record in occurrence_records
        ),
        "attempted": len(journals.mutations.records),
        "generated": status_count(journals.mutations, "generated"),
        "generation_rejected": status_count(journals.mutations, "rejected"),
        "accepted_failures": status_count(journals.verifications, "accepted_failure"),
        "verification_rejected": status_count(journals.verifications, "rejected"),
        "missing_goal_state": status_count(journals.verifications, "rejected", reason="missing_structured_proof_state"),
        "annotated": status_count(journals.annotations, "annotated"),
        "annotation_rejected": status_count(journals.annotations, "rejected"),
        "kept": len(output),
    }
    rejection_counts: dict[str, int] = {}
    for rejection in journals.rejections.records.values():
        label = f"{rejection.get('phase')}:{rejection.get('reason')}"
        rejection_counts[label] = rejection_counts.get(label, 0) + 1
    annotation_template = theorem_annotation_user_prompt(
        mutated_code="{mutated_code}", errors_formatted="{errors_formatted}", goal_state="{goal_state}",
        substitute_name="{substitute_name}", substitute_statement="{substitute_statement}",
        original_name="{original_name}", original_statement="{original_statement}", line=0,
    )
    manifest = {
        "protocol": PROTOCOL,
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "input_records": len(rows),
        "selected_theorems": sorted(row["theorem"] for row in rows),
        "source_commits": sorted({str(row.get("source_commit")) for row in rows}),
        "source_file_sha256": sorted({str(row.get("file_sha256")) for row in rows}),
        "source_dirty_values": sorted({bool(row.get("source_dirty")) for row in rows}),
        "lean_toolchains": sorted({str(row.get("lean_toolchain")) for row in rows}),
        "mathlib_rev": provenance.mathlib_rev,
        "verifier_fingerprints": sorted({str(row.get("verifier_fingerprint")) for row in rows}),
        "project_dir": str(args.project_dir),
        "verified_project": getattr(args, "verified_project", None),
        "lean_interact_version": package_version("lean-interact"),
        "lean_explore_version": package_version("lean-explore"),
        "openai_python_version": package_version("openai"),
        "seed": args.seed,
        "settings": {
            "compile_mode": args.compile_mode,
            "workers": args.workers,
            "lean_timeout_seconds": args.lean_timeout,
            "occurrence_policy": args.occurrence_policy,
            "max_candidates_per_proof": args.max_candidates_per_proof,
            "lemma": args.lemma,
            "force_substitute": args.force_substitute,
            "retry_if_compiles": args.retry_if_compiles,
            "reject_unknown_identifier": not args.keep_unknown_identifier,
            "drop_namespace_variants": not args.keep_namespace_variants,
            "drop_deprecated_aliases": not args.keep_deprecated_aliases,
            "library_packages": list(args.packages),
        },
        "retrieval": {
            "backend": provenance.retriever_backend,
            "version": provenance.retriever_version,
            "query_strategy": args.query_strategy,
            "k": args.neighbors_k,
        },
        "annotation": None if args.no_explain else {
            "model": args.model,
            "endpoint": provenance.endpoint,
            "temperature": None,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.annotation_max_tokens,
            "retries": args.annotation_retries,
            "version": THEOREM_ANNOTATION_PROMPT_VERSION,
            "system": THEOREM_ANNOTATION_SYSTEM_PROMPT,
            "template": annotation_template,
            "sha256": sha256_text(THEOREM_ANNOTATION_SYSTEM_PROMPT + "\0" + annotation_template),
        },
        "protocol_deviations": provenance.protocol_deviations,
        "output_sha256": {"thm": output_sha256},
        "counts": counts,
        "rejections_by_phase_and_reason": dict(sorted(rejection_counts.items())),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    return counts


# ------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=default_openai_math_input(REPO_ROOT))
    parser.add_argument("--project-dir", type=Path, default=Path("external/openai-math/lean"))
    parser.add_argument("--output-dir", type=Path, default=Path("contributions/openai-math-theorem-mutations"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/openai_math_theorem_mutations"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, help="Pilot limit applied before any Lean or API work")
    parser.add_argument("--theorem", action="append", default=[], help="Only process this theorem; repeatable")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Lean servers (several GB RAM each)")
    parser.add_argument("--lean-timeout", type=float, default=120.0, help="Seconds per declaration compile")
    parser.add_argument("--prefix-timeout", type=float, help="Seconds for the per-proof prefix (default 4x --lean-timeout)")
    parser.add_argument(
        "--compile-mode", choices=("env", "file"), default="env",
        help="env: elaborate the text before the declaration once (falls back to file per proof); file: whole file each time",
    )
    # Retrieval.
    parser.add_argument(
        "--retriever", choices=("leanexplore", "leanexplore-local", "name-overlap"), default="leanexplore"
    )
    parser.add_argument("--neighbors-k", type=int, default=5)
    parser.add_argument("--query-strategy", choices=QUERY_STRATEGIES, default="name")
    parser.add_argument("--packages", nargs="+", default=list(DEFAULT_PACKAGES),
                        help="Library packages whose lemmas may be mutated and retrieved")
    parser.add_argument("--leanexplore-timeout", type=float, default=30.0)
    parser.add_argument("--declarations-file", type=Path,
                        help="name-overlap only: one theorem name per line (exported from Lean when missing)")
    parser.add_argument("--declarations-header", default="import Mathlib",
                        help="name-overlap only: Lean header used to export the declarations file")
    # Occurrences and substitution.
    parser.add_argument("--occurrence-policy", choices=OCCURRENCE_POLICIES, default="each")
    parser.add_argument("--max-candidates-per-proof", type=int)
    parser.add_argument("--lemma", help="Only mutate occurrences of this lemma")
    parser.add_argument("--force-substitute", help="Debugging: skip retrieval and use this substitute (needs --lemma)")
    parser.add_argument("--retry-if-compiles", type=int, default=0,
                        help="Try up to this many other candidates when a mutation still compiles (paper: 0)")
    parser.add_argument("--keep-unknown-identifier", action="store_true",
                        help="Keep mutations whose only errors are unknown identifiers")
    parser.add_argument("--keep-namespace-variants", action="store_true")
    parser.add_argument("--keep-deprecated-aliases", action="store_true")
    # Explanations.
    parser.add_argument("--no-explain", action="store_true", help="Skip the OpenAI step; export explanation=null")
    parser.add_argument("--no-structured-output", action="store_true", help="Use plain completions instead of json_schema")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint; overrides --base-url-env")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="medium"
    )
    parser.add_argument("--annotation-max-tokens", type=int, default=2048,
                        help="Includes reasoning tokens on reasoning models")
    parser.add_argument("--annotation-retries", type=int, default=1)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name, minimum in (("workers", 1), ("neighbors_k", 1), ("retry_if_compiles", 0), ("annotation_retries", 0)):
        if getattr(args, name) < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be at least {minimum}")
    for name in ("limit", "max_candidates_per_proof"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.force_substitute and not args.lemma:
        raise ValueError("--force-substitute requires --lemma")
    if not args.no_explain and not args.model.strip():
        raise ValueError("--model must be nonempty")
    if not args.input.is_file():
        raise ValueError(f"input does not exist: {args.input}")
    if not args.project_dir.is_dir():
        raise ValueError(f"OpenAI Math Lean project does not exist: {args.project_dir}")


def ensure_run_config(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse to combine journals from incompatible settings."""
    config = {
        "protocol": PROTOCOL,
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "selected_sources": sorted(row["src_hash"] for row in rows),
        "selected_theorems": sorted(row["theorem"] for row in rows),
        "project_dir": str(args.project_dir),
        "verified_project": getattr(args, "verified_project", None),
        "seed": args.seed,
        "compile_mode": args.compile_mode,
        "retriever": args.retriever,
        "neighbors_k": args.neighbors_k,
        "query_strategy": args.query_strategy,
        "packages": list(args.packages),
        "occurrence_policy": args.occurrence_policy,
        "max_candidates_per_proof": args.max_candidates_per_proof,
        "lemma": args.lemma,
        "force_substitute": args.force_substitute,
        "retry_if_compiles": args.retry_if_compiles,
        "keep_unknown_identifier": args.keep_unknown_identifier,
        "keep_namespace_variants": args.keep_namespace_variants,
        "keep_deprecated_aliases": args.keep_deprecated_aliases,
        "model": args.model,
        "endpoint": sanitize_endpoint(endpoint_from_args(args)),
        "reasoning_effort": args.reasoning_effort,
        "annotation_max_tokens": args.annotation_max_tokens,
        "annotation_retries": args.annotation_retries,
        "prompt_version": THEOREM_ANNOTATION_PROMPT_VERSION,
    }
    path = args.work_dir / "run_config.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"work directory has incompatible settings: {path}; choose a new --work-dir")
        return
    if any((args.work_dir / name).exists() for name in JOURNAL_FILES):
        raise ValueError(f"journals exist without {path}; choose a new --work-dir")
    atomic_json(path, config)


def make_model(args: argparse.Namespace) -> OpenAIModel:
    return OpenAIModel(
        model=args.model,
        base_url=endpoint_from_args(args),
        api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort,
        timeout=args.api_timeout,
        retries=args.api_retries,
    )


def make_retriever(args: argparse.Namespace, session_factory: Callable[[], Any]) -> Any:
    if args.retriever == "leanexplore":
        return LeanExploreRetriever(
            packages=args.packages, query_strategy=args.query_strategy, timeout=args.leanexplore_timeout
        )
    if args.retriever == "leanexplore-local":
        return LeanExploreLocalRetriever(packages=args.packages, query_strategy=args.query_strategy)
    path = args.declarations_file or (args.work_dir / "theorem_names.txt")
    if not path.is_file():
        session = session_factory()
        try:
            count = session.export_theorem_names("declarations-export", args.declarations_header, path, args.packages)
        finally:
            session.restart()
        print(f"Exported {count} theorem names to {path}", file=sys.stderr)
    return NameOverlapRetriever(path)


def run_pipeline(
    args: argparse.Namespace,
    *,
    model: Any | None = None,
    session_factory: Callable[[], Any] | None = None,
    retriever: Any | None = None,
) -> dict[str, int]:
    validate_args(args)
    args.input = args.input.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    rows = OpenAIMathDataset(args.input).load(limit=args.limit, theorems=args.theorem)
    using_real_lean = session_factory is None
    args.verified_project = verify_project_provenance(rows, args.project_dir) if using_real_lean else None
    ensure_run_config(args, rows)
    journals = open_journals(args.work_dir)

    if session_factory is None:
        def session_factory() -> LeanEnvSession:
            return LeanEnvSession(args.project_dir, args.lean_timeout, prefix_timeout=args.prefix_timeout)

    if not args.no_explain and model is None:
        model = make_model(args)
    if retriever is None:
        retriever = make_retriever(args, session_factory)
    retriever = CachedRetriever(retriever, journals.neighbors, query_strategy=args.query_strategy)

    sessions = ThreadLocalSessions(session_factory)
    try:
        if args.workers == 1:
            for row in rows:
                process_proof(row, args, sessions.get(), retriever, journals)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(lambda row=row: process_proof(row, args, sessions.get(), retriever, journals))
                    for row in rows
                ]
                for future in as_completed(futures):
                    future.result()
    finally:
        sessions.close()

    if not args.no_explain:
        assert model is not None
        annotation_phase(args, model, journals)
    return export_outputs(rows, args, journals, retriever)


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
