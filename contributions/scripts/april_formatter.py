"""Convert verified proof pairs into the published APRIL row schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

try:
    from .llm_client import PAPER_MODEL
    from .mutation_utils import require_string, sha256_text
    from .proof_annotator import parse_annotation
except ImportError:  # Support direct execution by sibling scripts.
    from llm_client import PAPER_MODEL
    from mutation_utils import require_string, sha256_text
    from proof_annotator import parse_annotation


THEOREM_FIELDS = (
    "theorem",
    "correct_name",
    "incorrect_name",
    "correct_formal",
    "incorrect_formal",
    "correct_informal",
    "incorrect_informal",
)
APRIL_FIELDS = (
    "path",
    "correct_proof",
    "incorrect_proof",
    "src_hash",
    "line",
    "col",
    "error",
    "line_at_error",
    "state_at_error",
    "explanation",
    "fix_suggestion",
    "source",
    "error_type",
    "split",
    *THEOREM_FIELDS,
)
TypeFieldProvider = Callable[
    [Mapping[str, Any], Mapping[str, str]], Mapping[str, Any]
]
APRIL_ERROR_TYPES = frozenset({"lme", "mlme", "tme", "thme"})
UNION_NULL_FIELDS = THEOREM_FIELDS


def stable_id(src_hash: str, error_type: str, attempt_index: int, incorrect_proof: str) -> str:
    payload = f"{src_hash}\0{error_type}\0{attempt_index}\0{sha256_text(incorrect_proof)}"
    return "openai-math-" + error_type + "-" + sha256_text(payload)[:24]


def null_theorem_fields(
    original: Mapping[str, Any],
    pair: Mapping[str, str],
) -> Mapping[str, Any]:
    """Default LME, MLME, and TME union-schema fields."""
    del original, pair
    return {field: None for field in THEOREM_FIELDS}


def format_april_record(
    original: Mapping[str, Any],
    pair: Mapping[str, Any],
    *,
    error_type: str,
    verification: Mapping[str, Any],
    annotation: Mapping[str, Any],
    split: str | None = None,
    type_field_provider: TypeFieldProvider = null_theorem_fields,
) -> dict[str, Any]:
    """Build one strict APRIL row from pair-independent downstream evidence."""
    if set(pair) != {"correct_proof", "incorrect_proof"}:
        raise ValueError("proof pair must contain exactly correct_proof and incorrect_proof")
    if error_type not in APRIL_ERROR_TYPES:
        raise ValueError(f"unsupported APRIL error type: {error_type}")
    correct = require_string(pair, "correct_proof")
    incorrect = require_string(pair, "incorrect_proof")
    if correct != require_string(original, "correct_proof"):
        raise ValueError("proof pair does not belong to the supplied original")
    diagnostic = verification.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("accepted proof pair requires a structured diagnostic")
    for field in ("line", "col"):
        if not isinstance(diagnostic.get(field), int):
            raise ValueError(f"diagnostic field {field} must be an integer")
    for field in ("error", "line_at_error", "state_at_error"):
        require_string(diagnostic, field)
    theorem_fields = dict(type_field_provider(original, pair))
    if set(theorem_fields) != set(THEOREM_FIELDS):
        raise ValueError("type-field provider must return every theorem mutation field")
    source = original.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("original record requires a source")
    record: dict[str, Any] = {
        "path": require_string(original, "source_file"),
        "correct_proof": correct,
        "incorrect_proof": incorrect,
        "src_hash": require_string(original, "src_hash"),
        "line": diagnostic["line"],
        "col": diagnostic["col"],
        "error": diagnostic["error"],
        "line_at_error": diagnostic["line_at_error"],
        "state_at_error": diagnostic["state_at_error"],
        "explanation": require_string(annotation, "explanation"),
        "fix_suggestion": require_string(annotation, "fix_suggestion"),
        "source": source,
        "error_type": error_type,
        "split": split,
        **{field: theorem_fields[field] for field in THEOREM_FIELDS},
    }
    if tuple(record) != APRIL_FIELDS:
        raise AssertionError("APRIL formatter emitted an unexpected schema")
    return record


# Theorem mutation (THM). The annotator sees the intended and substituted lemma.

THEOREM_ANNOTATION_PROMPT_VERSION = "thm-reference-card-v1"
THEOREM_ANNOTATION_SYSTEM_PROMPT = (
    "You are an expert Lean 4 and Mathlib engineer. You are diagnosing a single\n"
    "proof that fails to compile. You are given: the failing proof, Lean's error\n"
    "message(s), the goal state where the failure occurs, and a reference card\n"
    "naming the lemma that was intended and the lemma that was used instead,\n"
    "with their full statements. Explain the failure precisely and concisely."
)
THEOREM_ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "fix_suggestion": {"type": "string"},
    },
    "required": ["explanation", "fix_suggestion"],
    "additionalProperties": False,
}
PAPER_LEAN_TOOLCHAIN = "v4.22.0-rc4"


def format_lean_errors(errors: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"line {error['line']}, col {error['col']}: {error['message']}" for error in errors]
    return "\n".join(lines) if lines else "(none)"


def fix_prefix(substitute_name: str, original_name: str) -> str:
    return f"Replace {substitute_name} with {original_name}"


def theorem_annotation_user_prompt(
    *,
    mutated_code: str,
    errors_formatted: str,
    goal_state: str | None,
    substitute_name: str,
    substitute_statement: str | None,
    original_name: str,
    original_statement: str | None,
    line: int,
) -> str:
    return f'''FAILING PROOF:
```lean
{mutated_code}
```

LEAN ERROR(S):
{errors_formatted}

GOAL STATE AT FAILURE:
{goal_state or "(unavailable)"}

REFERENCE CARD:
- Lemma used (incorrect): {substitute_name}
  Statement: {substitute_statement or "(unavailable)"}
- Lemma intended (correct): {original_name}
  Statement: {original_statement or "(unavailable)"}
- Location of the substitution: line {line}

Return a JSON object with:
- "explanation": 1 to 3 sentences giving the concrete reason the proof fails.
  Compare the two lemma statements (hypotheses, conclusion shape, types,
  direction of rewriting) and connect that to the error message and goal.
- "fix_suggestion": one sentence that begins exactly with
  "{fix_prefix(substitute_name, original_name)}" and then briefly says why
  this resolves the mismatch. No code blocks.'''


def parse_theorem_annotation(response: str, substitute_name: str, original_name: str) -> dict[str, str]:
    value = parse_annotation(response)
    for key, text in value.items():
        if "```" in text:
            raise ValueError(f"annotation field {key} contains a code block")
    prefix = fix_prefix(substitute_name, original_name)
    if not value["fix_suggestion"].startswith(prefix):
        raise ValueError(f"fix_suggestion must begin with {prefix!r}")
    return value


@dataclass(frozen=True)
class TheoremProvenance:
    annotation_model: str | None
    endpoint: str | None
    reasoning_effort: str | None
    annotation_max_tokens: int | None
    retriever_backend: str
    retriever_version: str | None
    query_strategy: str
    neighbors_k: int
    retry_if_compiles: int
    lean_toolchain: str | None
    mathlib_rev: str | None

    @property
    def protocol_deviations(self) -> list[str]:
        deviations: list[str] = []
        if self.annotation_model and self.annotation_model != PAPER_MODEL:
            deviations.append(f"annotation_model={self.annotation_model} (paper model is {PAPER_MODEL})")
        if not self.retriever_backend.startswith("leanexplore"):
            deviations.append(f"retriever={self.retriever_backend} (paper uses LeanExplore)")
        if self.retry_if_compiles:
            deviations.append(f"retry_if_compiles={self.retry_if_compiles} (paper substitutes once)")
        if self.lean_toolchain and PAPER_LEAN_TOOLCHAIN not in self.lean_toolchain:
            deviations.append(f"lean_toolchain={self.lean_toolchain} (paper uses {PAPER_LEAN_TOOLCHAIN})")
        if self.annotation_model:
            deviations.append(
                f"annotation_prompt={THEOREM_ANNOTATION_PROMPT_VERSION} (not from the paper)"
            )
        return deviations


def format_theorem_record(
    original: Mapping[str, Any],
    mutation: Mapping[str, Any],
    verification: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
    provenance: TheoremProvenance,
) -> dict[str, Any]:
    """One THM record; ``annotation`` is None with --no-explain."""
    correct = require_string(mutation, "correct_proof")
    incorrect = require_string(mutation, "incorrect_proof")
    if correct != require_string(original, "correct_proof"):
        raise ValueError("mutation does not belong to the supplied original")
    diagnostic = verification.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("accepted mutation requires a structured diagnostic")
    record: dict[str, Any] = {
        "selected_id": stable_id(original["src_hash"], "thm", mutation["attempt_index"], incorrect),
        "correct_proof": correct,
        "incorrect_proof": incorrect,
        "src_hash": original["src_hash"],
        "path": original.get("source_file"),
        **diagnostic,
        "errors": verification.get("errors", []),
        "explanation": annotation.get("explanation") if annotation else None,
        "fix_suggestion": annotation.get("fix_suggestion") if annotation else None,
        "source": "openai_math",
        "error_type": "thm",
        "split": None,
        "source_theorem": require_string(original, "theorem"),
        "source_selected_id": original.get("selected_id"),
        "source_commit": original.get("source_commit"),
        "source_dirty": original.get("source_dirty"),
        "source_file_sha256": original.get("file_sha256"),
        "lean_toolchain": original.get("lean_toolchain"),
        "mathlib_rev": provenance.mathlib_rev,
        "verifier_fingerprint": original.get("verifier_fingerprint"),
        "attempt_index": mutation["attempt_index"],
        "global_seed": mutation["global_seed"],
        "derived_seed": mutation["derived_seed"],
        "mutation": {
            "original_lemma": mutation["original_lemma"],
            "substitute_lemma": mutation["substitute_lemma"],
            "mutation_site": mutation["mutation_site"],
            "neighbors_considered": mutation["neighbors_considered"],
            "neighbors_after_filter": mutation["neighbors_after_filter"],
            "compile_mode": verification.get("compile_mode"),
        },
        "retriever_backend": provenance.retriever_backend,
        "retriever_version": provenance.retriever_version,
        "retrieval_query_strategy": provenance.query_strategy,
        "neighbors_k": provenance.neighbors_k,
        "annotation_model": provenance.annotation_model if annotation else None,
        "annotation_prompt_version": annotation.get("prompt_version") if annotation else None,
        "annotation_prompt_sha256": annotation.get("user_prompt_sha256") if annotation else None,
        "annotation_response_sha256": annotation.get("response_sha256") if annotation else None,
        "sampling_temperature": None,
        "annotation_max_tokens": provenance.annotation_max_tokens if annotation else None,
        "reasoning_effort": provenance.reasoning_effort if annotation else None,
        "model_endpoint": provenance.endpoint if annotation else None,
        "protocol_deviations": provenance.protocol_deviations,
    }
    record.update({field: None for field in UNION_NULL_FIELDS})
    return record
