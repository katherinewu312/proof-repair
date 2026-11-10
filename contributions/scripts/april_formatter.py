"""Convert verified proof pairs into APRIL-compatible dataset records."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

try:
    from .llm_client import PAPER_MODEL
    from .mutation_utils import require_string, sha256_text
except ImportError:  # Support direct execution by sibling scripts.
    from llm_client import PAPER_MODEL
    from mutation_utils import require_string, sha256_text


ANNOTATION_PROMPT_VERSION = "april-appendix-c1-diagnostic-json-repair-v1"
ANNOTATION_SYSTEM_PROMPT = (
    "You are a Lean 4 programmer diagnosing one failing proof. Assume you ONLY\n"
    "see the incorrect proof text, the infoview state near the failure, and\n"
    "Lean's error message."
)
UNION_NULL_FIELDS = (
    "theorem", "correct_name", "incorrect_name", "correct_formal", "incorrect_formal",
    "correct_informal", "incorrect_informal",
)


@dataclass(frozen=True)
class AprilProvenance:
    model: str
    endpoint: str
    attempts_per_proof: int
    max_tokens: int
    annotation_max_tokens: int
    reasoning_effort: str

    @property
    def protocol_deviations(self) -> list[str]:
        deviations: list[str] = []
        if self.attempts_per_proof != 1:
            deviations.append(f"attempts_per_proof={self.attempts_per_proof} (paper default is 1)")
        if self.model != PAPER_MODEL:
            deviations.append(f"generation_and_annotation_model={self.model} (paper model is {PAPER_MODEL})")
        return deviations


def annotation_user_prompt(incorrect_proof: str, state: str, error: str) -> str:
    return f'''Explain why the incorrect proof fails and how to correct it using only the
incorrect proof, infoview state, and error.
Return ONLY one JSON object with exactly these two fields:
{{
"explanation": "1--3 sentences explaining the concrete reason the proof fails",
"fix_suggestion": "1 sentence with a high-level fix (no code)"
}}
No code blocks. No extra fields. Both fields must be non-empty.

INCORRECT PROOF:
```lean4
{incorrect_proof}
```

INFOVIEW STATE:
{state}

LEAN ERROR:
{error}'''


def parse_annotation(response: str) -> dict[str, str]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("annotation response is not a JSON object") from error
    if not isinstance(value, dict) or set(value) != {"explanation", "fix_suggestion"}:
        raise ValueError("annotation must have exactly explanation and fix_suggestion")
    for key in ("explanation", "fix_suggestion"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"annotation field {key} must be a nonempty string")
        value[key] = value[key].strip()
    return value


def stable_id(src_hash: str, error_type: str, attempt_index: int, incorrect_proof: str) -> str:
    payload = f"{src_hash}\0{error_type}\0{attempt_index}\0{sha256_text(incorrect_proof)}"
    return "openai-math-" + error_type + "-" + sha256_text(payload)[:24]


def format_april_record(
    original: Mapping[str, Any],
    pair: Mapping[str, Any],
    verification: Mapping[str, Any],
    annotation: Mapping[str, Any],
    provenance: AprilProvenance,
) -> dict[str, Any]:
    """Build one APRIL record from a neutral pair and downstream evidence."""
    correct = require_string(pair, "correct_proof")
    incorrect = require_string(pair, "incorrect_proof")
    if correct != require_string(original, "correct_proof"):
        raise ValueError("proof pair does not belong to the supplied original")
    diagnostic = verification.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("accepted proof pair requires a structured diagnostic")
    record: dict[str, Any] = {
        "selected_id": stable_id(
            original["src_hash"], pair["error_type"], pair["attempt_index"], incorrect
        ),
        "correct_proof": correct,
        "incorrect_proof": incorrect,
        "src_hash": original["src_hash"],
        "path": original.get("source_file"),
        **diagnostic,
        "explanation": annotation["explanation"],
        "fix_suggestion": annotation["fix_suggestion"],
        "source": "openai_math",
        "error_type": pair["error_type"],
        "split": None,
        "source_theorem": require_string(original, "theorem"),
        "source_selected_id": original.get("selected_id"),
        "source_commit": original.get("source_commit"),
        "source_dirty": original.get("source_dirty"),
        "source_file_sha256": original.get("file_sha256"),
        "lean_toolchain": original.get("lean_toolchain"),
        "verifier_fingerprint": original.get("verifier_fingerprint"),
        "attempt_index": pair["attempt_index"],
        "global_seed": pair["global_seed"],
        "derived_seed": pair["derived_seed"],
        "mutation": pair["redaction"],
        "generation_model": provenance.model,
        "generation_prompt_version": pair["prompt_version"],
        "generation_prompt_sha256": pair["user_prompt_sha256"],
        "generation_response_sha256": pair["response_sha256"],
        "annotation_model": provenance.model,
        "annotation_prompt_version": annotation["prompt_version"],
        "annotation_prompt_sha256": annotation["user_prompt_sha256"],
        "annotation_response_sha256": annotation["response_sha256"],
        "sampling_temperature": None,
        "sampling_max_tokens": provenance.max_tokens,
        "annotation_max_tokens": provenance.annotation_max_tokens,
        "reasoning_effort": provenance.reasoning_effort,
        "model_endpoint": provenance.endpoint,
        "protocol_deviations": provenance.protocol_deviations,
    }
    record.update({field: None for field in UNION_NULL_FIELDS})
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
