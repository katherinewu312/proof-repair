"""Convert verified proof pairs into APRIL-compatible dataset records."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

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
