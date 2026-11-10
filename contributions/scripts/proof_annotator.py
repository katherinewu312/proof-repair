"""Generate and validate natural-language diagnostics for failing Lean proofs."""
from __future__ import annotations

import json
import re


ANNOTATION_PROMPT_VERSION = "april-appendix-c1-diagnostic-json-repair-v2"
ANNOTATION_SYSTEM_PROMPT = (
    "You are a Lean 4 programmer diagnosing one failing proof. Assume you ONLY\n"
    "see the incorrect proof text, the infoview state near the failure, and\n"
    "Lean's error message."
)


def annotation_user_prompt(
    incorrect_proof: str,
    state: str,
    line_at_error: str,
    error: str,
) -> str:
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

LINE AT ERROR:
{line_at_error}

LEAN ERROR:
{error}'''


def parse_annotation(response: str) -> dict[str, str]:
    candidate = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(?P<body>.*?)\n```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group("body").strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("annotation response is not a JSON object") from error
    if not isinstance(value, dict) or set(value) != {"explanation", "fix_suggestion"}:
        raise ValueError("annotation must have exactly explanation and fix_suggestion")
    for key in ("explanation", "fix_suggestion"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"annotation field {key} must be a nonempty string")
        value[key] = value[key].strip()
    return value
