"""Generate pair-only line and suffix mutations for Lean proofs."""
from __future__ import annotations

from dataclasses import dataclass
import random
import re
import textwrap
from typing import Any, Mapping, Protocol, Sequence

try:
    from .mutation_utils import require_string, sha256_text
except ImportError:  # Allow imports when a sibling script is run directly.
    from mutation_utils import require_string, sha256_text


LME_PROMPT_VERSION = "april-appendix-c1-lme-v1"
MLME_PROMPT_VERSION = "april-lme-suffix-adaptation-v1"
SYSTEM_PROMPT = "You are a Lean 4 programmer."
FORBIDDEN = re.compile(r"\b(?:sorry|admit)\b|REDACTED", re.IGNORECASE)


class CompletionModel(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]], max_tokens: int) -> str: ...


class MutationError(ValueError):
    """A rejected generation, optionally carrying its model response for audit."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        prompt_version: str | None = None,
        prompt_sha256: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.prompt_version = prompt_version
        self.prompt_sha256 = prompt_sha256


@dataclass(frozen=True)
class ProofLayout:
    proof_start: int
    proof_end: int
    by_offset: int
    by_line: int
    code_lines: tuple[int, ...]
    lines: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Redaction:
    error_type: str
    redacted_proof: str
    replace_start: int
    replace_end: int
    base_indent: str
    selected_line: int | None
    cutoff_line: int | None
    redacted_start_line: int
    redacted_end_line: int
    redacted_code_lines: int
    eligible_locations: int


def byte_to_char_index(text: str, byte_offset: int) -> int:
    raw = text.encode("utf-8")
    if byte_offset < 0 or byte_offset > len(raw):
        raise ValueError(f"byte offset {byte_offset} is outside source")
    try:
        return len(raw[:byte_offset].decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"byte offset {byte_offset} splits a UTF-8 character") from error


def mask_lean_source(text: str) -> str:
    """Mask comments and strings while retaining every character position."""
    out = list(text)
    index = 0
    comment_depth = 0
    in_string = False
    while index < len(text):
        if comment_depth:
            if text.startswith("/-", index):
                out[index:index + 2] = "  "
                comment_depth += 1
                index += 2
                continue
            if text.startswith("-/", index):
                out[index:index + 2] = "  "
                comment_depth -= 1
                index += 2
                continue
        elif in_string:
            if text[index] == "\\":
                out[index] = " "
                index += 1
                if index < len(text):
                    out[index] = "\n" if text[index] == "\n" else " "
                index += 1
                continue
            if text[index] == '"':
                in_string = False
        elif text.startswith("--", index):
            end = text.find("\n", index)
            if end < 0:
                end = len(text)
            out[index:end] = " " * (end - index)
            index = end
            continue
        elif text.startswith("/-", index):
            out[index:index + 2] = "  "
            comment_depth = 1
            index += 2
            continue
        elif text[index] == '"':
            in_string = True
        else:
            index += 1
            continue
        if text[index] != "\n":
            out[index] = " "
        index += 1
    if comment_depth or in_string:
        raise ValueError("unterminated Lean comment or string")
    return "".join(out)


def line_bounds(text: str) -> list[tuple[int, int, int]]:
    """Return (start, content_end, full_end) offsets for every physical line."""
    bounds: list[tuple[int, int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        full_end = cursor + len(line)
        content_end = full_end - len(line) + len(line.rstrip("\r\n"))
        bounds.append((cursor, content_end, full_end))
        cursor = full_end
    if not text or cursor == len(text) and text.endswith(("\n", "\r")):
        bounds.append((cursor, cursor, cursor))
    elif cursor < len(text):
        bounds.append((cursor, len(text), len(text)))
    return bounds


def proof_layout(record: Mapping[str, Any]) -> ProofLayout:
    source = require_string(record, "correct_proof")
    proof_text = require_string(record, "proof_text")
    span = record.get("proof_span")
    if not isinstance(span, Mapping):
        raise ValueError("proof_span must be an object")
    proof_start = byte_to_char_index(source, int(span["start_byte"]))
    proof_end = byte_to_char_index(source, int(span["end_byte"]))
    if not 0 <= proof_start < proof_end <= len(source):
        raise ValueError("invalid proof_span")
    if source[proof_start:proof_end] != proof_text:
        raise ValueError("proof_text does not match proof_span in correct_proof")
    masked = mask_lean_source(source)
    match = re.match(r"\s*by\b", masked[proof_start:proof_end])
    if not match:
        raise ValueError("target proof is not tactic mode with a main `by`")
    by_offset = proof_start + match.end() - 2
    lines = tuple(line_bounds(source))
    by_line = source.count("\n", 0, by_offset)
    code_lines: list[int] = []
    for line_number in range(by_line + 1, len(lines)):
        start, content_end, _ = lines[line_number]
        if start >= proof_end:
            break
        if masked[start:min(content_end, proof_end)].strip():
            code_lines.append(line_number)
    return ProofLayout(proof_start, proof_end, by_offset, by_line, tuple(code_lines), lines)


def indentation(value: str) -> str:
    return value[:len(value) - len(value.lstrip(" \t"))]


def make_lme_redaction(record: Mapping[str, Any], rng: random.Random) -> Redaction:
    source = require_string(record, "correct_proof")
    layout = proof_layout(record)
    if not layout.code_lines:
        raise ValueError("target proof has no eligible nonblank line after `by`")
    selected = rng.choice(layout.code_lines)
    start, content_end, _ = layout.lines[selected]
    replace_end = min(content_end, layout.proof_end)
    base_indent = indentation(source[start:replace_end])
    redacted = source[:start] + base_indent + "REDACTED" + source[replace_end:]
    return Redaction(
        "lme", redacted, start, replace_end, base_indent, selected + 1, None,
        selected + 1, selected + 1, 1, len(layout.code_lines),
    )


def mlme_cutoffs(record: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    """Return (cutoff line, first removed code line, removed code count)."""
    layout = proof_layout(record)
    maximum = len(layout.code_lines) // 2
    candidates: list[tuple[int, int, int]] = []
    for removed in range(1, maximum + 1):
        first_position = len(layout.code_lines) - removed
        candidates.append((
            layout.code_lines[first_position - 1],
            layout.code_lines[first_position],
            removed,
        ))
    return candidates


def make_mlme_redaction(record: Mapping[str, Any], rng: random.Random) -> Redaction:
    source = require_string(record, "correct_proof")
    layout = proof_layout(record)
    candidates = mlme_cutoffs(record)
    if not candidates:
        raise ValueError("target proof needs at least two nonblank body lines for MLME")
    cutoff, first_removed, removed = rng.choice(candidates)
    replace_start = layout.lines[cutoff][2]
    if replace_start > layout.proof_end:
        raise ValueError("MLME cutoff falls outside proof_span")
    first_start, first_end, _ = layout.lines[first_removed]
    base_indent = indentation(source[first_start:min(first_end, layout.proof_end)])
    redacted = source[:replace_start] + base_indent + "REDACTED" + source[layout.proof_end:]
    end_line = int(record["proof_span"]["end_line"])
    return Redaction(
        "mlme", redacted, replace_start, layout.proof_end, base_indent, None,
        cutoff + 1, cutoff + 2, end_line, removed, len(candidates),
    )


def lme_user_prompt(redacted_proof: str) -> str:
    return f'''One line has been redacted in this lean4 proof. Please complete the proof
by providing the correct contents of the redacted line. Your response will
be automatically searched for your answer. To facilitate this, please write
"MY ANSWER" before your answer. Your answer should be exactly one line long
and should contain no semicolons. For example, if you were given
```lean4
theorem very_simple: 1+1=2 := by
REDACTED
```
you might respond with
"""
This is very easy, `rfl` accomplishes this in Lean 4.
MY ANSWER
```lean4
rfl
```
"""
Now try this theorem
```lean4
{redacted_proof}
```'''


def mlme_user_prompt(redacted_proof: str) -> str:
    return f'''The final portion has been redacted in this lean4 proof. Please complete
the proof by providing the correct contents that replace REDACTED. Your
response will be automatically searched for your answer. To facilitate this,
please write "MY ANSWER" before your answer. Your answer may contain as many
Lean 4 lines as needed. For example, if you were given
```lean4
theorem very_simple: 1+1=2 := by
  have h : 1 = 1 := rfl
  REDACTED
```
you might respond with
"""
The remaining goal is immediate.
MY ANSWER
```lean4
norm_num
```
"""
Now try this theorem
```lean4
{redacted_proof}
```'''


def prompt_for(redaction: Redaction) -> tuple[str, str, str]:
    if redaction.error_type == "lme":
        return SYSTEM_PROMPT, lme_user_prompt(redaction.redacted_proof), LME_PROMPT_VERSION
    return SYSTEM_PROMPT, mlme_user_prompt(redaction.redacted_proof), MLME_PROMPT_VERSION


def answer_after_marker(response: str) -> str:
    position = response.rfind("MY ANSWER")
    if position < 0:
        raise ValueError("model response has no MY ANSWER marker")
    answer = response[position + len("MY ANSWER"):].strip()
    fenced = re.match(r"^```(?:lean4?|Lean4?)?\s*\n(?P<body>.*?)\n?```\s*$", answer, re.DOTALL)
    if fenced:
        answer = fenced.group("body")
    elif "```" in answer:
        raise ValueError("malformed or extra Markdown fence after MY ANSWER")
    return answer.strip("\r\n")


def parse_lme_response(response: str, base_indent: str) -> str:
    lines = answer_after_marker(response).splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise ValueError("LME answer must contain exactly one nonempty line")
    line = lines[0].strip()
    if ";" in line:
        raise ValueError("LME answer contains a semicolon")
    if FORBIDDEN.search(line):
        raise ValueError("LME answer contains a forbidden placeholder")
    return base_indent + line


def parse_mlme_response(response: str, base_indent: str) -> str:
    answer = answer_after_marker(response)
    if not answer.strip():
        raise ValueError("MLME answer is empty")
    if FORBIDDEN.search(answer):
        raise ValueError("MLME answer contains a forbidden placeholder")
    lines = textwrap.dedent(answer).strip("\r\n").splitlines()
    return "\n".join(base_indent + line if line.strip() else "" for line in lines)


def apply_completion(correct_proof: str, redaction: Redaction, completion: str) -> str:
    result = correct_proof[:redaction.replace_start] + completion + correct_proof[redaction.replace_end:]
    if result == correct_proof:
        raise ValueError("model reproduced the original proof unchanged")
    if FORBIDDEN.search(completion):
        raise ValueError("completion contains a forbidden placeholder")
    return result


def messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mutate_proof(
    record: Mapping[str, Any],
    *,
    model: CompletionModel,
    error_type: str,
    rng: random.Random,
    max_tokens: int = 2048,
) -> dict[str, str]:
    """Generate exactly one unverified correct/incorrect proof pair."""
    if error_type not in {"lme", "mlme"}:
        raise ValueError(f"unsupported mutation type: {error_type}")
    correct = require_string(record, "correct_proof")
    contains_admission = record.get("contains_sorry_or_admit")
    if not isinstance(contains_admission, bool):
        raise ValueError("contains_sorry_or_admit must be supplied by extraction")
    if contains_admission:
        raise ValueError("target_contains_sorry_or_admit")
    redaction = (
        make_lme_redaction(record, rng)
        if error_type == "lme"
        else make_mlme_redaction(record, rng)
    )
    system, user, version = prompt_for(redaction)
    response = model.complete(messages(system, user), max_tokens)
    try:
        completion = (
            parse_lme_response(response, redaction.base_indent)
            if error_type == "lme"
            else parse_mlme_response(response, redaction.base_indent)
        )
        incorrect = apply_completion(correct, redaction, completion)
    except ValueError as error:
        raise MutationError(
            str(error),
            raw_response=response,
            prompt_version=version,
            prompt_sha256=sha256_text(user),
        ) from error
    return {"correct_proof": correct, "incorrect_proof": incorrect}


def mutate_dataset(
    proofs: Sequence[Mapping[str, Any]],
    *,
    model: CompletionModel,
    error_type: str,
    rng: random.Random,
    max_tokens: int = 2048,
) -> list[dict[str, str]]:
    """Generate one pair per input using caller-owned randomness."""
    return [
        mutate_proof(
            proof,
            model=model,
            error_type=error_type,
            rng=rng,
            max_tokens=max_tokens,
        )
        for proof in proofs
    ]
