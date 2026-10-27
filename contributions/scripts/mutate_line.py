#!/usr/bin/env python3
"""Generate verified OpenAI Math line and multi-line proof mutations.

The pipeline is deliberately staged and journaled.  A completed generation,
Lean verification, or annotation is never repeated after an interrupted run.
Only fully verified and annotated records are copied from the ignored journals
to the tracked contribution JSONL files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit


PAPER_MODEL = "deepseek-ai/DeepSeek-V3-0324"
LME_PROMPT_VERSION = "april-appendix-c1-lme-v1"
MLME_PROMPT_VERSION = "april-lme-suffix-adaptation-v1"
ANNOTATION_PROMPT_VERSION = "april-appendix-c1-diagnostic-json-repair-v1"
SYSTEM_PROMPT = "You are a Lean 4 programmer."
ANNOTATION_SYSTEM_PROMPT = (
    "You are a Lean 4 programmer diagnosing one failing proof. Assume you ONLY\n"
    "see the incorrect proof text, the infoview state near the failure, and\n"
    "Lean's error message."
)
FORBIDDEN = re.compile(r"\b(?:sorry|admit)\b|REDACTED", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def derived_seed(global_seed: int, src_hash: str, error_type: str, attempt_index: int) -> int:
    """Derive an order-independent 64-bit seed for exactly one attempt."""
    material = f"{global_seed}\0{src_hash}\0{error_type}\0{attempt_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def stable_id(src_hash: str, error_type: str, attempt_index: int, incorrect_proof: str) -> str:
    payload = f"{src_hash}\0{error_type}\0{attempt_index}\0{sha256_text(incorrect_proof)}"
    return "openai-math-" + error_type + "-" + sha256_text(payload)[:24]


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
    elif cursor < len(text):  # Defensive; splitlines normally consumes all text.
        bounds.append((cursor, len(text), len(text)))
    return bounds


@dataclass(frozen=True)
class ProofLayout:
    proof_start: int
    proof_end: int
    by_offset: int
    by_line: int
    code_lines: tuple[int, ...]
    lines: tuple[tuple[int, int, int], ...]


def proof_layout(record: Mapping[str, Any]) -> ProofLayout:
    source = require_string(record, "correct_proof")
    span = record.get("proof_span")
    if not isinstance(span, Mapping):
        raise ValueError("proof_span must be an object")
    proof_start = byte_to_char_index(source, int(span["start_byte"]))
    proof_end = byte_to_char_index(source, int(span["end_byte"]))
    if not 0 <= proof_start < proof_end <= len(source):
        raise ValueError("invalid proof_span")
    masked = mask_lean_source(source)
    proof_mask = masked[proof_start:proof_end]
    match = re.match(r"\s*by\b", proof_mask)
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
        end = min(content_end, proof_end)
        if masked[start:end].strip():
            code_lines.append(line_number)
    return ProofLayout(proof_start, proof_end, by_offset, by_line, tuple(code_lines), lines)


def indentation(value: str) -> str:
    return value[: len(value) - len(value.lstrip(" \t"))]


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


def make_lme_redaction(record: Mapping[str, Any], seed: int) -> Redaction:
    source = require_string(record, "correct_proof")
    layout = proof_layout(record)
    if not layout.code_lines:
        raise ValueError("target proof has no eligible nonblank line after `by`")
    selected = random.Random(seed).choice(layout.code_lines)
    start, content_end, _ = layout.lines[selected]
    replace_end = min(content_end, layout.proof_end)
    original = source[start:replace_end]
    base_indent = indentation(original)
    redacted = source[:start] + base_indent + "REDACTED" + source[replace_end:]
    return Redaction(
        error_type="lme",
        redacted_proof=redacted,
        replace_start=start,
        replace_end=replace_end,
        base_indent=base_indent,
        selected_line=selected + 1,
        cutoff_line=None,
        redacted_start_line=selected + 1,
        redacted_end_line=selected + 1,
        redacted_code_lines=1,
        eligible_locations=len(layout.code_lines),
    )


def mlme_cutoffs(record: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    """Return (cutoff line, first removed code line, removed code count)."""
    layout = proof_layout(record)
    count = len(layout.code_lines)
    maximum = count // 2
    if maximum < 1:
        return []
    candidates: list[tuple[int, int, int]] = []
    for removed in range(1, maximum + 1):
        first_removed_position = count - removed
        cutoff = layout.code_lines[first_removed_position - 1]
        first_removed = layout.code_lines[first_removed_position]
        candidates.append((cutoff, first_removed, removed))
    return candidates


def make_mlme_redaction(record: Mapping[str, Any], seed: int) -> Redaction:
    source = require_string(record, "correct_proof")
    layout = proof_layout(record)
    candidates = mlme_cutoffs(record)
    if not candidates:
        raise ValueError("target proof needs at least two nonblank body lines for MLME")
    cutoff, first_removed, removed = random.Random(seed).choice(candidates)
    # Everything after the selected cutoff is the suffix, including intervening
    # comments or blank lines.  The marker adopts the next code line's indent.
    replace_start = layout.lines[cutoff][2]
    if replace_start > layout.proof_end:
        raise ValueError("MLME cutoff falls outside proof_span")
    first_start, first_end, _ = layout.lines[first_removed]
    base_indent = indentation(source[first_start:min(first_end, layout.proof_end)])
    redacted = source[:replace_start] + base_indent + "REDACTED" + source[layout.proof_end:]
    end_line = int(record["proof_span"]["end_line"])
    return Redaction(
        error_type="mlme",
        redacted_proof=redacted,
        replace_start=replace_start,
        replace_end=layout.proof_end,
        base_indent=base_indent,
        selected_line=None,
        cutoff_line=cutoff + 1,
        redacted_start_line=cutoff + 2,
        redacted_end_line=end_line,
        redacted_code_lines=removed,
        eligible_locations=len(candidates),
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


def annotation_user_prompt(incorrect_proof: str, state: str, error: str) -> str:
    # Appendix C.1 accidentally leaves the fix_suggestion example string
    # unterminated.  Close that quote so its simultaneously stated JSON-only
    # contract is satisfiable; the version name and manifest expose this repair.
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


def prompt_for(redaction: Redaction) -> tuple[str, str, str]:
    if redaction.error_type == "lme":
        return SYSTEM_PROMPT, lme_user_prompt(redaction.redacted_proof), LME_PROMPT_VERSION
    return SYSTEM_PROMPT, mlme_user_prompt(redaction.redacted_proof), MLME_PROMPT_VERSION


def answer_after_marker(response: str) -> str:
    marker = "MY ANSWER"
    position = response.rfind(marker)
    if position < 0:
        raise ValueError("model response has no MY ANSWER marker")
    answer = response[position + len(marker):].strip()
    fenced = re.match(r"^```(?:lean4?|Lean4?)?\s*\n(?P<body>.*?)\n?```\s*$", answer, re.DOTALL)
    if fenced:
        answer = fenced.group("body")
    elif "```" in answer:
        raise ValueError("malformed or extra Markdown fence after MY ANSWER")
    return answer.strip("\r\n")


def parse_lme_response(response: str, base_indent: str) -> str:
    answer = answer_after_marker(response)
    lines = answer.splitlines()
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
    normalized = textwrap.dedent(answer).strip("\r\n")
    lines = normalized.splitlines()
    return "\n".join(base_indent + line if line.strip() else "" for line in lines)


def apply_completion(correct_proof: str, redaction: Redaction, completion: str) -> str:
    result = correct_proof[:redaction.replace_start] + completion + correct_proof[redaction.replace_end:]
    if result == correct_proof:
        raise ValueError("model reproduced the original proof unchanged")
    if FORBIDDEN.search(result[redaction.replace_start:redaction.replace_start + len(completion)]):
        raise ValueError("completion contains a forbidden placeholder")
    return result


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


class ModelAdapter(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]], max_tokens: int) -> str: ...


class OpenAICompatibleModel:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key_env: str,
        temperature: float,
        timeout: float,
        retries: int,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install requirements.txt to use the model API") from error
        if not base_url:
            raise ValueError("set --base-url or OPENAI_BASE_URL for the DeepSeek-compatible endpoint")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"required API key environment variable {api_key_env} is unset")
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries)

    def complete(self, messages: Sequence[Mapping[str, str]], max_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            raise RuntimeError(f"model API request failed ({type(error).__name__})") from error
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model returned an empty completion")
        return content


@dataclass(frozen=True)
class Diagnostic:
    line: int
    col: int
    error: str
    line_at_error: str
    state_at_error: str


@dataclass(frozen=True)
class Verification:
    valid: bool
    diagnostic: Diagnostic | None = None


def _position_tuple(position: Any) -> tuple[int, int]:
    return int(position.line), int(position.column)


def _contains(start: tuple[int, int], end: tuple[int, int], point: tuple[int, int]) -> bool:
    return start <= point <= end


def state_from_response(response: Any, line: int, col: int) -> str | None:
    """Read an infoview goal from structured tactic/info-tree data only."""
    point = (line, col)
    containing: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    preceding: list[tuple[tuple[int, int], str]] = []
    for tactic in getattr(response, "tactics", None) or []:
        start = _position_tuple(tactic.start_pos)
        end = _position_tuple(tactic.end_pos)
        goals = str(getattr(tactic, "goals", "") or "").strip()
        if goals and _contains(start, end, point):
            containing.append((start, end, goals))
        if goals and start <= point:
            preceding.append((end, goals))
    if containing:
        containing.sort(key=lambda item: (item[1][0] - item[0][0], item[1][1] - item[0][1]))
        return containing[0][2]

    nodes: list[Any] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
            return
        info_node = getattr(node, "node", None)
        if info_node is not None and getattr(info_node, "goals_before", None) is not None:
            nodes.append(info_node)
        for child in getattr(node, "children", None) or []:
            visit(child)

    visit(getattr(response, "infotree", None))
    for node in nodes:
        syntax = getattr(node, "stx", None)
        node_range = getattr(syntax, "range", None)
        start_pos = getattr(node_range, "start", None)
        end_pos = getattr(node_range, "finish", None)
        if start_pos is None or end_pos is None:
            continue
        start, end = _position_tuple(start_pos), _position_tuple(end_pos)
        before_goals = getattr(node, "goals_before", None) or []
        after_goals = getattr(node, "goals_after", None) or []
        before = "\n\n".join(str(goal).strip() for goal in before_goals if str(goal).strip())
        after = "\n\n".join(str(goal).strip() for goal in after_goals if str(goal).strip())
        if _contains(start, end, point) and before:
            containing.append((start, end, before))
        if end <= point and (after or before):
            preceding.append((end, after or before))
    if containing:
        containing.sort(key=lambda item: (item[1][0] - item[0][0], item[1][1] - item[0][1]))
        return containing[0][2]
    if preceding:
        preceding.sort(key=lambda item: item[0])
        return preceding[-1][1]
    return None


class LeanInteractVerifier:
    """One reusable Lean-Interact server tied to the OpenAI Math project."""

    def __init__(self, project_dir: Path, timeout: float) -> None:
        try:
            from lean_interact import LeanREPLConfig, LeanServer, LocalProject
        except ImportError as error:
            raise RuntimeError("install lean-interact to verify mutations") from error
        self.project_dir = project_dir.resolve()
        self.timeout = timeout
        project = LocalProject(directory=self.project_dir, auto_build=False)
        config = LeanREPLConfig(
            project=project,
            enable_parallel_elaboration=False,
            enable_incremental_optimization=False,
        )
        self.server = LeanServer(config)

    def verify(self, source: str) -> Verification:
        from lean_interact import FileCommand

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".lean", prefix=".line-mutation-",
                delete=False,
            ) as handle:
                handle.write(source)
                temporary = Path(handle.name)
            response = self.server.run(
                FileCommand(path=str(temporary), all_tactics=True, root_goals=True, infotree="full"),
                timeout=self.timeout,
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if not hasattr(response, "messages"):
            raise RuntimeError(f"Lean-Interact failed before elaboration: {getattr(response, 'message', response)}")
        errors = [m for m in (response.messages or []) if m.severity == "error"]
        # OpenAI Math source files can contain admitted auxiliary declarations.
        # The pipeline separately rejects sorry/admit inside the extracted target
        # proof_span; unrelated source-level sorries must not disqualify a target
        # that elaborates successfully.
        if not errors:
            return Verification(valid=True)
        errors.sort(key=lambda message: _position_tuple(message.start_pos))
        first = errors[0]
        line, col = _position_tuple(first.start_pos)
        error_text = str(first.data).strip()
        source_lines = source.splitlines()
        line_text = source_lines[line - 1].strip() if 1 <= line <= len(source_lines) else ""
        state = state_from_response(response, line, col)
        diagnostic = None
        if state:
            diagnostic = Diagnostic(line, col, error_text, line_text, state)
        return Verification(valid=False, diagnostic=diagnostic)


class VerifierPool:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.local = threading.local()

    def verify(self, source: str) -> Verification:
        verifier = getattr(self.local, "verifier", None)
        if verifier is None:
            verifier = self.factory()
            self.local.verifier = verifier
        return verifier.verify(source)


class JsonlJournal:
    """Append-only journal indexed by a stable key; last record wins on read."""

    def __init__(self, path: Path, key: str) -> None:
        self.path = path
        self.key = key
        self.records: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self.records[str(record[key])] = record
                except (json.JSONDecodeError, KeyError) as error:
                    raise ValueError(f"invalid journal {path}:{number}: {error}") from error
        path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    def append(self, record: dict[str, Any]) -> None:
        value = str(record[self.key])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[value] = record


def require_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record field {key} must be a nonempty string")
    return value


def load_originals(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            src_hash = require_string(row, "src_hash")
            require_string(row, "correct_proof")
            require_string(row, "theorem")
            if src_hash in seen:
                raise ValueError(f"duplicate src_hash {src_hash}")
            seen.add(src_hash)
            rows.append(row)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"invalid input {path}:{number}: {error}") from error
        if limit is not None and len(rows) >= limit:
            break
    return rows


def iter_parallel(
    jobs: Iterable[tuple[str, str]],
    verify: Callable[[str], Verification],
    workers: int,
) -> Iterable[tuple[str, Verification]]:
    materialized = list(jobs)
    if workers == 1:
        for key, source in materialized:
            yield key, verify(source)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify, source): key for key, source in materialized}
        for future in as_completed(futures):
            yield futures[future], future.result()


def mutation_job_id(src_hash: str, error_type: str, attempt_index: int) -> str:
    return f"{error_type}:{src_hash}:{attempt_index}"


def messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def record_rejection(journal: JsonlJournal, key: str, phase: str, reason: str) -> None:
    if journal.get(key) is None:
        journal.append({"job_id": key, "phase": phase, "reason": reason})


def sanitize_endpoint(base_url: str | None) -> str | None:
    if not base_url:
        return None
    value = urlsplit(base_url)
    hostname = value.hostname or ""
    if value.port:
        hostname += f":{value.port}"
    return urlunsplit((value.scheme, hostname, value.path, "", ""))


def ensure_run_config(args: argparse.Namespace) -> None:
    """Refuse to combine journals produced by incompatible experiments."""
    config = {
        "protocol": "openai-math-line-mutations-v1",
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "project_dir": str(args.project_dir),
        "verified_project": getattr(args, "verified_project", None),
        "seed": args.seed,
        "model": args.model,
        "endpoint": sanitize_endpoint(endpoint_from_args(args)),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "annotation_max_tokens": args.annotation_max_tokens,
        "annotation_retries": args.annotation_retries,
        "prompt_versions": [LME_PROMPT_VERSION, MLME_PROMPT_VERSION, ANNOTATION_PROMPT_VERSION],
    }
    path = args.work_dir / "run_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"work directory was created with incompatible settings: {path}; choose a new --work-dir"
            )
        return
    journal_names = ("baseline.jsonl", "generations.jsonl", "verifications.jsonl", "annotations.jsonl")
    if any((args.work_dir / name).exists() for name in journal_names):
        raise ValueError(f"journals exist without {path}; choose a new --work-dir")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


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
    expected_fingerprints = {str(row.get("verifier_fingerprint")) for row in rows}
    if expected_fingerprints != {fingerprint}:
        raise ValueError("OpenAI Math verifier fingerprint does not match the extracted input")
    toolchain_path = project_dir / "lean-toolchain"
    if not toolchain_path.is_file():
        raise ValueError(f"missing OpenAI Math lean-toolchain: {toolchain_path}")
    toolchain = toolchain_path.read_text(encoding="utf-8").strip()
    expected_toolchains = {str(row.get("lean_toolchain")) for row in rows}
    if expected_toolchains != {toolchain}:
        raise ValueError("OpenAI Math lean-toolchain does not match the extracted input")
    return {"commit": commit, "fingerprint": fingerprint, "toolchain": toolchain}


def endpoint_from_args(args: argparse.Namespace) -> str | None:
    return args.base_url or os.environ.get(args.base_url_env)


def protocol_deviations(args: argparse.Namespace) -> list[str]:
    deviations: list[str] = []
    if args.attempts_per_proof != 1:
        deviations.append(f"attempts_per_proof={args.attempts_per_proof} (paper default is 1)")
    if args.model != PAPER_MODEL:
        deviations.append(f"generation_and_annotation_model={args.model} (paper model is {PAPER_MODEL})")
    return deviations


def make_redaction(record: Mapping[str, Any], error_type: str, seed: int) -> Redaction:
    return make_lme_redaction(record, seed) if error_type == "lme" else make_mlme_redaction(record, seed)


def generate_candidates(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    model: ModelAdapter,
    baseline: JsonlJournal,
    generations: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    for row in rows:
        src_hash = row["src_hash"]
        baseline_record = baseline.get(src_hash)
        if not baseline_record or baseline_record["status"] != "valid":
            continue
        target_mask = mask_lean_source(row.get("proof_text", ""))
        if row.get("contains_sorry_or_admit") or re.search(r"\b(?:sorry|admit)\b", target_mask):
            for error_type in args.mutation_types:
                for attempt in range(args.attempts_per_proof):
                    key = mutation_job_id(src_hash, error_type, attempt)
                    if generations.get(key) is None:
                        generations.append({"job_id": key, "status": "rejected", "reason": "target_contains_sorry_or_admit"})
                        record_rejection(rejections, key, "generation", "target_contains_sorry_or_admit")
            continue
        for error_type in args.mutation_types:
            for attempt in range(args.attempts_per_proof):
                key = mutation_job_id(src_hash, error_type, attempt)
                if generations.get(key) is not None:
                    continue
                seed = derived_seed(args.seed, src_hash, error_type, attempt)
                try:
                    redaction = make_redaction(row, error_type, seed)
                    system, user, version = prompt_for(redaction)
                except ValueError as error:
                    generations.append({"job_id": key, "status": "rejected", "reason": str(error)})
                    record_rejection(rejections, key, "generation", str(error))
                    continue
                # API exceptions escape without a journal entry because no
                # successful response exists and a resumed run may retry safely.
                response = model.complete(messages(system, user), args.max_tokens)
                try:
                    completion = (
                        parse_lme_response(response, redaction.base_indent)
                        if error_type == "lme"
                        else parse_mlme_response(response, redaction.base_indent)
                    )
                    incorrect = apply_completion(row["correct_proof"], redaction, completion)
                except ValueError as error:
                    generations.append({
                        "job_id": key,
                        "status": "rejected",
                        "reason": str(error),
                        "raw_response": response,
                        "response_sha256": sha256_text(response),
                        "prompt_version": version,
                        "user_prompt_sha256": sha256_text(user),
                    })
                    record_rejection(rejections, key, "generation", str(error))
                    continue
                generations.append({
                    "job_id": key,
                    "status": "generated",
                    "src_hash": src_hash,
                    "error_type": error_type,
                    "attempt_index": attempt,
                    "global_seed": args.seed,
                    "derived_seed": seed,
                    "incorrect_proof": incorrect,
                    "incorrect_sha256": sha256_text(incorrect),
                    "raw_response": response,
                    "response_sha256": sha256_text(response),
                    "prompt_version": version,
                    "system_prompt": system,
                    "user_prompt_sha256": sha256_text(user),
                    "redaction": asdict(redaction) | {"redacted_proof": None},
                })


def baseline_phase(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    verifier: VerifierPool,
    journal: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    jobs = [(row["src_hash"], row["correct_proof"]) for row in rows if journal.get(row["src_hash"]) is None]
    for src_hash, result in iter_parallel(jobs, verifier.verify, args.workers):
        if result.valid:
            journal.append({"src_hash": src_hash, "status": "valid"})
        else:
            reason = "baseline_failed" if result.diagnostic else "baseline_failed_without_structured_state"
            journal.append({
                "src_hash": src_hash,
                "status": "rejected",
                "reason": reason,
                "diagnostic": asdict(result.diagnostic) if result.diagnostic else None,
            })
            record_rejection(rejections, "baseline:" + src_hash, "baseline", reason)


def verification_phase(
    args: argparse.Namespace,
    verifier: VerifierPool,
    generations: JsonlJournal,
    verifications: JsonlJournal,
    rejections: JsonlJournal,
) -> None:
    generated = [record for record in generations.records.values() if record.get("status") == "generated"]
    generated.sort(key=lambda record: record["job_id"])
    owner_by_hash: dict[str, str] = {}
    jobs: list[tuple[str, str]] = []
    for record in generated:
        key = record["job_id"]
        proof_hash = record["incorrect_sha256"]
        owner = owner_by_hash.setdefault(proof_hash, key)
        if verifications.get(key) is not None:
            continue
        if owner != key:
            verifications.append({"job_id": key, "status": "rejected", "reason": "duplicate_incorrect_proof", "duplicate_of": owner})
            record_rejection(rejections, key, "verification", "duplicate_incorrect_proof")
            continue
        jobs.append((key, record["incorrect_proof"]))
    for key, result in iter_parallel(jobs, verifier.verify, args.workers):
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
    model: ModelAdapter,
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
        raw_responses: list[str] = list(existing.get("raw_responses", [])) if existing else []
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
                "job_id": key, "status": "rejected", "reason": parse_error,
                "raw_responses": raw_responses,
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


UNION_NULL_FIELDS = (
    "theorem", "correct_name", "incorrect_name", "correct_formal", "incorrect_formal",
    "correct_informal", "incorrect_informal",
)


def output_record(
    original: Mapping[str, Any],
    generated: Mapping[str, Any],
    verified: Mapping[str, Any],
    annotated: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    diagnostic = verified["diagnostic"]
    redaction = generated["redaction"]
    record: dict[str, Any] = {
        "selected_id": stable_id(
            original["src_hash"], generated["error_type"], generated["attempt_index"], generated["incorrect_proof"]
        ),
        "correct_proof": original["correct_proof"],
        "incorrect_proof": generated["incorrect_proof"],
        "src_hash": original["src_hash"],
        "path": original.get("source_file"),
        **diagnostic,
        "explanation": annotated["explanation"],
        "fix_suggestion": annotated["fix_suggestion"],
        "source": "openai_math",
        "error_type": generated["error_type"],
        "split": None,
        "source_theorem": original["theorem"],
        "source_selected_id": original.get("selected_id"),
        "source_commit": original.get("source_commit"),
        "source_dirty": original.get("source_dirty"),
        "source_file_sha256": original.get("file_sha256"),
        "lean_toolchain": original.get("lean_toolchain"),
        "verifier_fingerprint": original.get("verifier_fingerprint"),
        "attempt_index": generated["attempt_index"],
        "global_seed": generated["global_seed"],
        "derived_seed": generated["derived_seed"],
        "mutation": {
            key: value for key, value in redaction.items() if key != "redacted_proof"
        },
        "generation_model": args.model,
        "generation_prompt_version": generated["prompt_version"],
        "generation_prompt_sha256": generated["user_prompt_sha256"],
        "generation_response_sha256": generated["response_sha256"],
        "annotation_model": args.model,
        "annotation_prompt_version": annotated["prompt_version"],
        "annotation_prompt_sha256": annotated["user_prompt_sha256"],
        "annotation_response_sha256": annotated["response_sha256"],
        "sampling_temperature": args.temperature,
        "sampling_max_tokens": args.max_tokens,
        "annotation_max_tokens": args.annotation_max_tokens,
        "model_endpoint": sanitize_endpoint(endpoint_from_args(args)),
        "protocol_deviations": protocol_deviations(args),
    }
    record.update({field: None for field in UNION_NULL_FIELDS})
    return record


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256_text(payload)


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256_text(payload)


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
        if original is None:
            continue
        outputs[generated["error_type"]].append(output_record(original, generated, verified, annotated, args))
    for values in outputs.values():
        values.sort(key=lambda value: value["selected_id"])
    output_hashes = {
        error_type: atomic_jsonl(args.output_dir / f"{error_type}.jsonl", outputs[error_type])
        for error_type in ("lme", "mlme")
    }
    counts: dict[str, int] = {
        "input_records": len(rows),
        "baseline_valid": sum(record.get("status") == "valid" for record in baseline.records.values()),
        "baseline_rejected": sum(record.get("status") == "rejected" for record in baseline.records.values()),
        "lme_accepted": len(outputs["lme"]),
        "mlme_accepted": len(outputs["mlme"]),
        "generated": sum(record.get("status") == "generated" for record in generations.records.values()),
        "generation_rejected": sum(record.get("status") == "rejected" for record in generations.records.values()),
        "accepted_failures": sum(
            record.get("status") == "accepted_failure" for record in verifications.records.values()
        ),
        "verification_rejected": sum(record.get("status") == "rejected" for record in verifications.records.values()),
        "annotated": sum(record.get("status") == "annotated" for record in annotations.records.values()),
        "annotation_rejected": sum(record.get("status") == "rejected" for record in annotations.records.values()),
    }
    source_commits = sorted({str(row.get("source_commit")) for row in rows})
    fingerprints = sorted({str(row.get("verifier_fingerprint")) for row in rows})
    toolchains = sorted({str(row.get("lean_toolchain")) for row in rows})
    rejection_counts: dict[str, int] = {}
    for rejection in rejections.records.values():
        label = f"{rejection.get('phase')}:{rejection.get('reason')}"
        rejection_counts[label] = rejection_counts.get(label, 0) + 1
    lme_template = lme_user_prompt("{broken_proof}")
    mlme_template = mlme_user_prompt("{broken_proof}")
    annotation_template = annotation_user_prompt("{incorrect_proof}", "{state_at_error}", "{error}")
    manifest: dict[str, Any] = {
        "protocol": "openai-math-line-mutations-v1",
        "input": str(args.input),
        "input_sha256": sha256_bytes(args.input.read_bytes()),
        "input_records": len(rows),
        "source_commits": source_commits,
        "source_file_sha256": sorted({str(row.get("file_sha256")) for row in rows}),
        "source_dirty_values": sorted({bool(row.get("source_dirty")) for row in rows}),
        "lean_toolchains": toolchains,
        "verifier_fingerprints": fingerprints,
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
        "protocol_deviations": protocol_deviations(args),
        "sampling": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "annotation_max_tokens": args.annotation_max_tokens,
            "annotation_retries": args.annotation_retries,
        },
        "verification": {"workers": args.workers, "timeout_seconds": args.lean_timeout},
        "prompts": {
            "lme": {
                "version": LME_PROMPT_VERSION,
                "system": SYSTEM_PROMPT,
                "template": lme_template,
                "sha256": sha256_text(SYSTEM_PROMPT + "\0" + lme_template),
            },
            "mlme": {
                "version": MLME_PROMPT_VERSION,
                "system": SYSTEM_PROMPT,
                "template": mlme_template,
                "sha256": sha256_text(SYSTEM_PROMPT + "\0" + mlme_template),
            },
            "annotation": {
                "version": ANNOTATION_PROMPT_VERSION,
                "system": ANNOTATION_SYSTEM_PROMPT,
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
    parser.add_argument("--input", type=Path, default=Path("data/raw/openai_math/originals.jsonl"))
    parser.add_argument("--project-dir", type=Path, default=Path("external/openai-math/lean"))
    parser.add_argument("--output-dir", type=Path, default=Path("contributions/openai-math-line-mutations"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/openai_math_line_mutations"))
    parser.add_argument("--mutation-types", nargs="+", choices=("lme", "mlme"), default=("lme", "mlme"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts-per-proof", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Pilot limit applied before any verification or API call")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Lean-Interact servers")
    parser.add_argument("--lean-timeout", type=float, default=120.0)
    parser.add_argument("--model", default=PAPER_MODEL)
    parser.add_argument("--allow-model-deviation", action="store_true")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint; overrides --base-url-env")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    if args.model != PAPER_MODEL and not args.allow_model_deviation:
        raise ValueError(
            f"the paper-faithful model is exactly {PAPER_MODEL}; pass --allow-model-deviation to label a replacement"
        )
    if not args.input.is_file():
        raise ValueError(f"input does not exist: {args.input}")
    if not args.project_dir.is_dir():
        raise ValueError(f"OpenAI Math Lean project does not exist: {args.project_dir}")


def run_pipeline(
    args: argparse.Namespace,
    *,
    model: ModelAdapter | None = None,
    verifier_pool: VerifierPool | None = None,
) -> dict[str, int]:
    validate_args(args)
    args.input = args.input.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    rows = load_originals(args.input, args.limit)
    using_real_verifier = verifier_pool is None
    args.verified_project = (
        verify_project_provenance(rows, args.project_dir) if using_real_verifier else None
    )
    ensure_run_config(args)

    def make_model() -> OpenAICompatibleModel:
        return OpenAICompatibleModel(
            model=args.model,
            base_url=endpoint_from_args(args),
            api_key_env=args.api_key_env,
            temperature=args.temperature,
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
    generation_pending = any(
        baseline.get(row["src_hash"]) is not None
        and baseline.get(row["src_hash"])["status"] == "valid"
        and generations.get(mutation_job_id(row["src_hash"], error_type, attempt)) is None
        for row in rows
        for error_type in args.mutation_types
        for attempt in range(args.attempts_per_proof)
    )
    if generation_pending:
        if model is None:
            model = make_model()
        generate_candidates(rows, args, model, baseline, generations, rejections)
    verification_phase(args, verifier_pool, generations, verifications, rejections)
    annotation_pending = any(
        verified.get("status") == "accepted_failure"
        and (
            annotations.get(key) is None
            or annotations.get(key).get("status") not in {"annotated", "rejected"}
        )
        for key, verified in verifications.records.items()
    )
    if annotation_pending:
        if model is None:
            model = make_model()
        annotation_phase(args, model, generations, verifications, annotations, rejections)
    return export_outputs(rows, args, generations, verifications, annotations, baseline, rejections)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        counts = run_pipeline(args)
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
