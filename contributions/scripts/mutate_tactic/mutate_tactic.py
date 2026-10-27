#!/usr/bin/env python3
"""Generate compiler-verified tactic mutation errors from OpenAI Math originals.

Only a tactic's name is replaced: its arguments, imports, and surrounding source
stay intact. Lean's elaborator identifies tactic occurrences and supplies proof
states; a text search for tactic-like words would also match ordinary terms.
Requires a prepared Lake project and the adjacent ``tactic_probe.lean`` helper.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = Path(__file__).resolve().with_name("tactic_probe.lean")
PROBE_MARKER = "__TACTIC_PROBE__"

# These are the role-based families discussed in the tactic taxonomy. They are
# intentionally broad: replacement tactics need not accept the same arguments.
TACTIC_CLASSES: dict[str, tuple[str, ...]] = {
    "arithmetic": (
        "linarith", "nlinarith", "norm_num", "norm_num1", "ring", "ring_nf",
        "omega", "positivity", "abel", "field_simp", "linear_combination",
    ),
    "rewriting": (
        "rw", "rewrite", "erw", "rwa", "simp", "simpa", "simp_all", "simp_rw",
    ),
    "introduction": ("intro", "intros", "rintro"),
    "proof_construction": (
        "apply", "refine", "refine'", "exact", "assumption", "apply_assumption",
    ),
    "constructors": ("constructor", "left", "right", "exists", "use"),
    "case_analysis": (
        "cases", "rcases", "obtain", "by_cases", "split", "split_ifs",
        "interval_cases", "fin_cases",
    ),
    "induction": ("induction", "induction'", "fun_induction"),
    "contradiction": ("exfalso", "contradiction", "by_contra", "contrapose"),
    "substitution": ("subst", "subst_eqs", "subst_vars"),
    "extensionality": ("ext", "ext1", "funext"),
    "congruence": ("congr", "congrm", "gcongr", "mono"),
    "definitional_reduction": ("unfold", "delta", "dsimp", "change", "show", "cbv"),
    "casts": (
        "norm_cast", "push_cast", "exact_mod_cast", "apply_mod_cast",
        "rw_mod_cast", "assumption_mod_cast",
    ),
    "proof_search": ("aesop", "grind", "solve_by_elim", "apply_rules"),
    "decision": ("decide", "native_decide", "decide_cbv", "bv_decide"),
    "suggestions": ("exact?", "apply?", "rw?", "simp?", "aesop?"),
    "context": (
        "have", "let", "suffices", "specialize", "generalize", "revert", "clear",
        "rename_i",
    ),
    "control": ("try", "first", "repeat", "all_goals", "any_goals", "focus", "case"),
}
TACTIC_TO_CLASS = {
    tactic: family for family, tactics in TACTIC_CLASSES.items() for tactic in tactics
}
RESOURCE_ERRORS = (
    "maximum number of heartbeats", "maximum recursion depth", "maxrecdepth",
    "(deterministic) timeout", "out of memory", "memory allocation failed",
    "excessive memory consumption", "maximum memory", "stack overflow",
)


class ProbeError(RuntimeError):
    """The compiler/helper could not run, rather than rejecting a proof."""


@dataclass(frozen=True)
class Tactic:
    name: str
    start: int  # UTF-8 byte offsets, as reported by Lean.
    end: int
    syntax_start: int
    syntax_end: int
    before: str
    after: str
    has_missing: bool = False


@dataclass(frozen=True)
class ProbeResult:
    tactics: tuple[Tactic, ...]
    errors: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]

    @property
    def compiled(self) -> bool:
        return not self.errors


def proof_bounds(row: dict[str, Any], source: str) -> tuple[int, int]:
    """Validate extractor offsets without confusing Unicode bytes and characters."""
    raw = source.encode("utf-8")
    span = row.get("proof_span")
    if isinstance(span, dict):
        start, end = span.get("start_byte"), span.get("end_byte")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
            raise ValueError("proof_span must contain valid start_byte/end_byte offsets")
        try:
            body = raw[start:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("proof_span splits a UTF-8 character") from error
        if "proof_text" in row and body != row["proof_text"]:
            raise ValueError("proof_span does not match proof_text")
        return start, end
    body = row.get("proof_text")
    if not isinstance(body, str) or not body:
        raise ValueError("a proof_span or an unambiguous proof_text is required")
    first = source.find(body)
    if first < 0 or source.find(body, first + 1) >= 0:
        raise ValueError("proof_text must occur exactly once in correct_proof")
    start = len(source[:first].encode("utf-8"))
    return start, start + len(body.encode("utf-8"))


def source_position(source: str, offset: int) -> tuple[int, int]:
    """Return a one-based line and Unicode character column for a byte offset."""
    prefix = source.encode("utf-8")[:offset].decode("utf-8")
    return prefix.count("\n") + 1, len(prefix.rsplit("\n", 1)[-1]) + 1


def diagnostic_offset(source: str, diagnostic: dict[str, Any]) -> int:
    if type(diagnostic.get("start_byte")) is int:
        return diagnostic["start_byte"]
    pos = diagnostic["pos"]
    lines = source.splitlines(keepends=True)
    line = int(pos["line"]) - 1
    if not 0 <= line < len(lines):
        raise ProbeError("Lean returned a diagnostic outside the source file")
    column = int(pos["column"])
    return len(("".join(lines[:line]) + lines[line][:column]).encode("utf-8"))


class LeanProbe:
    def __init__(self, project: Path, timeout: float, scratch: Path):
        self.project = project
        self.timeout = timeout
        self.path = scratch / "Mutation.lean"

    def compile(self, source: str, bounds: tuple[int, int]) -> ProbeResult:
        self.path.write_text(source, encoding="utf-8")
        command = [
            "lake", "env", "lean", "--run", str(PROBE_PATH), str(self.path),
            str(bounds[0]), str(bounds[1]),
        ]
        # Kill the entire private process group on timeout, including Lake's
        # compiler child, so an expensive tactic cannot leave orphan workers.
        with subprocess.Popen(
            command, cwd=self.project, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            start_new_session=(os.name == "posix"),
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.communicate()
                raise
        if process.returncode != 0:
            detail = (stderr + "\n" + stdout).strip()
            raise ProbeError(f"Lean helper failed (exit {process.returncode}):\n{detail}")
        payload = None
        for line in stdout.splitlines():
            if line.startswith(PROBE_MARKER):
                payload = json.loads(line[len(PROBE_MARKER):])
        if not isinstance(payload, dict) or "messages" not in payload or "tactics" not in payload:
            raise ProbeError(f"Lean helper returned no result:\n{stderr}\n{stdout}")
        raw = source.encode("utf-8")
        tactics = []
        for item in payload["tactics"]:
            start, end = item["start"], item["end"]
            # Use the actual original spelling. Macro-generated tactics may
            # inherit source references while having a different syntax head.
            spelling = raw[start:end].decode("utf-8")
            if spelling != item["head"]:
                continue
            tactics.append(Tactic(
                spelling, start, end, item["syntax_start"], item["syntax_end"],
                item["before"], item["after"], item.get("has_missing", False),
            ))
        messages = tuple(payload["messages"])
        errors = tuple(message for message in messages if message["severity"] == "error")
        if payload.get("compiled") != (not errors):
            raise ProbeError("Lean helper returned inconsistent compiler diagnostics")
        for error in errors:
            text = error["data"]
            if "unknown module prefix" in text or ("object file" in text and "does not exist" in text):
                raise ProbeError(
                    f"Lean dependencies are not built: {text.splitlines()[0]}\n"
                    f"Build the imported modules in {self.project} before running mutations."
                )
        return ProbeResult(tuple(tactics), errors, messages)


def swappable_tactics(result: ProbeResult, bounds: tuple[int, int]) -> list[Tactic]:
    """Deduplicate elaborator/macro wrappers around the same source occurrence."""
    by_range: dict[tuple[int, int], Tactic] = {}
    for tactic in result.tactics:
        if tactic.name not in TACTIC_TO_CLASS or not bounds[0] <= tactic.start < tactic.end <= bounds[1]:
            continue
        key = (tactic.start, tactic.end)
        previous = by_range.get(key)
        if previous is None or tactic.syntax_end - tactic.syntax_start < previous.syntax_end - previous.syntax_start:
            by_range[key] = tactic
    return [by_range[key] for key in sorted(by_range)]


def choose_error(result: ProbeResult, source: str, changed: Tactic) -> dict[str, Any]:
    """Prefer the error at the changed tactic over subsequent recovery errors."""
    def rank(error: dict[str, Any]) -> tuple[int, int]:
        offset = diagnostic_offset(source, error)
        if changed.start <= offset <= changed.start + len(changed.name.encode("utf-8")):
            return 0, offset
        if offset >= changed.start:
            return 1, offset
        return 2, offset
    return min(result.errors, key=rank)


def error_state(result: ProbeResult, source: str, error: dict[str, Any], changed: Tactic) -> str:
    message = error["data"]
    # Unsolved-goal diagnostics themselves contain the final, actual goal state.
    if message.startswith("unsolved goals\n"):
        return message[len("unsolved goals\n"):].strip()
    offset = diagnostic_offset(source, error)
    enclosing = [t for t in result.tactics if t.syntax_start <= offset < t.syntax_end]
    if enclosing:
        closest = min(enclosing, key=lambda t: t.syntax_end - t.syntax_start)
        return closest.before
    preceding = [t for t in result.tactics if t.syntax_end <= offset]
    if preceding:
        # For an unexpected token after a parsed tactic (e.g. changing
        # `exact h` to `assumption h`), report the state AFTER that tactic.
        previous = max(preceding, key=lambda t: (t.syntax_end, t.syntax_start))
        # Missing syntax may elaborate through Lean's error recovery and
        # appear to close goals. Such an after-state is not a real solution.
        return previous.before if previous.has_missing else previous.after
    at_mutation = [t for t in result.tactics if t.start == changed.start]
    if at_mutation:
        return min(at_mutation, key=lambda t: t.syntax_end - t.syntax_start).before
    # A syntax error can stop elaboration before a TacticInfo node exists.
    # The original state before the changed token is still valid: its prefix
    # is byte-for-byte identical. Never invent a goal state from error prose.
    return changed.before


def resource_limited(result: ProbeResult) -> bool:
    return any(
        fragment in error["data"].lower()
        for error in result.errors for fragment in RESOURCE_ERRORS
    )


def mutate_proof(
    row: dict[str, Any], probe: LeanProbe, rng: random.Random,
) -> tuple[dict[str, Any] | None, str, int]:
    source = row.get("correct_proof")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("correct_proof must be a nonempty string")
    bounds = proof_bounds(row, source)
    original = probe.compile(source, bounds)
    if resource_limited(original):
        return None, "original_resource_limit", 0
    if not original.compiled:
        return None, "original_does_not_compile", 0
    if row.get("contains_sorry_or_admit") or any(
        message.get("kind") == "hasSorry" or "declaration uses `sorry`" in message["data"]
        for message in original.messages
    ):
        return None, "original_uses_sorry", 0
    occurrences = swappable_tactics(original, bounds)
    if not occurrences:
        return None, "no_swappable_tactics", 0
    # Shuffling gives the first occurrence uniform probability. Once its
    # replacements are exhausted, exclude that entire line as requested.
    rng.shuffle(occurrences)
    attempted_lines: set[int] = set()
    attempts = 0
    timeouts = 0
    resource_failures = 0
    raw = source.encode("utf-8")
    for occurrence in occurrences:
        line, _ = source_position(source, occurrence.start)
        if line in attempted_lines:
            continue
        attempted_lines.add(line)
        replacements = [
            name for name in TACTIC_CLASSES[TACTIC_TO_CLASS[occurrence.name]]
            if name != occurrence.name
        ]
        rng.shuffle(replacements)
        for replacement in replacements:
            replacement_bytes = replacement.encode("utf-8")
            incorrect = (raw[:occurrence.start] + replacement_bytes + raw[occurrence.end:]).decode("utf-8")
            changed = Tactic(
                replacement, occurrence.start, occurrence.start + len(replacement_bytes),
                occurrence.syntax_start,
                occurrence.syntax_end + len(replacement_bytes) - (occurrence.end - occurrence.start),
                occurrence.before, occurrence.after,
            )
            attempts += 1
            try:
                result = probe.compile(incorrect, (
                    bounds[0], bounds[1] + len(replacement_bytes) - (occurrence.end - occurrence.start),
                ))
            except subprocess.TimeoutExpired:
                timeouts += 1
                continue  # Resource exhaustion is not a tactic mutation error.
            if result.compiled:
                continue
            if resource_limited(result):
                resource_failures += 1
                continue
            error = choose_error(result, incorrect, changed)
            error_line = int(error["pos"]["line"])
            record = {
                "correct_proof": source,
                "incorrect_proof": incorrect,
                "src_hash": row.get("src_hash") or hashlib.sha256(
                    row.get("declaration_text", source).encode("utf-8")
                ).hexdigest(),
                "line": error_line,
                "col": int(error["pos"]["column"]) + 1,
                "error": error["data"],
                # A parser error may point at the empty line after a trailing
                # newline. splitlines() discards precisely that valid line.
                "line_at_error": incorrect.split("\n")[error_line - 1],
                "state_at_error": error_state(result, incorrect, error, changed),
                "error_type": "tme",
            }
            return record, "mutated", attempts
    reason = "exhausted_with_timeouts" if timeouts else (
        "exhausted_with_resource_limits" if resource_failures else "exhausted"
    )
    return None, reason, attempts


def default_input() -> Path:
    requested = REPO_ROOT / "data/openai_math/originals.jsonl"
    extracted = REPO_ROOT / "data/raw/openai_math/originals.jsonl"
    return requested if requested.is_file() else extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input())
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/openai_math/tme.jsonl")
    parser.add_argument("--project-dir", type=Path, default=REPO_ROOT / "external/openai-math/lean")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible selection and replacement order")
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds allowed for each Lean invocation")
    parser.add_argument("--limit", type=int, help="Process only the first N input records (for a smoke run)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--timeout and --limit must be positive")
    project = args.project_dir.resolve()
    input_path, output_path = args.input.resolve(), args.output.resolve()
    if input_path == output_path:
        raise ValueError("--output must differ from --input")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if not PROBE_PATH.is_file():
        raise FileNotFoundError(f"Lean helper not found: {PROBE_PATH}")
    toolchain_file = project / "lean-toolchain"
    if not toolchain_file.is_file():
        raise FileNotFoundError(f"No lean-toolchain in --project-dir: {project}")
    toolchain = toolchain_file.read_text(encoding="utf-8").strip()
    rng = random.Random(args.seed)
    counts: dict[str, int] = {}
    processed = written = attempts = 0
    with tempfile.TemporaryDirectory(prefix="mutate-tactic-") as scratch:
        probe = LeanProbe(project, args.timeout, Path(scratch))
        preflight = "example : True := by trivial\n"
        if not probe.compile(preflight, (0, len(preflight.encode("utf-8")))).compiled:
            raise ProbeError("Lean helper preflight failed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Input: {input_path}\nProject: {project}\nOutput: {output_path}", file=sys.stderr)
        with input_path.open(encoding="utf-8") as stream, output_path.open("w", encoding="utf-8") as output:
            for input_line, text in enumerate(stream, start=1):
                if not text.strip():
                    continue
                if args.limit is not None and processed >= args.limit:
                    break
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError(f"{input_path}:{input_line}: expected a JSON object")
                if row.get("lean_toolchain", toolchain) != toolchain:
                    raise ValueError(
                        f"{input_path}:{input_line}: toolchain mismatch: "
                        f"{row['lean_toolchain']} versus {toolchain}"
                    )
                processed += 1
                label = row.get("theorem") or row.get("src_hash") or f"input line {input_line}"
                try:
                    record, reason, used = mutate_proof(row, probe, rng)
                except subprocess.TimeoutExpired:
                    record, reason, used = None, "original_timeout", 0
                except ValueError as error:
                    raise ValueError(f"{input_path}:{input_line}: {error}") from error
                attempts += used
                counts[reason] = counts.get(reason, 0) + 1
                if record is not None:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    written += 1
                print(f"[{processed}] {label}: {reason} ({used} replacement attempts)", file=sys.stderr, flush=True)
    print(json.dumps({
        "processed": processed, "written": written, "replacement_attempts": attempts,
        "outcomes": counts, "output": str(output_path),
    }, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ProbeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
