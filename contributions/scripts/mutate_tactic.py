"""Generate syntactic tactic replacements; verification belongs to lean_verifier.

Each result contains only correct_proof and incorrect_proof. A replacement is
not guaranteed to fail compilation. No Lean installation or model is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    return "".join(out)

# Match whole Lean identifiers, including prime and suggestion suffixes.
TOKEN = re.compile(r"[^\W\d][\w'?!]*", re.UNICODE)
HEAD_PREFIX = re.compile(r"(?:\bby|;|<;>|=>|[·{}]|\b(?:try|repeat|all_goals|any_goals|focus))\s*$")


def proof_region(record: Mapping[str, Any], source: str) -> tuple[int, int]:
    """Use extractor metadata when supplied; otherwise scan the source."""
    span = record.get("proof_span")
    if span:
        raw = source.encode("utf-8")
        return (
            len(raw[:span["start_byte"]].decode("utf-8")),
            len(raw[:span["end_byte"]].decode("utf-8")),
        )
    body = record.get("proof_text")
    if body:
        start = source.find(body)
        if start >= 0:
            return start, start + len(body)
    return 0, len(source)


def eligible_tactics(record: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """Find likely tactic heads, ignoring comments, strings and qualified names.

    This is a lexical heuristic, not a Lean parser. In particular, custom macros
    and unusual tactic layouts may be missed.
    """
    source = record["correct_proof"]
    start, end = proof_region(record, source)
    masked = mask_lean_source(source)
    occurrences = []
    for match in TOKEN.finditer(masked, start, end):
        name = match.group()
        if name not in TACTIC_TO_CLASS:
            continue
        # Qualified identifiers are terms rather than tactic heads.
        if match.start() and masked[match.start() - 1] == ".":
            continue
        prefix = masked[max(start, masked.rfind("\n", start, match.start()) + 1):match.start()]
        stripped = prefix.strip()
        if not stripped or stripped == "|" or HEAD_PREFIX.search(prefix):
            occurrences.append((match.start(), match.end(), name))
    return occurrences


def mutate_proof(
    record: Mapping[str, Any], *, global_seed: int = 0,
    rng: random.Random | None = None,
) -> dict[str, str] | None:
    """Replace one likely tactic head; return None when none is eligible."""
    rng = rng if rng is not None else random.Random(global_seed)
    occurrences = eligible_tactics(record)
    if not occurrences:
        return None
    start, end, name = rng.choice(occurrences)
    replacement = rng.choice([
        tactic for tactic in TACTIC_CLASSES[TACTIC_TO_CLASS[name]] if tactic != name
    ])
    correct = record["correct_proof"]
    return {
        "correct_proof": correct,
        "incorrect_proof": correct[:start] + replacement + correct[end:],
    }


def mutate_dataset(
    proofs: Sequence[Mapping[str, Any]], *, global_seed: int = 0,
) -> list[dict[str, str]]:
    """Generate one pair per eligible record, omitting records without tactics."""
    rng = random.Random(global_seed)
    pairs = (mutate_proof(record, rng=rng) for record in proofs)
    return [pair for pair in pairs if pair is not None]


def default_input() -> Path:
    preferred = REPO_ROOT / "data/openai_math/originals.jsonl"
    return preferred if preferred.is_file() else REPO_ROOT / "data/raw/openai_math/originals.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input())
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/openai_math/tme.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    # Protect the input from accidental truncation; no proof/environment checks.
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    rng = random.Random(args.seed)
    processed = written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as stream, args.output.open("w", encoding="utf-8") as output:
        for text in stream:
            if not text.strip():
                continue
            if args.limit is not None and processed >= args.limit:
                break
            pair = mutate_proof(json.loads(text), rng=rng)
            processed += 1
            if pair is not None:
                output.write(json.dumps(pair, ensure_ascii=False) + "\n")
                output.flush()
                written += 1
    print(f"Processed {processed}; wrote {written} pairs to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
