"""Theorem mutation: replace one library-lemma reference with a similar lemma.

Pure logic only; Lean and retrieval are injected by the driver.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import random
import re
from typing import Any, Iterable, Mapping, Sequence

try:
    from .mutate_line import byte_to_char_index, mask_lean_source
    from .mutate_tactic import HEAD_PREFIX, TACTIC_CLASSES
    from .mutation_orchestrator import derive_seed as derived_seed
    from .mutation_utils import require_string
except ImportError:  # Support direct execution from contributions/scripts.
    from mutate_line import byte_to_char_index, mask_lean_source
    from mutate_tactic import HEAD_PREFIX, TACTIC_CLASSES
    from mutation_orchestrator import derive_seed as derived_seed
    from mutation_utils import require_string


ERROR_TYPE = "thm"
LIBRARY_PACKAGES = ("Mathlib", "Batteries", "Init", "Std")
OCCURRENCE_POLICIES = ("each", "random_one_per_lemma")

KEYWORDS = frozenset({
    "_", "at", "by", "calc", "case", "deriving", "do", "else", "end", "exists", "for", "from",
    "fun", "generalizing", "have", "if", "in", "let", "match", "mut", "next", "only", "open",
    "return", "show", "suffices", "then", "this", "using", "where", "with", "λ", "Type",
    "Prop", "Sort", "sorry", "admit", "termination_by", "decreasing_by", "set_option",
})
EXTRA_TACTICS = frozenset({
    "absurd", "bound", "by_contra!", "calc", "cases'", "casesm", "choose", "compute_degree",
    "congr!", "constructorm", "continuity", "conv", "convert", "convert_to", "exacts",
    "existsi", "filter_upwards", "fun_prop", "infer_instance", "injection", "lift",
    "measurability", "nth_rewrite", "nth_rw", "peel", "polyrith", "push_neg", "qify", "rel",
    "rfl", "set", "simp_arith", "split_ands", "symm", "tauto", "trans", "trivial", "use!",
    "wlog", "zify",
})
TACTIC_NAMES = frozenset(t for tactics in TACTIC_CLASSES.values() for t in tactics) | EXTRA_TACTICS

IDENT_PART = r"(?:«[^»\n]+»|[^\W\d][\w'!?]*)"
QUALIFIED_IDENT = re.compile(rf"(?<![\w'!?.»]){IDENT_PART}(?:\.{IDENT_PART})*")
SIMPLE_IDENT = re.compile(r"[^\W\d][\w'!?]*")

BINDER_KEYWORD = re.compile(
    r"(?<![\w'.!?])(?:intro|intros|rintro|fun|λ|obtain|choose|by_contra!?|by_cases|ext|funext|"
    r"generalize|next|case|have|let|set|with|rename_i)(?![\w'!?])"
)
BINDER_STOP = re.compile(r":=|=>|↦|:|\busing\b|\bfrom\b|\bwith\b|<;>|;|\n")
MATCH_ARM = re.compile(r"\|\s*([^\n|]*?)\s*=>")
STATEMENT_BINDER = re.compile(r"[(\[{⦃]\s*([^:()\[\]{}⦃⦄]+?)\s*:(?!=)")

IN_MODIFIER_LINE = re.compile(r"^[ \t]*(?:set_option|open|omit|include|unseal|seal)\b.*\bin[ \t]*$")
ATTRIBUTE_LINE = re.compile(r"^[ \t]*@\[")
UNKNOWN_NAME = re.compile(r"^\s*unknown (?:identifier|constant)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class TargetSpans:
    """Character offsets in ``correct_proof``.

    The prefix ``[:split]`` is compiled once; ``[split:declaration_end]`` is the
    declaration (trailing comments dropped). ``line_offset`` = lines before split.
    """

    declaration_start: int
    proof_start: int
    proof_end: int
    declaration_end: int
    split: int
    line_offset: int


@dataclass(frozen=True)
class IdentToken:
    start: int
    end: int
    text: str
    prefixes: tuple[str, ...]  # longest first: ("foo.symm", "foo")


@dataclass(frozen=True)
class LeanConstant:
    name: str
    kind: str
    module: str | None = None
    statement: str = ""
    deprecated_alias_of: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeanConstant":
        return cls(
            name=str(value["name"]),
            kind=str(value.get("kind") or "other"),
            module=value.get("module"),
            statement=str(value.get("statement") or value.get("type") or ""),
            deprecated_alias_of=value.get("deprecated_alias_of"),
        )


@dataclass(frozen=True)
class LemmaOccurrence:
    occurrence_index: int
    start: int
    end: int
    token: str
    lemma: LeanConstant
    line: int
    col_start: int
    col_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LemmaOccurrence":
        fields = dict(value)
        fields["lemma"] = LeanConstant.from_dict(value["lemma"])
        return cls(**fields)


@dataclass(frozen=True)
class NeighborFilter:
    considered: tuple[str, ...]
    kept: tuple[LeanConstant, ...]
    dropped: tuple[tuple[str, str], ...]  # (candidate name, reason)

    @property
    def kept_names(self) -> list[str]:
        return [constant.name for constant in self.kept]

    def to_dict(self) -> dict[str, Any]:
        return {
            "considered": list(self.considered),
            "kept": [asdict(constant) for constant in self.kept],
            "dropped": [list(item) for item in self.dropped],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NeighborFilter":
        return cls(
            considered=tuple(value["considered"]),
            kept=tuple(LeanConstant.from_dict(item) for item in value["kept"]),
            dropped=tuple((str(name), str(reason)) for name, reason in value["dropped"]),
        )


# ------------------------------------------------------------------ declaration


def offset_to_line_col(text: str, offset: int) -> tuple[int, int]:
    """Return a 1-based line and 0-based column (code points), as Lean reports."""
    line_start = text.rfind("\n", 0, offset) + 1
    return text.count("\n", 0, offset) + 1, offset - line_start


def block_comment_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every top-level block comment."""
    spans: list[tuple[int, int]] = []
    index = 0
    depth = 0
    start = 0
    in_string = False
    while index < len(text):
        if depth:
            if text.startswith("/-", index):
                depth += 1
                index += 2
            elif text.startswith("-/", index):
                depth -= 1
                index += 2
                if depth == 0:
                    spans.append((start, index))
            else:
                index += 1
            continue
        if in_string:
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == '"':
                in_string = False
            index += 1
            continue
        if text.startswith("--", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif text.startswith("/-", index):
            depth = 1
            start = index
            index += 2
        else:
            if text[index] == '"':
                in_string = True
            index += 1
    return spans


def env_split_point(source: str, declaration_start: int) -> int:
    """Move the split before a doc comment, attribute or `... in` modifier."""
    masked = mask_lean_source(source)
    comment_starts = {end: start for start, end in block_comment_spans(source)}
    split = source.rfind("\n", 0, declaration_start) + 1
    while split > 0:
        before_end = len(source[:split].rstrip())
        if before_end == 0:
            break
        line_start = source.rfind("\n", 0, before_end) + 1
        masked_line = masked[line_start:before_end]
        if source.endswith("-/", 0, before_end) and before_end in comment_starts:
            comment_start = comment_starts[before_end]
            comment_line = source.rfind("\n", 0, comment_start) + 1
            if not source.startswith("/--", comment_start) or source[comment_line:comment_start].strip():
                break
            split = comment_line
            continue
        if IN_MODIFIER_LINE.match(masked_line) or ATTRIBUTE_LINE.match(masked_line):
            split = line_start
            continue
        break
    return split


def target_spans(record: Mapping[str, Any]) -> TargetSpans:
    """Validate the extractor spans and compute the split."""
    source = require_string(record, "correct_proof")
    proof = record.get("proof_span")
    declaration = record.get("declaration_span")
    if not isinstance(proof, Mapping) or not isinstance(declaration, Mapping):
        raise ValueError("record needs proof_span and declaration_span objects")
    proof_start = byte_to_char_index(source, int(proof["start_byte"]))
    proof_end = byte_to_char_index(source, int(proof["end_byte"]))
    declaration_start = byte_to_char_index(source, int(declaration["start_byte"]))
    if not 0 <= declaration_start <= proof_start < proof_end <= len(source):
        raise ValueError("invalid declaration_span/proof_span")
    proof_text = record.get("proof_text")
    if proof_text is not None and source[proof_start:proof_end] != proof_text:
        raise ValueError("proof_text does not match proof_span in correct_proof")
    masked = mask_lean_source(source)
    declaration_end = max(proof_start + 1, len(masked[:proof_end].rstrip()))
    split = env_split_point(source, declaration_start)
    return TargetSpans(
        declaration_start, proof_start, proof_end, declaration_end, split, source.count("\n", 0, split)
    )


def declaration_text(source: str, spans: TargetSpans, length_delta: int = 0) -> str:
    """Declaration text; ``length_delta`` accounts for a substitution."""
    return source[spans.split:spans.declaration_end + length_delta]


# --------------------------------------------------------------------- tokens


def at_tactic_head(masked: str, region_start: int, position: int) -> bool:
    line_start = max(region_start, masked.rfind("\n", 0, position) + 1)
    prefix = masked[line_start:position]
    stripped = prefix.strip()
    return not stripped or stripped == "|" or bool(HEAD_PREFIX.search(prefix))


def bound_names(masked_proof: str, masked_statement: str = "") -> set[str]:
    """Local names bound in the proof or the statement (heuristic)."""
    names: set[str] = set()
    for match in BINDER_KEYWORD.finditer(masked_proof):
        stop = BINDER_STOP.search(masked_proof, match.end())
        window = masked_proof[match.end():stop.start() if stop else len(masked_proof)]
        names.update(SIMPLE_IDENT.findall(window))
    for match in MATCH_ARM.finditer(masked_proof):
        names.update(SIMPLE_IDENT.findall(match.group(1))[1:])  # skip the constructor
    for match in STATEMENT_BINDER.finditer(masked_statement):
        names.update(SIMPLE_IDENT.findall(match.group(1)))
    return names - KEYWORDS


def identifier_tokens(record: Mapping[str, Any], spans: TargetSpans) -> list[IdentToken]:
    """Identifier tokens in the proof body, minus keywords, binders and tactic heads."""
    source = require_string(record, "correct_proof")
    masked = mask_lean_source(source)
    bound = bound_names(
        masked[spans.proof_start:spans.proof_end], masked[spans.declaration_start:spans.proof_start]
    )
    tokens: list[IdentToken] = []
    for match in QUALIFIED_IDENT.finditer(masked, spans.proof_start, spans.proof_end):
        text = match.group()
        if "«" in text:
            continue
        parts = text.split(".")
        if parts[0] in KEYWORDS or parts[0] in bound:
            continue
        if text in TACTIC_NAMES and at_tactic_head(masked, spans.proof_start, match.start()):
            continue
        prefixes = tuple(".".join(parts[:count]) for count in range(len(parts), 0, -1))
        tokens.append(IdentToken(match.start(), match.end(), text, prefixes))
    return tokens


def resolution_queries(tokens: Iterable[IdentToken]) -> list[str]:
    return sorted({prefix for token in tokens for prefix in token.prefixes})


# ---------------------------------------------------------------- occurrences


def is_library_module(module: str | None, packages: Sequence[str]) -> bool:
    return module is not None and any(module == package or module.startswith(package + ".") for package in packages)


def enclosing_used_constants(tactics: Sequence[Any], line: int, col: int) -> set[str] | None:
    """Constants used by tactics enclosing the position; None if unknown."""
    used: set[str] = set()
    for tactic in tactics:
        if (tactic.start_line, tactic.start_col) <= (line, col) < (tactic.end_line, tactic.end_col):
            used.update(tactic.used_constants)
    return used or None


def select_occurrences(
    source: str,
    tokens: Sequence[IdentToken],
    resolutions: Mapping[str, Sequence[LeanConstant]],
    tactics: Sequence[Any] = (),
    *,
    packages: Sequence[str] = LIBRARY_PACKAGES,
) -> tuple[list[LemmaOccurrence], dict[str, int]]:
    """Tokens that resolve to a library theorem (longest dotted prefix wins)."""
    skipped: Counter[str] = Counter()
    occurrences: list[LemmaOccurrence] = []
    for token in tokens:
        prefix = next((p for p in token.prefixes if resolutions.get(p)), None)
        if prefix is None:
            skipped["unresolved"] += 1
            continue
        matches = list(resolutions[prefix])
        start, end = token.start, token.start + len(prefix)
        line, col = offset_to_line_col(source, start)
        used = enclosing_used_constants(tactics, line, col)
        if len(matches) > 1 and used:
            matches = [match for match in matches if match.name in used]
        if len(matches) != 1:
            skipped["ambiguous"] += 1
            continue
        lemma = matches[0]
        if lemma.kind != "theorem":
            skipped["not_theorem"] += 1
        elif not is_library_module(lemma.module, packages):
            skipped["not_library"] += 1
        elif used is not None and lemma.name not in used:
            skipped["not_used_by_tactic"] += 1
        else:
            occurrences.append(LemmaOccurrence(
                occurrence_index=len(occurrences), start=start, end=end, token=prefix, lemma=lemma,
                line=line, col_start=col, col_end=col + (end - start),
            ))
    return occurrences, dict(sorted(skipped.items()))


def lemma_matches(occurrence: LemmaOccurrence, name: str) -> bool:
    resolved = occurrence.lemma.name
    return name in {occurrence.token, resolved} or resolved.endswith("." + name)


def choose_occurrences(
    occurrences: Sequence[LemmaOccurrence],
    *,
    src_hash: str,
    global_seed: int,
    policy: str = "each",
    max_candidates: int | None = None,
    lemma: str | None = None,
) -> list[LemmaOccurrence]:
    """Apply --lemma, the occurrence policy and the per-proof cap."""
    if policy not in OCCURRENCE_POLICIES:
        raise ValueError(f"unsupported occurrence policy: {policy}")
    selected = [o for o in occurrences if lemma is None or lemma_matches(o, lemma)]
    if policy == "random_one_per_lemma":
        groups: dict[str, list[LemmaOccurrence]] = defaultdict(list)
        for occurrence in selected:
            groups[occurrence.lemma.name].append(occurrence)
        rng = random.Random(derived_seed(global_seed, src_hash, f"{ERROR_TYPE}-one-per-lemma", 0))
        selected = [rng.choice(groups[name]) for name in sorted(groups)]
    if max_candidates is not None and len(selected) > max_candidates:
        rng = random.Random(derived_seed(global_seed, src_hash, f"{ERROR_TYPE}-occurrence-cap", 0))
        selected = rng.sample(selected, max_candidates)
    return sorted(selected, key=lambda occurrence: occurrence.occurrence_index)


# ------------------------------------------------------------------ neighbours


def last_component(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def filter_neighbors(
    original: LeanConstant,
    candidate_names: Sequence[str],
    resolutions: Mapping[str, Sequence[LeanConstant]],
    *,
    drop_namespace_variants: bool = True,
    drop_deprecated_aliases: bool = True,
    forced: bool = False,
) -> NeighborFilter:
    """Drop candidates that are unavailable, the same lemma, a namespace variant, not a theorem, or a deprecated alias."""
    considered: list[str] = []
    kept: list[LeanConstant] = []
    dropped: list[tuple[str, str]] = []
    for name in candidate_names:
        if name in considered:
            continue
        considered.append(name)
        found = list(resolutions.get(name) or ())
        exact = [constant for constant in found if constant.name == name]
        reason: str | None = None
        if not exact:
            reason = "unavailable"
        elif len(found) > 1:
            reason = "ambiguous"
        else:
            resolved = exact[0]
            if resolved.kind != "theorem":
                reason = "not_theorem"
            elif forced:
                reason = None
            elif resolved.name == original.name:
                reason = "same_constant"
            elif drop_namespace_variants and last_component(resolved.name) == last_component(original.name):
                reason = "namespace_variant"
            elif drop_deprecated_aliases and (
                resolved.deprecated_alias_of == original.name or original.deprecated_alias_of == resolved.name
            ):
                reason = "deprecated_alias"
            elif any(constant.name == resolved.name for constant in kept):
                reason = "duplicate_candidate"
        if reason is None:
            kept.append(exact[0])
        else:
            dropped.append((name, reason))
    return NeighborFilter(tuple(considered), tuple(kept), tuple(dropped))


def choose_substitute(
    kept: Sequence[LeanConstant],
    *,
    global_seed: int,
    src_hash: str,
    occurrence_index: int,
    attempt: int,
    exclude: Iterable[str] = (),
) -> tuple[LeanConstant, int] | None:
    """Seeded uniform choice among candidates not yet tried."""
    excluded = set(exclude)
    pool = sorted((c for c in kept if c.name not in excluded), key=lambda constant: constant.name)
    if not pool:
        return None
    seed = derived_seed(global_seed, src_hash, f"{ERROR_TYPE}:{occurrence_index}", attempt)
    return random.Random(seed).choice(pool), seed


# ----------------------------------------------------------------- substitute


def substitute(source: str, start: int, end: int, replacement: str) -> str:
    if not 0 <= start < end <= len(source):
        raise ValueError("substitution span is outside the source")
    if not replacement:
        raise ValueError("replacement must be nonempty")
    mutated = source[:start] + replacement + source[end:]
    if mutated[:start] != source[:start] or mutated[start + len(replacement):] != source[end:]:
        raise AssertionError("substitution changed text outside the chosen span")
    if mutated == source:
        raise ValueError("substitution left the proof unchanged")
    return mutated


def lemma_card(constant: LeanConstant, written_as: str | None = None) -> dict[str, Any]:
    card: dict[str, Any] = {
        "name": written_as or constant.name,
        "resolved_name": constant.name,
        "statement": f"theorem {constant.name} : {constant.statement}" if constant.statement else None,
        "module": constant.module,
    }
    return card


def build_mutation(
    record: Mapping[str, Any],
    occurrence: LemmaOccurrence,
    replacement: LeanConstant,
    neighbor_filter: NeighborFilter,
    *,
    attempt: int,
    global_seed: int,
    derived: int,
) -> dict[str, Any]:
    """Mutation record with correct and mutated proof."""
    correct = require_string(record, "correct_proof")
    incorrect = substitute(correct, occurrence.start, occurrence.end, replacement.name)
    return {
        "src_hash": require_string(record, "src_hash"),
        "error_type": ERROR_TYPE,
        "occurrence_index": occurrence.occurrence_index,
        "attempt_index": attempt,
        "global_seed": global_seed,
        "derived_seed": derived,
        "correct_proof": correct,
        "incorrect_proof": incorrect,
        "length_delta": len(incorrect) - len(correct),
        "original_lemma": lemma_card(occurrence.lemma, occurrence.token),
        "substitute_lemma": lemma_card(replacement),
        "mutation_site": {
            "line": occurrence.line,
            "col_start": occurrence.col_start,
            "col_end": occurrence.col_end,
            "occurrence_index": occurrence.occurrence_index,
            "start_char": occurrence.start,
            "end_char": occurrence.end,
        },
        "neighbors_considered": list(neighbor_filter.considered),
        "neighbors_after_filter": neighbor_filter.kept_names,
        "neighbors_dropped": [list(item) for item in neighbor_filter.dropped],
    }


# -------------------------------------------------------------------- compile


def normalize_proof(text: str) -> str:
    return " ".join(text.split())


def classify_compile(result: Any, *, reject_unknown_identifier: bool = True) -> str | None:
    """Rejection reason, or None to keep."""
    if getattr(result, "timed_out", False):
        return "timeout"
    if getattr(result, "crashed", False) or getattr(result, "prefix_failed", False):
        return "repl_crash"
    errors = list(getattr(result, "errors", ()) or ())
    if not errors:
        return "still_compiles"
    if reject_unknown_identifier and all(UNKNOWN_NAME.match(error.message) for error in errors):
        return "unknown_identifier"
    if not getattr(result, "state_at_error", None):
        return "missing_structured_proof_state"
    return None
