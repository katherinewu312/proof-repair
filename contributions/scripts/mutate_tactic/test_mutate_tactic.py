"""Behavioral tests for mutation selection; no Lean installation is required.

Run with ``python3 -m unittest discover -s contributions/scripts -p 'test_*.py'``.
Real Lean compilation and syntax-tree extraction are checked separately.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("mutate_tactic.py")
SPEC = importlib.util.spec_from_file_location("mutate_tactic_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mutation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mutation
SPEC.loader.exec_module(mutation)


class InOrder:
    """Fix the randomized order so retry behavior can be asserted exactly."""

    def shuffle(self, values):
        pass


class FakeProbe:
    def __init__(self, original, replacements):
        self.original = original
        self.replacements = replacements
        self.calls = []

    def compile(self, source, bounds):
        self.calls.append((source, bounds))
        if len(self.calls) == 1:
            return self.original
        return self.replacements(source, bounds)


def fixture(source, body):
    start = len(source[:source.index(body)].encode("utf-8"))
    return {
        "correct_proof": source,
        "proof_text": body,
        "proof_span": {
            "start_byte": start,
            "end_byte": start + len(body.encode("utf-8")),
        },
        "src_hash": "the-source-theorem-hash",
    }


def tactic(source, token, *, occurrence=0, before="P : Prop\nh : P\n⊢ P", after="no goals"):
    start_char = -1
    for _ in range(occurrence + 1):
        start_char = source.index(token, start_char + 1)
    start = len(source[:start_char].encode("utf-8"))
    end = start + len(token.encode("utf-8"))
    return mutation.Tactic(token, start, end, start, end, before, after)


def result(*tactics, errors=(), messages=()):
    return mutation.ProbeResult(tuple(tactics), tuple(errors), tuple(messages))


def diagnostic(source, token, text="tactic failed", *, occurrence=0):
    at = tactic(source, token, occurrence=occurrence)
    line, column = mutation.source_position(source, at.start)
    return {
        "severity": "error",
        "start_byte": at.start,
        "pos": {"line": line, "column": column - 1},
        "data": text,
    }


class MutationBehaviorTests(unittest.TestCase):
    def test_changes_one_tactic_and_preserves_unicode_context_and_output_schema(self):
        source = (
            "import Mathlib\n"
            "-- θ 🌱 context\n"
            "example (P : Prop) (h : P) : P := by\n"
            "  exact h -- exact must stay in this comment\n"
        )
        row = fixture(source, "by\n  exact h -- exact must stay in this comment\n")
        original_tactic = tactic(source, "exact")
        expected = source.replace("  exact h", "  apply h", 1)
        expected_error = "application failed at the changed tactic"

        def replacement(incorrect, bounds):
            changed = tactic(incorrect, "apply", before="P : Prop\nh : P\n⊢ P")
            # A recovery diagnostic can precede the actual error in the list.
            recovery = diagnostic(incorrect, "by", "unsolved goals\n⊢ P")
            return result(changed, errors=(
                recovery, diagnostic(incorrect, "apply", expected_error),
            ))

        probe = FakeProbe(result(original_tactic), replacement)
        record, reason, attempts = mutation.mutate_proof(row, probe, InOrder())
        self.assertEqual((reason, attempts), ("mutated", 1))
        self.assertEqual(set(record), {
            "correct_proof", "incorrect_proof", "src_hash", "line", "col",
            "error", "line_at_error", "state_at_error", "error_type",
        })
        self.assertEqual(record["correct_proof"], source)
        self.assertEqual(record["incorrect_proof"], expected)
        self.assertEqual(record["src_hash"], row["src_hash"])
        self.assertEqual((record["line"], record["col"]), (4, 3))
        self.assertEqual(record["error"], expected_error)
        self.assertEqual(record["line_at_error"], expected.splitlines()[3])
        self.assertEqual(record["state_at_error"], "P : Prop\nh : P\n⊢ P")
        self.assertEqual(record["error_type"], "tme")
        self.assertEqual(probe.calls[1][1][1], row["proof_span"]["end_byte"])

    def test_exhausts_selected_class_then_excludes_the_whole_attempted_line(self):
        source = (
            "example (P Q : Prop) : P → Q → P := by\n"
            "  intro h; have hx := h\n"
            "  rintro k\n"
            "  exact hx\n"
        )
        row = fixture(source, "by\n  intro h; have hx := h\n  rintro k\n  exact hx\n")
        baseline = result(
            tactic(source, "intro"),
            tactic(source, "have"),
            tactic(source, "rintro"),
        )

        def replacement(incorrect, bounds):
            if "\n  intro k\n" in incorrect:
                return result(errors=(diagnostic(incorrect, "intro", occurrence=1),))
            return result()

        probe = FakeProbe(baseline, replacement)
        record, reason, attempts = mutation.mutate_proof(row, probe, InOrder())
        self.assertEqual((reason, attempts), ("mutated", 3))
        self.assertEqual([text for text, _ in probe.calls[1:]], [
            source.replace("intro h", "intros h", 1),
            source.replace("intro h", "rintro h", 1),
            source.replace("rintro k", "intro k", 1),
        ])
        self.assertIn("intro h; have hx := h", record["incorrect_proof"])
        self.assertEqual(record["line"], 3)
        # A shorter token must also shorten the proof's byte range.
        self.assertEqual(probe.calls[-1][1][1], row["proof_span"]["end_byte"] - 1)

    def test_invalid_original_is_omitted_without_trying_replacements(self):
        source = "example : False := by\n  assumption\n"
        row = fixture(source, "by\n  assumption\n")
        probe = FakeProbe(
            result(tactic(source, "assumption"), errors=(diagnostic(source, "assumption"),)),
            lambda *_: self.fail("A broken original must not be mutated"),
        )
        self.assertEqual(
            mutation.mutate_proof(row, probe, InOrder()),
            (None, "original_does_not_compile", 0),
        )
        self.assertEqual(len(probe.calls), 1)

    def test_all_successful_replacements_omit_the_proof(self):
        source = "example : True → True := by\n  intro h\n  exact h\n"
        row = fixture(source, "by\n  intro h\n  exact h\n")
        probe = FakeProbe(result(tactic(source, "intro")), lambda *_: result())
        self.assertEqual(
            mutation.mutate_proof(row, probe, InOrder()),
            (None, "exhausted", 2),
        )
        self.assertEqual(len(probe.calls), 3)

    def test_timeout_is_skipped_until_a_real_compiler_error_is_found(self):
        source = "example : True → True := by\n  intro h\n  exact h\n"
        row = fixture(source, "by\n  intro h\n  exact h\n")
        original_tactic = tactic(source, "intro", before="⊢ True → True")

        def replacement(incorrect, bounds):
            if "intros" in incorrect:
                raise subprocess.TimeoutExpired(["lean"], 1)
            return result(errors=(diagnostic(incorrect, "rintro", "unexpected token"),))

        probe = FakeProbe(result(original_tactic), replacement)
        record, reason, attempts = mutation.mutate_proof(row, probe, InOrder())
        self.assertEqual((reason, attempts), ("mutated", 2))
        self.assertIn("rintro h", record["incorrect_proof"])
        self.assertEqual(record["error"], "unexpected token")
        # Parsing may fail before new tactic info exists; preserve the real
        # original pre-state, since the preceding source has not changed.
        self.assertEqual(record["state_at_error"], original_tactic.before)

    def test_only_timeouts_do_not_produce_a_mutation_record(self):
        source = "example : True → True := by\n  intro h\n  exact h\n"
        row = fixture(source, "by\n  intro h\n  exact h\n")

        def timeout(*_):
            raise subprocess.TimeoutExpired(["lean"], 1)

        probe = FakeProbe(result(tactic(source, "intro")), timeout)
        self.assertEqual(
            mutation.mutate_proof(row, probe, InOrder()),
            (None, "exhausted_with_timeouts", 2),
        )

    def test_infrastructure_failure_propagates_instead_of_becoming_a_tme(self):
        source = "example : True → True := by\n  intro h\n  exact h\n"
        row = fixture(source, "by\n  intro h\n  exact h\n")

        def infrastructure_error(*_):
            raise mutation.ProbeError("missing imported .olean")

        probe = FakeProbe(result(tactic(source, "intro")), infrastructure_error)
        with self.assertRaisesRegex(mutation.ProbeError, "missing imported"):
            mutation.mutate_proof(row, probe, InOrder())

    def test_eof_diagnostic_preserves_the_empty_line_after_a_trailing_newline(self):
        source = "example : True := by\n  exact True.intro\n"
        row = fixture(source, "by\n  exact True.intro\n")

        def replacement(incorrect, bounds):
            return result(errors=({
                "severity": "error",
                "start_byte": len(incorrect.encode("utf-8")),
                "pos": {"line": incorrect.count("\n") + 1, "column": 0},
                "data": "unexpected end of input",
            },))

        probe = FakeProbe(result(tactic(source, "exact")), replacement)
        record, reason, attempts = mutation.mutate_proof(row, probe, InOrder())
        self.assertEqual((reason, attempts), ("mutated", 1))
        self.assertEqual((record["line"], record["col"]), (3, 1))
        self.assertEqual(record["line_at_error"], "")
        self.assertEqual(record["error"], "unexpected end of input")

    def test_resource_limits_are_retried_and_never_saved_as_tactic_errors(self):
        source = "example : True → True := by\n  intro h\n  exact h\n"
        row = fixture(source, "by\n  intro h\n  exact h\n")
        for resource_error in (
            "(deterministic) timeout at `whnf`, maximum number of heartbeats (200000) has been reached",
            "maximum recursion depth has been reached",
            "out of memory",
        ):
            with self.subTest(resource_error=resource_error):
                def replacement(incorrect, bounds):
                    if "intros h" in incorrect:
                        return result(errors=(diagnostic(incorrect, "intros", resource_error),))
                    return result(errors=(diagnostic(incorrect, "rintro", "real tactic failure"),))

                probe = FakeProbe(result(tactic(source, "intro")), replacement)
                record, reason, attempts = mutation.mutate_proof(row, probe, InOrder())
                self.assertEqual((reason, attempts), ("mutated", 2))
                self.assertEqual(record["error"], "real tactic failure")
                self.assertIn("rintro h", record["incorrect_proof"])

        probe = FakeProbe(
            result(tactic(source, "intro")),
            lambda incorrect, _: result(errors=(diagnostic(incorrect, "intro", "out of memory"),)),
        )
        self.assertEqual(
            mutation.mutate_proof(row, probe, InOrder()),
            (None, "exhausted_with_resource_limits", 2),
        )

    def test_baseline_sorry_warning_is_omitted_even_when_compilation_succeeds(self):
        source = "example : True := by\n  exact True.intro\n"
        row = fixture(source, "by\n  exact True.intro\n")
        warnings = (
            {"severity": "warning", "kind": "hasSorry", "data": "declaration uses `sorry`"},
            {"severity": "warning", "kind": "hasSorry", "data": "proof placeholder"},
            {"severity": "warning", "data": "declaration uses `sorry`"},
        )
        for warning in warnings:
            with self.subTest(warning=warning):
                probe = FakeProbe(
                    result(tactic(source, "exact"), messages=(warning,)),
                    lambda *_: self.fail("An admitted original must not be mutated"),
                )
                self.assertEqual(
                    mutation.mutate_proof(row, probe, InOrder()),
                    (None, "original_uses_sorry", 0),
                )
                self.assertEqual(len(probe.calls), 1)


class ProofBoundaryTests(unittest.TestCase):
    def test_utf8_bounds_are_bytes_and_source_columns_are_characters(self):
        source = "theorem α : True := by\n  exact True.intro\n"
        body = "by\n  exact True.intro\n"
        row = fixture(source, body)
        bounds = mutation.proof_bounds(row, source)
        self.assertEqual(source.encode()[bounds[0]:bounds[1]].decode(), body)
        self.assertNotEqual(bounds[0], source.index(body))
        at = len("theorem α : True := ".encode())
        self.assertEqual(mutation.source_position(source, at), (1, 21))

    def test_invalid_spans_cannot_silently_change_another_source_region(self):
        source = "α by exact True.intro"
        cases = [
            ({"start_byte": 1, "end_byte": len(source.encode())}, "UTF-8"),
            ({"start_byte": 0, "end_byte": len(source.encode()) + 1}, "valid"),
            ({"start_byte": True, "end_byte": 3}, "valid"),
            ({"start_byte": 0, "end_byte": 2}, "match proof_text"),
        ]
        for span, message in cases:
            with self.subTest(span=span), self.assertRaisesRegex(ValueError, message):
                mutation.proof_bounds({"proof_span": span, "proof_text": "different"}, source)

    def test_fallback_requires_unambiguous_proof_text(self):
        source = "-- α\nexample : True := by trivial\n"
        body = "by trivial"
        start, end = mutation.proof_bounds({"proof_text": body}, source)
        self.assertEqual(source.encode()[start:end].decode(), body)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            mutation.proof_bounds({"proof_text": body}, source + source)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            mutation.proof_bounds({"proof_text": "by assumption"}, source)

    def test_target_filter_removes_context_tactics_and_duplicate_macro_wrappers(self):
        source = (
            "example : True := by exact True.intro\n"
            "example : True := by intro h; exact h\n"
            "example : True := by exact True.intro\n"
        )
        body = "by intro h; exact h"
        row = fixture(source, body)
        selected = tactic(source, "exact", occurrence=1)
        wrapper = mutation.Tactic(
            selected.name, selected.start, selected.end,
            selected.syntax_start - 10, selected.syntax_end + 10,
            "wrapper state", "wrapper end",
        )
        candidates = mutation.swappable_tactics(result(
            tactic(source, "exact"), wrapper, selected,
            tactic(source, "exact", occurrence=2),
        ), mutation.proof_bounds(row, source))
        self.assertEqual(candidates, [selected])


if __name__ == "__main__":
    unittest.main()
