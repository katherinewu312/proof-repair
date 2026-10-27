"""Unit tests for safe, incremental dataset-module build planning.

These tests use tiny source fixtures and mock build execution; they never run Lake.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "build_modules_under_test", Path(__file__).with_name("build_modules.py"),
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class BuildFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="test-build-modules-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "lean"
        self.project.mkdir()
        self.toolchain = "leanprover/lean4:v4.34.1"
        (self.project / "lean-toolchain").write_text(self.toolchain + "\n")
        self.dataset = self.root / "originals.jsonl"
        self.rows = []
        for filename in ("OAI/First.lean", "OAI/Second.lean", "OAI/Third.lean"):
            path = self.project / filename
            path.parent.mkdir(exist_ok=True)
            source = f"-- {filename}\nexample : True := by trivial\n"
            path.write_text(source, encoding="utf-8")
            self.rows.append({
                "source_file": filename,
                "correct_proof": source,
                "file_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "lean_toolchain": self.toolchain,
            })
        self.write_dataset(self.rows)

    def write_dataset(self, rows):
        self.dataset.write_text(
            "\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    def args(self, **overrides):
        fields = dict(
            input=self.dataset, project_dir=self.project, threads=2, batch_size=1,
            limit=None, full=False, get_cache=False, keep_going=False,
            timeout=None, log_dir=self.root / "logs", dry_run=False,
        )
        fields.update(overrides)
        return argparse.Namespace(**fields)

    @staticmethod
    def outcome(status):
        return {
            "command": ["mock-lake"], "status": status,
            "returncode": 0 if status == "completed" else 1,
            "elapsed_seconds": 0, "log": "mock.log",
        }

    def invoke_main(self, args, outcomes=()):
        output, errors = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(builder, "parse_args", return_value=args),
            mock.patch.object(builder.shutil, "which", return_value="/mock/lake"),
            mock.patch.object(builder, "run_command", side_effect=outcomes) as run,
            mock.patch.object(builder.subprocess, "Popen") as popen,
            redirect_stdout(output), redirect_stderr(errors),
        ):
            status = builder.main()
        popen.assert_not_called()
        return status, run, output.getvalue(), errors.getvalue()


class DatasetValidationTests(BuildFixture):
    def test_preserves_first_seen_module_order_and_counts_duplicate_records(self):
        self.write_dataset([self.rows[1], self.rows[0], self.rows[1], self.rows[2]])
        modules, records = builder.load_modules(self.dataset, self.project, None)
        self.assertEqual(modules, ["OAI.Second", "OAI.First", "OAI.Third"])
        self.assertEqual(records, 4)

    def test_limit_counts_records_including_duplicates_and_ignores_blank_lines(self):
        self.write_dataset([self.rows[1], self.rows[1], self.rows[0]])
        self.assertEqual(
            builder.load_modules(self.dataset, self.project, 2),
            (["OAI.Second"], 2),
        )

    def test_stale_source_is_rejected_even_if_its_module_is_duplicated(self):
        stale = dict(self.rows[0], file_sha256="0" * 64)
        self.write_dataset([self.rows[0], stale])
        with self.assertRaisesRegex(ValueError, "changed; re-extract"):
            builder.load_modules(self.dataset, self.project, None)

    def test_correct_proof_is_checked_in_addition_to_file_hash(self):
        self.write_dataset([dict(self.rows[0], correct_proof="different source")])
        with self.assertRaisesRegex(ValueError, "does not match correct_proof"):
            builder.load_modules(self.dataset, self.project, None)

    def test_toolchain_mismatch_is_rejected_before_building(self):
        self.write_dataset([dict(self.rows[0], lean_toolchain="leanprover/lean4:v4.28.0")])
        with self.assertRaisesRegex(ValueError, "toolchain differs"):
            builder.load_modules(self.dataset, self.project, None)

    def test_source_path_traversal_absolute_paths_and_external_symlinks_are_rejected(self):
        outside = self.root / "Outside.lean"
        outside.write_text("example : True := by trivial\n")
        symlink = self.project / "Escape.lean"
        symlink.symlink_to(outside)
        for filename in ("../Outside.lean", str(outside), "Escape.lean"):
            with self.subTest(filename=filename):
                self.write_dataset([{"source_file": filename}])
                with self.assertRaisesRegex(ValueError, "project-relative|outside the Lean project"):
                    builder.load_modules(self.dataset, self.project, None)

    def test_dataset_requires_objects_and_lean_source_paths(self):
        for row, message in (
            ([], "expected a JSON object"),
            ({}, "source_file must be a nonempty string"),
            ({"source_file": "OAI/First.txt"}, "project-relative .lean path"),
        ):
            with self.subTest(row=row):
                self.write_dataset([row])
                with self.assertRaisesRegex(ValueError, message):
                    builder.load_modules(self.dataset, self.project, None)

    def test_lake_target_punctuation_in_filenames_is_rejected(self):
        filename = "OAI/Bad:Target.lean"
        (self.project / filename).write_text("example : True := by trivial\n")
        self.write_dataset([{"source_file": filename}])
        with self.assertRaisesRegex(ValueError, "unsupported module path"):
            builder.load_modules(self.dataset, self.project, None)


class BuildPlanTests(unittest.TestCase):
    def test_default_plan_requests_dependencies_in_bounded_ordered_batches(self):
        self.assertEqual(builder.build_commands(["OAI.A", "OAI.B", "OAI.C"], 2, False), [
            ["lake", "--no-ansi", "build", "+OAI.A:deps", "+OAI.B:deps"],
            ["lake", "--no-ansi", "build", "+OAI.C:deps"],
        ])

    def test_full_plan_requests_source_oleans(self):
        self.assertEqual(builder.build_commands(["OAI.A", "OAI.B"], 8, True), [
            ["lake", "--no-ansi", "build", "+OAI.A:olean", "+OAI.B:olean"],
        ])


class MainControlFlowTests(BuildFixture):
    def test_default_stops_after_a_failed_batch_and_records_failure(self):
        args = self.args()
        status, run, _, _ = self.invoke_main(args, [self.outcome("failed")])
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 1)
        report = json.loads((args.log_dir / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["batches"]), 1)
        self.assertEqual(report["modules"], ["OAI.First", "OAI.Second", "OAI.Third"])

    def test_keep_going_runs_later_batches_but_preserves_failed_exit_status(self):
        args = self.args(keep_going=True)
        status, run, _, _ = self.invoke_main(args, [
            self.outcome("timed_out"), self.outcome("completed"), self.outcome("completed"),
        ])
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 3)
        report = json.loads((args.log_dir / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual([batch["status"] for batch in report["batches"]], [
            "timed_out", "completed", "completed",
        ])

    def test_dry_run_prints_cache_and_build_plan_without_subprocesses_or_logs(self):
        args = self.args(dry_run=True, get_cache=True, full=True)
        status, run, output, _ = self.invoke_main(args)
        self.assertEqual(status, 0)
        run.assert_not_called()
        self.assertFalse(args.log_dir.exists())
        self.assertIn("lake --no-ansi exe cache get", output)
        self.assertIn("+OAI.First:olean", output)
        self.assertIn("LEAN_NUM_THREADS=2", output)

    def test_interruption_stops_even_with_keep_going(self):
        args = self.args(keep_going=True)
        status, run, _, _ = self.invoke_main(args, [
            self.outcome("completed"), self.outcome("interrupted"),
        ])
        self.assertEqual(status, 130)
        self.assertEqual(run.call_count, 2)
        report = json.loads((args.log_dir / "report.json").read_text())
        self.assertEqual(report["status"], "interrupted")

    def test_failed_cache_step_stops_before_build_batches(self):
        args = self.args(get_cache=True, keep_going=True)
        status, run, _, _ = self.invoke_main(args, [self.outcome("failed")])
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["lake", "--no-ansi", "exe", "cache", "get"])
        report = json.loads((args.log_dir / "report.json").read_text())
        self.assertEqual(report["batches"], [])
        self.assertEqual(report["cache"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
