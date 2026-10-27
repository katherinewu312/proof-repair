#!/usr/bin/env python3
"""Build the Lean dependencies needed by the OpenAI Math mutation dataset.

By default, request each source module's :deps facet. Lake builds its imports
and required plugins, while the mutation script later checks the source itself.
Rerunning is incremental: Lake decides which artifacts are still up to date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def default_input() -> Path:
    requested = REPO_ROOT / "data/openai_math/originals.jsonl"
    return requested if requested.is_file() else REPO_ROOT / "data/raw/openai_math/originals.jsonl"


def load_modules(input_path: Path, project: Path, limit: int | None) -> tuple[list[str], int]:
    """Keep dataset order, deduplicate files, and check the checkout matches."""
    toolchain = (project / "lean-toolchain").read_text(encoding="utf-8").strip()
    modules: dict[str, str] = {}
    records = 0
    with input_path.open(encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, 1):
            if not text.strip():
                continue
            if limit is not None and records >= limit:
                break
            row = json.loads(text)
            prefix = f"{input_path}:{line_number}"
            if not isinstance(row, dict):
                raise ValueError(f"{prefix}: expected a JSON object")
            records += 1
            if row.get("lean_toolchain", toolchain) != toolchain:
                raise ValueError(f"{prefix}: dataset toolchain differs from {project / 'lean-toolchain'}")
            filename = row.get("source_file")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{prefix}: source_file must be a nonempty string")
            relative = PurePosixPath(filename)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".lean":
                raise ValueError(f"{prefix}: source_file must be a project-relative .lean path")
            path = (project / filename).resolve()
            if not path.is_relative_to(project):
                raise ValueError(f"{prefix}: source_file resolves outside the Lean project")
            raw = path.read_bytes()
            expected_hash = row.get("file_sha256")
            if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError(f"{prefix}: {filename} changed; re-extract the dataset before building")
            if "correct_proof" in row and raw.decode("utf-8") != row["correct_proof"]:
                raise ValueError(f"{prefix}: {filename} does not match correct_proof; re-extract the dataset")
            module = ".".join(relative.with_suffix("").parts)
            if any(character in module for character in ":@+\\\n\r\t "):
                raise ValueError(f"{prefix}: unsupported module path: {filename}")
            modules.setdefault(module, filename)
    if not modules:
        raise ValueError(f"No source modules found in {input_path}")
    return list(modules), records


def build_commands(modules: list[str], batch_size: int, full: bool) -> list[list[str]]:
    facet = "olean" if full else "deps"
    return [
        ["lake", "--no-ansi", "build", *[f"+{module}:{facet}" for module in modules[start:start + batch_size]]]
        for start in range(0, len(modules), batch_size)
    ]


def stop_process(process: subprocess.Popen) -> None:
    """Stop only this build's private process group, including compiler children."""
    if os.name == "posix":
        try:
            # The Lake leader can exit while its compiler children still run.
            # Its private group remains ours even after poll() reports an exit.
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + 5
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Give children the grace period even if Lake exits first. Signal 0
        # group probes are blocked by some execution sandboxes, so cleanup
        # does not rely on being able to enumerate/probe surviving children.
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_command(
    command: list[str], project: Path, threads: int, log_path: Path,
    timeout: float | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["LEAN_NUM_THREADS"] = str(threads)
    started = time.monotonic()
    status = "completed"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"LEAN_NUM_THREADS={threads} {shlex.join(command)}\n")
        log.flush()
        with subprocess.Popen(
            command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        ) as process:
            # Tail the log instead of using a pipe: a canceled child cannot
            # keep a pipe open and prevent interruption from completing.
            with log_path.open(encoding="utf-8", errors="replace") as reader:
                try:
                    while True:
                        chunk = reader.read()
                        if chunk:
                            print(chunk, end="", flush=True)
                        if timeout is not None and time.monotonic() - started >= timeout:
                            status = "timed_out"
                            stop_process(process)
                            break
                        try:
                            process.wait(timeout=0.5)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                except KeyboardInterrupt:
                    status = "interrupted"
                    stop_process(process)
                finally:
                    # Also stop children if an output/write exception occurs.
                    stop_process(process)
                    remaining = reader.read()
                    if remaining:
                        print(remaining, end="", flush=True)
    if status == "completed" and process.returncode != 0:
        status = "failed"
    return {
        "command": command, "status": status, "returncode": process.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3), "log": str(log_path),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input())
    parser.add_argument("--project-dir", type=Path, default=REPO_ROOT / "external/openai-math/lean")
    parser.add_argument("--threads", type=int, default=2, help="LEAN_NUM_THREADS worker-pool setting (default: 2)")
    parser.add_argument("--batch-size", type=int, default=8, help="Source modules per Lake invocation (default: 8)")
    parser.add_argument("--limit", type=int, help="Prepare only the first N dataset records, matching mutate_tactic --limit")
    parser.add_argument("--full", action="store_true", help="Build source modules' OLeans too; default builds only dependencies")
    parser.add_argument("--get-cache", action="store_true", help="Run 'lake exe cache get' before building (requires network)")
    parser.add_argument("--keep-going", action="store_true", help="Continue later batches after a failed or timed-out batch")
    parser.add_argument("--timeout", type=float, help="Optional seconds per batch; default has no time limit")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "runs/openai_math_build")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without invoking Lake or writing logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--threads, --batch-size, and --limit must be positive")
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("--timeout must be a finite positive number")
    project, input_path = args.project_dir.resolve(), args.input.resolve()
    modules, records = load_modules(input_path, project, args.limit)
    commands = build_commands(modules, args.batch_size, args.full)
    print(f"Project: {project}\nInput: {input_path}")
    print(f"Preparing {len(modules)} unique source modules from {records} records in {len(commands)} batches.")
    print(f"Mode: {'source OLeans and dependencies' if args.full else 'dependencies only'}; LEAN_NUM_THREADS={args.threads}.")
    print("Existing build artifacts are reused and checked by Lake.", flush=True)
    if args.dry_run:
        if args.get_cache:
            print(f"LEAN_NUM_THREADS={args.threads} lake --no-ansi exe cache get")
        for command in commands:
            print(f"LEAN_NUM_THREADS={args.threads} {shlex.join(command)}")
        return 0
    if shutil.which("lake") is None:
        raise FileNotFoundError("lake was not found on PATH; install/activate elan first")
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "report.json"
    report: dict[str, Any] = {
        "input": str(input_path), "project": str(project), "records": records,
        "modules": modules, "threads": args.threads,
        "mode": "olean" if args.full else "deps", "status": "running", "batches": [],
    }
    write_report(report_path, report)
    if args.get_cache:
        report["cache"] = run_command(
            ["lake", "--no-ansi", "exe", "cache", "get"], project, args.threads,
            log_dir / "cache.log", args.timeout,
        )
        if report["cache"]["status"] != "completed":
            report["status"] = report["cache"]["status"]
            write_report(report_path, report)
            print(f"Cache step failed or was interrupted; see {log_dir / 'cache.log'}", file=sys.stderr)
            return 130 if report["status"] == "interrupted" else 1
        write_report(report_path, report)
    failed = 0
    for index, command in enumerate(commands, 1):
        print(f"\n[Batch {index}/{len(commands)}] {len(command) - 3} source modules", flush=True)
        result = run_command(
            command, project, args.threads, log_dir / f"batch-{index:04d}.log", args.timeout,
        )
        report["batches"].append(result)
        if result["status"] != "completed":
            failed += 1
        report["status"] = "running"
        write_report(report_path, report)
        if result["status"] == "interrupted":
            report["status"] = "interrupted"
            write_report(report_path, report)
            print(f"\nInterrupted. Rerun the same command to reuse completed work. Report: {report_path}")
            return 130
        if result["status"] != "completed" and not args.keep_going:
            break
    report["status"] = "failed" if failed else "completed"
    write_report(report_path, report)
    print(f"\nFinished {len(report['batches'])}/{len(commands)} batches; {failed} failed. Report: {report_path}")
    if failed:
        print("Fix the reported errors and rerun; Lake will reuse completed artifacts.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
