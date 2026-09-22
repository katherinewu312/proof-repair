#!/usr/bin/env python3
"""Compile generated APRIL repairs and report single-shot pass@1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_PREDICTIONS = Path("runs/zero_shot/predictions.jsonl")
DEFAULT_RESULTS = Path("runs/zero_shot/compilation_results.jsonl")
DEFAULT_SUMMARY = Path("runs/zero_shot/score.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile generated Lean repairs with the pinned APRIL verifier."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--verifier-dir", type=Path, default=Path("verifier"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--expected", type=int, default=100)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path, *, require_unique_ids: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            selected_id = row.get("selected_id")
            if not isinstance(selected_id, str) or not selected_id:
                raise ValueError(f"{path}:{line_number}: missing selected_id")
            if require_unique_ids and selected_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate {selected_id}")
            seen.add(selected_id)
            rows.append(row)
    return rows


def verifier_fingerprint(verifier_dir: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("lean-toolchain", "lakefile.toml", "lake-manifest.json"):
        path = verifier_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_cached_results(
    path: Path, environment: str
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path, require_unique_ids=False):
        if row.get("verifier_fingerprint") != environment:
            continue
        key = (row["selected_id"], row["candidate_sha256"])
        cached[key] = row
    return cached


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def compile_candidate(
    prediction: dict[str, Any],
    verifier_dir: Path,
    environment: str,
    timeout: float,
) -> dict[str, Any]:
    candidate = prediction.get("candidate")
    if not isinstance(candidate, str):
        raise ValueError(f"{prediction['selected_id']}: candidate must be a string")

    candidate_sha256 = sha256_text(candidate)
    if not candidate.strip():
        return {
            "selected_id": prediction["selected_id"],
            "candidate_sha256": candidate_sha256,
            "compiled": False,
            "exit_code": None,
            "timed_out": False,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": "empty generated candidate",
            "verifier_fingerprint": environment,
        }
    temporary_path: Path | None = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".lean", delete=False
        ) as temporary:
            temporary.write(candidate)
            if candidate and not candidate.endswith("\n"):
                temporary.write("\n")
            temporary_path = Path(temporary.name).resolve()

        completed = subprocess.run(
            ["lake", "env", "lean", str(temporary_path)],
            cwd=verifier_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        return {
            "selected_id": prediction["selected_id"],
            "candidate_sha256": candidate_sha256,
            "compiled": completed.returncode == 0,
            "exit_code": completed.returncode,
            "timed_out": False,
            "elapsed_seconds": round(elapsed, 6),
            "stdout": completed.stdout.replace(str(temporary_path), "<TEMP_FILE>"),
            "stderr": completed.stderr.replace(str(temporary_path), "<TEMP_FILE>"),
            "verifier_fingerprint": environment,
        }
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        if temporary_path is not None:
            stdout = stdout.replace(str(temporary_path), "<TEMP_FILE>")
            stderr = stderr.replace(str(temporary_path), "<TEMP_FILE>")
        return {
            "selected_id": prediction["selected_id"],
            "candidate_sha256": candidate_sha256,
            "compiled": False,
            "exit_code": None,
            "timed_out": True,
            "elapsed_seconds": round(elapsed, 6),
            "stdout": stdout,
            "stderr": stderr,
            "verifier_fingerprint": environment,
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.expected <= 0:
        raise ValueError("--expected must be positive")

    predictions = read_jsonl(args.predictions)
    if len(predictions) != args.expected:
        raise ValueError(
            f"Expected {args.expected} predictions, found {len(predictions)}; "
            "finish generation before scoring"
        )

    verifier_dir = args.verifier_dir.resolve()
    environment = verifier_fingerprint(verifier_dir)
    cached = load_cached_results(args.results, environment)
    final_results: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions, start=1):
        candidate = prediction.get("candidate")
        if not isinstance(candidate, str):
            raise ValueError(f"{prediction['selected_id']}: candidate must be a string")
        key = (prediction["selected_id"], sha256_text(candidate))
        if key in cached:
            result = cached[key]
            cache_label = " cached"
        else:
            result = compile_candidate(
                prediction, verifier_dir, environment, args.timeout
            )
            append_jsonl(args.results, result)
            cached[key] = result
            cache_label = ""
        final_results.append(result)
        status = "PASS" if result["compiled"] else "FAIL"
        print(
            f"[{index}/{args.expected}] {prediction['selected_id']}: {status} "
            f"(exit={result['exit_code']}, {result['elapsed_seconds']:.3f}s{cache_label})",
            flush=True,
        )

    successes = sum(result["compiled"] for result in final_results)
    timeouts = sum(result["timed_out"] for result in final_results)
    summary = {
        "successful_compilations": successes,
        "total": args.expected,
        "pass_at_1": successes / args.expected,
        "score": f"{successes}/{args.expected}",
        "timeouts": timeouts,
        "predictions": str(args.predictions),
        "results": str(args.results),
        "verifier_fingerprint": environment,
    }
    write_json(args.summary, summary)
    print(f"\nScore: {successes}/{args.expected} ({100 * successes / args.expected:.1f}%)")
    print(f"Wrote score to {args.summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
