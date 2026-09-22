#!/usr/bin/env python3
"""Compile-audit the sampled APRIL benchmark in its pinned Lean environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from prepare_data import DEFAULT_SEED, canonical_row, to_sft_chat, write_json, write_jsonl


SPLITS = (
    ("train", "train/tme_train.jsonl", 500),
    ("valid", "val/tme_val.jsonl", 100),
    ("test", "test/tme_test.jsonl", 100),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that APRIL gold proofs pass and mutated proofs fail."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--sft-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/april"))
    parser.add_argument("--verifier-dir", type=Path, default=Path("verifier"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/verification"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent Lean processes (default: %(default)s)",
    )
    parser.add_argument(
        "--show-compiler-output",
        action="store_true",
        help="Print captured Lean stdout/stderr after every compiler invocation",
    )
    return parser.parse_args()


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verifier_fingerprint(verifier_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("lean-toolchain", "lakefile.toml", "lake-manifest.json"):
        path = verifier_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def selected_id(split_name: str, row: dict[str, Any]) -> str:
    original = {key: value for key, value in row.items() if key != "selected_id"}
    digest = sha256_text(canonical_row(original))
    return f"tme-{split_name}-{row['src_hash'][:16]}-{digest[:12]}"


def cache_key(environment: str, kind: str, proof: str) -> str:
    return sha256_text(f"{environment}\0{kind}\0{proof}")


def compile_proof(
    *,
    environment: str,
    verifier_dir: Path,
    selected: str,
    kind: str,
    proof: str,
    timeout: float,
) -> dict[str, Any]:
    key = cache_key(environment, kind, proof)
    temporary_path: Path | None = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".lean", delete=False
        ) as temporary:
            temporary.write(proof)
            if not proof.endswith("\n"):
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
        stdout = completed.stdout.replace(str(temporary_path), "<TEMP_FILE>")
        stderr = completed.stderr.replace(str(temporary_path), "<TEMP_FILE>")
        return {
            "cache_key": key,
            "selected_id": selected,
            "kind": kind,
            "proof_sha256": sha256_text(proof),
            "exit_code": completed.returncode,
            "timed_out": False,
            "elapsed_seconds": round(elapsed, 6),
            "stdout": stdout,
            "stderr": stderr,
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
            "cache_key": key,
            "selected_id": selected,
            "kind": kind,
            "proof_sha256": sha256_text(proof),
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


def load_cache(path: Path, environment: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for result in jsonl_rows(path):
        if result.get("verifier_fingerprint") == environment:
            cache[result["cache_key"]] = result
    return cache


def append_results(path: Path, results: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            stream.flush()


def print_result(
    result: dict[str, Any],
    position: int,
    total: int,
    *,
    cached: bool,
    show_compiler_output: bool,
) -> None:
    kind = result["kind"]
    exit_code = result["exit_code"]
    if result["timed_out"]:
        status = "TIMEOUT"
    elif kind == "correct" and exit_code == 0:
        status = "PASS"
    elif kind == "correct":
        status = "UNEXPECTED FAIL"
    elif exit_code != 0:
        status = "EXPECTED FAIL"
    else:
        status = "UNEXPECTED PASS"

    cache_label = " cached" if cached else ""
    print(
        f"[{position}/{total}] {result['selected_id']} {kind}: {status} "
        f"(exit={exit_code}, {result['elapsed_seconds']:.3f}s{cache_label})",
        flush=True,
    )

    unexpected = status in {"TIMEOUT", "UNEXPECTED FAIL", "UNEXPECTED PASS"}
    if show_compiler_output or unexpected:
        for stream_name in ("stdout", "stderr"):
            output = result[stream_name].rstrip()
            if output:
                print(f"  {stream_name}:", flush=True)
                for line in output.splitlines():
                    print(f"    {line}", flush=True)


def verify_rows(
    rows: list[dict[str, Any]],
    *,
    environment: str,
    verifier_dir: Path,
    timeout: float,
    workers: int,
    cache: dict[str, dict[str, Any]],
    results_path: Path,
    show_compiler_output: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    jobs: list[tuple[dict[str, Any], str, str, str]] = []
    paired: dict[str, dict[str, dict[str, Any]]] = {}

    cached_results: list[dict[str, Any]] = []
    for row in rows:
        sid = row["selected_id"]
        paired.setdefault(sid, {})
        for kind, field in (("correct", "correct_proof"), ("incorrect", "incorrect_proof")):
            proof = row[field]
            key = cache_key(environment, kind, proof)
            if key in cache:
                paired[sid][kind] = cache[key]
                cached_results.append(cache[key])
            else:
                jobs.append((row, sid, kind, proof))

    total = len(cached_results) + len(jobs)
    position = 0
    for result in cached_results:
        position += 1
        print_result(
            result,
            position,
            total,
            cached=True,
            show_compiler_output=show_compiler_output,
        )

    if jobs:
        mode = "sequentially" if workers == 1 else f"with {workers} workers"
        print(f"Compiling {len(jobs)} uncached proofs {mode}...", flush=True)

        if workers == 1:
            for _, sid, kind, proof in jobs:
                result = compile_proof(
                    environment=environment,
                    verifier_dir=verifier_dir,
                    selected=sid,
                    kind=kind,
                    proof=proof,
                    timeout=timeout,
                )
                cache[result["cache_key"]] = result
                paired[sid][kind] = result
                append_results(results_path, [result])
                position += 1
                print_result(
                    result,
                    position,
                    total,
                    cached=False,
                    show_compiler_output=show_compiler_output,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_jobs = {
                    executor.submit(
                        compile_proof,
                        environment=environment,
                        verifier_dir=verifier_dir,
                        selected=sid,
                        kind=kind,
                        proof=proof,
                        timeout=timeout,
                    ): (sid, kind)
                    for _, sid, kind, proof in jobs
                }
                for future in as_completed(future_jobs):
                    sid, kind = future_jobs[future]
                    result = future.result()
                    cache[result["cache_key"]] = result
                    paired[sid][kind] = result
                    append_results(results_path, [result])
                    position += 1
                    print_result(
                        result,
                        position,
                        total,
                        cached=False,
                        show_compiler_output=show_compiler_output,
                    )

    return paired


def pair_is_valid(results: dict[str, dict[str, Any]]) -> bool:
    correct = results["correct"]
    incorrect = results["incorrect"]
    return (
        not correct["timed_out"]
        and correct["exit_code"] == 0
        and not incorrect["timed_out"]
        and incorrect["exit_code"] != 0
    )


def deterministic_candidates(
    raw_path: Path,
    split_name: str,
    seed: int,
    excluded_hashes: set[str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in jsonl_rows(raw_path):
        if row.get("error_type") != "tme":
            raise ValueError(f"Non-TME row found in {raw_path}")
        groups.setdefault(row["src_hash"], []).append(row)

    def ranked(value: str) -> str:
        return sha256_text(f"{seed}\0{split_name}\0{value}")

    candidates: list[dict[str, Any]] = []
    for src_hash in sorted(groups, key=ranked):
        if src_hash in excluded_hashes:
            continue
        rows = sorted(groups[src_hash], key=lambda row: ranked(canonical_row(row)))
        candidate = dict(rows[0])
        candidate["selected_id"] = selected_id(split_name, candidate)
        candidates.append(candidate)
    return candidates


def failure_reason(results: dict[str, dict[str, Any]]) -> str:
    correct = results["correct"]
    incorrect = results["incorrect"]
    if correct["timed_out"]:
        return "correct_timeout"
    if correct["exit_code"] != 0:
        return "correct_failed"
    if incorrect["timed_out"]:
        return "incorrect_timeout"
    if incorrect["exit_code"] == 0:
        return "incorrect_compiled"
    return "unknown"


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    root = Path.cwd().resolve()
    verifier_dir = (root / args.verifier_dir).resolve()
    environment = verifier_fingerprint(verifier_dir)
    results_path = args.output_dir / "compiler_results.jsonl"
    cache = load_cache(results_path, environment)
    print(f"Loaded {len(cache)} cached compiler results", flush=True)

    all_rejected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "verifier_fingerprint": environment,
        "lean_toolchain": (verifier_dir / "lean-toolchain").read_text().strip(),
        "timeout_seconds": args.timeout,
        "workers": args.workers,
        "seed": args.seed,
        "splits": {},
    }

    manifest_path = args.benchmark_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for split_name, raw_relative, target_count in SPLITS:
        benchmark_path = args.benchmark_dir / f"{split_name}.jsonl"
        initial = jsonl_rows(benchmark_path)
        if len(initial) != target_count:
            raise ValueError(
                f"{benchmark_path} has {len(initial)} rows; expected {target_count}"
            )

        initial_results = verify_rows(
            initial,
            environment=environment,
            verifier_dir=verifier_dir,
            timeout=args.timeout,
            workers=args.workers,
            cache=cache,
            results_path=results_path,
            show_compiler_output=args.show_compiler_output,
        )

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for row in initial:
            results = initial_results[row["selected_id"]]
            if pair_is_valid(results):
                accepted.append(row)
            else:
                rejected.append(
                    {
                        "selected_id": row["selected_id"],
                        "src_hash": row["src_hash"],
                        "split": split_name,
                        "reason": failure_reason(results),
                        "replacement": False,
                    }
                )

        needed = target_count - len(accepted)
        if needed:
            used_hashes = {row["src_hash"] for row in accepted}
            candidates = deterministic_candidates(
                args.raw_dir / raw_relative,
                split_name,
                args.seed,
                used_hashes,
            )
            cursor = 0
            while needed and cursor < len(candidates):
                # Validate modest batches so we do not compile the entire unused pool.
                batch = candidates[cursor : cursor + max(needed * 2, 8)]
                cursor += len(batch)
                replacement_results = verify_rows(
                    batch,
                    environment=environment,
                    verifier_dir=verifier_dir,
                    timeout=args.timeout,
                    workers=args.workers,
                    cache=cache,
                    results_path=results_path,
                    show_compiler_output=args.show_compiler_output,
                )
                for row in batch:
                    results = replacement_results[row["selected_id"]]
                    if pair_is_valid(results) and row["src_hash"] not in used_hashes and needed:
                        accepted.append(row)
                        used_hashes.add(row["src_hash"])
                        needed -= 1
                    elif not pair_is_valid(results):
                        rejected.append(
                            {
                                "selected_id": row["selected_id"],
                                "src_hash": row["src_hash"],
                                "split": split_name,
                                "reason": failure_reason(results),
                                "replacement": True,
                            }
                        )

            if needed:
                raise RuntimeError(
                    f"Could not find {target_count} valid {split_name} examples; "
                    f"still need {needed}"
                )

        accepted.sort(key=lambda row: row["selected_id"])
        write_jsonl(benchmark_path, accepted)
        write_jsonl(args.sft_dir / f"{split_name}.jsonl", [to_sft_chat(row) for row in accepted])
        all_rejected.extend(rejected)

        manifest_split = manifest["splits"][split_name]
        manifest_split["selected_count"] = len(accepted)
        manifest_split["selected_ids"] = [row["selected_id"] for row in accepted]
        manifest_split["verification_rejections"] = len(rejected)

        summary["splits"][split_name] = {
            "selected": len(accepted),
            "correct_compiled": len(accepted),
            "incorrect_failed": len(accepted),
            "rejected_candidates": len(rejected),
            "replacements_used": sum(
                1 for row in accepted if row["selected_id"] not in {x["selected_id"] for x in initial}
            ),
        }
        print(
            f"{split_name}: {len(accepted)}/{target_count} gold compile; "
            f"{len(accepted)}/{target_count} incorrect fail; "
            f"{len(rejected)} rejected",
            flush=True,
        )

    write_jsonl(args.output_dir / "rejected.jsonl", all_rejected)
    write_json(args.output_dir / "summary.json", summary)
    write_json(manifest_path, manifest)
    print(f"Wrote verification report to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
