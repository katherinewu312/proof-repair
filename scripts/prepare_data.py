#!/usr/bin/env python3
"""Build a small, deterministic TME proof-repair benchmark from APRIL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260921
SYSTEM_PROMPT = (
    "You are a Lean 4 proof repair assistant. Return only the complete corrected "
    "Lean file. Do not use Markdown fences and do not include an explanation."
)

SPLITS = (
    ("train", "train/tme_train.jsonl", 500),
    ("valid", "val/tme_val.jsonl", 100),
    ("test", "test/tme_test.jsonl", 100),
)

REQUIRED_FIELDS = (
    "correct_proof",
    "incorrect_proof",
    "src_hash",
    "error",
    "state_at_error",
    "error_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select theorem-distinct APRIL TME examples and produce chat SFT JSONL."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/april"),
        help="APRIL directory containing train/, val/, and test/ (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Output directory for sampled original records (default: %(default)s)",
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("data/sft"),
        help="Output directory for chat SFT JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: %(default)s)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_row(row).encode("utf-8")).hexdigest()


def load_and_group(path: Path, expected_split: str) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    if not path.is_file():
        raise FileNotFoundError(f"APRIL input file not found: {path}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_count = 0

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error

            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing required fields: {', '.join(missing)}"
                )

            if row["error_type"] != "tme":
                raise ValueError(
                    f"{path}:{line_number}: expected error_type='tme', "
                    f"got {row['error_type']!r}"
                )

            source_split = row.get("split")
            accepted_splits = {expected_split}
            if expected_split == "valid":
                accepted_splits.add("val")
            if source_split is not None and source_split not in accepted_splits:
                raise ValueError(
                    f"{path}:{line_number}: expected split in "
                    f"{sorted(accepted_splits)!r}, got {source_split!r}"
                )

            src_hash = row["src_hash"]
            if not isinstance(src_hash, str) or not src_hash:
                raise ValueError(f"{path}:{line_number}: src_hash must be a non-empty string")

            for field in ("correct_proof", "incorrect_proof", "error", "state_at_error"):
                if not isinstance(row[field], str):
                    raise ValueError(f"{path}:{line_number}: {field} must be a string")

            groups[src_hash].append(row)
            row_count += 1

    return row_count, groups


def select_rows(
    groups: dict[str, list[dict[str, Any]]],
    count: int,
    split_name: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    hashes = sorted(groups)
    if len(hashes) < count:
        raise ValueError(
            f"Cannot select {count} theorem-distinct {split_name} examples: "
            f"only {len(hashes)} unique src_hash values are available"
        )

    selected: list[dict[str, Any]] = []
    for src_hash in rng.sample(hashes, count):
        # Sorting first makes the choice stable even if the source JSONL is reordered.
        candidates = sorted(groups[src_hash], key=canonical_row)
        original = rng.choice(candidates)
        digest = row_digest(original)
        enriched = dict(original)
        enriched["selected_id"] = (
            f"tme-{split_name}-{src_hash[:16]}-{digest[:12]}"
        )
        selected.append(enriched)

    return sorted(selected, key=lambda row: row["selected_id"])


def user_prompt(row: dict[str, Any]) -> str:
    return (
        "Repair the following Lean 4 file.\n\n"
        f"LEAN ERROR:\n{row['error']}\n\n"
        f"PROOF STATE:\n{row['state_at_error']}\n\n"
        f"INCORRECT FILE:\n{row['incorrect_proof']}"
    )


def to_sft_chat(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_id": row["selected_id"],
        "src_hash": row["src_hash"],
        "source": row.get("source"),
        "error_type": row["error_type"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row)},
            {"role": "assistant", "content": row["correct_proof"]},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    manifest: dict[str, Any] = {
        "dataset": "uw-math-ai/APRIL",
        "error_type": "tme",
        "seed": args.seed,
        "splits": {},
    }

    for output_name, relative_input, sample_count in SPLITS:
        input_path = args.input_dir / relative_input
        row_count, groups = load_and_group(input_path, output_name)
        selected = select_rows(groups, sample_count, output_name, rng)

        benchmark_path = args.benchmark_dir / f"{output_name}.jsonl"
        sft_path = args.sft_dir / f"{output_name}.jsonl"
        write_jsonl(benchmark_path, selected)
        write_jsonl(sft_path, [to_sft_chat(row) for row in selected])

        selected_ids = [row["selected_id"] for row in selected]
        manifest["splits"][output_name] = {
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "source_rows": row_count,
            "unique_src_hashes": len(groups),
            "selected_count": len(selected),
            "selected_ids": selected_ids,
            "benchmark_output": str(benchmark_path),
            "sft_output": str(sft_path),
        }

        print(
            f"{output_name}: selected {len(selected)} of {row_count} rows "
            f"from {len(groups)} unique src_hash values"
        )

    write_json(args.benchmark_dir / "manifest.json", manifest)
    print(f"Wrote manifest to {args.benchmark_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
