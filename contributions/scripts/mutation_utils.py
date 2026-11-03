"""Shared persistence, hashing, and JSONL helpers for mutation pipelines."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def require_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record field {key} must be a nonempty string")
    return value


class JsonlJournal:
    """Append-only journal indexed by a stable key; the last record wins."""

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


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256_text(payload)


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return sha256_text(payload)
