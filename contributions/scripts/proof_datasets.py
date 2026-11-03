"""Dataset adapters for proof-mutation inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .mutation_utils import require_string
except ImportError:  # Support direct execution by sibling scripts.
    from mutation_utils import require_string


def default_openai_math_input(repo_root: Path) -> Path:
    requested = repo_root / "data/openai_math/originals.jsonl"
    extracted = repo_root / "data/raw/openai_math/originals.jsonl"
    return requested if requested.is_file() else extracted


class JsonlProofDataset:
    """Load neutral proof records from JSONL with stable source identities."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _validate(self, row: dict[str, Any]) -> None:
        require_string(row, "src_hash")
        require_string(row, "correct_proof")

    def load(
        self,
        *,
        limit: int | None = None,
        source_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        if not self.path.is_file():
            raise ValueError(f"input does not exist: {self.path}")
        requested = set(source_ids)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                self._validate(row)
                src_hash = row["src_hash"]
                if requested and src_hash not in requested:
                    continue
                if src_hash in seen:
                    raise ValueError(f"duplicate src_hash {src_hash}")
                seen.add(src_hash)
                rows.append(row)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"invalid input {self.path}:{number}: {error}") from error
            if limit is not None and len(rows) >= limit:
                break
        if requested - seen:
            raise ValueError(f"requested source IDs not found: {sorted(requested - seen)}")
        return rows


class OpenAIMathDataset(JsonlProofDataset):
    """Validate and select records produced by ``extract_openai_math.py``."""

    def _validate(self, row: dict[str, Any]) -> None:
        super()._validate(row)
        require_string(row, "theorem")
        require_string(row, "proof_text")
        require_string(row, "source_file")
        if not isinstance(row.get("proof_span"), dict):
            raise ValueError("proof_span must be an object")
        if not isinstance(row.get("contains_sorry_or_admit"), bool):
            raise ValueError("contains_sorry_or_admit must be a boolean")

    def load(
        self,
        *,
        limit: int | None = None,
        source_ids: Iterable[str] = (),
        theorems: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        requested_theorems = set(theorems)
        rows = super().load(limit=None, source_ids=source_ids)
        if requested_theorems:
            by_theorem = {row["theorem"]: row for row in rows}
            missing = requested_theorems - by_theorem.keys()
            if missing:
                raise ValueError(f"requested theorems not found: {sorted(missing)}")
            rows = [row for row in rows if row["theorem"] in requested_theorems]
        return rows[:limit] if limit is not None else rows
