"""Mutation-agnostic jobs and deterministic sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Callable, Mapping, Sequence


APRIL_ERROR_TYPES = frozenset({"lme", "mlme", "tme", "thme"})
SEED_DERIVATION_VERSION = "sha256-global-source-mutator-attempt-v1"
ProofPair = dict[str, str]
MutationFunction = Callable[[Mapping[str, Any], random.Random], ProofPair | None]


def derive_seed(
    global_seed: int,
    src_hash: str,
    mutator_id: str,
    attempt_index: int,
) -> int:
    """Derive an order-independent seed for one mutator attempt."""
    material = f"{global_seed}\0{src_hash}\0{mutator_id}\0{attempt_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def mutation_job_id(mutator_id: str, src_hash: str, attempt_index: int) -> str:
    return f"{mutator_id}:{src_hash}:{attempt_index}"


def validate_pair(pair: Mapping[str, Any], correct_proof: str) -> ProofPair:
    if set(pair) != {"correct_proof", "incorrect_proof"}:
        raise ValueError("mutator result must contain exactly correct_proof and incorrect_proof")
    correct, incorrect = pair["correct_proof"], pair["incorrect_proof"]
    if not isinstance(correct, str) or not isinstance(incorrect, str):
        raise ValueError("proof-pair values must be strings")
    if correct != correct_proof:
        raise ValueError("mutator changed or replaced the supplied correct proof")
    if incorrect == correct:
        raise ValueError("mutator returned an unchanged proof")
    return {"correct_proof": correct, "incorrect_proof": incorrect}


@dataclass(frozen=True)
class MutationAdapter:
    """Register one pair-only mutator with its APRIL classification."""

    mutator_id: str
    april_error_type: str
    mutate_function: MutationFunction = field(repr=False, compare=False)
    requires_model: bool = False
    manifest_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mutator_id or any(char.isspace() for char in self.mutator_id):
            raise ValueError("mutator_id must be a nonempty identifier without whitespace")
        if self.april_error_type not in APRIL_ERROR_TYPES:
            raise ValueError(f"unsupported APRIL error type: {self.april_error_type}")

    def mutate(
        self,
        record: Mapping[str, Any],
        *,
        rng: random.Random,
    ) -> ProofPair | None:
        pair = self.mutate_function(record, rng)
        if pair is None:
            return None
        correct = record.get("correct_proof")
        if not isinstance(correct, str):
            raise ValueError("source record has no correct_proof string")
        return validate_pair(pair, correct)


@dataclass(frozen=True)
class MutationJob:
    """Caller-owned context wrapped around a pair-only mutation call."""

    source: Mapping[str, Any]
    adapter: MutationAdapter
    attempt_index: int
    global_seed: int

    @property
    def src_hash(self) -> str:
        value = self.source.get("src_hash")
        if not isinstance(value, str) or not value:
            raise ValueError("source record has no src_hash")
        return value

    @property
    def job_id(self) -> str:
        return mutation_job_id(self.adapter.mutator_id, self.src_hash, self.attempt_index)

    @property
    def seed(self) -> int:
        return derive_seed(
            self.global_seed,
            self.src_hash,
            self.adapter.mutator_id,
            self.attempt_index,
        )

    def run(self) -> ProofPair | None:
        return self.adapter.mutate(self.source, rng=random.Random(self.seed))


def make_jobs(
    rows: Sequence[Mapping[str, Any]],
    adapters: Sequence[MutationAdapter],
    *,
    attempts_per_mutator: int,
    global_seed: int,
) -> list[MutationJob]:
    if attempts_per_mutator < 1:
        raise ValueError("attempts_per_mutator must be at least 1")
    return [
        MutationJob(row, adapter, attempt, global_seed)
        for row in rows
        for adapter in adapters
        for attempt in range(attempts_per_mutator)
    ]
