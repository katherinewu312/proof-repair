"""Registry entries that adapt pair-only mutation functions."""
from __future__ import annotations

import random
from typing import Any, Mapping

try:
    from . import mutate_line, mutate_tactic
    from .mutation_orchestrator import MutationAdapter
except ImportError:  # Support direct execution by sibling scripts.
    import mutate_line
    import mutate_tactic
    from mutation_orchestrator import MutationAdapter


LINE_SINGLE = "line.single.v1"
LINE_SUFFIX = "line.suffix.v1"
TACTIC_REPLACE = "tactic.replace.v1"
DEFAULT_MUTATORS = (LINE_SINGLE, LINE_SUFFIX)

MUTATOR_DESCRIPTORS: dict[str, dict[str, Any]] = {
    LINE_SINGLE: {"april_error_type": "lme", "requires_model": True},
    LINE_SUFFIX: {"april_error_type": "mlme", "requires_model": True},
    TACTIC_REPLACE: {"april_error_type": "tme", "requires_model": False},
}


def manifest_metadata(mutator_id: str) -> dict[str, Any]:
    """Return static algorithm metadata without constructing model dependencies."""
    if mutator_id == LINE_SINGLE:
        return {
            "prompt_version": mutate_line.LME_PROMPT_VERSION,
            "system_prompt": mutate_line.SYSTEM_PROMPT,
            "prompt_template": mutate_line.lme_user_prompt("{broken_proof}"),
        }
    if mutator_id == LINE_SUFFIX:
        return {
            "prompt_version": mutate_line.MLME_PROMPT_VERSION,
            "system_prompt": mutate_line.SYSTEM_PROMPT,
            "prompt_template": mutate_line.mlme_user_prompt("{broken_proof}"),
        }
    if mutator_id == TACTIC_REPLACE:
        return {"algorithm": "same-tactic-family lexical replacement"}
    raise ValueError(f"unknown mutator: {mutator_id}")


def build_adapter(
    mutator_id: str,
    *,
    model: Any | None,
    max_tokens: int,
) -> MutationAdapter:
    """Construct one registered adapter with its dependencies injected."""
    if mutator_id not in MUTATOR_DESCRIPTORS:
        raise ValueError(f"unknown mutator: {mutator_id}")
    descriptor = MUTATOR_DESCRIPTORS[mutator_id]
    if descriptor["requires_model"] and model is None:
        raise ValueError(f"mutator {mutator_id} requires a completion model")

    if mutator_id in {LINE_SINGLE, LINE_SUFFIX}:
        error_type = descriptor["april_error_type"]

        def mutate(record: Mapping[str, Any], rng: random.Random) -> dict[str, str]:
            return mutate_line.mutate_proof(
                record,
                model=model,
                error_type=error_type,
                rng=rng,
                max_tokens=max_tokens,
            )

        metadata = manifest_metadata(mutator_id)
    else:
        def mutate(record: Mapping[str, Any], rng: random.Random) -> dict[str, str] | None:
            return mutate_tactic.mutate_proof(record, rng=rng)

        metadata = manifest_metadata(mutator_id)

    return MutationAdapter(
        mutator_id=mutator_id,
        april_error_type=descriptor["april_error_type"],
        mutate_function=mutate,
        requires_model=descriptor["requires_model"],
        manifest_metadata=metadata,
    )
