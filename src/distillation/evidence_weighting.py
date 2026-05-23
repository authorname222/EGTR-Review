"""Evidence-weighted loss (paper Sec. 3.3.3, Eq. 4).

Aggregates unit-level reliability flags Flag_{i,n} into a sample weight α_i
used to scale the multi-task distillation loss. Flag labels follow the
canonical names defined in `docs/prompts/`.
"""
from __future__ import annotations

from typing import Iterable


EVIDENCE_WEIGHT_MAP: dict[str, float] = {
    "Strong Evidence-Supports": 1.0,
    "Strong Evidence-Refutes": 1.0,
    "Weak Evidence-Metadata Only": 0.5,
    "No Evidence": 0.3,
    "Non-verifiable Item": 0.3,
}


def compute_sample_weight(flags: Iterable[str]) -> float:
    """Aggregate unit-level evidence flags into a sample weight α_i.

    Falls back to the lowest reliability weight if a flag is missing from the
    map. Returns 0.3 (the default low-reliability weight) for an empty input.
    """
    flags = list(flags)
    if not flags:
        return 0.3
    fallback = EVIDENCE_WEIGHT_MAP["No Evidence"]
    return sum(EVIDENCE_WEIGHT_MAP.get(f, fallback) for f in flags) / len(flags)
