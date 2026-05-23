"""Distillation: data construction, task prefixing, evidence weighting, student training."""
from .evidence_weighting import compute_sample_weight, EVIDENCE_WEIGHT_MAP
from .task_prefix import TASK_PREFIXES, format_with_prefix

__all__ = [
    "compute_sample_weight",
    "EVIDENCE_WEIGHT_MAP",
    "TASK_PREFIXES",
    "format_with_prefix",
]
