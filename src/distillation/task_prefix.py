"""Task-prefix multi-task formatting (paper §3.3.2).

Paper Eq. 3: the student receives [τ] ⊕ L'_i where τ ∈ {Reasoning, Review}.
We use the bracketed token form to match the paper's notation and to make
the prefix robust to tokenizers that would otherwise merge "Reasoning:"
with surrounding whitespace.
"""
from __future__ import annotations

TASK_PREFIXES: dict[str, str] = {
    "reasoning": "[Reasoning] ",
    "review": "[Review] ",
}


def format_with_prefix(task: str, x: str) -> str:
    """Prepend the task prefix to the model input for multi-task supervision."""
    if task not in TASK_PREFIXES:
        raise ValueError(f"Unknown task '{task}', expected one of {list(TASK_PREFIXES)}")
    return TASK_PREFIXES[task] + x
