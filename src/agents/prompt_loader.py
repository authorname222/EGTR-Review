"""Single-source-of-truth prompt loader.

All five teacher-side agent prompts live in `docs/prompts/` as per-agent
Markdown files. This module extracts the fenced code block from each file
at import time so prompts only need to be edited in one place.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "prompts"

# Map each agent to its file under docs/prompts/.
_AGENT_FILES: Dict[str, str] = {
    "structure_parser":      "01_structure_parser.md",
    "key_element_extractor": "02_key_element_extractor.md",
    "evidence_retriever":    "03_evidence_retriever.md",
    "verification_reasoner": "04_verification_reasoner.md",
    "review_synthesizer":    "05_review_synthesizer.md",
}


def _load_all() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, fname in _AGENT_FILES.items():
        path = PROMPTS_DIR / fname
        if not path.exists():
            raise RuntimeError(f"missing prompt file: {path}")
        text = path.read_text(encoding="utf-8")
        m = re.search(r"```(?:\w+)?\n(.+?)```", text, re.DOTALL)
        if not m:
            raise RuntimeError(f"no fenced prompt block in {path}")
        out[key] = m.group(1).rstrip()
    return out


_PROMPTS = _load_all()


def get_prompt(agent_key: str) -> str:
    return _PROMPTS[agent_key]


FLAGS = (
    "Strong Evidence-Supports",
    "Strong Evidence-Refutes",
    "Weak Evidence-Metadata Only",
    "No Evidence",
    "Non-verifiable Item",
)
