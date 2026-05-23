"""Agent IV: Verification Reasoner.

Consumes the evidence-enhanced paper representation L_i' and produces the
intermediate reasoning trajectory T_i, where each entry covers one paper unit
with a verification path chosen by its Flag_i_n. System prompt is loaded
verbatim from `docs/prompts/` (Section 4).

Output schema per reasoning unit (see prompts/ §4):
  unit_id, P_i_n, C_i_n, E_i_n, Flag_i_n,
  verification_path, verification_question, reasoning_process,
  preliminary_review_point, review_point_type, traceability_basis

The Flag → verification_path mapping below mirrors the prompt's per-flag
branching and is exposed so downstream code and the ablation harness can
inspect / override individual strategies.
"""
from __future__ import annotations

import json
from typing import Any

from .prompt_loader import get_prompt


SYSTEM_PROMPT = get_prompt("verification_reasoner")


FLAG_TO_PATH: dict[str, str] = {
    "Strong Evidence-Supports": "Support Verification",
    "Strong Evidence-Refutes": "Refutation Verification",
    "Weak Evidence-Metadata Only": "Weak Evidence Probing",
    "No Evidence": "Internal Consistency Checking",
    "Non-verifiable Item": "Non-verifiable Quality Assessment",
}


class VerificationReasonerAgent:
    def __init__(self, config: dict, llm_client: Any):
        self.config = config
        self.llm = llm_client

    def run(self, retrieved_units: list[dict]) -> list[dict]:
        prompt = SYSTEM_PROMPT.replace(
            "{evidence_enhanced_paper_representation}",
            json.dumps(retrieved_units, ensure_ascii=False),
        )
        response = self.llm.chat(system=prompt, user="", response_format="json")
        records = self._parse(response)
        # Backfill the deterministic per-flag path in case the LLM omitted it.
        for rec in records:
            if not rec.get("verification_path"):
                rec["verification_path"] = FLAG_TO_PATH.get(
                    rec.get("Flag_i_n", ""), "Internal Consistency Checking"
                )
        return records

    @staticmethod
    def _parse(response: str) -> list[dict]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return data
        for key in ("reasoning_units", "trajectory", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return []
