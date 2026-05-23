"""Agent II: Key Element Extractor.

For each (P_i_n, C_i_n) unit produced by the Structure Parser, extracts the
review-relevant key elements, claim type, and a single retrieval query
Q_i_n. System prompt is loaded verbatim from `docs/prompts/` (Section 2).

Output schema per unit (see prompts/ §2):
  unit_id, P_i_n, C_i_n, key_elements, claim_type, retrieval_query,
  feedback_to_structure_parser_agent, feedback_rationale
"""
from __future__ import annotations

import json
from typing import Any

from .prompt_loader import get_prompt


SYSTEM_PROMPT = get_prompt("key_element_extractor")


class KeyElementExtractorAgent:
    def __init__(self, config: dict, llm_client: Any):
        self.config = config
        self.llm = llm_client

    def run(self, units: list[dict]) -> list[dict]:
        prompt = SYSTEM_PROMPT.replace(
            "{paper_units}", json.dumps(units, ensure_ascii=False)
        )
        response = self.llm.chat(system=prompt, user="", response_format="json")
        return self._parse(response)

    def request_resegmentation(self, units: list[dict]) -> dict | None:
        """Aggregate feedback to the Structure Parser per prompts/ §2."""
        issues = [
            {
                "unit_id": u.get("unit_id"),
                "P_i_n": u.get("P_i_n"),
                "feedback": u.get("feedback_to_structure_parser_agent"),
                "rationale": u.get("feedback_rationale"),
            }
            for u in units
            if u.get("feedback_to_structure_parser_agent") not in (None, "", "None")
        ]
        return {"issues": issues} if issues else None

    @staticmethod
    def _parse(response: str) -> list[dict]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return data
        for key in ("units", "paper_units", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return []
