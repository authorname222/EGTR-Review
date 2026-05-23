"""Agent V: Review Synthesizer.

Combines the evidence-enhanced paper representation L_i' and the intermediate
reasoning trajectory T_i into the structured final review Y_i. System prompt
is loaded verbatim from `docs/prompts/` (Section 5).

Output schema (see prompts/ §5):
  {
    "summary": str,
    "strengths":   [{comment, impact, evidence_grounding, trace_ids}],
    "weaknesses":  [{comment, impact, evidence_grounding, trace_ids}],
    "questions":   [{question, type, trace_ids}],
    "suggestions": [{suggestion, related_to, trace_ids}],
    "traceability_notes": [{review_point_id, source_position, paper_fragment,
                            evidence_set, evidence_state_label,
                            reasoning_unit_id, rationale}]
  }
"""
from __future__ import annotations

import json
from typing import Any

from .prompt_loader import get_prompt


SYSTEM_PROMPT = get_prompt("review_synthesizer")


EMPTY_REVIEW: dict[str, Any] = {
    "summary": "",
    "strengths": [],
    "weaknesses": [],
    "questions": [],
    "suggestions": [],
    "traceability_notes": [],
}


class ReviewSynthesizerAgent:
    def __init__(self, config: dict, llm_client: Any):
        self.config = config
        self.llm = llm_client

    def run(self, evidence_enhanced: list[dict], trajectory: list[dict]) -> dict:
        body = SYSTEM_PROMPT.replace(
            "{evidence_enhanced_paper_representation}",
            json.dumps(evidence_enhanced, ensure_ascii=False),
        ).replace(
            "{intermediate_reasoning_trajectory}",
            json.dumps(trajectory, ensure_ascii=False),
        )
        response = self.llm.chat(system=body, user="", response_format="json")
        return self._parse(response)

    @staticmethod
    def _parse(response: str) -> dict:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return dict(EMPTY_REVIEW)
        if not isinstance(data, dict):
            return dict(EMPTY_REVIEW)
        for key, default in EMPTY_REVIEW.items():
            data.setdefault(key, default if isinstance(default, str) else list(default))
        return data
