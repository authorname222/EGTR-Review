"""Agent I: Structure Parser.

Decomposes a paper D_i into locatable, semantically coherent units. The system
prompt is loaded verbatim from `docs/prompts/` (Section 1) at import time.

Output schema per unit (see prompts/ §1):
  unit_id, P_i_n, C_i_n, structural_type,
  segmentation_decision, segmentation_rationale, suggested_operation
"""
from __future__ import annotations

import json
from typing import Any

from .prompt_loader import get_prompt


SYSTEM_PROMPT = get_prompt("structure_parser")


class StructureParserAgent:
    def __init__(self, config: dict, llm_client: Any):
        self.config = config
        self.llm = llm_client
        self.max_chunk_tokens = config.get("max_chunk_tokens", 800)
        self.overlap_tokens = config.get("overlap_tokens", 80)

    def run(self, paper: dict) -> list[dict]:
        prompt = SYSTEM_PROMPT.replace("{paper_text}", self._format_paper(paper))
        response = self.llm.chat(system=prompt, user="", response_format="json")
        return self._parse(response)

    def revise(self, paper: dict, feedback: dict) -> list[dict]:
        body = (
            self._format_paper(paper)
            + "\n\nFeedback from downstream agent:\n"
            + json.dumps(feedback, ensure_ascii=False)
            + "\nPlease re-segment, addressing the issues above. Preserve the "
            "output JSON schema."
        )
        prompt = SYSTEM_PROMPT.replace("{paper_text}", body)
        response = self.llm.chat(system=prompt, user="", response_format="json")
        return self._parse(response)

    def _format_paper(self, paper: dict) -> str:
        lines = [f"# {paper.get('title', '')}", "", paper.get("abstract", "")]
        for sec in paper.get("sections", []):
            lines.append("")
            lines.append(f"## {sec.get('heading', '')}")
            for para in sec.get("paragraphs", []):
                lines.append(para)
        return "\n".join(lines)

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
