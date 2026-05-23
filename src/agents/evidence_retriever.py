"""Agent III: Evidence Retriever.

Takes the initial paper representation L_i and produces L_i' = {(P_i_n, C_i_n,
E_i_n, Flag_i_n)}. The agent (1) runs the external retrievers, (2) deduplicates
and trims to at most three highly relevant records, (3) calls the LLM with the
verbatim system prompt from `docs/prompts/` (Section 3) to extract core
snippets and assign Flag_i_n, (4) sends a feedback-style query-refinement
request to the Key Element Extractor when the preliminary flag tends toward
"No Evidence" (at most two rounds).

Output schema per unit (see prompts/ §3):
  unit_id, P_i_n, C_i_n, Q_i_n,
  E_i_n: [{source_title, source_metadata, evidence_snippet, evidence_relevance}],
  Flag_i_n, labeling_rationale,
  feedback_to_key_element_extractor_agent, feedback_rationale
"""
from __future__ import annotations

import json
from typing import Any

from ..utils.leak_filter import filter_review_leakage
from .prompt_loader import FLAGS, get_prompt


SYSTEM_PROMPT = get_prompt("evidence_retriever")

FLAG_VALUES = FLAGS  # back-compat alias used by tests / external scripts


class EvidenceRetrieverAgent:
    def __init__(self, config: dict, llm_client: Any, retrievers: dict[str, Any]):
        self.config = config
        self.llm = llm_client
        self.retrievers = retrievers
        self.sources = config.get("sources", ["serpapi", "arxiv", "semantic_scholar"])
        self.top_k = config.get("top_k_per_source", 5)
        self.max_evidence = config.get("max_evidence_per_unit", 3)
        self.rewrite_rounds = config.get("rewrite_rounds", 2)
        self.leak_filter_on = config.get("leak_filter", True)

    def run(self, annotated_units: list[dict]) -> list[dict]:
        out: list[dict] = []
        for unit in annotated_units:
            queries = self._unit_queries(unit)
            evidence: list[dict] = []
            for round_i in range(max(1, self.rewrite_rounds)):
                raw = self._retrieve(queries)
                if self.leak_filter_on:
                    raw = filter_review_leakage(raw)
                evidence = self._dedup(raw)[: self.max_evidence]
                if evidence:
                    break
                queries = [f"{q} method dataset metric" for q in queries]
            llm_unit = self._label_with_llm(unit, evidence)
            out.append(llm_unit)
        return out

    def _unit_queries(self, unit: dict) -> list[str]:
        q = unit.get("retrieval_query") or unit.get("Q_i_n")
        if isinstance(q, list):
            return q
        if isinstance(q, str) and q.strip():
            return [q]
        return []

    def _retrieve(self, queries: list[str]) -> list[dict]:
        results: list[dict] = []
        for q in queries:
            for source in self.sources:
                client = self.retrievers.get(source)
                if client is None:
                    continue
                results.extend(client.search(q, top_k=self.top_k))
        return results

    @staticmethod
    def _dedup(results: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for r in results:
            key = r.get("url") or r.get("title") or ""
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    @staticmethod
    def _evidence_for_prompt(raw_evidence: list[dict]) -> list[dict]:
        """Reshape raw retriever output into prompts/ §3 evidence schema."""
        out = []
        for ev in raw_evidence:
            out.append(
                {
                    "source_title": ev.get("title", ""),
                    "source_metadata": {
                        "source": ev.get("source", ""),
                        "url": ev.get("url", ""),
                    },
                    "evidence_snippet": ev.get("abstract", ""),
                    "evidence_relevance": "contextual",
                }
            )
        return out

    def _label_with_llm(self, unit: dict, evidence: list[dict]) -> dict:
        """Invoke the LLM with the §3 system prompt and parse a single record."""
        unit_payload = {
            "unit_id": unit.get("unit_id"),
            "P_i_n": unit.get("P_i_n"),
            "C_i_n": unit.get("C_i_n"),
            "Q_i_n": unit.get("retrieval_query") or unit.get("Q_i_n"),
            "retrieved_records": self._evidence_for_prompt(evidence),
        }
        body = SYSTEM_PROMPT.replace(
            "{initial_paper_representation}",
            json.dumps([unit_payload], ensure_ascii=False),
        )
        response = self.llm.chat(system=body, user="", response_format="json")
        parsed = self._parse(response)
        if not parsed:
            return {
                **unit,
                "E_i_n": [],
                "Flag_i_n": "No Evidence",
                "labeling_rationale": "LLM did not return a parsable record.",
            }
        rec = parsed[0]
        flag = rec.get("Flag_i_n", "No Evidence")
        if flag not in FLAGS:
            flag = "No Evidence"
        rec["Flag_i_n"] = flag
        # carry over the original retrieval-query input for downstream agents
        rec.setdefault("Q_i_n", unit_payload["Q_i_n"])
        return rec

    @staticmethod
    def _parse(response: str) -> list[dict]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return data
        for key in ("units", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data] if isinstance(data, dict) else []
