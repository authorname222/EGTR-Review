"""arXiv evidence client."""
from __future__ import annotations

from typing import Any


class ArxivClient:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.max_results = cfg.get("max_results", 10)
        self.sort_by = cfg.get("sort_by", "relevance")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        try:
            import arxiv
        except ImportError:
            return []
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        s = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=sort_map.get(self.sort_by, arxiv.SortCriterion.Relevance),
        )
        out = []
        for r in s.results():
            out.append(
                {
                    "source": "arxiv",
                    "title": r.title,
                    "abstract": r.summary,
                    "url": r.entry_id,
                }
            )
            if len(out) >= top_k:
                break
        return out
