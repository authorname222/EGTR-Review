"""Semantic Scholar evidence client."""
from __future__ import annotations

import os
from typing import Any


class SemanticScholarClient:
    def __init__(self, config: dict[str, Any] | None = None):
        self.api_key = os.environ.get("S2_API_KEY", "")
        cfg = config or {}
        self.fields = cfg.get("fields", ["title", "abstract", "year", "venue"])
        self.limit = cfg.get("limit", 10)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        try:
            from semanticscholar import SemanticScholar
        except ImportError:
            return []
        client = SemanticScholar(api_key=self.api_key or None)
        results = client.search_paper(query, limit=self.limit, fields=self.fields)
        out = []
        for r in results:
            out.append(
                {
                    "source": "semantic_scholar",
                    "title": getattr(r, "title", "") or "",
                    "abstract": getattr(r, "abstract", "") or "",
                    "url": f"https://www.semanticscholar.org/paper/{getattr(r, 'paperId', '')}",
                }
            )
            if len(out) >= top_k:
                break
        return out
