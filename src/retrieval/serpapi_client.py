"""SerpApi (Google Scholar) evidence client."""
from __future__ import annotations

import os
from typing import Any


class SerpApiClient:
    def __init__(self, config: dict[str, Any] | None = None):
        self.api_key = os.environ.get("SERPAPI_API_KEY", "")
        cfg = config or {}
        self.engine = cfg.get("engine", "google_scholar")
        self.num = cfg.get("num", 10)
        self.timeout = cfg.get("timeout", 30)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.api_key:
            return []
        try:
            from serpapi import GoogleSearch
        except ImportError:
            return []
        params = {"engine": self.engine, "q": query, "num": self.num, "api_key": self.api_key}
        results = GoogleSearch(params).get_dict().get("organic_results", [])
        return [
            {
                "source": "serpapi",
                "title": r.get("title", ""),
                "abstract": r.get("snippet", ""),
                "url": r.get("link", ""),
            }
            for r in results[:top_k]
        ]
