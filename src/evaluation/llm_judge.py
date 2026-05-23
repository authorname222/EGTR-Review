"""LLM-as-Judge over 6 dimensions, with optional cached judgments.

Also exposes `cross_judge_summary` for aggregating per-judge per-paper
scores into the Table 3 row reported in the paper, and `icc_2k` for
the inter-judge reliability statistic (paper §4.2.2 reports ICC=0.9136
across all (paper, dimension) cells × 3 judges).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


DIMENSIONS: tuple[str, ...] = (
    "pertinency",
    "usefulness",
    "evidence_groundedness",
    "traceability",
    "depth",
    "comprehensiveness",
)


def icc_2k(matrix: Sequence[Sequence[float]]) -> float:
    """Intraclass Correlation Coefficient ICC(2,k) — two-way random effects,
    absolute agreement, mean of k raters (Shrout & Fleiss, 1979).

    `matrix` has shape (n_items, k_raters). Returns float('nan') if the
    matrix is empty or degenerate (n<2 or k<2).
    """
    rows = [list(r) for r in matrix]
    if not rows or len(rows) < 2:
        return float("nan")
    k = len(rows[0])
    if any(len(r) != k for r in rows) or k < 2:
        return float("nan")
    n = len(rows)
    flat = [v for r in rows for v in r]
    grand = sum(flat) / (n * k)
    row_means = [sum(r) / k for r in rows]
    col_means = [sum(rows[i][j] for i in range(n)) / n for j in range(k)]
    ss_r = k * sum((rm - grand) ** 2 for rm in row_means)
    ss_c = n * sum((cm - grand) ** 2 for cm in col_means)
    ss_total = sum((v - grand) ** 2 for v in flat)
    ss_e = ss_total - ss_r - ss_c
    df_r = n - 1
    df_c = k - 1
    df_e = df_r * df_c
    if df_e <= 0:
        return float("nan")
    ms_r = ss_r / df_r
    ms_c = ss_c / df_c
    ms_e = ss_e / df_e
    denom = ms_r + (ms_c - ms_e) / n
    if denom == 0:
        return float("nan")
    return (ms_r - ms_e) / denom


def cross_judge_summary(
    per_judge_per_paper: dict[str, dict[str, dict[str, float]]],
    paper_ids: Iterable[str],
    dimensions: Sequence[str] = DIMENSIONS,
) -> dict[str, Any]:
    """Aggregate per-judge per-paper 6-dim scores into Table 3 figures.

    Args:
        per_judge_per_paper: {judge_name: {paper_id: {dim: score, ...}}}
        paper_ids: ordered iterable of paper IDs to consider.
        dimensions: list of dimension keys (defaults to the canonical 6).

    Returns a dict with:
        - cross_judge_per_paper: {paper_id: {dim: mean_over_judges}}
        - average:               {dim: mean_over_papers, ..., 'overall': mean_over_dims}
        - ICC_2k:                inter-judge reliability across all (paper, dim) cells
        - n_papers, n_judges
    """
    paper_ids = list(paper_ids)
    judges = list(per_judge_per_paper)
    if not judges or not paper_ids:
        return {
            "cross_judge_per_paper": {},
            "average": {},
            "ICC_2k": float("nan"),
            "n_papers": 0,
            "n_judges": 0,
        }

    cross_judge_per_paper: dict[str, dict[str, float]] = {}
    for pid in paper_ids:
        cross_judge_per_paper[pid] = {}
        for dim in dimensions:
            vals = [
                float(per_judge_per_paper[j].get(pid, {}).get(dim, 0.0))
                for j in judges
            ]
            cross_judge_per_paper[pid][dim] = sum(vals) / len(vals)

    avg_per_dim: dict[str, float] = {}
    for dim in dimensions:
        dim_vals = [cross_judge_per_paper[pid][dim] for pid in paper_ids]
        avg_per_dim[dim] = sum(dim_vals) / len(dim_vals) if dim_vals else 0.0
    if dimensions:
        avg_per_dim["overall"] = sum(avg_per_dim[d] for d in dimensions) / len(dimensions)

    # ICC matrix: one row per (paper, dimension) cell, one column per judge.
    icc_matrix = [
        [float(per_judge_per_paper[j].get(pid, {}).get(dim, 0.0)) for j in judges]
        for pid in paper_ids
        for dim in dimensions
    ]
    return {
        "cross_judge_per_paper": cross_judge_per_paper,
        "average": avg_per_dim,
        "ICC_2k": icc_2k(icc_matrix),
        "n_papers": len(paper_ids),
        "n_judges": len(judges),
    }


JUDGE_PROMPT = """You are an expert peer-review judge. Score the candidate review against
the paper on the following 6 dimensions on a {scale_lo}-{scale_hi} integer scale.

Dimensions:
  1. Pertinency             — focused on the paper's specifics
  2. Usefulness             — actionable for authors
  3. Evidence-groundedness  — claims tied to evidence
  4. Traceability           — comments linked to positions
  5. Depth                  — substantive analysis
  6. Comprehensiveness      — covers the paper as a whole

Return JSON: {{"pertinency": int, "usefulness": int, "evidence_groundedness": int,
"traceability": int, "depth": int, "comprehensiveness": int}}."""


class LLMJudge:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.judges = cfg.get("judges", ["gemini"])
        self.scale = cfg.get("scale", [1, 10])
        self.use_cache = cfg.get("use_cache", True)
        self.cache_path = cfg.get("cache_path", "")
        self._cache: dict[str, Any] = {}
        if self.use_cache and self.cache_path and Path(self.cache_path).exists():
            with open(self.cache_path) as f:
                self._cache = json.load(f)

    def judge(self, paper_id: str, paper: str, review: str, judge_name: str) -> dict:
        key = f"{judge_name}:{paper_id}"
        if self.use_cache and key in self._cache:
            return self._cache[key]
        scores = self._call_judge(paper, review, judge_name)
        self._cache[key] = scores
        if self.use_cache and self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        return scores

    def _call_judge(self, paper: str, review: str, judge_name: str) -> dict:
        """Pluggable judge backend. For anonymity, this skeleton uses the OpenAI-compatible API.

        Reviewers can swap in Gemini / DeepSeek / Claude by setting the corresponding
        base_url + api_key environment variables.
        """
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get(f"{judge_name.upper()}_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=os.environ.get(f"{judge_name.upper()}_BASE_URL") or None,
        )
        sys = JUDGE_PROMPT.format(scale_lo=self.scale[0], scale_hi=self.scale[1])
        user = json.dumps({"paper": paper[:8000], "review": review[:4000]})
        model = os.environ.get(f"{judge_name.upper()}_MODEL", "gpt-5.1")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        try:
            return json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {}

    def aggregate(self, scores: list[dict]) -> dict:
        if not scores:
            return {}
        keys = scores[0].keys()
        out = {k: sum(s.get(k, 0) for s in scores) / len(scores) for k in keys}
        out["overall"] = sum(out.values()) / max(1, len(out))
        return out
