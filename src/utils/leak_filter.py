"""Review-leakage filter (Red-line 3 in the construction guide).

Removes any retrieved evidence that originates from OpenReview reviews,
rebuttals, meta-reviews, or accept/reject decisions, so the teacher pipeline
cannot indirectly observe ground-truth reviews during evidence gathering.
"""
from __future__ import annotations

from typing import Iterable


BLOCKED_DOMAINS: tuple[str, ...] = (
    "openreview.net",
)

BLOCKED_KEYWORDS: tuple[str, ...] = (
    "rebuttal",
    "official review",
    "meta-review",
    "decision: accept",
    "decision: reject",
    "reviewer 1",
    "reviewer 2",
    "reviewer 3",
)


def filter_review_leakage(evidence_list: Iterable[dict]) -> list[dict]:
    """Filter out OpenReview reviews, rebuttals, and decisions from retrieved evidence.

    Args:
        evidence_list: iterable of evidence dicts (expects keys: url, title, abstract)
    Returns:
        list of evidence dicts that pass both the domain filter and the keyword filter
    """
    filtered: list[dict] = []
    for ev in evidence_list:
        url = ev.get("url", "") or ""
        if any(domain in url for domain in BLOCKED_DOMAINS):
            continue
        text = ((ev.get("title", "") or "") + " " + (ev.get("abstract", "") or "")).lower()
        if any(keyword in text for keyword in BLOCKED_KEYWORDS):
            continue
        filtered.append(ev)
    return filtered
