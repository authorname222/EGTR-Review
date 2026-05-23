"""Best-effort anonymizer for free-text fields.

Strips e-mail addresses, common author affiliations, and ORCIDs from text fields
that may end up in distillation data. This is a safety net only — the primary
defense is curation at data-prep time.
"""
from __future__ import annotations

import re


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
AFFIL_HEAD_RE = re.compile(
    r"(Affiliations?|Author Information|Correspondence)\s*[:：].*?(?=\n\n|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def scrub(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = ORCID_RE.sub("[ORCID]", text)
    text = AFFIL_HEAD_RE.sub("[AFFILIATION REMOVED]", text)
    return text
