"""Evidence-quality metrics (paper §4.3 + Table 5).

The four metrics are defined verbatim from the paper. The granularity of the
denominator follows the paper's wording, which in turn explains why Student
and Teacher report identical EA / LC but different ECR / SLA in Table 5:

  - EA  (Evidence Authenticity)
      "外部证据是否真实存在且内容准确"
      Per *evidence entry* (an item inside `E_i_n`). Both Student and
      Teacher share the retrieval stage, so EA is invariant under
      distillation — Table 5 reports 0.912 for both.

  - LC  (Label Consistency)
      "Flag_i_n 是否与人工判断一致"
      Per *paper unit* (one entry in `L_prime`). Also retrieval-side, so
      Student == Teacher (0.846 in Table 5).

  - ECR (Evidence-Comment Relevance)
      "最终评审意见是否与其引用证据语义相关"
      Per *review comment* in Y_i.

  - SLA (Source Localization Accuracy)
      "评审意见是否能够对应到论文中的章节、段落或实验位置"
      Per *review comment* in Y_i.

Each annotated field is a boolean. Items missing the corresponding annotation
are excluded from that metric's denominator. Returns 0.0 for an empty
denominator (rather than raising) so partial annotations still produce a
report.

Annotation schema
-----------------
Predictions for human evaluation are extended with boolean labels::

    {
      "paper_id": "...",
      "L_prime": [
        {
          "unit_id": "1", "P_i_n": "...", "C_i_n": "...", "Flag_i_n": "...",
          "flag_correct": true,                       # ← LC
          "E_i_n": [
            {"source_title": "...", "evidence_authentic": true},   # ← EA
            ...
          ]
        }
      ],
      "Y": {
        "strengths":  [{"comment": "...", "evidence_relevant": true,    # ECR
                                          "position_correct":  true}],  # SLA
        "weaknesses": [...], "questions": [...], "suggestions": [...]
      }
    }

If an annotator can only label `evidence_authentic` at the *comment* level
(i.e. "are the citations in this comment authentic?"), set
`evidence_authentic` on comment dicts; the report will include both
`EA` (per-evidence, paper definition) and `EA_per_comment` (alternative
annotator workflow). Whichever denominator is non-zero indicates which
labelling style was used.
"""
from __future__ import annotations

from typing import Iterable


def compute_evidence_quality(
    predictions: list[dict], references: list[dict] | None = None
) -> dict[str, float]:
    """Compute EA / LC / ECR / SLA over human-annotated paper-level predictions.

    Accepts both the canonical §3/§5 shape (`L_prime` / `Y`) and the
    legacy/baseline shape (`units` / `comments` or `review.comments`).
    """
    # Per-evidence (EA) and per-unit (LC) — retrieval-side
    n_ea_ev = d_ea_ev = 0
    n_lc = d_lc = 0
    # Per-comment (ECR, SLA) — synthesizer-side
    n_ecr = d_ecr = 0
    n_sla = d_sla = 0
    # Optional comment-level EA fallback
    n_ea_co = d_ea_co = 0

    for pred in predictions:
        for unit in _iter_units(pred):
            if "flag_correct" in unit:
                d_lc += 1
                n_lc += int(bool(unit["flag_correct"]))
            for ev in _iter_evidence(unit):
                if "evidence_authentic" in ev:
                    d_ea_ev += 1
                    n_ea_ev += int(bool(ev["evidence_authentic"]))

        for comment in _iter_comments(pred):
            if "evidence_relevant" in comment:
                d_ecr += 1
                n_ecr += int(bool(comment["evidence_relevant"]))
            if "position_correct" in comment:
                d_sla += 1
                n_sla += int(bool(comment["position_correct"]))
            # Comment-level EA only used as a fallback when the annotator
            # didn't label individual evidence entries.
            if "evidence_authentic" in comment:
                d_ea_co += 1
                n_ea_co += int(bool(comment["evidence_authentic"]))

    def _frac(num: int, den: int) -> float:
        return num / den if den else 0.0

    result: dict[str, float] = {
        "EA": _frac(n_ea_ev, d_ea_ev),
        "LC": _frac(n_lc, d_lc),
        "ECR": _frac(n_ecr, d_ecr),
        "SLA": _frac(n_sla, d_sla),
        "_denominators": {
            "EA": d_ea_ev, "LC": d_lc, "ECR": d_ecr, "SLA": d_sla,
        },
    }
    if d_ea_co:  # only surface the fallback when there's data for it
        result["EA_per_comment"] = _frac(n_ea_co, d_ea_co)
        result["_denominators"]["EA_per_comment"] = d_ea_co
    return result


def _iter_units(pred: dict) -> Iterable[dict]:
    if isinstance(pred.get("units"), list):
        yield from pred["units"]
    elif isinstance(pred.get("L_prime"), list):
        yield from pred["L_prime"]


def _iter_evidence(unit: dict) -> Iterable[dict]:
    for key in ("E_i_n", "E", "evidence"):
        v = unit.get(key)
        if isinstance(v, list):
            yield from v
            return


def _iter_comments(pred: dict) -> Iterable[dict]:
    if isinstance(pred.get("comments"), list):
        yield from pred["comments"]
        return
    review = pred.get("review")
    if isinstance(review, dict):
        if isinstance(review.get("comments"), list):
            yield from review["comments"]
            return
        for bucket in ("strengths", "weaknesses", "questions", "suggestions"):
            yield from review.get(bucket, []) or []
        return
    y = pred.get("Y")
    if isinstance(y, list):
        yield from y
    elif isinstance(y, dict):
        for bucket in ("strengths", "weaknesses", "questions", "suggestions"):
            yield from y.get(bucket, []) or []
