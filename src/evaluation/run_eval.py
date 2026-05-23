"""CLI wrapper that runs auto_metrics + (optional) LLM-as-Judge + evidence_quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .auto_metrics import compute_auto_metrics
from .evidence_quality import compute_evidence_quality
from .llm_judge import LLMJudge, cross_judge_summary


def _flatten_review(review: dict | list | str) -> str:
    """Render a Y_i payload as a plain-text review string for auto metrics.

    Accepts (a) the structured Y_i object produced by the Review Synthesizer
    Agent (schema in docs/prompts/05_review_synthesizer.md), (b) a flat list of comment dicts
    (legacy/baselines), or (c) a raw string.
    """
    if isinstance(review, str):
        return review
    if isinstance(review, list):
        return "\n".join(
            c.get("comment", "") if isinstance(c, dict) else str(c) for c in review
        )
    if isinstance(review, dict):
        # New §5 schema: summary + strengths + weaknesses + questions + suggestions.
        if any(k in review for k in ("summary", "strengths", "weaknesses",
                                     "questions", "suggestions")):
            parts: list[str] = []
            if review.get("summary"):
                parts.append(str(review["summary"]))
            for bucket, key in (
                ("strengths", "comment"),
                ("weaknesses", "comment"),
                ("questions", "question"),
                ("suggestions", "suggestion"),
            ):
                for item in review.get(bucket, []) or []:
                    if isinstance(item, dict):
                        parts.append(item.get(key, "") or "")
                    else:
                        parts.append(str(item))
            return "\n".join(p for p in parts if p)
        if "comments" in review:
            return _flatten_review(review["comments"])
        if "raw_text" in review:
            return str(review["raw_text"])
    return json.dumps(review, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--references", required=True)
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--out", default="outputs/eval_metrics.json")
    args = ap.parse_args()

    preds = json.load(open(args.predictions))
    refs = json.load(open(args.references))
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["eval"]

    pred_index = {p.get("paper_id"): p for p in preds}
    ref_index = {r.get("paper_id"): r for r in refs}
    paper_ids = sorted(set(pred_index) & set(ref_index))

    pred_texts = [_flatten_review(pred_index[pid].get("review", {})) for pid in paper_ids]
    ref_texts = [_flatten_review(ref_index[pid].get("review", {})) for pid in paper_ids]

    auto = compute_auto_metrics(pred_texts, ref_texts, cfg.get("auto_metrics", {}))
    ev = compute_evidence_quality(
        [pred_index[pid] for pid in paper_ids]
    )
    metrics = {"auto": auto, "evidence_quality": ev}

    if cfg.get("llm_judge", {}).get("judges"):
        judge = LLMJudge(cfg["llm_judge"])
        # per_judge_scores[j][pid] = {pertinency, usefulness, ...} for that judge.
        per_judge_scores: dict[str, dict[str, dict[str, float]]] = {}
        per_judge_summary: dict[str, dict[str, float]] = {}
        for j in cfg["llm_judge"]["judges"]:
            per_judge_scores[j] = {}
            scores_list: list[dict] = []
            for i, pid in enumerate(paper_ids):
                s = judge.judge(pid, ref_texts[i], pred_texts[i], j)
                per_judge_scores[j][pid] = s
                scores_list.append(s)
            per_judge_summary[j] = judge.aggregate(scores_list)
        cross = cross_judge_summary(per_judge_scores, paper_ids)
        metrics["llm_judge"] = {
            "per_judge": per_judge_summary,
            "cross_judge_mean": cross["average"],
            "ICC_2k": cross["ICC_2k"],
            "n_papers": cross["n_papers"],
            "n_judges": cross["n_judges"],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
