"""Migrate ~/distill_project/distill_data/distill_triplets.jsonl into the
EGTR-Review per-paper teacher-output schema (data/samples/distilled/*.json).

The legacy distill_project file stores one record per paper with:
  - input    : a single string with marked segments
                "=== 论文原文片段 (P_i) ===", "=== 论文核心描述 (C_i) ===",
                "=== 外部证据与状态标记 (E_i + Flag_i) ===", "=== 输入结束 ==="
  - reasoning: Markdown blocks beginning with "[Claim C_xxx]" and step lines
  - review   : Markdown with "### Summary / Strengths / Weaknesses / Questions"

This script converts each record into the per-paper JSON schema documented in
`docs/prompts/` (sections 3–5). Output fields per unit:

  unit_id, P_i_n, C_i_n, Q_i_n, E_i_n, Flag_i_n,
  labeling_rationale, feedback_to_key_element_extractor_agent, feedback_rationale

T_i is a list of reasoning units (see prompts/ §4), and Y_i is the
structured review object (see prompts/ §5).

Notes
-----
- The legacy file has empty external-evidence sections ("暂无外部证据") for all
  854 papers. Without re-running retrieval, every unit is assigned
  Flag_i_n = "No Evidence" (verifiable claims) or "Non-verifiable Item"
  (writing-style claims). Pass `--rerun_retrieval` to invoke the EGTR-Review
  evidence retriever and overwrite E_i_n / Flag_i_n.
- `source_position` for review comments is recovered from the Markdown
  "Location & Basis: …" line when present; otherwise it defaults to "Unknown".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distillation.evidence_weighting import compute_sample_weight  # noqa: E402


CLAIM_RE = re.compile(r"\[Claim\s+(C_\d+)\][^\n]*\n", re.IGNORECASE)
LOC_RE = re.compile(
    r"Location\s*&\s*Basis\**\s*[:：]\s*([^\n]+)", re.IGNORECASE
)
SECTION_RE = re.compile(
    r"(Section\s+\d+(?:\.\d+)*|Abstract|Introduction|Related Work|Method|Methods|"
    r"Approach|Experiments?|Experimental|Results?|Analysis|Discussion|"
    r"Limitations?|Conclusion|Appendix|Figure\s*\d+|Table\s*\d+|"
    r"摘要|引言|方法|实验|结果|结论|附录|讨论|相关工作)",
    re.IGNORECASE,
)
NONVERIFIABLE_HINTS = (
    "writing", "presentation", "exposition", "readability", "clarity",
    "subjective", "limitation",
)

FLAG_TO_PATH: dict[str, str] = {
    "Strong Evidence-Supports": "Support Verification",
    "Strong Evidence-Refutes": "Refutation Verification",
    "Weak Evidence-Metadata Only": "Weak Evidence Probing",
    "No Evidence": "Internal Consistency Checking",
    "Non-verifiable Item": "Non-verifiable Quality Assessment",
}


def split_input(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    m = re.search(r"=== 论文原文片段 \(P_i\) ===\n(.+?)\n=== 论文核心描述", text, re.DOTALL)
    out["P_i"] = m.group(1).strip() if m else ""
    m = re.search(r"=== 论文核心描述 \(C_i\) ===\n(.+?)\n=== 外部证据", text, re.DOTALL)
    out["C_i"] = m.group(1).strip() if m else ""
    m = re.search(
        r"=== 外部证据与状态标记 \(E_i \+ Flag_i\) ===\n(.+?)\n=== 输入结束 ===",
        text,
        re.DOTALL,
    )
    out["E_i"] = m.group(1).strip() if m else ""
    return out


def parse_claims(reasoning: str) -> list[dict[str, str]]:
    matches = list(CLAIM_RE.finditer(reasoning))
    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reasoning)
        spans.append((start, end, m.group(1)))
    claims = []
    for start, end, cid in spans:
        block = reasoning[start:end].strip()
        loc = LOC_RE.search(block)
        claims.append(
            {
                "claim_id": cid,
                "block": block,
                "location_text": loc.group(1).strip() if loc else "",
            }
        )
    return claims


def parse_position(loc_text: str) -> str:
    if not loc_text:
        return "Unknown"
    sec = SECTION_RE.search(loc_text)
    if sec:
        para = re.search(r"paragraph\s*(\d+)", loc_text, re.IGNORECASE)
        if para:
            return f"{sec.group(1)} / Paragraph {para.group(1)}"
        return sec.group(1)
    return loc_text[:80]


def default_flag(claim_block: str) -> str:
    txt = claim_block.lower()
    if any(h in txt for h in NONVERIFIABLE_HINTS):
        return "Non-verifiable Item"
    return "No Evidence"


def claim_to_unit(idx: int, claim: dict[str, str]) -> dict[str, Any]:
    fragment = re.split(r"→\s*Step", claim["block"], maxsplit=1)[0]
    fragment = re.sub(r"\[Claim\s+C_\d+\]\s*", "", fragment).strip()
    flag = default_flag(claim["block"])
    return {
        "unit_id": str(idx),
        "P_i_n": parse_position(claim["location_text"]),
        "C_i_n": fragment[:2000],
        "Q_i_n": "",
        "E_i_n": [],
        "Flag_i_n": flag,
        "labeling_rationale": "Inherited from legacy distill_triplets; no external evidence was retrieved.",
        "feedback_to_key_element_extractor_agent": "None",
        "feedback_rationale": "Legacy record had no retrieval-side feedback.",
    }


def claim_to_trajectory(idx: int, claim: dict[str, str], unit: dict[str, Any]) -> dict[str, Any]:
    reasoning = re.split(r"\[Claim\s+C_\d+\]", claim["block"], maxsplit=1)[-1].strip()[:2000]
    return {
        "unit_id": str(idx),
        "P_i_n": unit["P_i_n"],
        "C_i_n": unit["C_i_n"],
        "E_i_n": unit["E_i_n"],
        "Flag_i_n": unit["Flag_i_n"],
        "verification_path": FLAG_TO_PATH[unit["Flag_i_n"]],
        "verification_question": "",
        "reasoning_process": reasoning,
        "preliminary_review_point": "",
        "review_point_type": "Clarification Request",
        "traceability_basis": "Reasoning Process",
    }


def parse_review_md(review_md: str) -> dict[str, Any]:
    sections = re.split(r"^###\s+", review_md, flags=re.MULTILINE)
    result: dict[str, Any] = {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "suggestions": [],
        "traceability_notes": [],
    }
    for sec in sections:
        if not sec.strip():
            continue
        head, _, body = sec.partition("\n")
        kind = head.strip().lower()
        if "summary" in kind:
            result["summary"] = body.strip()
            continue
        bucket: str | None
        if "strength" in kind:
            bucket = "strengths"
        elif "weakness" in kind:
            bucket = "weaknesses"
        elif "question" in kind:
            bucket = "questions"
        elif "suggestion" in kind:
            bucket = "suggestions"
        else:
            bucket = None
        if not bucket:
            continue
        for bullet in re.findall(
            r"(?ms)^(?:\d+\.|-|\*)\s+(.+?)(?=^(?:\d+\.|-|\*)\s+|\Z)", body
        ):
            text = re.sub(r"\*+", "", bullet.strip())[:1200]
            if not text:
                continue
            loc = LOC_RE.search(bullet)
            position = parse_position(loc.group(1)) if loc else "Unknown"
            if bucket == "questions":
                result[bucket].append(
                    {"question": text, "type": "Clarification", "trace_ids": []}
                )
            elif bucket == "suggestions":
                result[bucket].append(
                    {"suggestion": text, "related_to": "", "trace_ids": []}
                )
            else:
                result[bucket].append(
                    {
                        "comment": text,
                        "impact": "Medium",
                        "evidence_grounding": "Legacy distill_triplets reasoning trace.",
                        "trace_ids": [],
                    }
                )
            if bucket == "weaknesses":
                result["traceability_notes"].append(
                    {
                        "review_point_id": f"W{len(result['weaknesses'])}",
                        "source_position": position,
                        "paper_fragment": "",
                        "evidence_set": "None",
                        "evidence_state_label": "No Evidence",
                        "reasoning_unit_id": "",
                        "rationale": "Position recovered from legacy 'Location & Basis' line.",
                    }
                )
    return result


def convert(record: dict[str, Any]) -> dict[str, Any]:
    parts = split_input(record.get("input", ""))
    reasoning = record.get("reasoning", "") or ""
    review_md = record.get("review", "") or ""

    claims = parse_claims(reasoning)
    units = [claim_to_unit(i + 1, c) for i, c in enumerate(claims)]
    if not units:
        units = [
            {
                "unit_id": "1",
                "P_i_n": "Abstract",
                "C_i_n": parts["C_i"][:2000],
                "Q_i_n": "",
                "E_i_n": [],
                "Flag_i_n": "No Evidence",
                "labeling_rationale": "No claim segments parsed; falling back to whole paper core description.",
                "feedback_to_key_element_extractor_agent": "None",
                "feedback_rationale": "Single-unit fallback.",
            }
        ]
        trajectory = [
            {
                "unit_id": "1",
                "P_i_n": "Abstract",
                "C_i_n": units[0]["C_i_n"],
                "E_i_n": [],
                "Flag_i_n": "No Evidence",
                "verification_path": "Internal Consistency Checking",
                "verification_question": "",
                "reasoning_process": parts["C_i"][:500],
                "preliminary_review_point": "",
                "review_point_type": "Clarification Request",
                "traceability_basis": "Reasoning Process",
            }
        ]
    else:
        trajectory = [
            claim_to_trajectory(i + 1, c, u) for i, (c, u) in enumerate(zip(claims, units))
        ]

    review_obj = parse_review_md(review_md)
    alpha = compute_sample_weight([u["Flag_i_n"] for u in units])
    return {
        "paper_id": record.get("sample_id") or record.get("id") or "unknown",
        "L_prime": units,
        "T": trajectory,
        "Y": review_obj,
        "alpha": alpha,
        "_provenance": {
            "source_file": "distill_project/distill_data/distill_triplets.jsonl",
            "had_external_evidence": "暂无" not in parts["E_i"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_file",
        default=str(Path.home() / "distill_project/distill_data/distill_triplets.jsonl"),
    )
    ap.add_argument(
        "--out_dir",
        default="data/full/distilled",
        help="Per-paper JSON files will be written here.",
    )
    ap.add_argument(
        "--rerun_retrieval",
        action="store_true",
        help="Re-run the EGTR-Review evidence retriever on each unit and overwrite "
             "E_i_n / Flag_i_n using the current configs/teacher_pipeline.yaml settings.",
    )
    ap.add_argument("--config", default="configs/teacher_pipeline.yaml")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever = None
    if args.rerun_retrieval:
        import yaml
        from src.agents import EvidenceRetrieverAgent
        from src.retrieval import ArxivClient, SemanticScholarClient, SerpApiClient
        from src.teacher_pipeline import LLMClient

        with open(args.config) as f:
            cfg = yaml.safe_load(f)["teacher"]
        llm_cfg = cfg["llm"]
        llm = LLMClient(
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_output_tokens", 4096),
        )
        retriever = EvidenceRetrieverAgent(
            cfg["evidence_retriever"],
            llm,
            {
                "serpapi": SerpApiClient(),
                "arxiv": ArxivClient(),
                "semantic_scholar": SemanticScholarClient(),
            },
        )

    written = 0
    with open(args.in_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = convert(json.loads(line))
            if retriever is not None:
                rerun_units = [
                    {**u, "retrieval_query": u.get("C_i_n", "")[:160]}
                    for u in rec["L_prime"]
                ]
                retrieved = retriever.run(rerun_units)
                for ru, original in zip(retrieved, rec["L_prime"]):
                    original["E_i_n"] = ru.get("E_i_n", [])
                    original["Flag_i_n"] = ru.get("Flag_i_n", original["Flag_i_n"])
                rec["alpha"] = compute_sample_weight([u["Flag_i_n"] for u in rec["L_prime"]])
                rec["_provenance"]["retrieval_rerun"] = True
            out_path = out_dir / f"{rec['paper_id']}.json"
            with open(out_path, "w") as g:
                json.dump(rec, g, ensure_ascii=False, indent=2)
            written += 1

    print(f"[migrate] wrote {written} files to {out_dir}")


if __name__ == "__main__":
    main()
