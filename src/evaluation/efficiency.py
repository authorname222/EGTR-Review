"""Measure tokens / paper and inference time / paper — paper Table 8.

Teacher
-------
Wraps `TeacherPipeline.run(paper)` so the cumulative prompt + completion
tokens across all five agents (Structure Parser → Key Element Extractor →
Evidence Retriever → Verification Reasoner → Review Synthesizer) are summed
into `tokens_in / tokens_out / tokens_total`. Wall-clock time covers the
whole 5-agent pipeline.

Student
-------
Loads the distilled student model, encodes the paper with the inference-
time serializer, runs a single `model.generate()`, and records:
  - tokens_in:    number of tokens fed into the model
  - tokens_out:   number of tokens generated
  - tokens_total: in + out
  - calls:        always 1 (single generation per paper)
  - time_s:       wall-clock around `model.generate()`

The output JSON contains a per-paper record list plus a summary
(`avg_tokens_total`, `avg_time_s`, etc.) that maps directly to the columns
of Table 8.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..distillation.task_prefix import format_with_prefix
from ..utils.io_utils import load_paper, save_json


def _iter_papers(paper_dir: str):
    """Yield (path, paper_dict) for every JSON file under `paper_dir` that
    has the per-paper schema. Files such as `references.json` (a list of
    review references) are skipped automatically.
    """
    for path in sorted(Path(paper_dir).glob("*.json")):
        obj = load_paper(path)
        if isinstance(obj, dict) and (
            "sections" in obj or "abstract" in obj or "title" in obj
        ):
            yield path, obj


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------
def measure_teacher(paper_dir: str, config_path: str) -> list[dict]:
    """Run the full Multi-Agent Teacher per paper and report token+time."""
    from ..teacher_pipeline import TeacherPipeline

    pipeline = TeacherPipeline(config_path)
    records: list[dict] = []
    for path, paper in _iter_papers(paper_dir):
        pipeline.llm.reset_counters()
        t0 = time.time()
        pipeline.run(paper)
        elapsed = time.time() - t0
        records.append(
            {
                "paper_id": paper.get("paper_id", path.stem),
                "tokens_in": pipeline.llm.prompt_tokens,
                "tokens_out": pipeline.llm.completion_tokens,
                "tokens_total": pipeline.llm.prompt_tokens
                + pipeline.llm.completion_tokens,
                "calls": pipeline.llm.call_count,
                "time_s": round(elapsed, 3),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------
def _serialize_paper(paper: dict) -> str:
    parts = [f"# {paper.get('title', '')}", paper.get("abstract", "")]
    for sec in paper.get("sections", []):
        parts.append(f"## {sec.get('heading', '')}")
        parts.extend(sec.get("paragraphs", []))
    return "\n\n".join(parts)


def measure_student(
    paper_dir: str, model_path: str, max_new_tokens: int = 768
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    records: list[dict] = []
    for path, paper in _iter_papers(paper_dir):
        prompt = format_with_prefix("review", _serialize_paper(paper))
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
        prompt_len = int(enc.input_ids.shape[1])
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False
            )
        elapsed = time.time() - t0
        out_len = int(out_ids.shape[1] - prompt_len)
        records.append(
            {
                "paper_id": paper.get("paper_id", path.stem),
                "tokens_in": prompt_len,
                "tokens_out": out_len,
                "tokens_total": prompt_len + out_len,
                "calls": 1,
                "time_s": round(elapsed, 3),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarize(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    n = len(records)
    keys = ("tokens_in", "tokens_out", "tokens_total", "calls", "time_s")
    summary = {f"avg_{k}": sum(r[k] for r in records) / n for k in keys}
    summary["n_papers"] = n
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_dir", default="data/samples/eval")
    ap.add_argument(
        "--mode",
        choices=["teacher", "student", "both"],
        default="both",
        help="Measure teacher tokens, student tokens, or both.",
    )
    ap.add_argument("--teacher_config", default="configs/teacher_pipeline.yaml")
    ap.add_argument("--student_path", default="checkpoints/egtr_student")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--out", default="outputs/efficiency.json")
    args = ap.parse_args()

    report: dict = {}
    if args.mode in ("teacher", "both"):
        recs = measure_teacher(args.paper_dir, args.teacher_config)
        report["teacher"] = {"per_paper": recs, "summary": summarize(recs)}
    if args.mode in ("student", "both"):
        recs = measure_student(
            args.paper_dir, args.student_path, args.max_new_tokens
        )
        report["student"] = {"per_paper": recs, "summary": summarize(recs)}

    save_json(report, args.out)
    print(
        json.dumps(
            {k: v["summary"] for k, v in report.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
