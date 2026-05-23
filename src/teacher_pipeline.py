"""End-to-end Multi-Agent Teacher pipeline.

Composes the five agents into the teacher flow that produces
L'_i (evidence-enhanced paper representation), T_i (reasoning trajectory)
and Y_i (final review) — the supervision triple for distillation.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .agents import (
    EvidenceRetrieverAgent,
    KeyElementExtractorAgent,
    ReviewSynthesizerAgent,
    StructureParserAgent,
    VerificationReasonerAgent,
)
from .retrieval import ArxivClient, SemanticScholarClient, SerpApiClient
from .utils.anonymizer import scrub
from .utils.io_utils import load_paper, save_json


class LLMClient:
    """Thin wrapper around the OpenAI-compatible Chat Completions API.

    Tracks prompt and completion tokens across every `chat()` call so the
    teacher-side efficiency measurement (Table 8) can sum tokens across all
    five agents per paper without instrumenting every agent separately.
    """

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 4096):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    def reset_counters(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    def chat(self, system: str, user: str, response_format: str | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.call_count += 1
        return resp.choices[0].message.content or ""


class TeacherPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["teacher"]
        llm_cfg = cfg["llm"]
        self.llm = LLMClient(
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_output_tokens", 4096),
        )
        retrievers = {
            "serpapi": SerpApiClient(),
            "arxiv": ArxivClient(),
            "semantic_scholar": SemanticScholarClient(),
        }
        self.parser = StructureParserAgent(cfg["structure_parser"], self.llm)
        self.extractor = KeyElementExtractorAgent(cfg["key_element_extractor"], self.llm)
        self.retriever = EvidenceRetrieverAgent(
            cfg["evidence_retriever"], self.llm, retrievers
        )
        self.reasoner = VerificationReasonerAgent(cfg["verification_reasoner"], self.llm)
        self.synth = ReviewSynthesizerAgent(cfg["review_synthesizer"], self.llm)
        # Cap the Structure Parser ↔ Key Element Extractor feedback loop at two
        # rounds, per paper §3.2 ("This feedback-based revision is performed
        # for at most two rounds").
        self.max_resegmentation_rounds = int(
            cfg.get("max_resegmentation_rounds", 2)
        )

    def run(self, paper: dict) -> dict:
        # Structure Parser ↔ Key Element Extractor feedback loop (paper §3.2).
        # The Key Element Extractor may request re-segmentation or merging from
        # the Structure Parser; the paper caps this loop at two rounds.
        units = self.parser.run(paper)
        annotated = self.extractor.run(units)
        for _ in range(self.max_resegmentation_rounds):
            feedback = self.extractor.request_resegmentation(annotated)
            if not feedback:
                break
            units = self.parser.revise(paper, feedback)
            annotated = self.extractor.run(units)
        retrieved = self.retriever.run(annotated)
        for u in retrieved:
            if isinstance(u.get("C_i_n"), str):
                u["C_i_n"] = scrub(u["C_i_n"])
        trajectory = self.reasoner.run(retrieved)
        review = self.synth.run(retrieved, trajectory)
        return {
            "paper_id": paper.get("paper_id"),
            "L_prime": retrieved,   # list of L_i' units, schema per prompts/ §3
            "T": trajectory,         # list of reasoning units, schema per prompts/ §4
            "Y": review,             # structured review object, schema per prompts/ §5
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_log(log_path: Path, entry: dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(input_dir: str, output_dir: str, config: str = "configs/teacher_pipeline.yaml") -> dict:
    """Run the teacher over every paper in `input_dir`.

    Resume support (docs/faq.md Q8): per-paper outputs are written to
    `<output_dir>/<paper_id>.json`; a paper whose output already exists is
    skipped. Every paper produces one line in `<output_dir>/resume_log.jsonl`
    so a partial run can be audited or resumed.

    Returns a summary dict counting completed/skipped/failed papers.
    """
    pipeline = TeacherPipeline(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "resume_log.jsonl"

    paper_paths = sorted(Path(input_dir).glob("*.json"))
    completed = skipped = failed = 0
    run_started = _utc_now()
    _append_log(
        log_path,
        {
            "event": "run_start",
            "ts": run_started,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "config": str(config),
            "n_inputs": len(paper_paths),
        },
    )

    for path in paper_paths:
        try:
            paper = load_paper(path)
            pid = paper.get("paper_id", path.stem)
        except Exception as e:  # malformed JSON or unreadable file
            failed += 1
            _append_log(
                log_path,
                {
                    "event": "paper_failed",
                    "ts": _utc_now(),
                    "paper_path": str(path),
                    "stage": "load_paper",
                    "error": repr(e),
                },
            )
            print(f"[teacher] FAILED to load {path.name}: {e}", flush=True)
            continue

        out_path = out_dir / f"{pid}.json"
        if out_path.exists():
            skipped += 1
            _append_log(
                log_path,
                {
                    "event": "paper_skipped",
                    "ts": _utc_now(),
                    "paper_id": pid,
                    "reason": "output_exists",
                    "output_path": str(out_path),
                },
            )
            print(f"[teacher] skip {pid} (already exists)", flush=True)
            continue

        t0 = time.time()
        started_at = _utc_now()
        pipeline.llm.reset_counters()
        try:
            out = pipeline.run(paper)
        except Exception as e:
            failed += 1
            _append_log(
                log_path,
                {
                    "event": "paper_failed",
                    "ts": _utc_now(),
                    "paper_id": pid,
                    "stage": "pipeline.run",
                    "error": repr(e),
                    "duration_s": round(time.time() - t0, 3),
                    "started_at": started_at,
                },
            )
            print(f"[teacher] FAILED {pid}: {e}", flush=True)
            continue
        save_json(out, out_path)
        duration = time.time() - t0
        completed += 1
        _append_log(
            log_path,
            {
                "event": "paper_completed",
                "ts": _utc_now(),
                "paper_id": pid,
                "output_path": str(out_path),
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_s": round(duration, 3),
                "llm_prompt_tokens": pipeline.llm.prompt_tokens,
                "llm_completion_tokens": pipeline.llm.completion_tokens,
                "llm_calls": pipeline.llm.call_count,
            },
        )
        print(
            f"[teacher] done {pid} ({duration:.1f}s, "
            f"{pipeline.llm.call_count} calls, "
            f"{pipeline.llm.prompt_tokens + pipeline.llm.completion_tokens} tok)",
            flush=True,
        )

    summary = {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "total": len(paper_paths),
        "started_at": run_started,
        "finished_at": _utc_now(),
    }
    _append_log(log_path, {"event": "run_end", **summary})
    print(
        f"[teacher] summary: completed={completed} "
        f"skipped={skipped} failed={failed} total={len(paper_paths)}",
        flush=True,
    )
    return summary


def status_from_log(output_dir: str) -> dict[str, Any]:
    """Read `<output_dir>/resume_log.jsonl` and summarize the run state.

    Used by `python -m src.teacher_pipeline --status` to answer FAQ Q8
    without re-running anything.
    """
    log_path = Path(output_dir) / "resume_log.jsonl"
    if not log_path.exists():
        return {"completed": [], "skipped": [], "failed": [], "n_events": 0}
    completed: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    last_start = last_end = None
    n_events = 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_events += 1
            kind = entry.get("event")
            if kind == "paper_completed":
                completed.append(entry)
            elif kind == "paper_skipped":
                skipped.append(entry)
            elif kind == "paper_failed":
                failed.append(entry)
            elif kind == "run_start":
                last_start = entry
            elif kind == "run_end":
                last_end = entry
    return {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "n_events": n_events,
        "last_run_start": last_start,
        "last_run_end": last_end,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/teacher_pipeline.yaml")
    ap.add_argument(
        "--status",
        action="store_true",
        help="Print a summary of <output>/resume_log.jsonl and exit without "
             "running the pipeline (FAQ Q8).",
    )
    args = ap.parse_args()
    if args.status:
        s = status_from_log(args.output)
        print(
            json.dumps(
                {
                    "completed_ids": [e["paper_id"] for e in s["completed"]],
                    "skipped_ids": [e["paper_id"] for e in s["skipped"]],
                    "failed_ids": [
                        e.get("paper_id") or e.get("paper_path") for e in s["failed"]
                    ],
                    "counts": {
                        "completed": len(s["completed"]),
                        "skipped": len(s["skipped"]),
                        "failed": len(s["failed"]),
                    },
                    "last_run_start": s["last_run_start"],
                    "last_run_end": s["last_run_end"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if not args.input:
            ap.error("--input is required when not using --status")
        main(args.input, args.output, args.config)
