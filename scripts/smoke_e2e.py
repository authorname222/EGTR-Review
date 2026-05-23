"""End-to-end smoke test for EGTR-Review.

Drives the full pipeline using:
  - Stub LLMClient with canned JSON responses per agent
  - Stub retrievers returning fixed evidence records
  - Stub student model inference (no torch / no GPU)
  - Real code paths for everything else (build_distill_data, run_eval auto
    metrics, evidence_quality, ablation harness, efficiency measurement,
    resume_log, --status CLI)

Verifies that the wiring produces the expected outputs without ever
calling an external API.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO = Path("/home/run/EGTR-Review")
sys.path.insert(0, str(REPO))


# Inject stub `openai` module + heavy deps so imports succeed without the
# real packages being installed.
def _install_stub_modules():
    if "openai" not in sys.modules:
        m = types.ModuleType("openai")
        class _OpenAI:
            def __init__(self, *a, **k):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **kw: None)
                )
        m.OpenAI = _OpenAI
        sys.modules["openai"] = m

_install_stub_modules()

SCRATCH = Path("/tmp/egtr_e2e")
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)

INPUT_DIR  = SCRATCH / "train_papers"
OUTPUT_DIR = SCRATCH / "distilled"
EVAL_DIR   = SCRATCH / "eval"
INPUT_DIR.mkdir(); OUTPUT_DIR.mkdir(); EVAL_DIR.mkdir()

# --- Copy demo paper + reference into the scratch dirs ---
shutil.copy(REPO / "data/samples/eval/demo_papers.json",  INPUT_DIR / "demo.json")
shutil.copy(REPO / "data/samples/eval/demo_papers.json",  EVAL_DIR  / "demo.json")
shutil.copy(REPO / "data/samples/eval/references.json",   EVAL_DIR  / "references.json")

print("=" * 70)
print("STAGE 0 — Inputs")
print("=" * 70)
for p in sorted(INPUT_DIR.iterdir()):
    print(f"  {p.relative_to(SCRATCH)}  ({p.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# STAGE 1: stub LLMClient + retrievers, run teacher pipeline
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 1 — Teacher pipeline (stubbed LLM + retrievers)")
print("=" * 70)

# Canned LLM responses keyed by which prompt is in the system message.
# Each agent's prompt has a distinct opening phrase, so we route on that.
CANNED = {
    # Key = the agent's *opening* sentence (each agent's prompt starts with
    # "You are the <Agent Name> Agent in EGTR-Review"). Routing on this
    # is unambiguous, whereas matching just "Evidence Retriever Agent"
    # would collide with the references in §4/§5 prompts.
    "You are the Structure Parser Agent": [
        {
            "unit_id": "1",
            "P_i_n": "Section 1 Introduction / Paragraph 1",
            "C_i_n": "We propose a difficulty-aware sampler that requests teacher labels only for examples where the student's predictive entropy exceeds a threshold.",
            "structural_type": "Method",
            "segmentation_decision": "Keep",
            "segmentation_rationale": "Self-contained methodological claim.",
            "suggested_operation": "None",
        },
        {
            "unit_id": "2",
            "P_i_n": "Section 2 Experiments / Paragraph 1",
            "C_i_n": "On CIFAR-100, ImageNet-1k, and a 3M-example text classification benchmark, our method matches dense distillation within 0.4 accuracy points while reducing teacher compute by 51%.",
            "structural_type": "Result",
            "segmentation_decision": "Keep",
            "segmentation_rationale": "Coherent multi-benchmark result claim.",
            "suggested_operation": "None",
        },
    ],
    "You are the Key Element Extractor Agent": [
        {
            "unit_id": "1",
            "P_i_n": "Section 1 Introduction / Paragraph 1",
            "C_i_n": "We propose a difficulty-aware sampler that requests teacher labels only for examples where the student's predictive entropy exceeds a threshold.",
            "key_elements": ["difficulty-aware sampler", "entropy threshold"],
            "claim_type": "Methodological Claim",
            "retrieval_query": "difficulty-aware sampling knowledge distillation entropy threshold",
            "feedback_to_structure_parser_agent": "None",
            "feedback_rationale": "Fragment is well-bounded.",
        },
        {
            "unit_id": "2",
            "P_i_n": "Section 2 Experiments / Paragraph 1",
            "C_i_n": "On CIFAR-100, ImageNet-1k, and a 3M-example text classification benchmark, our method matches dense distillation within 0.4 accuracy points while reducing teacher compute by 51%.",
            "key_elements": ["CIFAR-100", "ImageNet-1k", "51% teacher compute reduction"],
            "claim_type": "Result Claim",
            "retrieval_query": "selective distillation teacher compute reduction CIFAR-100 ImageNet",
            "feedback_to_structure_parser_agent": "None",
            "feedback_rationale": "Fragment is well-bounded.",
        },
    ],
    "You are the Evidence Retriever Agent": {
        # Per-unit response (called once per unit by EvidenceRetrieverAgent)
        "default": [{
            "unit_id": "1",
            "P_i_n": "Section 1 Introduction / Paragraph 1",
            "C_i_n": "We propose a difficulty-aware sampler...",
            "Q_i_n": "difficulty-aware sampling knowledge distillation entropy threshold",
            "E_i_n": [
                {
                    "source_title": "Active Learning for Knowledge Distillation",
                    "source_metadata": {"source": "arxiv", "url": "https://arxiv.org/abs/example-1"},
                    "evidence_snippet": "We use uncertainty-based active sampling to reduce teacher query budget.",
                    "evidence_relevance": "support",
                }
            ],
            "Flag_i_n": "Strong Evidence-Supports",
            "labeling_rationale": "External evidence directly supports the entropy-based sampler.",
            "feedback_to_key_element_extractor_agent": "None",
            "feedback_rationale": "Query was sufficient.",
        }],
    },
    "You are the Verification Reasoner Agent": [
        {
            "unit_id": "1", "P_i_n": "Section 1 Introduction / Paragraph 1",
            "C_i_n": "We propose a difficulty-aware sampler...",
            "E_i_n": [], "Flag_i_n": "Strong Evidence-Supports",
            "verification_path": "Support Verification",
            "verification_question": "Does the cited active-learning literature support entropy-based selection?",
            "reasoning_process": "Cited paper supports uncertainty-based selection.",
            "preliminary_review_point": "The entropy-based sampler is well-grounded in active-learning literature.",
            "review_point_type": "Strength",
            "traceability_basis": "P_i_n / C_i_n / E_i_n",
        },
        {
            "unit_id": "2", "P_i_n": "Section 2 Experiments / Paragraph 1",
            "C_i_n": "On CIFAR-100, ImageNet-1k...",
            "E_i_n": [], "Flag_i_n": "Strong Evidence-Supports",
            "verification_path": "Support Verification",
            "verification_question": "Are the reported 0.4-point gap and 51% compute reduction internally consistent?",
            "reasoning_process": "Numbers are coherent within the paper.",
            "preliminary_review_point": "Compute savings are reported with appropriate qualifications.",
            "review_point_type": "Strength",
            "traceability_basis": "C_i_n / Reasoning Process",
        },
    ],
    "You are the Review Synthesizer Agent": {
        "summary": "The paper proposes a difficulty-aware sampler that requests teacher labels only when the student's predictive entropy exceeds a threshold. Across CIFAR-100, ImageNet-1k, and a 3M-example text classification benchmark, the method matches dense distillation within 0.4 accuracy points while reducing teacher compute by 51%.",
        "strengths": [{
            "comment": "The entropy-based sampler is intuitive and supported by prior active-learning work.",
            "impact": "Medium",
            "evidence_grounding": "arXiv:example-1",
            "trace_ids": ["T_i-1", "U1"],
        }],
        "weaknesses": [{
            "comment": "Robustness of the entropy threshold across {0.1, 0.5, 0.9} is not ablated.",
            "impact": "Medium",
            "evidence_grounding": "Internal reasoning.",
            "trace_ids": ["T_i-2", "U2"],
        }],
        "questions": [{
            "question": "Is teacher compute measured in wall-clock or FLOPs?",
            "type": "Clarification",
            "trace_ids": ["T_i-2"],
        }],
        "suggestions": [{
            "suggestion": "Add a random-subset distillation baseline to isolate the contribution of the difficulty-aware criterion.",
            "related_to": "Weakness W1",
            "trace_ids": ["T_i-2"],
        }],
        "traceability_notes": [{
            "review_point_id": "S1",
            "source_position": "Section 1 Introduction / Paragraph 1",
            "paper_fragment": "difficulty-aware sampler",
            "evidence_set": "arXiv:example-1",
            "evidence_state_label": "Strong Evidence-Supports",
            "reasoning_unit_id": "T_i-1",
            "rationale": "Strength is grounded in directly supporting external record.",
        }],
    },
}


class FakeUsage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class FakeResp:
    def __init__(self, content, p, c):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = FakeUsage(p, c)


class FakeOpenAI:
    """Mimics openai.OpenAI() for src.teacher_pipeline.LLMClient."""
    def __init__(self, *a, **k):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    @staticmethod
    def _create(**kwargs):
        system = (kwargs.get("messages") or [{}])[0].get("content", "")
        # Route on the agent's opening phrase
        for key, payload in CANNED.items():
            if key in system:
                if "Evidence Retriever Agent" in key:
                    body = json.dumps(payload["default"], ensure_ascii=False)
                elif isinstance(payload, dict):
                    body = json.dumps(payload, ensure_ascii=False)
                else:
                    body = json.dumps(payload, ensure_ascii=False)
                # token usage estimate: roughly 1 token per 4 chars
                prompt_tokens = max(1, len(system) // 4)
                completion_tokens = max(1, len(body) // 4)
                return FakeResp(body, prompt_tokens, completion_tokens)
        return FakeResp("[]", 1, 1)


class FakeRetriever:
    def __init__(self, *a, **k):
        pass
    def search(self, query, top_k=5):
        return [{
            "source": "stub",
            "title": "Stub Evidence Record",
            "abstract": "Stub abstract for E2E smoke test.",
            "url": "https://stub.example.com/record",
        }]


# Patch external dependencies and run the teacher pipeline
with patch("openai.OpenAI", FakeOpenAI), \
     patch("src.retrieval.SerpApiClient", FakeRetriever), \
     patch("src.retrieval.ArxivClient", FakeRetriever), \
     patch("src.retrieval.SemanticScholarClient", FakeRetriever):
    from src.teacher_pipeline import main as teacher_main, status_from_log
    summary = teacher_main(str(INPUT_DIR), str(OUTPUT_DIR),
                           str(REPO / "configs/teacher_pipeline.yaml"))
    print(f"  summary: {summary}")
    assert summary["completed"] == 1, summary
    assert summary["failed"] == 0, summary

# Verify the per-paper output file conforms to the §3 schema
demo_out = json.load(open(OUTPUT_DIR / "iclr2019_demo.json"))
print(f"  iclr2019_demo.json keys: {list(demo_out.keys())}")
assert demo_out["paper_id"] == "iclr2019_demo"
assert isinstance(demo_out["L_prime"], list) and len(demo_out["L_prime"]) >= 1
unit = demo_out["L_prime"][0]
assert {"unit_id", "P_i_n", "C_i_n", "E_i_n", "Flag_i_n"} <= unit.keys(), unit
assert unit["Flag_i_n"] in {"Strong Evidence-Supports","Strong Evidence-Refutes",
                            "Weak Evidence-Metadata Only","No Evidence","Non-verifiable Item"}
assert isinstance(demo_out["T"], list) and len(demo_out["T"]) >= 1
assert isinstance(demo_out["Y"], dict)
assert {"summary","strengths","weaknesses","questions","suggestions","traceability_notes"} <= demo_out["Y"].keys()
print("  schema check ✓")

# Resume log
status = status_from_log(str(OUTPUT_DIR))
print(f"  resume_log: {len(status['completed'])} completed, "
      f"{len(status['skipped'])} skipped, {len(status['failed'])} failed")
assert len(status["completed"]) == 1

# Re-run to verify the skip path
with patch("openai.OpenAI", FakeOpenAI), \
     patch("src.retrieval.SerpApiClient", FakeRetriever), \
     patch("src.retrieval.ArxivClient", FakeRetriever), \
     patch("src.retrieval.SemanticScholarClient", FakeRetriever):
    summary2 = teacher_main(str(INPUT_DIR), str(OUTPUT_DIR),
                            str(REPO / "configs/teacher_pipeline.yaml"))
assert summary2["completed"] == 0 and summary2["skipped"] == 1, summary2
print("  resume / skip path ✓")


# ---------------------------------------------------------------------------
# STAGE 2: build D_distill from teacher outputs
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 2 — build_distill_data")
print("=" * 70)

from src.distillation import build_distill_data
DDISTILL = SCRATCH / "D_distill.jsonl"
n = build_distill_data.build(str(OUTPUT_DIR), str(DDISTILL))
print(f"  wrote {n} record(s) to {DDISTILL.name}")
assert n == 1
record = json.loads(open(DDISTILL).readline())
assert {"paper_id","L_prime","T","Y","alpha"} <= record.keys()
print(f"  paper_id={record['paper_id']}  alpha={record['alpha']}  "
      f"n_units={len(record['L_prime'])}")


# ---------------------------------------------------------------------------
# STAGE 3: build_examples (verifies new [Reasoning]/[Review] task prefix)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 3 — build_examples (task-prefix multi-task formatting)")
print("=" * 70)

from src.distillation.train_student import build_examples
exs = build_examples([record], {
    "use_reasoning_supervision": True, "use_task_prefix": True,
    "use_evidence_labels": True, "use_evidence_weighting": True,
})
print(f"  produced {len(exs)} examples: {[e['task'] for e in exs]}")
print(f"  reasoning input head: {exs[0]['input'][:60]!r}")
print(f"  review    input head: {exs[1]['input'][:60]!r}")
assert exs[0]["input"].startswith("[Reasoning] ")
assert exs[1]["input"].startswith("[Review] ")
assert exs[0]["alpha"] == record["alpha"]


# ---------------------------------------------------------------------------
# STAGE 4: evaluation pipeline (auto metrics + evidence quality + run_eval)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 4 — Evaluation pipeline (auto metrics + evidence quality)")
print("=" * 70)

# Make a fake prediction that mirrors the teacher's Y, with human annotations
predictions = [{
    "paper_id": "iclr2019_demo",
    "review": demo_out["Y"],
    "L_prime": [
        {**u, "flag_correct": True,
         "E_i_n": [{**e, "evidence_authentic": True} for e in u.get("E_i_n", [])]}
        for u in demo_out["L_prime"]
    ],
}]
# Add per-comment annotations
for bucket in ("strengths","weaknesses","questions","suggestions"):
    for c in predictions[0]["review"].get(bucket, []):
        c["evidence_relevant"] = True
        c["position_correct"]  = True

PRED_PATH = SCRATCH / "predictions.json"
json.dump(predictions, open(PRED_PATH, "w"), ensure_ascii=False, indent=2)

# Evidence quality (no external deps)
from src.evaluation.evidence_quality import compute_evidence_quality
eq = compute_evidence_quality(predictions)
print(f"  evidence_quality: EA={eq['EA']} LC={eq['LC']} ECR={eq['ECR']} SLA={eq['SLA']}")
assert eq["EA"] == 1.0 and eq["LC"] == 1.0 and eq["ECR"] == 1.0 and eq["SLA"] == 1.0

# Auto metrics: ROUGE/BERTScore/SN-F1 need heavy packages. Test the bits
# that work without them: ITF-IDF + flatten_review.
from src.evaluation.auto_metrics import _itf_idf
from src.evaluation.run_eval import _flatten_review
flat = _flatten_review(demo_out["Y"])
assert "difficulty-aware sampler" in flat or "entropy" in flat
itf = _itf_idf([flat], [flat])
print(f"  flatten_review len: {len(flat)} chars")
print(f"  ITF-IDF self-compare: {itf:.3f}")
try:
    from src.evaluation.auto_metrics import _sn_f1
    sn = _sn_f1([flat], [flat])
    assert sn["SN-F1"] == 1.0
    print(f"  SN-F1 self-compare: {sn}")
except ImportError as e:
    print(f"  SN-F1 skipped (missing dep: {e.name})")


# ---------------------------------------------------------------------------
# STAGE 5: cross-judge averaging + ICC (LLM-as-Judge stubbed)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 5 — LLM-as-Judge cross-judge mean + ICC(2,k)")
print("=" * 70)
from src.evaluation.llm_judge import cross_judge_summary, DIMENSIONS
pj = {
    "gemini":   {"iclr2019_demo": {d: 8.5 for d in DIMENSIONS}},
    "deepseek": {"iclr2019_demo": {d: 8.0 for d in DIMENSIONS}},
    "claude":   {"iclr2019_demo": {d: 8.5 for d in DIMENSIONS}},
}
cj = cross_judge_summary(pj, paper_ids=["iclr2019_demo"])
print(f"  cross_judge_mean overall: {cj['average']['overall']:.3f}")
print(f"  per-dim: {dict((k, round(v,2)) for k,v in cj['average'].items())}")
expected = (8.5 + 8.0 + 8.5) / 3
assert abs(cj["average"]["overall"] - expected) < 1e-9
# ICC NaN with single paper (no between-subject variance possible at scale)
# is allowed; just confirm shape
print(f"  ICC(2,k): {cj['ICC_2k']}  (NaN expected on 1-paper smoke)")


# ---------------------------------------------------------------------------
# STAGE 6: efficiency measurement (teacher tokens; student stubbed)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 6 — Efficiency: teacher token + time per paper")
print("=" * 70)

with patch("openai.OpenAI", FakeOpenAI), \
     patch("src.retrieval.SerpApiClient", FakeRetriever), \
     patch("src.retrieval.ArxivClient", FakeRetriever), \
     patch("src.retrieval.SemanticScholarClient", FakeRetriever):
    from src.evaluation.efficiency import measure_teacher, summarize
    recs = measure_teacher(str(INPUT_DIR), str(REPO / "configs/teacher_pipeline.yaml"))

print(f"  teacher records ({len(recs)}):")
for r in recs:
    print(f"    paper_id={r['paper_id']}  tokens_in={r['tokens_in']}  "
          f"tokens_out={r['tokens_out']}  calls={r['calls']}  time_s={r['time_s']}")
print(f"  summary: {summarize(recs)}")
assert recs[0]["calls"] >= 4   # at least Parser + KEE + Reasoner + Synth
assert recs[0]["tokens_in"] > 0 and recs[0]["tokens_out"] > 0


# ---------------------------------------------------------------------------
# STAGE 7: ablation runner end-to-end (subprocess.run patched)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 7 — Ablation runner (5 variants, subprocess stubbed)")
print("=" * 70)

import argparse
import src.evaluation.run_ablation as ra

# Fake subprocess.run that satisfies train / inference / eval calls.
def fake_run(cmd, check=True):
    if "src.inference" in cmd:
        out = cmd[cmd.index("--output") + 1]
        json.dump({"paper_id": "iclr2019_demo", "review": demo_out["Y"]},
                  open(out, "w"))
    elif "src.evaluation.run_eval" in cmd:
        out = cmd[cmd.index("--out") + 1]
        json.dump({"auto": {"rougeL": 0.5}, "evidence_quality": eq,
                   "llm_judge": {"cross_judge_mean": cj["average"], "ICC_2k": None}},
                  open(out, "w"))
    # train_student: no-op

abl_out = SCRATCH / "ablation"
abl_out.mkdir()
args = argparse.Namespace(
    data=str(DDISTILL),
    base_config=str(REPO / "configs/student_train.yaml"),
    paper_dir=str(EVAL_DIR),
    references=str(EVAL_DIR / "references.json"),
    eval_config=str(REPO / "configs/evaluation.yaml"),
    out_dir=str(abl_out),
    max_new_tokens=128,
    skip_training=True,
    variants=None,
)

with patch.object(ra.subprocess, "run", side_effect=fake_run):
    for name, override in ra.VARIANTS.items():
        m = ra.run_variant(name, override, args, abl_out)
        assert m.exists()

table = {n: json.load(open(abl_out / f"{n}.metrics.json")) for n in ra.VARIANTS}
print(f"  ran {len(table)} variants: {list(table)}")
print(f"  variant metrics keys: {list(next(iter(table.values())).keys())}")


# ---------------------------------------------------------------------------
# STAGE 8: CLI / shell-script wiring
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STAGE 8 — CLI: --status and reproduce shell scripts")
print("=" * 70)

# --status CLI mode
r = subprocess.run([sys.executable, "-m", "src.teacher_pipeline",
                    "--output", str(OUTPUT_DIR), "--status"],
                   cwd=REPO, capture_output=True, text=True)
assert r.returncode == 0, r.stderr
status_payload = json.loads(r.stdout)
print(f"  --status: completed={status_payload['counts']['completed']} "
      f"skipped={status_payload['counts']['skipped']} "
      f"failed={status_payload['counts']['failed']}")
assert status_payload["counts"]["completed"] >= 1
assert status_payload["counts"]["skipped"]  >= 1   # the resume run

# STATUS_ONLY env var
r = subprocess.run(["bash", str(REPO / "scripts/run_teacher_pipeline.sh"),
                    "--output", str(OUTPUT_DIR), "--status"],
                   cwd=REPO, capture_output=True, text=True)
assert r.returncode == 0, r.stderr
assert "completed_ids" in r.stdout


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("END-TO-END SMOKE TEST: PASS")
print("=" * 70)
print(f"  Scratch dir: {SCRATCH}")
print(f"  Teacher outputs: {OUTPUT_DIR}")
print(f"  D_distill: {DDISTILL}")
print(f"  Predictions: {PRED_PATH}")
print(f"  Ablation outputs: {abl_out}")
print()
print("Stages: 0 inputs, 1 teacher, 2 build_distill, 3 build_examples,")
print("        4 evaluation, 5 cross-judge+ICC, 6 efficiency,")
print("        7 ablation, 8 CLI — all passed.")
