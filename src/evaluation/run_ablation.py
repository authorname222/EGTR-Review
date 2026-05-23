"""End-to-end ablation harness (paper Table 6).

For each single-switch variant defined in `VARIANTS`:

  1. Patch the `ablation:` block of the base student config.
  2. Train the student (skippable with --skip_training to evaluate an
     existing checkpoint, e.g. for CI / smoke tests).
  3. Run student inference on every paper in --paper_dir.
  4. Aggregate per-paper predictions into a single JSON list and score
     them against --references via `src.evaluation.run_eval`.
  5. Collect the resulting metrics into one summary file
     (`<out_dir>/table6.json`).

The single-switch list mirrors §4 / Table 6 of the paper:
  full                  — baseline (all switches on)
  w/o reasoning         — drop reasoning supervision
  w/o task prefix       — drop the [Reasoning]/[Review] task prefixes
  w/o evidence labels   — drop Flag_i_n from the student input
  w/o evidence weighting— uniform α = 1 in the loss
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


VARIANTS: dict[str, tuple[str, str, bool] | None] = {
    "full": None,
    "wo_reasoning": ("ablation", "use_reasoning_supervision", False),
    "wo_prefix": ("ablation", "use_task_prefix", False),
    "wo_evidence_labels": ("ablation", "use_evidence_labels", False),
    "wo_evidence_weighting": ("ablation", "use_evidence_weighting", False),
}


def _patched_config(base_path: str, name: str, override) -> str:
    cfg = yaml.safe_load(open(base_path))
    if override is not None:
        section, key, value = override
        cfg.setdefault(section, {})[key] = value
    cfg.setdefault("train", {})["output_dir"] = f"checkpoints/ablation/{name}"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
    tmp.close()
    return tmp.name


def _iter_paper_files(paper_dir: str):
    """Yield paper JSON paths under `paper_dir` that look like a paper schema."""
    for path in sorted(Path(paper_dir).glob("*.json")):
        try:
            obj = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and (
            "sections" in obj or "abstract" in obj or "title" in obj
        ):
            yield path


def _run(cmd: list[str]) -> None:
    print("[ablation] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_variant(
    name: str,
    override,
    args: argparse.Namespace,
    out_root: Path,
) -> Path:
    cfg_path = _patched_config(args.base_config, name, override)
    ckpt = Path(f"checkpoints/ablation/{name}")

    if not args.skip_training:
        _run(
            [
                sys.executable,
                "-m",
                "src.distillation.train_student",
                "--data",
                args.data,
                "--config",
                cfg_path,
            ]
        )

    # Inference: collect per-paper predictions into a single list.
    pred_path = out_root / f"{name}.predictions.json"
    predictions: list[dict] = []
    for paper_path in _iter_paper_files(args.paper_dir):
        tmp_out = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        _run(
            [
                sys.executable,
                "-m",
                "src.inference",
                "--paper",
                str(paper_path),
                "--model_path",
                str(ckpt),
                "--output",
                tmp_out,
                "--max_new_tokens",
                str(args.max_new_tokens),
            ]
        )
        predictions.append(json.load(open(tmp_out)))
    with open(pred_path, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Evaluation: auto metrics + evidence quality (+ LLM-Judge if enabled).
    metrics_path = out_root / f"{name}.metrics.json"
    _run(
        [
            sys.executable,
            "-m",
            "src.evaluation.run_eval",
            "--predictions",
            str(pred_path),
            "--references",
            args.references,
            "--config",
            args.eval_config,
            "--out",
            str(metrics_path),
        ]
    )
    return metrics_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples/distilled/D_distill.jsonl")
    ap.add_argument("--base_config", default="configs/student_train.yaml")
    ap.add_argument("--paper_dir", default="data/samples/eval")
    ap.add_argument("--references", default="data/samples/eval/references.json")
    ap.add_argument("--eval_config", default="configs/evaluation.yaml")
    ap.add_argument("--out_dir", default="outputs/ablation")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip the training step and only run inference + evaluation "
             "against pre-existing checkpoints/ablation/<name>/ directories.",
    )
    ap.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Subset of variant names to run (default: all). "
             f"Choices: {list(VARIANTS)}",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    chosen = args.variants or list(VARIANTS)
    table: dict[str, dict] = {}
    for name in chosen:
        if name not in VARIANTS:
            print(f"[ablation] unknown variant {name!r}, skipping", file=sys.stderr)
            continue
        metrics_path = run_variant(name, VARIANTS[name], args, out_root)
        table[name] = json.load(open(metrics_path))

    summary_path = out_root / "table6.json"
    with open(summary_path, "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"[ablation] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
