"""Build the distillation dataset D_distill (paper §3.3.1).

Each line is one paper's training instance:

  - L_prime : list of evidence-enhanced units (schema in docs/prompts/03_evidence_retriever.md)
  - T       : list of reasoning units            (schema in docs/prompts/04_verification_reasoner.md)
  - Y       : structured final-review object     (schema in docs/prompts/05_review_synthesizer.md)
  - alpha   : aggregated evidence weight from Flag_i_n
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence_weighting import compute_sample_weight
from ..utils.io_utils import save_jsonl


def build(teacher_outputs_dir: str, out_path: str) -> int:
    items = []
    for path in sorted(Path(teacher_outputs_dir).glob("*.json")):
        with open(path) as f:
            obj = json.load(f)
        units = obj.get("L_prime", []) or []
        flags = [u.get("Flag_i_n", "No Evidence") for u in units]
        items.append(
            {
                "paper_id": obj.get("paper_id", path.stem),
                "L_prime": units,
                "T": obj.get("T", []),
                "Y": obj.get("Y", {}),
                "alpha": compute_sample_weight(flags),
            }
        )
    save_jsonl(items, out_path)
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_outputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = build(args.teacher_outputs, args.out)
    print(f"Wrote {n} distillation samples to {args.out}")


if __name__ == "__main__":
    main()
