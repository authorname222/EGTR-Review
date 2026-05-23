"""End-to-end inference: paper -> review comments.

Loads the distilled student model and emits a JSON review with
(position, evidence, reasoning, comment) tuples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .distillation.task_prefix import format_with_prefix
from .utils.io_utils import load_paper, save_json


def serialize_paper(paper: dict) -> str:
    parts = [f"# {paper.get('title', '')}", paper.get("abstract", "")]
    for sec in paper.get("sections", []):
        parts.append(f"## {sec.get('heading', '')}")
        parts.extend(sec.get("paragraphs", []))
    return "\n\n".join(parts)


def run_inference(model_path: str, paper_path: str, output_path: str, max_new_tokens: int = 1024) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paper = load_paper(paper_path)
    x = serialize_paper(paper)
    prompt = format_with_prefix("review", x)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(out_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

    try:
        review = json.loads(text)
    except json.JSONDecodeError:
        review = {"raw_text": text}

    save_json(
        {
            "paper_id": paper.get("paper_id", Path(paper_path).stem),
            "review": review,
        },
        output_path,
    )
    print(f"[inference] wrote {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()
    run_inference(args.model_path, args.paper, args.output, args.max_new_tokens)


if __name__ == "__main__":
    main()
