"""Student-model distillation training entry point.

Implements task-prefix multi-task learning and evidence-weighted loss
(paper Sec. 3.3.3, Eq. 4):

  L_i      = alpha_i * (L_reason_i + lambda_balance * L_review_i)
  L_total  = mean over papers i

The reasoning and review supervisions share the evidence-enhanced input but
use different task prefixes; alpha_i is the per-sample (per-paper) weight
aggregated from unit-level evidence-state labels by `compute_sample_weight`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .evidence_weighting import compute_sample_weight
from .task_prefix import TASK_PREFIXES, format_with_prefix
from ..utils.io_utils import load_jsonl


def build_examples(records: list[dict], ablation: dict) -> list[dict]:
    """Expand each paper into (Reasoning, Review) supervised examples."""
    use_prefix = ablation.get("use_task_prefix", True)
    use_reasoning = ablation.get("use_reasoning_supervision", True)
    use_evidence_labels = ablation.get("use_evidence_labels", True)
    use_evidence_weighting = ablation.get("use_evidence_weighting", True)

    examples = []
    for rec in records:
        paper_id = rec["paper_id"]
        x = serialize_input(rec["L_prime"], include_flags=use_evidence_labels)
        flags = [u.get("Flag_i_n", "No Evidence") for u in rec.get("L_prime", [])]
        alpha = compute_sample_weight(flags) if use_evidence_weighting else 1.0

        if use_reasoning:
            examples.append(
                {
                    "paper_id": paper_id,
                    "task": "reasoning",
                    "input": format_with_prefix("reasoning", x) if use_prefix else x,
                    "target": json.dumps(rec.get("T", []), ensure_ascii=False),
                    "alpha": alpha,
                }
            )
        examples.append(
            {
                "paper_id": paper_id,
                "task": "review",
                "input": format_with_prefix("review", x) if use_prefix else x,
                "target": json.dumps(rec.get("Y", {}), ensure_ascii=False),
                "alpha": alpha,
            }
        )
    return examples


def serialize_input(l_prime: list[dict], include_flags: bool) -> str:
    parts = []
    for u in l_prime:
        block = [
            f"[Position] {json.dumps(u.get('P_i_n', ''), ensure_ascii=False)}",
            f"[Content] {u.get('C_i_n', '')}",
        ]
        if u.get("E_i_n"):
            block.append(
                f"[Evidence] {json.dumps(u.get('E_i_n', []), ensure_ascii=False)}"
            )
        if include_flags and u.get("Flag_i_n"):
            block.append(f"[Flag] {u.get('Flag_i_n')}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def train(config_path: str, data_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["train"]
    distill_cfg = cfg["distillation"]
    ablation = cfg.get("ablation", {})

    records = load_jsonl(data_path)
    examples = build_examples(records, ablation)
    print(f"[train_student] loaded {len(records)} papers -> {len(examples)} examples")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(cfg["student"]["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["student"]["base_model"],
        torch_dtype=torch.bfloat16,
    )
    if cfg["student"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    lora_cfg = cfg["student"]["lora"]
    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    class SFTDataset(Dataset):
        def __init__(self, items: list[dict]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            it = self.items[idx]
            text = it["input"] + "\n\n" + it["target"]
            enc = tokenizer(
                text,
                truncation=True,
                max_length=train_cfg["max_length"],
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "alpha": float(it["alpha"]),
                "task": it["task"],
            }

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids = torch.tensor(b["input_ids"], dtype=torch.long)
            input_ids[i, :n] = ids
            attn[i, :n] = torch.tensor(b["attention_mask"], dtype=torch.long)
            labels[i, :n] = ids
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "alpha": torch.tensor([b["alpha"] for b in batch], dtype=torch.float32),
            "task": [b["task"] for b in batch],
        }

    dataset = SFTDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        collate_fn=collate,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    lambda_balance = distill_cfg["lambda_balance"]
    for epoch in range(train_cfg["num_epochs"]):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # Per-sample (paper) weight α_i; per-task multiplier on review loss.
            alpha = batch["alpha"].to(device)
            task_mult = torch.tensor(
                [lambda_balance if t == "review" else 1.0 for t in batch["task"]],
                device=device,
                dtype=alpha.dtype,
            )
            # Use reduction=mean per-example, then weight per-example.
            # HF returns scalar loss = mean over batch; recover per-example via
            # token-level CE with reduction="none".
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Treat -100 (pad) as ignored; CE with reduction='none' must use a
            # valid index, so we replace -100 with 0 then mask afterwards.
            valid_mask = (shift_labels != -100).to(logits.dtype)
            safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
            ce = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), safe_labels, reduction="none"
            )
            per_example = (ce * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            loss = (alpha * task_mult * per_example).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step % train_cfg["logging_steps"] == 0:
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"alpha_mean {alpha.mean().item():.3f}"
                )

    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train_student] saved to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config, args.data)


if __name__ == "__main__":
    main()
