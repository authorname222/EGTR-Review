# Reproducing the Paper

This page expands on README "Reproducing Paper Tables" with full step-by-step
instructions. All paths are repo-relative.

## 0. Environment

```bash
conda create -n egtr python=3.10 -y
conda activate egtr
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## 1. Build the full dataset (~6 hours, single machine)

The anonymous repository only ships data samples. The full dataset is
reconstructed locally:

```bash
# Released upon acceptance — placeholder script
# bash scripts/build_full_dataset.sh \
#     --peerread_dir data/raw/peerread \
#     --openreview_year 2019 2020 \
#     --out data/full
```

PeerRead is BSD-3 licensed and can be redistributed; OpenReview is not.
For this reason, only the fetcher is released, not the raw data.

## 2. Run the Multi-Agent Teacher pipeline

```bash
bash scripts/run_teacher_pipeline.sh \
    --input  data/full/train_papers \
    --output data/full/distilled
```

The pipeline is parallelizable per-paper; on 8 workers it processes ~1000
papers in ~9 hours of wall-clock time at the default GPT-5.1 budget.

To avoid re-issuing SerpApi / arXiv / Semantic Scholar calls, the
retriever caches results under
`data/full/distilled/cached_evidence/` (see `configs/retrieval.yaml`).

## 3. Build the distillation dataset

```bash
python -m src.distillation.build_distill_data \
    --teacher_outputs data/full/distilled \
    --out             data/full/D_distill.jsonl
```

## 4. Train the student

```bash
bash scripts/run_student_train.sh \
    --data   data/full/D_distill.jsonl \
    --config configs/student_train.yaml
```

Hardware: 2 × A100 (40 GB), bfloat16, gradient checkpointing.
Wall-clock: ~14 hours for 3 epochs over 997 papers (~6 minutes per step
at the default batch configuration).

## 5. Inference on the test set

```bash
mkdir -p outputs/test_predictions
for p in data/full/test_papers/*.json; do
  bash scripts/run_inference.sh "$p"
done
```

Aggregate predictions into a single JSON file with `paper_id` keys:

```bash
python -m src.utils.io_utils  # or a small aggregator script
```

## 6. Evaluation

```bash
bash scripts/run_eval.sh \
    --predictions outputs/test_predictions/egtr_student.json \
    --references  data/full/test_references.json \
    --out         outputs/test_metrics.json
```

The LLM-as-Judge step is the slowest. If you only want to verify the
metric computation, the cached judgments in
`data/samples/cached_llm_judgments.json` allow you to recompute the
six-dimension aggregates without re-issuing API calls.

## 6b. End-to-end smoke test (no API keys, no GPU)

```bash
python scripts/smoke_e2e.py
```

The smoke test exercises every wiring path — teacher pipeline → resume log
→ build_distill_data → task-prefixed multi-task examples → evidence-quality
metrics → cross-judge / ICC aggregation → efficiency token counter →
ablation harness → `--status` CLI — using stub LLM clients and stub
retrievers, so it runs in seconds without any external service or model
weight. Use it as a regression check after schema or pipeline changes.

## 7. Reproducing specific tables

| Table | Script                                  | Notes                                |
|-------|-----------------------------------------|--------------------------------------|
| 1     | `scripts/reproduce_table1.sh`           | ROUGE + BERTScore                   |
| 2     | `scripts/reproduce_table2.sh`           | SN-F1 + ITF-IDF (same entry point)  |
| 3     | `scripts/reproduce_table3.sh`           | LLM-as-Judge (cacheable)            |
| 5     | `scripts/reproduce_table5.sh`           | Evidence quality                    |
| 6     | `scripts/reproduce_ablation.sh`         | Trains the 5 variants (full + 4 single-switch), runs inference, scores against `--references`, and writes `outputs/ablation/table6.json`. Use `--skip_training` if `checkpoints/ablation/<name>/` already exists. |
| 8     | `scripts/reproduce_efficiency.sh`       | Tokens + wall-clock per paper. Teacher tokens are accumulated across all 5-agent `LLMClient.chat()` calls; student tokens come from the HF tokenizer + `generate()` length. Set `MODE=teacher|student|both`. |
