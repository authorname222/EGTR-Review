#!/bin/bash
# Reproduce Table 8: tokens / paper + inference time / paper.
#
# Teacher row : sums prompt + completion tokens across all five agents per
#               paper (via LLMClient counters in src/teacher_pipeline.py)
#               and times the whole 5-agent pipeline.
# Student row : counts prompt and generated tokens directly via the HF
#               tokenizer and times a single model.generate() call.
#
# The two rows are written to a single JSON report.
set -euo pipefail

PAPER_DIR="${PAPER_DIR:-data/samples/eval}"
MODE="${MODE:-both}"                       # teacher | student | both
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/teacher_pipeline.yaml}"
MODEL_PATH="${MODEL_PATH:-checkpoints/egtr_student}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
OUT="${OUT:-outputs/efficiency.json}"

python -m src.evaluation.efficiency \
  --paper_dir       "$PAPER_DIR" \
  --mode            "$MODE" \
  --teacher_config  "$TEACHER_CONFIG" \
  --student_path    "$MODEL_PATH" \
  --max_new_tokens  "$MAX_NEW_TOKENS" \
  --out             "$OUT"

echo "Efficiency report saved to $OUT"
