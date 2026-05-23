#!/bin/bash
# End-to-end inference demo on a single paper.
set -euo pipefail

PAPER="${1:-data/samples/eval/demo_papers.json}"
MODEL_PATH="${MODEL_PATH:-checkpoints/egtr_student}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

mkdir -p "$OUTPUT_DIR"
OUT="$OUTPUT_DIR/$(basename "$PAPER" .json)_review.json"

python -m src.inference \
  --paper "$PAPER" \
  --model_path "$MODEL_PATH" \
  --output "$OUT"

echo "Wrote review to $OUT"
