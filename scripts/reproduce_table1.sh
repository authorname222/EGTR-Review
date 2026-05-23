#!/bin/bash
# Reproduce Table 1: ROUGE + BERTScore on the test set across all baselines.
set -euo pipefail

PRED_DIR="${PRED_DIR:-outputs/test_predictions}"
REF="${REF:-data/samples/eval/references.json}"
OUT_DIR="${OUT_DIR:-outputs/table1}"

mkdir -p "$OUT_DIR"

for pred in "$PRED_DIR"/*.json; do
  name=$(basename "$pred" .json)
  python -m src.evaluation.run_eval \
    --predictions "$pred" \
    --references "$REF" \
    --config configs/evaluation.yaml \
    --out "$OUT_DIR/$name.json"
done

echo "Table 1 metrics saved to $OUT_DIR"
