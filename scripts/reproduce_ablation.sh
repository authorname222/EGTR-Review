#!/bin/bash
# Reproduce Table 6: ablation over the four single-switch variants on the
# student side (reasoning supervision / task prefix / evidence labels /
# evidence weighting). The runner trains each variant, runs inference on
# the eval paper set, scores against references, and aggregates results
# into outputs/ablation/table6.json.
#
# To rebuild the table without re-training (e.g. when checkpoints already
# exist under checkpoints/ablation/<variant>/), pass --skip_training.
set -euo pipefail

DATA="${DATA:-data/samples/distilled/D_distill.jsonl}"
BASE_CONFIG="${BASE_CONFIG:-configs/student_train.yaml}"
PAPER_DIR="${PAPER_DIR:-data/samples/eval}"
REFERENCES="${REFERENCES:-data/samples/eval/references.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/evaluation.yaml}"
OUT_DIR="${OUT_DIR:-outputs/ablation}"

python -m src.evaluation.run_ablation \
  --data         "$DATA" \
  --base_config  "$BASE_CONFIG" \
  --paper_dir    "$PAPER_DIR" \
  --references   "$REFERENCES" \
  --eval_config  "$EVAL_CONFIG" \
  --out_dir      "$OUT_DIR" \
  "$@"

echo "Ablation table written to $OUT_DIR/table6.json"
