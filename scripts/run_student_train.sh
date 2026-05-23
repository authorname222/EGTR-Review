#!/bin/bash
# Train the distilled student model with task-prefix multi-task + evidence-weighted loss.
set -euo pipefail

DATA="data/samples/distilled/D_distill.jsonl"
CONFIG="configs/student_train.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)   DATA="$2";   shift 2;;
    --config) CONFIG="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

python -m src.distillation.train_student --data "$DATA" --config "$CONFIG"
