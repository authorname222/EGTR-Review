#!/bin/bash
# Run the full evaluation suite (auto metrics + LLM-as-Judge + evidence quality).
set -euo pipefail

PREDICTIONS=""
REFERENCES="data/samples/eval/references.json"
CONFIG="configs/evaluation.yaml"
OUT="outputs/eval_metrics.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions) PREDICTIONS="$2"; shift 2;;
    --references)  REFERENCES="$2";  shift 2;;
    --config)      CONFIG="$2";      shift 2;;
    --out)         OUT="$2";         shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$PREDICTIONS" ]]; then
  echo "usage: $0 --predictions <path> [--references <path>] [--config <path>] [--out <path>]"
  exit 1
fi

python -m src.evaluation.run_eval \
  --predictions "$PREDICTIONS" \
  --references "$REFERENCES" \
  --config "$CONFIG" \
  --out "$OUT"
