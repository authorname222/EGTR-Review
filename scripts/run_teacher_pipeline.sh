#!/bin/bash
# Run the Multi-Agent Teacher pipeline, or print resume-log status.
#
# Standard run:
#     bash scripts/run_teacher_pipeline.sh --input <papers> --output <distilled>
#
# Resume-log status (no API calls — reads <output>/resume_log.jsonl):
#     bash scripts/run_teacher_pipeline.sh --output <distilled> --status
#     STATUS_ONLY=1 bash scripts/run_teacher_pipeline.sh --output <distilled>
set -euo pipefail

INPUT="data/samples/train_papers"
OUTPUT="data/samples/distilled"
CONFIG="configs/teacher_pipeline.yaml"
STATUS_ONLY="${STATUS_ONLY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)  INPUT="$2";  shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --status) STATUS_ONLY=1; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

if [[ -n "$STATUS_ONLY" ]]; then
  python -m src.teacher_pipeline --output "$OUTPUT" --status
else
  python -m src.teacher_pipeline --input "$INPUT" --output "$OUTPUT" --config "$CONFIG"
fi
