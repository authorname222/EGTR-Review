#!/bin/bash
# Reproduce Table 3: LLM-as-Judge cross-judge mean over 6 dimensions + ICC(2,k).
#
# The eval entry point produces a full metrics JSON; this script then prints
# just the row that goes into Table 3 (cross-judge mean per dimension,
# overall, and inter-judge ICC), so the numbers can be copy-pasted into the
# paper without further post-processing.
#
# By default uses the cached LLM-as-Judge results in
# `configs/evaluation.yaml :: llm_judge.use_cache`.
set -euo pipefail

PRED="${PRED:-outputs/test_predictions/egtr_student.json}"
REF="${REF:-data/samples/eval/references.json}"
OUT="${OUT:-outputs/table3.json}"

python -m src.evaluation.run_eval \
  --predictions "$PRED" \
  --references  "$REF" \
  --config      configs/evaluation.yaml \
  --out         "$OUT"

echo
echo "Table 3 metrics saved to $OUT"
echo
echo "=================  Table 3 row  ================="
python - <<PY
import json, math
m = json.load(open("$OUT"))
lj = m.get("llm_judge", {})
xj = lj.get("cross_judge_mean", {}) or {}
order = ["pertinency", "usefulness", "evidence_groundedness",
         "traceability", "depth", "comprehensiveness", "overall"]
labels = {"pertinency": "P", "usefulness": "U", "evidence_groundedness": "E",
          "traceability": "T", "depth": "D", "comprehensiveness": "C",
          "overall": "Overall"}
if not xj:
    print("(no llm_judge.cross_judge_mean — was llm_judge.judges configured?)")
else:
    header = " | ".join(f"{labels[k]:>7s}" for k in order)
    values = " | ".join(f"{xj.get(k, float('nan')):7.2f}" for k in order)
    print(header)
    print("-" * len(header))
    print(values)
    icc = lj.get("ICC_2k", float("nan"))
    icc_str = f"{icc:.4f}" if isinstance(icc, (int, float)) and not math.isnan(icc) else "n/a"
    print()
    print(f"ICC(2,k) = {icc_str}  (n_papers={lj.get('n_papers')}, n_judges={lj.get('n_judges')})")
PY
