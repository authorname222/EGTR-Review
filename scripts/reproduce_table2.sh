#!/bin/bash
# Reproduce Table 2: SN-F1 + ITF-IDF across all baselines.
# Shares the same evaluation entry point as Table 1; the metrics are reported jointly.
set -euo pipefail
exec bash scripts/reproduce_table1.sh "$@"
