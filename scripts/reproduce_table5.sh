#!/bin/bash
# Reproduce Table 5: Evidence quality (EA / LC / ECR / SLA).
# Shares the same evaluation entry point; evidence_quality is always computed.
set -euo pipefail
exec bash scripts/reproduce_table3.sh "$@"
