#!/usr/bin/env bash
set -euo pipefail

python src/xxz_mps_study.py \
  --L 64 \
  --Ds 4 8 12 16 24 32 40 \
  --records 1000 \
  --candidate-D 32 \
  --output output/L64_D_convergence
