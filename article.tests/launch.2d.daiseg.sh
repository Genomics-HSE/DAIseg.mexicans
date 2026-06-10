#!/usr/bin/env bash
set -euo pipefail

for i in $(seq 7 50); do
  BASE_SEED=$((123456789 + (i - 1) * 1000000))
  SIM_NAME="2d.daiseg.seed${i}"

  echo "Run ${i}/50"
  echo "SIM_NAME=${SIM_NAME}"
  echo "BASE_SEED=${BASE_SEED}"
  echo "Started at $(date)"

  env BASE_SEED="${BASE_SEED}" SIM_NAME="${SIM_NAME}" ./2d.daiseg.sh > "seed${i}.log" 2>&1

done
