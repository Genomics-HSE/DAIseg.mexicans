#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Minimal DAIseg.mexicans test run: simulate a reduced dataset, run DAIseg, and evaluate predictions.

SIM_NAME="${1:-test.em}"
BASE_DIR="${BASE_DIR:-.}"
YAML_FILE="${YAML_FILE:-mexicans.demography.yml}"

N_CHR=10
CHR_LENGTH=30000000
BASE_SEED="${BASE_SEED:-12345}"
N_JOBS_SIM="${N_JOBS_SIM:-4}"

SIM_N_MEXICANS="${SIM_N_MEXICANS:-1}"
SIM_N_EU="${SIM_N_EU:-250}"
SIM_N_NA="${SIM_N_NA:-250}"
SIM_N_AF="${SIM_N_AF:-250}"
SIM_N_ND="${SIM_N_ND:-10}"

MEXICANS_EU_REF="${MEXICANS_EU_REF:-250}"
MEXICANS_NA_REF="${MEXICANS_NA_REF:-250}"
MEXICANS_AF_REF="${MEXICANS_AF_REF:-250}"
MEXICANS_ND_REF="${MEXICANS_ND_REF:-3}"

THREADS="${THREADS:-4}"

step() {
  echo
  echo "[$(date '+%F %T')] $*"
}

echo "Minimal DAIseg.mexicans run"
echo "SIM_NAME=${SIM_NAME}"
echo "BASE_DIR=${BASE_DIR}"
echo "N_CHR=${N_CHR}"
echo "CHR_LENGTH=${CHR_LENGTH}"
echo "BASE_SEED=${BASE_SEED}"
echo "YAML_FILE=${YAML_FILE}"

step "simulate ${N_CHR} chromosomes x ${CHR_LENGTH} bp"

python simulate_mexicans.py \
  --sim-name "${SIM_NAME}" \
  --base-dir "${BASE_DIR}" \
  --yaml "${YAML_FILE}" \
  --n-chr "${N_CHR}" \
  --chr-length "${CHR_LENGTH}" \
  --base-seed "${BASE_SEED}" \
  --n-jobs "${N_JOBS_SIM}" \
  --n-mexicans "${SIM_N_MEXICANS}" \
  --n-eu "${SIM_N_EU}" \
  --n-na "${SIM_N_NA}" \
  --n-af "${SIM_N_AF}" \
  --n-nd "${SIM_N_ND}" \
  --clean

step "prepare_daiseg_mexicans"

python prepare_daiseg_mexicans.py \
  --sim-name "${SIM_NAME}" \
  --base-dir "${BASE_DIR}" \
  --n-eu-ref "${MEXICANS_EU_REF}" \
  --n-na-ref "${MEXICANS_NA_REF}" \
  --n-af-ref "${MEXICANS_AF_REF}" \
  --n-nd-ref "${MEXICANS_ND_REF}" \
  --threads "${THREADS}" \
  --force

step "run_daiseg_mexicans"

python run_daiseg_mexicans.py \
  --sim-name "${SIM_NAME}" \
  --base-dir "${BASE_DIR}" \
  --n-eu-ref "${MEXICANS_EU_REF}" \
  --n-na-ref "${MEXICANS_NA_REF}" \
  --n-af-ref "${MEXICANS_AF_REF}" \
  --n-nd-ref "${MEXICANS_ND_REF}" \
  --threads "${THREADS}" \
  --force

step "evaluate daiseg_mexicans"

python evaluate_methods.py daiseg_mexicans \
  --sim-name "${SIM_NAME}" \
  --base-dir "${BASE_DIR}" \
  --mexicans-eu-ref "${MEXICANS_EU_REF}" \
  --mexicans-na-ref "${MEXICANS_NA_REF}" \
  --mexicans-af-ref "${MEXICANS_AF_REF}" \
  --mexicans-nd-ref "${MEXICANS_ND_REF}"
