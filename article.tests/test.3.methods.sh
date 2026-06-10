#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SIM_NAME="${1:-${SIM_NAME:-test.3.methods}}"

N_CHR="${N_CHR:-60}"
CHR_LENGTH="${CHR_LENGTH:-50000000}"
BASE_SEED="${BASE_SEED:-12345}"
N_JOBS_SIM="${N_JOBS_SIM:-10}"

RFMIX_EU_REF="${RFMIX_EU_REF:-50}"
RFMIX_NA_REF="${RFMIX_NA_REF:-50}"
RFMIX_AF_REF="${RFMIX_AF_REF:-50}"
RFMIX_JOBS="${RFMIX_JOBS:-8}"
RFMIX_THREADS="${RFMIX_THREADS:-8}"
RFMIX_GENERATIONS="${RFMIX_GENERATIONS:-16}"

HMMIX_AF_REF="${HMMIX_AF_REF:-250}"
HMMIX_THREADS="${HMMIX_THREADS:-8}"
HMMIX_THRESHOLDS="${HMMIX_THRESHOLDS:-0.5 0.6 0.7 0.8 0.85 0.87 0.88 0.89 0.9 0.91 0.92 0.95 0.99}"
RUN_HMMIX_VITERBI="${RUN_HMMIX_VITERBI:-1}"

SIMPLE_AF_REF="${SIMPLE_AF_REF:-250}"
SIMPLE_ND_REF="${SIMPLE_ND_REF:-3}"
SIMPLE_THREADS="${SIMPLE_THREADS:-16}"

MEXICANS_EU_REF="${MEXICANS_EU_REF:-250}"
MEXICANS_NA_REF="${MEXICANS_NA_REF:-250}"
MEXICANS_AF_REF="${MEXICANS_AF_REF:-250}"
MEXICANS_ND_REF="${MEXICANS_ND_REF:-3}"
MEXICANS_THREADS="${MEXICANS_THREADS:-16}"

YAML_FILE="${YAML_FILE:-mexicans.demography.yml}"
HMMIX_THRESHOLDS_CSV="$(echo "${HMMIX_THRESHOLDS}" | tr ' ' ',')"

step() {
  echo
  echo "[$(date '+%F %T')] $*"
}

echo "Benchmark start"
echo "SIM_NAME=${SIM_NAME}"
echo "N_CHR=${N_CHR}"
echo "CHR_LENGTH=${CHR_LENGTH}"
echo "BASE_SEED=${BASE_SEED}"

step "simulate"
python simulate_mexicans.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --yaml "${YAML_FILE}" \
  --n-chr "${N_CHR}" \
  --chr-length "${CHR_LENGTH}" \
  --base-seed "${BASE_SEED}" \
  --n-jobs "${N_JOBS_SIM}" \
  --clean

step "prepare_rfmix"
python prepare_rfmix.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-eu-ref "${RFMIX_EU_REF}" \
  --n-na-ref "${RFMIX_NA_REF}" \
  --n-af-ref "${RFMIX_AF_REF}" \
  --threads "${RFMIX_THREADS}" \
  --force

step "run_rfmix"
python run_rfmix.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-eu-ref "${RFMIX_EU_REF}" \
  --n-na-ref "${RFMIX_NA_REF}" \
  --n-af-ref "${RFMIX_AF_REF}" \
  --jobs "${RFMIX_JOBS}" \
  --threads "${RFMIX_THREADS}" \
  --generations "${RFMIX_GENERATIONS}" \
  --force

step "prepare_hmmix"
python prepare_hmmix.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-af-ref "${HMMIX_AF_REF}" \
  --threads "${HMMIX_THREADS}" \
  --use-mutrates-file \
  --mutrate-window-size 100000 \
  --force

step "run_hmmix posterior"
python run_hmmix.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-af-ref "${HMMIX_AF_REF}" \
  --thresholds "${HMMIX_THRESHOLDS_CSV}" \
  --force

for thr in ${HMMIX_THRESHOLDS}; do
  step "combine + evaluate rfmix_hmmix posterior threshold=${thr}"

  python combine_predictions.py rfmix_hmmix \
    --sim-name "${SIM_NAME}" \
    --base-dir . \
    --rfmix-eu-ref "${RFMIX_EU_REF}" \
    --rfmix-na-ref "${RFMIX_NA_REF}" \
    --rfmix-af-ref "${RFMIX_AF_REF}" \
    --hmmix-af-ref "${HMMIX_AF_REF}" \
    --hmmix-threshold "${thr}" \
    --force

  python evaluate_methods.py rfmix_hmmix \
    --sim-name "${SIM_NAME}" \
    --base-dir . \
    --rfmix-eu-ref "${RFMIX_EU_REF}" \
    --rfmix-na-ref "${RFMIX_NA_REF}" \
    --rfmix-af-ref "${RFMIX_AF_REF}" \
    --hmmix-af-ref "${HMMIX_AF_REF}" \
    --hmmix-threshold "${thr}"
done

if [[ "${RUN_HMMIX_VITERBI}" == "1" ]]; then
  step "run_hmmix viterbi"

  python run_hmmix.py \
    --sim-name "${SIM_NAME}" \
    --base-dir . \
    --n-af-ref "${HMMIX_AF_REF}" \
    --viterbi

  step "combine + evaluate rfmix_hmmix viterbi"

  python combine_predictions.py rfmix_hmmix \
    --sim-name "${SIM_NAME}" \
    --base-dir . \
    --rfmix-eu-ref "${RFMIX_EU_REF}" \
    --rfmix-na-ref "${RFMIX_NA_REF}" \
    --rfmix-af-ref "${RFMIX_AF_REF}" \
    --hmmix-af-ref "${HMMIX_AF_REF}" \
    --viterbi \
    --force

  python evaluate_methods.py rfmix_hmmix \
    --sim-name "${SIM_NAME}" \
    --base-dir . \
    --rfmix-eu-ref "${RFMIX_EU_REF}" \
    --rfmix-na-ref "${RFMIX_NA_REF}" \
    --rfmix-af-ref "${RFMIX_AF_REF}" \
    --hmmix-af-ref "${HMMIX_AF_REF}" \
    --viterbi
fi

step "prepare_daiseg_simple"
python prepare_daiseg_simple.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-af-ref "${SIMPLE_AF_REF}" \
  --n-nd-ref "${SIMPLE_ND_REF}" \
  --threads "${SIMPLE_THREADS}" \
  --force

step "run_daiseg_simple"
python run_daiseg_simple.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-af-ref "${SIMPLE_AF_REF}" \
  --n-nd-ref "${SIMPLE_ND_REF}" \
  --threads "${SIMPLE_THREADS}" \
  --force

step "combine + evaluate rfmix_daiseg_simple"
python combine_predictions.py rfmix_daiseg_simple \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --rfmix-eu-ref "${RFMIX_EU_REF}" \
  --rfmix-na-ref "${RFMIX_NA_REF}" \
  --rfmix-af-ref "${RFMIX_AF_REF}" \
  --simple-af-ref "${SIMPLE_AF_REF}" \
  --simple-nd-ref "${SIMPLE_ND_REF}" \
  --force

python evaluate_methods.py rfmix_daiseg_simple \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --rfmix-eu-ref "${RFMIX_EU_REF}" \
  --rfmix-na-ref "${RFMIX_NA_REF}" \
  --rfmix-af-ref "${RFMIX_AF_REF}" \
  --simple-af-ref "${SIMPLE_AF_REF}" \
  --simple-nd-ref "${SIMPLE_ND_REF}"

step "prepare_daiseg_mexicans"
python prepare_daiseg_mexicans.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-eu-ref "${MEXICANS_EU_REF}" \
  --n-na-ref "${MEXICANS_NA_REF}" \
  --n-af-ref "${MEXICANS_AF_REF}" \
  --n-nd-ref "${MEXICANS_ND_REF}" \
  --threads "${MEXICANS_THREADS}" \
  --force

step "run_daiseg_mexicans"
python run_daiseg_mexicans.py \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --n-eu-ref "${MEXICANS_EU_REF}" \
  --n-na-ref "${MEXICANS_NA_REF}" \
  --n-af-ref "${MEXICANS_AF_REF}" \
  --n-nd-ref "${MEXICANS_ND_REF}" \
  --threads "${MEXICANS_THREADS}" \
  --force

step "evaluate daiseg_mexicans"
python evaluate_methods.py daiseg_mexicans \
  --sim-name "${SIM_NAME}" \
  --base-dir . \
  --mexicans-eu-ref "${MEXICANS_EU_REF}" \
  --mexicans-na-ref "${MEXICANS_NA_REF}" \
  --mexicans-af-ref "${MEXICANS_AF_REF}" \
  --mexicans-nd-ref "${MEXICANS_ND_REF}"

step "final summaries"

python - <<PY
import json
from pathlib import Path

sim = "${SIM_NAME}"
base = Path(sim) / "metrics"

files = []
files.append(("DAIseg.mexicans", base / "daiseg_mexicans" / "summary.ref.eu${MEXICANS_EU_REF}.na${MEXICANS_NA_REF}.af${MEXICANS_AF_REF}.nd${MEXICANS_ND_REF}.json"))
files.append(("RFMix + DAIseg.simple", base / "rfmix_daiseg_simple" / "summary.rfmix.eu${RFMIX_EU_REF}.na${RFMIX_NA_REF}.af${RFMIX_AF_REF}__simple.af${SIMPLE_AF_REF}.nd${SIMPLE_ND_REF}.json"))

for thr in "${HMMIX_THRESHOLDS}".split():
    thr_tag = f"{float(thr):.2f}".replace(".", "_")
    files.append((f"RFMix + HMMix posterior thr={thr}", base / "rfmix_hmmix" / f"summary.rfmix.eu${RFMIX_EU_REF}.na${RFMIX_NA_REF}.af${RFMIX_AF_REF}__hmmix.af${HMMIX_AF_REF}.thr{thr_tag}.json"))

if "${RUN_HMMIX_VITERBI}" == "1":
    files.append(("RFMix + HMMix viterbi", base / "rfmix_hmmix" / "summary.rfmix.eu${RFMIX_EU_REF}.na${RFMIX_NA_REF}.af${RFMIX_AF_REF}__hmmix.af${HMMIX_AF_REF}.viterbi.json"))

for name, path in files:
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        print(f"{name}: accuracy={d['accuracy']:.6f}")
    else:
        print(f"{name}: missing summary -> {path}")
PY
