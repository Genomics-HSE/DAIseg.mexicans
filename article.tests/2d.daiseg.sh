#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SIM_NAME="${1:-${SIM_NAME:-2d.daiseg.seed*}}"
BASE_DIR="${BASE_DIR:-.}"

N_CHR="${N_CHR:-60}"
CHR_LENGTH="${CHR_LENGTH:-50000000}"
BASE_SEED="${BASE_SEED:-123456}"
N_JOBS_SIM="${N_JOBS_SIM:-12}"
YAML_FILE="${YAML_FILE:-mexicans.demography.yml}"

SIM_N_MEXICANS="${SIM_N_MEXICANS:-1}"
SIM_N_EU="${SIM_N_EU:-250}"
SIM_N_NA="${SIM_N_NA:-250}"
SIM_N_AF="${SIM_N_AF:-250}"
SIM_N_ND="${SIM_N_ND:-10}"

MODERN_GRID="${MODERN_GRID:-25 50 100 150 200 250}"
ND_GRID="${ND_GRID:-0 1 3 6 10}"

PREP_THREADS="${PREP_THREADS:-24}"
RUN_THREADS="${RUN_THREADS:-24}"

step() {
  echo
  echo "[$(date '+%F %T')] $*"
}

echo "SIM_NAME         = ${SIM_NAME}"
echo "BASE_DIR         = ${BASE_DIR}"
echo "N_CHR            = ${N_CHR}"
echo "CHR_LENGTH       = ${CHR_LENGTH}"
echo "BASE_SEED        = ${BASE_SEED}"
echo "SIM_N_EU         = ${SIM_N_EU}"
echo "SIM_N_NA         = ${SIM_N_NA}"
echo "SIM_N_AF         = ${SIM_N_AF}"
echo "SIM_N_ND         = ${SIM_N_ND}"
echo "MODERN_GRID      = ${MODERN_GRID}"
echo "ND_GRID          = ${ND_GRID}"
echo "PREP_THREADS     = ${PREP_THREADS}"
echo "RUN_THREADS      = ${RUN_THREADS}"

step "simulate maximal dataset once"
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

for ND_REF in ${ND_GRID}; do
  for MODERN_REF in ${MODERN_GRID}; do

    step "prepare modern_ref=${MODERN_REF} nd_ref=${ND_REF}"
    python prepare_daiseg_mexicans.py \
      --sim-name "${SIM_NAME}" \
      --base-dir "${BASE_DIR}" \
      --n-eu-ref "${MODERN_REF}" \
      --n-na-ref "${MODERN_REF}" \
      --n-af-ref "${MODERN_REF}" \
      --n-nd-ref "${ND_REF}" \
      --threads "${PREP_THREADS}" \
      --force

    step "run modern_ref=${MODERN_REF} nd_ref=${ND_REF}"
    python run_daiseg_mexicans.py \
      --sim-name "${SIM_NAME}" \
      --base-dir "${BASE_DIR}" \
      --n-eu-ref "${MODERN_REF}" \
      --n-na-ref "${MODERN_REF}" \
      --n-af-ref "${MODERN_REF}" \
      --n-nd-ref "${ND_REF}" \
      --threads "${RUN_THREADS}" \
      --force

    step "evaluate modern_ref=${MODERN_REF} nd_ref=${ND_REF}"
    python evaluate_methods.py daiseg_mexicans \
      --sim-name "${SIM_NAME}" \
      --base-dir "${BASE_DIR}" \
      --mexicans-eu-ref "${MODERN_REF}" \
      --mexicans-na-ref "${MODERN_REF}" \
      --mexicans-af-ref "${MODERN_REF}" \
      --mexicans-nd-ref "${ND_REF}"

  done
done

step "collect grid summaries into TSV"

python - <<PY
import json
import re
from pathlib import Path

sim_name = "${SIM_NAME}"
base_dir = Path("${BASE_DIR}")
metrics_dir = base_dir / sim_name / "metrics" / "daiseg_mexicans"
out_path = metrics_dir / "grid_metrics.long.tsv"

pattern = re.compile(r"summary\\.ref\\.eu(\\d+)\\.na(\\d+)\\.af(\\d+)\\.nd(\\d+)\\.json$")

rows = []
for f in sorted(metrics_dir.glob("summary.ref.eu*.na*.af*.nd*.json")):
    m = pattern.search(f.name)
    if not m:
        continue

    eu_ref, na_ref, af_ref, nd_ref = map(int, m.groups())

    with open(f) as fh:
        d = json.load(fh)

    for state, stats in d.get("per_state", {}).items():
        rows.append({
            "summary_file": str(f),
            "modern_ref": eu_ref,
            "eu_ref": eu_ref,
            "na_ref": na_ref,
            "af_ref": af_ref,
            "nd_ref": nd_ref,
            "state": state,
            "precision": stats.get("precision"),
            "recall": stats.get("recall"),
            "f1": stats.get("f1"),
            "support_bp": stats.get("support_bp"),
            "accuracy": d.get("accuracy"),
            "total_bp_scored": d.get("total_bp_scored"),
        })

if not rows:
    raise SystemExit(f"No summary files found in {metrics_dir}")

cols = [
    "summary_file",
    "modern_ref",
    "eu_ref",
    "na_ref",
    "af_ref",
    "nd_ref",
    "state",
    "precision",
    "recall",
    "f1",
    "support_bp",
    "accuracy",
    "total_bp_scored",
]

with open(out_path, "w") as out:
    out.write("\\t".join(cols) + "\\n")
    for r in rows:
        out.write("\\t".join(str(r[c]) for c in cols) + "\\n")

print(f"Saved long-form grid metrics to: {out_path}")
PY

echo "Results: ${BASE_DIR}/${SIM_NAME}"
echo "Long metrics table: ${BASE_DIR}/${SIM_NAME}/metrics/daiseg_mexicans/grid_metrics.long.tsv"
