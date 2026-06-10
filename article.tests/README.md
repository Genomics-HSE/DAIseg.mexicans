# 2D grid benchmark

## Goals

This benchmark evaluates DAIseg on independent whole-genome-scale simulations.

Each seed simulates 60 chromosomes of 50 Mb, giving 3 Gb of sequence per replicate.

The analysis has two parts:

1. test how performance changes across the `(modern_ref, nd_ref)` grid;
2. test whether performance depends on the length of the true tract.

## Scripts

### `launch.2d.daiseg.sh`

Runs multiple simulation seeds.

For each seed, it sets:

```text
SIM_NAME=2d.daiseg.seedN
BASE_SEED

and launches:

2d.daiseg.sh
2d.daiseg.sh

Runs the full pipeline for one seed:

simulate data
prepare DAIseg inputs
run DAIseg
evaluate predictions
collect per-seed grid metrics

Main output:

2d.daiseg.seedN/metrics/daiseg_mexicans/grid_metrics.long.tsv
collect.2d.runs.py

Parses completed seed folders using:

2d.daiseg.seed*/metrics/daiseg_mexicans/grid_metrics.long.tsv

Combines per-seed results into:

all_runs.long.tsv

Final outputs:

archaic.tileplot.pdf
modern.tileplot.pdf
eval_len_bin.py

Uses the same 2d.daiseg.seedN folders, but analyzes only one grid point:

ref.eu250.na250.af250.nd3

For each seed, it compares:

2d.daiseg.seedN/raw/truth.all.tsv

with:

2d.daiseg.seedN/runs/daiseg_mexicans/ref.eu250.na250.af250.nd3/all.inferred.daiseg_mexicans.em.tsv

The goal is to evaluate performance by true tract length.

Output directory:

length_bin_analysis.ref250.nd3
