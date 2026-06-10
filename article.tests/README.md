# 2D grid benchmark

## Goals

This benchmark evaluates DAIseg on independent whole-genome-scale simulations.

Each seed simulates 60 chromosomes of 50 Mb, giving 3 Gb of sequence per replicate.

The analysis has two parts:

1. test how performance changes across the `(modern_ref, nd_ref)` grid;
2. test whether performance depends on the length of the true tract.

## Scripts and file flow

### `launch.2d.daiseg.sh`

Runs multiple simulation seeds.

For each seed, it sets `SIM_NAME=2d.daiseg.seedN` and `BASE_SEED`, then launches:

`launch.2d.daiseg.sh` → `2d.daiseg.sh`

### `2d.daiseg.sh`

Runs the full pipeline for one seed:

`simulate data` → `prepare DAIseg inputs` → `run DAIseg` → `evaluate predictions` → `collect per-seed grid metrics`

Main per-seed output:

`2d.daiseg.sh` → `2d.daiseg.seedN/metrics/daiseg_mexicans/grid_metrics.long.tsv`

### `collect.2d.runs.py`

Combines grid metrics across completed seeds:

`2d.daiseg.seed*/metrics/daiseg_mexicans/grid_metrics.long.tsv` → `all_runs.long.tsv`

Then it averages precision and recall across seeds and writes:

`collect.2d.runs.py` → `archaic.tileplot.pdf`  
`collect.2d.runs.py` → `modern.tileplot.pdf`

### `eval_len_bin.py`

Runs a length-stratified analysis for one grid point:

`ref.eu250.na250.af250.nd3`

For each seed, it compares:

`2d.daiseg.seedN/raw/truth.all.tsv`  
with  
`2d.daiseg.seedN/runs/daiseg_mexicans/ref.eu250.na250.af250.nd3/all.inferred.daiseg_mexicans.em.tsv`

The goal is to evaluate performance by true tract length.

Main output:

`eval_len_bin.py` → `length_bin_analysis.ref250.nd3/length_bin_confusion.mean_across_runs.pdf`

Additional outputs:

`eval_len_bin.py` → `length_bin_analysis.ref250.nd3/length_bin_summary.json`  
`eval_len_bin.py` → `length_bin_analysis.ref250.nd3/mean_confusion_*.txt`

### `plot.confusion.py`

Builds 5-state confusion matrices for selected grid points:

`modern_ref = 25, 100, 250`  
`nd_ref = 0, 1, 3`

For each seed, it compares:

`2d.daiseg.seedN/raw/truth.all.tsv`

with the corresponding DAIseg prediction file:

`2d.daiseg.seedN/runs/daiseg_mexicans/ref.eu{modern_ref}.na{modern_ref}.af{modern_ref}.nd{nd_ref}/all.inferred.daiseg_mexicans.em.tsv`

The script aggregates confusion counts across available seeds and writes:

`plot.confusion.py` → `confusion.selected.grid.pdf`
