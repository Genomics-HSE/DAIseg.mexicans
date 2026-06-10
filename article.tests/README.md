# 2D grid benchmark

## Goals

This benchmark evaluates DAIseg on independent whole-genome-scale simulations. Each seed simulates 60 chromosomes of 50 Mb, giving 3 Gb of sequence per replicate.

The analysis has three parts: grid-level performance across `(modern_ref, nd_ref)`, selected 5-state confusion matrices, and performance by true tract length.

## Scripts

`launch.2d.daiseg.sh` runs multiple simulation seeds by setting `SIM_NAME=2d.daiseg.seedN` and `BASE_SEED`, then launching `2d.daiseg.sh`.

`2d.daiseg.sh` runs the full per-seed pipeline: simulation, DAIseg input preparation, inference, evaluation, and per-seed metric collection.

`collect.2d.runs.py` combines completed per-seed grid metrics and produces the main grid summary plots.

`analysis_utils.py` contains shared loading, formatting, and confusion-matrix functions used by `plot.confusion.py` and `eval_len_bin.py`.

`plot.confusion.py` builds row-normalized 5-state confusion matrices for selected grid points.

`eval_len_bin.py` runs length-stratified analysis for `ref.eu250.na250.af250.nd3` to test whether performance depends on true tract length.

## File dependency tree

```text
launch.2d.daiseg.sh
└── 2d.daiseg.sh
    └── 2d.daiseg.seedN/
        ├── raw/truth.all.tsv
        ├── runs/daiseg_mexicans/ref.eu*.na*.af*.nd*/all.inferred.daiseg_mexicans.em.tsv
        └── metrics/daiseg_mexicans/grid_metrics.long.tsv

collect.2d.runs.py
└── grid_metrics.long.tsv → all_runs.long.tsv → archaic.tileplot.pdf, modern.tileplot.pdf

analysis_utils.py
├── plot.confusion.py → confusion.selected.grid.pdf
└── eval_len_bin.py → length_bin_analysis.ref250.nd3/length_bin_confusion.mean_across_runs.pdf
```

# Three-method comparison benchmark

## Goal

This benchmark compares three approaches on the same whole-genome-scale simulation replicate.

Each run simulates 60 chromosomes of 50 Mb, giving 3 Gb of sequence, and evaluates:

1. `RFMix + HMMix`
2. `RFMix + DAIseg.simple`
3. `DAIseg.mexicans`

## Scripts

`test.3.methods.sh` runs the full comparison pipeline for one simulation replicate.

It first simulates the dataset, then runs RFMix, HMMix, DAIseg.simple, and DAIseg.mexicans. Combined predictions are evaluated with `evaluate_methods.py`. 
The default simulation name is:

`comparison.3.methods`


## File dependency tree

```text
test.3.methods.sh
├── simulate_mexicans.py
├── prepare_rfmix.py
├── run_rfmix.py
├── prepare_hmmix.py
├── run_hmmix.py
├── prepare_daiseg_simple.py
├── run_daiseg_simple.py
├── prepare_daiseg_mexicans.py
├── run_daiseg_mexicans.py
├── combine_predictions.py
└── evaluate_methods.py


outputs:

simulate_mexicans.py → comparison.3.methods/raw/truth.all.tsv
run_rfmix.py → comparison.3.methods/runs/rfmix/
run_hmmix.py → comparison.3.methods/runs/hmmix/
combine_predictions.py rfmix_hmmix → comparison.3.methods/runs/rfmix_hmmix/
combine_predictions.py rfmix_daiseg_simple → comparison.3.methods/runs/rfmix_daiseg_simple/
run_daiseg_mexicans.py → comparison.3.methods/runs/daiseg_mexicans/
evaluate_methods.py → comparison.3.methods/metrics/{rfmix_hmmix,rfmix_daiseg_simple,daiseg_mexicans}/
```
