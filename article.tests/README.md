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
    ├── simulate_mexicans.py
    ├── prepare_daiseg_mexicans.py
    ├── run_daiseg_mexicans.py
    ├── evaluate_methods.py
    └── 2d.daiseg.seedN/
        ├── raw/truth.all.tsv
        ├── runs/daiseg_mexicans/ref.eu*.na*.af*.nd*/all.inferred.daiseg_mexicans.em.tsv
        └── metrics/daiseg_mexicans/
            ├── summary.ref.eu*.na*.af*.nd*.json
            └── grid_metrics.long.tsv

collect.2d.runs.py
├── reads  → 2d.daiseg.seed*/metrics/daiseg_mexicans/grid_metrics.long.tsv
└── writes → all_runs.long.tsv
          → archaic.tileplot.pdf
          → modern.tileplot.pdf

analysis_utils.py
├── plot.confusion.py
│   ├── reads  → 2d.daiseg.seedN/raw/truth.all.tsv
│   ├── reads  → 2d.daiseg.seedN/runs/daiseg_mexicans/ref.eu{25,100,250}.na{25,100,250}.af{25,100,250}.nd{0,1,3}/all.inferred.daiseg_mexicans.em.tsv
│   └── writes → confusion.selected.grid.pdf
└── eval_len_bin.py
    ├── reads  → 2d.daiseg.seedN/raw/truth.all.tsv
    ├── reads  → 2d.daiseg.seedN/runs/daiseg_mexicans/ref.eu250.na250.af250.nd3/all.inferred.daiseg_mexicans.em.tsv
    └── writes → length_bin_analysis.ref250.nd3/length_bin_confusion.mean_across_runs.pdf
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


# Minimal DAIseg.mexicans test run

## Goal

This script runs a reduced DAIseg.mexicans test pipeline to check that simulation, inference, and evaluation work end-to-end.

## Script

`simpl.sims.sh` simulates a smaller dataset, prepares DAIseg.mexicans inputs, runs DAIseg, and evaluates the prediction.

By default, it simulates 20 chromosomes of 30 Mb each, giving 600 Mb of sequence.

## File flow

```text
simpl.sims.sh
├── simulate_mexicans.py
├── prepare_daiseg_mexicans.py
├── run_daiseg_mexicans.py
└── evaluate_methods.py

simulate_mexicans.py → test.em/raw/truth.all.tsv
run_daiseg_mexicans.py → test.em/runs/daiseg_mexicans/
evaluate_methods.py → test.em/metrics/daiseg_mexicans/
```


# Empirical callability-mask benchmark

## Goal

This benchmark evaluates DAIseg.mexicans under empirical modern and Neanderthal callability masks. It simulates chromosomes 1--22 using human autosomal lengths, applies 1000 Genomes modern callability masks and Neanderthal masks, runs masked inference, and evaluates archaic-state recovery under different post-inference callability filters.

The analysis has three parts: simulation and masked inference, post-inference filtering for the `pooled_union` regime, and a callability-stratified precision/recall heatmap.

## Scripts

`experiment.with.mask.py` runs the full simulation and inference pipeline. It simulates chromosomes, extracts five-state truth, constructs masked observations, runs EM/Viterbi decoding, and writes truth, prediction, confusion, and summary files.

`grid_pooled_union_filter.py` performs the post-inference filtering grid for the `pooled_union` regime. It reads `predictions.pooled_union.tsv` and the chromosome-level truth files, applies thresholds on modern callability \(L_{\mathrm{mod}}\) and Neanderthal union coverage \(U_{\mathrm{ND}}=L_{\mathrm{arch}}/L_{\mathrm{mod}}\), and writes the filtering-grid summary used for the main callability-filter table.

`stratify_callability.py` is a post-processing script for the heatmap. It reads the generated truth and `predictions.pooled_union.tsv`, recomputes \(L_{\mathrm{mod}}\) and \(U_{\mathrm{ND}}\), and plots archaic precision and recall across callability bins.

## File dependency tree

```text
experiment.with.mask.py
├── reads  → demographic model YAML
├── reads  → chromosome lengths
├── reads  → 1000 Genomes modern callability masks
├── reads  → genomic gap mask
├── reads  → Vindija / Altai / Chagyrskaya Neanderthal masks
└── writes → masked_matrix_all/
            ├── sim_chr*_seed_*.trees
            ├── ground_truth_5state_chr*_seed_*.tsv
            ├── ground_truth_5state.all.tsv
            ├── predictions.<regime>.tsv
            ├── predictions.pooled_union.tsv
            ├── confusion.<regime>.txt
            ├── class_report.<regime>.json
            ├── binary_arch_report.<regime>.json
            └── summary.tsv

grid_pooled_union_filter.py
├── reads  → masked_matrix_all/ground_truth_5state_chr*_seed_*.tsv
├── reads  → masked_matrix_all/predictions.pooled_union.tsv
├── reads  → 1000 Genomes modern callability masks
├── reads  → genomic gap mask
├── reads  → Vindija / Altai / Chagyrskaya Neanderthal masks
└── writes → masked_matrix_all/pooled_union_filter_grid/
            ├── pooled_union_filter_grid_summary.tsv
            ├── callable_space.lmod_*.union_*.tsv
            ├── confusion.lmod_*.union_*.txt
            └── class_report.lmod_*.union_*.json

stratify_callability.py
├── reads  → masked_matrix_all/ground_truth_5state_chr*_seed_*.tsv
├── reads  → masked_matrix_all/predictions.pooled_union.tsv
├── reads  → 1000 Genomes modern callability masks
├── reads  → genomic gap mask
├── reads  → Vindija / Altai / Chagyrskaya Neanderthal masks
└── writes → masked_matrix_all/archaic_precision_recall_by_callability.png
          → masked_matrix_all/archaic_precision_recall_by_callability.pdf
````







