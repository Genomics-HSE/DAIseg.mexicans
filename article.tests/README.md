
# 2D grid benchmark

This benchmark runs independent whole-genome-scale simulations. Each seed simulates 60 chromosomes of 50 Mb, giving 3 Gb of sequence per replicate.

## Main pipeline

`launch.2d.daiseg.sh` runs multiple seeds. For each seed, it sets `SIM_NAME=2d.daiseg.seedN` and `BASE_SEED`, then launches `2d.daiseg.sh`.

`2d.daiseg.sh` runs the full per-seed pipeline: simulation, DAIseg input preparation, DAIseg inference, and evaluation for each `(modern_ref, nd_ref)` grid point. 
Per-seed metrics are written to:

```text
2d.daiseg.seedN/metrics/daiseg_mexicans/grid_metrics.long.tsv
````

## Grid summary

`collect.2d.runs.py` searches for:

```text
2d.daiseg.seed*/metrics/daiseg_mexicans/grid_metrics.long.tsv
```

It combines available seed results into `all_runs.long.tsv`, averages precision and recall across seeds, and produces:

```text
archaic.tileplot.pdf
modern.tileplot.pdf
```

## Length-bin analysis

`eval_len_bin.py` uses the same `2d.daiseg.seedN` folders, but analyzes one grid point:

```text
ref.eu250.na250.af250.nd3
```

For each seed, it compares:

```text
2d.daiseg.seedN/raw/truth.all.tsv
```

with:

```text
2d.daiseg.seedN/runs/daiseg_mexicans/ref.eu250.na250.af250.nd3/all.inferred.daiseg_mexicans.em.tsv
```

It groups true tracts by length bin and computes 5-state and archaic/non-archaic confusion matrices.

Outputs are written to:

```text
length_bin_analysis.ref250.nd3
```


