# 2D grid

`launch.2d.daiseg.sh` runs multiple independent simulation replicates. For each seed, it sets `SIM_NAME=2d.daiseg.seedN` and `BASE_SEED`, then launches `2d.daiseg.sh`, writing the log to `seedN.log`.

`2d.daiseg.sh` runs the full pipeline for one seed: simulates a 3 Gb dataset, prepares DAIseg inputs for each `(modern_ref, nd_ref)` grid point, runs DAIseg, evaluates predictions, and writes per-seed metrics to:

`2d.daiseg.seedN/metrics/daiseg_mexicans/grid_metrics.long.tsv`

Each `2d.daiseg.seedN` directory corresponds to one independent whole-genome-scale simulation replicate. In the full run, there are 50 replicates, each with 60 chromosomes of 50 Mb.

`collect.2d.runs.py` searches the current directory for completed seed folders matching:

`2d.daiseg.seed*/metrics/daiseg_mexicans/grid_metrics.long.tsv`

It parses these per-seed metric files, adds the seed number, combines them into:

`all_runs.long.tsv`

Then it averages precision and recall across seeds for each `(state, modern_ref, nd_ref)` combination and produces the final figures:

`archaic.tileplot.pdf`  
`modern.tileplot.pdf`

`eval_len_bin.py` uses the same `2d.daiseg.seedN` folders, but performs a separate length-stratified analysis for one grid point:

`ref.eu250.na250.af250.nd3`

For each available seed, it reads the truth file:

`2d.daiseg.seedN/raw/truth.all.tsv`

and the corresponding DAIseg prediction file:

`2d.daiseg.seedN/runs/daiseg_mexicans/ref.eu250.na250.af250.nd3/all.inferred.daiseg_mexicans.em.tsv`

It groups true tracts by length bins, compares true and inferred ancestry states by overlapping base pairs, and computes 5-state and binary archaic/non-archaic confusion matrices.

The results are written to:

`length_bin_analysis.ref250.nd3`

including `length_bin_summary.json`, `length_bin_confusion.mean_across_runs.pdf`, and per-bin confusion matrix text files.
