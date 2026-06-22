# Neutral threshold scripts

## 1. neutral_drift_thresholds_truth_level.py

Simulates neutral genomes or reuses existing truth-level simulations.

Main outputs:
- `validation_batch_run/neutral_effects_by_window.tsv`
- `validation_batch_run/selected_truth_level_delta_thresholds.tsv`

Run:

```bash
python neutral_drift_thresholds_truth_level.py
```

## 2. neutral_pvalue_thresholds.py

Uses `neutral_effects_by_window.tsv` to calibrate directional binomial p-value thresholds.

Run:

```bash
python neutral_pvalue_thresholds.py \
  --sim-dir validation_batch_run \
  --out-prefix neutral_pvalue_thresholds
```

Main output:
- `neutral_pvalue_thresholds.selected_neutral_pvalue_thresholds.tsv`

View thresholds:

```bash
column -t -s $'\t' validation_batch_run/selected_truth_level_delta_thresholds.tsv
column -t -s $'\t' neutral_pvalue_thresholds.selected_neutral_pvalue_thresholds.tsv
```

## 3. Minimum neutral p-values

To write per-window neutral p-values, run:

```bash
python neutral_pvalue_thresholds.py \
  --sim-dir validation_batch_run \
  --out-prefix neutral_pvalue_thresholds.with_windows \
  --write-neutral-pvalues
```

To find minimum p-values among informative windows with `p_IBS > 0`:

```bash
awk -F'\t' '
NR==1 {
  for (i=1;i<=NF;i++) {
    if ($i=="p_excess") pe=i
    if ($i=="p_depletion") pd=i
    if ($i=="p_IBS") pibs=i
  }
  next
}
$pibs+0 > 0 && $pe!="nan" && $pe!="" {
  if (min_e=="" || $pe+0 < min_e) min_e=$pe+0
}
$pibs+0 > 0 && $pd!="nan" && $pd!="" {
  if (min_d=="" || $pd+0 < min_d) min_d=$pd+0
}
END {
  print "min p_excess =", min_e
  print "min p_depletion =", min_d
}' neutral_pvalue_thresholds.with_windows.neutral_pvalues_by_window.tsv
```
