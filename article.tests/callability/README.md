# Empirical callability-mask benchmark

This directory contains the empirical callability-mask benchmark used to test
DAIseg under modern-human and Neanderthal genomic masks. The analysis uses
chromosomes 1--22, the 1000 Genomes strict mask, genomic gaps, and the union of
the Vindija, Altai, and Chagyrskaya Neanderthal masks.

The modern-callability fraction in window \(t\) is
\(\rho_{\mathrm{mod},t}=L_{\mathrm{mod},t}/L\). Neanderthal union coverage is
\(U_{\mathrm{ND},t}=L_{\mathrm{arch},t}/L_{\mathrm{mod},t}\).

## Files

- `experiment.with.mask.py` simulates masked data, runs EM/Viterbi inference,
  and writes truth, predictions, and evaluation summaries.
- `grid_pooled_union_filter.py` evaluates the post-inference filtering grid and
  writes its TSV and LaTeX summary tables.
- `stratify_callability.py` produces the precision/recall heatmap.
- `mexicans.demography.yml` defines the demographic model.

## Run

Run all commands from this directory.

### 1. Simulation and masked inference

```bash
PYTHONPATH="$HOME/DAIseg.mexicans" python experiment.with.mask.py \
  --yaml mexicans.demography.yml \
  --chrom_lengths /home/share/human.data/ref.fa/hg19.chr.lengths/hg19.chrom.len \
  --modern_dir /home/share/human.data/1000GP/1000GP.grch37/bed \
  --gaps_file /home/share/human.data/ref.fa/gaps.grch37/gap.renamed.txt \
  --vindija_dir /home/share/human.data/neand/33.19/bed \
  --altai_dir /home/share/human.data/neand/altai/bed \
  --chagyr_dir /home/share/human.data/neand/Chagyrskaya/bed \
  --out_dir masked_matrix_all \
  --base_seed 1234567 \
  --n_threads 8 \
  --resimulate true
```

Use `--resimulate false` to reuse existing tree-sequence and truth files.

### 2. Post-inference filtering grid

```bash
python grid_pooled_union_filter.py \
  --chrom_lengths /home/share/human.data/ref.fa/hg19.chr.lengths/hg19.chrom.len \
  --modern_dir /home/share/human.data/1000GP/1000GP.grch37/bed \
  --gaps_file /home/share/human.data/ref.fa/gaps.grch37/gap.renamed.txt \
  --vindija_dir /home/share/human.data/neand/33.19/bed \
  --altai_dir /home/share/human.data/neand/altai/bed \
  --chagyr_dir /home/share/human.data/neand/Chagyrskaya/bed \
  --out_dir masked_matrix_all \
  --base_seed 1234567
```

### 3. Callability-stratified heatmap

```bash
python stratify_callability.py
```

## Main outputs

```text
masked_matrix_all/summary.tsv
masked_matrix_all/predictions.pooled_union.tsv
masked_matrix_all/pooled_union_filter_grid/pooled_union_filter_grid_summary.tsv
masked_matrix_all/pooled_union_filter_grid/pooled_union_filter_grid_table.tex
masked_matrix_all/archaic_precision_recall_by_callability.pdf
masked_matrix_all/archaic_precision_recall_by_callability.png
```
