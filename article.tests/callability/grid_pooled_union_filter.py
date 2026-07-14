#!/usr/bin/env python3


import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STATE_ORDER = ("EU", "ND_EU", "NA", "ND_NA", "AF")
WINDOW_SIZE = 1000


def open_maybe_gzip(path, mode="rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def normalize_chrom(chrom):
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom


def load_chrom_lengths(path):
    out = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            chrom = normalize_chrom(parts[0])
            if chrom.lower() in {"chrom", "#chrom"}:
                continue
            out[chrom] = int(parts[1])
    return out


def modern_bed_path(modern_dir, chrom):
    p = Path(modern_dir) / f"chr{chrom}.renamed.bed"
    if not p.exists():
        raise FileNotFoundError(f"Modern BED not found: {p}")
    return p


def nd_bed_path(nd_dir, chrom):
    gz = Path(nd_dir) / f"chr{chrom}_mask.bed.gz"
    plain = Path(nd_dir) / f"chr{chrom}_mask.bed"
    if gz.exists():
        return gz
    if plain.exists():
        return plain
    raise FileNotFoundError(f"Neanderthal BED not found for chr{chrom} in {nd_dir}")


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def clip_intervals(intervals, chrom_len):
    out = []
    for s, e in intervals:
        s = max(0, s)
        e = min(chrom_len, e)
        if s < e:
            out.append((s, e))
    return merge_intervals(out)


def intersect_two(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def union_lists(lists):
    all_intervals = []
    for x in lists:
        all_intervals.extend(x)
    return merge_intervals(all_intervals)


def subtract_intervals(base, sub):
    out = []
    j = 0
    for bs, be in base:
        cur = bs
        while j < len(sub) and sub[j][1] <= bs:
            j += 1
        k = j
        while k < len(sub) and sub[k][0] < be:
            ss, se = sub[k]
            if ss > cur:
                out.append((cur, min(ss, be)))
            cur = max(cur, se)
            if cur >= be:
                break
            k += 1
        if cur < be:
            out.append((cur, be))
    return out


def load_standard_bed_intervals(path, chrom, chrom_len):
    target = normalize_chrom(chrom)
    out = []
    with open_maybe_gzip(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            c = normalize_chrom(parts[0])
            if c != target:
                continue
            s = int(parts[1])
            e = int(parts[2])
            if e > s:
                out.append((s, e))
    return clip_intervals(merge_intervals(out), chrom_len)


def load_gap_intervals_ucsc_like(path, chrom, chrom_len):
    target = normalize_chrom(chrom)
    out = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            c = normalize_chrom(parts[1])
            if c != target:
                continue
            s = int(parts[2])
            e = int(parts[3])
            if e > s:
                out.append((s, e))
    return clip_intervals(merge_intervals(out), chrom_len)


def intervals_to_window_fractions(intervals, chrom_len, window_len=WINDOW_SIZE):
    nwin = int(math.ceil(chrom_len / window_len))
    arr = np.zeros(nwin, dtype=np.float64)
    for s, e in intervals:
        w0 = s // window_len
        w1 = (e - 1) // window_len
        for w in range(w0, w1 + 1):
            ws = w * window_len
            we = min((w + 1) * window_len, chrom_len)
            ov = max(0, min(e, we) - max(s, ws))
            if ov > 0:
                arr[w] += ov
    for w in range(nwin):
        ws = w * window_len
        we = min((w + 1) * window_len, chrom_len)
        denom = max(1, we - ws)
        arr[w] /= denom
    return arr


def intervals_from_window_vector(lengths, window_len=WINDOW_SIZE, min_fraction=0.5):
    intervals = []
    for i, x in enumerate(lengths):
        if x > min_fraction:
            s = i * window_len
            e = (i + 1) * window_len
            intervals.append((s, e))
    return merge_intervals(intervals)


def calculate_masked_metrics_by_intersections(gt_full_df, pred_full_df, eval_map, state_order=STATE_ORDER):
    state_map = {s: i for i, s in enumerate(state_order)}
    n_states = len(state_order)

    def _cleanup_state_col(df, col="State"):
        df = df.copy()
        return df[df[col].isin(state_map.keys())]

    def _prep_gt(df):
        df = _cleanup_state_col(df, "State").copy()
        df["CHROM"] = df["CHR"].astype(str)
        df["Sample"] = df["Sample"].astype(str)
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
        df["state_id"] = df["State"].map(state_map).astype(int)
        return df[["Sample", "CHROM", "Start", "End", "state_id"]]

    def _prep_pred(df):
        df = _cleanup_state_col(df, "State").copy()
        df["CHROM"] = df["CHROM"].astype(str)
        df["Sample"] = df["Sample"].astype(str)
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int) + 1
        df["state_id"] = df["State"].map(state_map).astype(int)
        return df[["Sample", "CHROM", "Start", "End", "state_id"]]

    def _df_to_intervals(df, sample, chrom):
        sub = df[(df["Sample"] == sample) & (df["CHROM"] == chrom)].sort_values("Start", kind="mergesort")
        if sub.empty:
            return []
        return list(zip(sub["Start"].to_numpy(), sub["End"].to_numpy(), sub["state_id"].to_numpy()))

    def _clip_state_intervals_to_mask(intervals, mask_intervals):
        out = []
        i = j = 0
        while i < len(intervals) and j < len(mask_intervals):
            s1, e1, st = intervals[i]
            s2, e2 = mask_intervals[j]
            left = max(s1, s2)
            right = min(e1, e2)
            if left < right:
                out.append((left, right, st))
            if e1 <= e2:
                i += 1
            else:
                j += 1
        return out

    def _confmat_sweepline(gt_int, pr_int):
        conf = np.zeros((n_states, n_states), dtype=np.int64)
        i = j = 0
        while i < len(gt_int) and j < len(pr_int):
            gs, ge, gk = gt_int[i]
            ps, pe, pk = pr_int[j]
            left = max(gs, ps)
            right = min(ge, pe)
            if left < right:
                conf[gk, pk] += (right - left)
            if ge <= pe:
                i += 1
            else:
                j += 1
        return conf

    def _report_from_conf(conf):
        rep = {"state_order": list(state_order)}
        total = conf.sum()
        rep["total_bp_scored"] = int(total)
        rep["accuracy"] = float(conf.trace() / total) if total > 0 else float("nan")
        for k, name in enumerate(state_order):
            tp = conf[k, k]
            fp = conf[:, k].sum() - tp
            fn = conf[k, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec) > 0) else float("nan")
            rep[name] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support_bp": int(conf[k, :].sum()),
            }
        return rep

    gt = _prep_gt(gt_full_df)
    pr = _prep_pred(pred_full_df)

    samples = sorted(set(gt["Sample"]).intersection(set(pr["Sample"])))
    chroms = sorted(set(gt["CHROM"]).intersection(set(pr["CHROM"])))
    conf_total = np.zeros((n_states, n_states), dtype=np.int64)

    for sample in samples:
        for chrom in chroms:
            if chrom not in eval_map:
                continue
            mask_intervals = eval_map[chrom]
            if not mask_intervals:
                continue
            gt_int = _df_to_intervals(gt, sample, chrom)
            pr_int = _df_to_intervals(pr, sample, chrom)
            if not gt_int or not pr_int:
                continue
            gt_clip = _clip_state_intervals_to_mask(gt_int, mask_intervals)
            pr_clip = _clip_state_intervals_to_mask(pr_int, mask_intervals)
            if gt_clip and pr_clip:
                conf_total += _confmat_sweepline(gt_clip, pr_clip)

    return conf_total, _report_from_conf(conf_total)


def per_chrom_window_metrics(chrom, chrom_len, modern_dir, gaps_file, nd_dirs, window_len=WINDOW_SIZE):
    modern_callable = load_standard_bed_intervals(modern_bed_path(modern_dir, chrom), chrom, chrom_len)
    gaps = load_gap_intervals_ucsc_like(gaps_file, chrom, chrom_len)
    modern_eval = subtract_intervals(modern_callable, gaps)

    nd_lists = []
    for nd_dir in nd_dirs:
        nd_lists.append(load_standard_bed_intervals(nd_bed_path(nd_dir, chrom), chrom, chrom_len))
    nd_union = union_lists(nd_lists)
    archaic_eval = intersect_two(modern_eval, nd_union)

    l_mod = intervals_to_window_fractions(modern_eval, chrom_len, window_len)
    l_arch = intervals_to_window_fractions(archaic_eval, chrom_len, window_len)

    # union_eff relative to strict/modern evaluable part
    union_eff = np.zeros_like(l_mod)
    mask = l_mod > 0
    union_eff[mask] = l_arch[mask] / l_mod[mask]
    union_eff[~mask] = 0.0
    return l_mod, union_eff


def tag(x):
    return str(x).replace(".", "p")

def write_filter_grid_latex(summary_df, out_path):
    df = summary_df.sort_values(
        ["lmod_threshold", "union_threshold"]
    ).copy()

    baseline = df.loc[
        np.isclose(df["lmod_threshold"], 0.5)
        & np.isclose(df["union_threshold"], 0.0),
        "total_bp_scored",
    ].iloc[0]

    df["retained"] = df["total_bp_scored"] / baseline

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\caption{",
        r"Post-inference filtering grid for the "
        r"\textit{modern-plus-Neanderthal} experiment.",
        r'The ``Retained'' column is normalized to the least stringent '
        r"filter combination, \(\rho^{\ast}_{\mathrm{mod}}=0.5\) and "
        r"\(U^{\ast}_{\mathrm{ND}}=0.0\), which is set to 1.0.",
        r"Here \(U_{\mathrm{ND},t}="
        r"L_{\mathrm{arch},t}/L_{\mathrm{mod},t}\) denotes the fraction "
        r"of the modern-callable portion of a window that is also "
        r"covered by the pooled Neanderthal mask.",
        r"}",
        r"\label{tab:callability_filter_grid}",
        r"\begin{tabular}{ccccccc}",
        r"\hline",
        r"\(\rho^*_{\mathrm{mod}}\) & \(U^*_{\mathrm{ND}}\) & Retained &",
        r"\multicolumn{2}{c}{\texttt{ND\_EU}} &",
        r"\multicolumn{2}{c}{\texttt{ND\_NA}} \\",
        r" & & & Prec. & Rec. & Prec. & Rec. \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        lines.append(
            f'{row["lmod_threshold"]:.1f} & '
            f'{row["union_threshold"]:.1f} & '
            f'{row["retained"]:.3f} & '
            f'{row["ND_EU_precision"]:.3f} & '
            f'{row["ND_EU_recall"]:.3f} & '
            f'{row["ND_NA_precision"]:.3f} & '
            f'{row["ND_NA_recall"]:.3f} \\\\'
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom_lengths", required=True)
    ap.add_argument("--modern_dir", required=True)
    ap.add_argument("--gaps_file", required=True)
    ap.add_argument("--vindija_dir", required=True)
    ap.add_argument("--altai_dir", required=True)
    ap.add_argument("--chagyr_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_seed", type=int, default=1234567)
    ap.add_argument("--chroms", nargs="*", default=[str(i) for i in range(1, 23)])
    ap.add_argument("--lmod_thresholds", nargs="+", type=float, default=[0.5, 0.8])
    ap.add_argument("--union_thresholds", nargs="+", type=float, default=[0.0, 0.5, 0.7, 0.8])
    ap.add_argument("--window_len", type=int, default=WINDOW_SIZE)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    eval_dir = out_dir / "pooled_union_filter_grid"
    eval_dir.mkdir(parents=True, exist_ok=True)

    pred_file = out_dir / "predictions.pooled_union.tsv"
    if not pred_file.exists():
        raise SystemExit(f"Missing file: {pred_file}")

    chrom_lengths = load_chrom_lengths(args.chrom_lengths)

    chrom_info = []
    for i, chrom in enumerate(args.chroms):
        chrom = normalize_chrom(chrom)
        if chrom not in chrom_lengths:
            continue
        chrom_info.append({
            "chrom": chrom,
            "seed": args.base_seed + i,
            "chrom_len": chrom_lengths[chrom],
        })
    if not chrom_info:
        raise SystemExit("No valid chromosomes selected.")

    gt_all = []
    for item in chrom_info:
        gt_file = out_dir / f"ground_truth_5state_chr{item['chrom']}_seed_{item['seed']}.tsv"
        if gt_file.exists():
            gt_all.append(pd.read_csv(gt_file, sep="\t", keep_default_na=False))
    if not gt_all:
        raise SystemExit("No ground truth files found.")
    gt_full_df = pd.concat(gt_all, ignore_index=True)
    pred_df = pd.read_csv(pred_file, sep="\t", keep_default_na=False)

    nd_dirs = [args.vindija_dir, args.altai_dir, args.chagyr_dir]

    metrics_by_seed = {}
    for item in chrom_info:
        l_mod, union_eff = per_chrom_window_metrics(
            item["chrom"], item["chrom_len"], args.modern_dir, args.gaps_file, nd_dirs, args.window_len
        )
        metrics_by_seed[str(item["seed"])] = {
            "l_mod": l_mod,
            "union_eff": union_eff,
            "chrom_len": item["chrom_len"],
        }

    summary_rows = []

    for t_mod in args.lmod_thresholds:
        for t_union in args.union_thresholds:
            eval_map = {}
            callable_rows = []

            for item in chrom_info:
                seed = str(item["seed"])
                chrom_len = item["chrom_len"]
                l_mod = metrics_by_seed[seed]["l_mod"]
                union_eff = metrics_by_seed[seed]["union_eff"]

                keep = (l_mod > t_mod) & (union_eff >= t_union)

                intervals = []
                for i, ok in enumerate(keep):
                    if ok:
                        s = i * args.window_len
                        e = min((i + 1) * args.window_len, chrom_len)
                        intervals.append((s, e))
                intervals = merge_intervals(intervals)
                eval_map[seed] = intervals

                callable_rows.append({
                    "chrom": item["chrom"],
                    "seed": item["seed"],
                    "lmod_threshold": t_mod,
                    "union_threshold": t_union,
                    "n_windows_kept": int(keep.sum()),
                    "bp_kept": int(sum(e - s for s, e in intervals)),
                    "fraction_kept": float(sum(e - s for s, e in intervals) / chrom_len) if chrom_len > 0 else np.nan,
                })

            conf, rep = calculate_masked_metrics_by_intersections(gt_full_df, pred_df, eval_map)

            with open(eval_dir / f"class_report.lmod_{tag(t_mod)}.union_{tag(t_union)}.json", "w") as f:
                json.dump(rep, f, indent=2)
            np.savetxt(eval_dir / f"confusion.lmod_{tag(t_mod)}.union_{tag(t_union)}.txt", conf, fmt="%d")
            pd.DataFrame(callable_rows).to_csv(
                eval_dir / f"callable_space.lmod_{tag(t_mod)}.union_{tag(t_union)}.tsv",
                sep="\t",
                index=False
            )

            summary_rows.append({
                "lmod_threshold": t_mod,
                "union_threshold": t_union,
                "accuracy": rep.get("accuracy", np.nan),
                "total_bp_scored": rep.get("total_bp_scored", np.nan),

                "ND_EU_precision": rep.get("ND_EU", {}).get("precision", np.nan),
                "ND_EU_recall": rep.get("ND_EU", {}).get("recall", np.nan),
                "ND_EU_f1": rep.get("ND_EU", {}).get("f1", np.nan),

                "ND_NA_precision": rep.get("ND_NA", {}).get("precision", np.nan),
                "ND_NA_recall": rep.get("ND_NA", {}).get("recall", np.nan),
                "ND_NA_f1": rep.get("ND_NA", {}).get("f1", np.nan),
            })

            print(
                f"[OK] L_mod>{t_mod:.2f}, union>={t_union:.2f} | "
                f"ND_EU R={summary_rows[-1]['ND_EU_recall']:.3f}, "
                f"ND_NA R={summary_rows[-1]['ND_NA_recall']:.3f}"
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["lmod_threshold", "union_threshold"])
    summary_path = eval_dir / "pooled_union_filter_grid_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    table_path = eval_dir / "pooled_union_filter_grid_table.tex"
    write_filter_grid_latex(summary_df, table_path)
    print(f"[DONE] Saved LaTeX table: {table_path}")

    print(f"[DONE] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
