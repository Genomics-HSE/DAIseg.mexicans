#!/usr/bin/env python3
"""

Use neutral simulations to choose empirical p-value thresholds for the real
MXL-vs-IBS scan.

The script reads neutral_effects_by_window.tsv produced by the neutral
truth-level simulation pipeline and recomputes the same directional binomial
p-values used in the real scan:

    excess:    P[X >= k_MX]
    depletion: P[X <= k_MX]

where

    X ~ Binomial(N_MX_EUbg, p_IBS)

Then it writes empirical lower-tail quantiles of these p-values. These
quantiles can be used as simulation-calibrated p-value cutoffs.

Default calibration:
    excess:    use windows with p_IBS > 0
    depletion: use windows with p_IBS > 0

This avoids non-informative/degenerate cases where p_IBS = 0.


"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


PVALUE_FLOOR = np.nextafter(0, 1)


def find_file(sim_dir, filename):
    sim_dir = Path(sim_dir)
    candidates = [
        sim_dir / filename,
        sim_dir / "validation_batch_run" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(sim_dir.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} under {sim_dir}")


def binom_tail_pvalues(k, n, p0):
    k = np.asarray(k, dtype=int)
    n = np.asarray(n, dtype=int)
    p0 = np.asarray(p0, dtype=float)

    p_excess = np.full(len(k), np.nan, dtype=float)
    p_depletion = np.full(len(k), np.nan, dtype=float)

    ok = (n > 0) & np.isfinite(p0) & (p0 >= 0) & (p0 <= 1)
    for i in np.where(ok)[0]:
        ni = int(n[i])
        ki = int(k[i])
        pi = float(p0[i])

        # P[X >= k] = survival(k-1)
        if ki <= 0:
            p_excess[i] = 1.0
        elif pi == 0.0:
            p_excess[i] = 0.0
        elif pi == 1.0:
            p_excess[i] = 1.0
        else:
            p_excess[i] = float(binom.sf(ki - 1, ni, pi))

        # P[X <= k] = cdf(k)
        if ki >= ni:
            p_depletion[i] = 1.0
        elif pi == 0.0:
            p_depletion[i] = 1.0
        elif pi == 1.0:
            p_depletion[i] = 0.0
        else:
            p_depletion[i] = float(binom.cdf(ki, ni, pi))

    return p_excess, p_depletion


def load_neutral_counts(sim_dir, window_size, min_eu_bg_count):
    path = find_file(sim_dir, "neutral_effects_by_window.tsv")
    print(f"[INFO] Reading {path}", flush=True)

    df = pd.read_csv(path, sep="\t")

    required = [
        "EU_background_bp_MX",
        "ND_EU_MX",
        "p_IBS",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    n_mx = np.rint(df["EU_background_bp_MX"].to_numpy(dtype=float) / window_size).astype(int)
    k_mx = np.rint(df["ND_EU_MX"].to_numpy(dtype=float) / window_size).astype(int)
    p_ibs = df["p_IBS"].to_numpy(dtype=float)

    k_mx = np.minimum(np.maximum(k_mx, 0), n_mx)

    out = pd.DataFrame({
        "CHR": df["CHR"] if "CHR" in df.columns else np.nan,
        "Window": df["Window"] if "Window" in df.columns else np.arange(len(df)),
        "N_MX_EUbg": n_mx,
        "K_MX_ND_EU": k_mx,
        "p_IBS": p_ibs,
    })

    out = out[(out["N_MX_EUbg"] >= min_eu_bg_count) & np.isfinite(out["p_IBS"])].copy()
    return out


def compute_neutral_pvalues(neutral_df):
    p_excess, p_depletion = binom_tail_pvalues(
        neutral_df["K_MX_ND_EU"].to_numpy(),
        neutral_df["N_MX_EUbg"].to_numpy(),
        neutral_df["p_IBS"].to_numpy(),
    )
    out = neutral_df.copy()
    out["p_excess"] = p_excess
    out["p_depletion"] = p_depletion
    out["neglog10_p_excess"] = -np.log10(np.maximum(p_excess, PVALUE_FLOOR))
    out["neglog10_p_depletion"] = -np.log10(np.maximum(p_depletion, PVALUE_FLOOR))
    return out


def threshold_table(neutral_pvals, quantiles, out_prefix,
                    excess_require_pibs_positive=True,
                    depletion_require_pibs_positive=True):
    rows = []

    sets = []
    sets.append(("all_windows", neutral_pvals.copy()))
    sets.append(("pIBS_positive", neutral_pvals[neutral_pvals["p_IBS"] > 0].copy()))

    for set_name, sub in sets:
        for direction, pcol in [
            ("excess", "p_excess"),
            ("depletion", "p_depletion"),
        ]:
            # Recommended sets
            if direction == "excess" and excess_require_pibs_positive and set_name != "pIBS_positive":
                recommended = False
            elif direction == "depletion" and depletion_require_pibs_positive and set_name != "pIBS_positive":
                recommended = False
            elif direction == "excess" and (not excess_require_pibs_positive) and set_name != "all_windows":
                recommended = False
            elif direction == "depletion" and (not depletion_require_pibs_positive) and set_name != "all_windows":
                recommended = False
            else:
                recommended = True

            p = sub[pcol].to_numpy(dtype=float)
            p = p[np.isfinite(p)]
            p = p[(p >= 0) & (p <= 1)]

            if len(p) == 0:
                continue

            for q in quantiles:
                thr = float(np.quantile(p, q))
                rows.append({
                    "calibration_set": set_name,
                    "direction": direction,
                    "quantile": q,
                    "pvalue_threshold": thr,
                    "neglog10_threshold": float(-np.log10(max(thr, PVALUE_FLOOR))),
                    "n_windows": len(p),
                    "recommended_for_direction": recommended,
                    "rule": f"p_{direction} < {thr:.6g}",
                })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{out_prefix}.neutral_pvalue_quantile_thresholds.tsv", sep="\t", index=False)

    # Selected thresholds
    selected_q = [q for q in [0.001, 0.0001] if q in quantiles]
    selected = tbl[
        (tbl["recommended_for_direction"]) &
        (tbl["quantile"].isin(selected_q))
    ].copy()

    strength_map = {
        0.001: "suggestive",
        0.0001: "strong",
    }
    selected["strength"] = selected["quantile"].map(strength_map).fillna("custom")
    selected = selected[
        [
            "direction",
            "strength",
            "calibration_set",
            "quantile",
            "pvalue_threshold",
            "neglog10_threshold",
            "n_windows",
            "rule",
        ]
    ]
    selected.to_csv(f"{out_prefix}.selected_neutral_pvalue_thresholds.tsv", sep="\t", index=False)

    return tbl, selected


def apply_to_real(real_path, selected, out_prefix, min_eu_bg_count):
    print(f"[INFO] Applying thresholds to real file: {real_path}", flush=True)
    real = pd.read_csv(real_path, sep="\t")

    required = ["p_excess", "p_depletion"]
    missing = [c for c in required if c not in real.columns]
    if missing:
        raise ValueError(f"Missing required columns in real file: {missing}")

    if "final_callable_mask" in real.columns:
        tested = real["final_callable_mask"].astype(bool).copy()
    else:
        tested = pd.Series(True, index=real.index)

    if "eu_bg_count" in real.columns:
        tested &= real["eu_bg_count"] >= min_eu_bg_count

    pibs_col = None
    for c in ["expected_prop", "p_IBS", "ibs_freq"]:
        if c in real.columns:
            pibs_col = c
            break

    # initialize
    for direction in ["excess", "depletion"]:
        for strength in ["suggestive", "strong"]:
            real[f"neutral_pvalue_{direction}_{strength}"] = False

    for row in selected.itertuples(index=False):
        direction = row.direction
        strength = row.strength
        threshold = row.pvalue_threshold
        pcol = f"p_{direction}"

        mask = tested & real[pcol].notna() & (real[pcol] < threshold)

        # Recommended default: both excess and depletion thresholds are calibrated on pIBS>0.
        if pibs_col is not None:
            mask &= real[pibs_col] > 0

        real[f"neutral_pvalue_{direction}_{strength}"] = mask

    out_file = f"{out_prefix}.real_windows_with_neutral_pvalue_flags.tsv"
    real.to_csv(out_file, sep="\t", index=False)

    summary_rows = []
    for direction in ["excess", "depletion"]:
        for strength in ["suggestive", "strong"]:
            col = f"neutral_pvalue_{direction}_{strength}"
            if col in real.columns:
                summary_rows.append({
                    "direction": direction,
                    "strength": strength,
                    "n_windows": int(real[col].sum()),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{out_prefix}.real_neutral_pvalue_flag_summary.tsv", sep="\t", index=False)

    return out_file, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True,
                    help="Directory containing neutral_effects_by_window.tsv")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--window-size", type=int, default=1000)
    ap.add_argument("--min-eu-bg-count", type=int, default=10)

    ap.add_argument("--quantiles", nargs="+", type=float,
                    default=[0.0001, 0.001, 0.005, 0.01, 0.05],
                    help="Lower-tail p-value quantiles to report.")

    ap.add_argument("--use-all-windows-for-depletion", action="store_true",
                    help="Use all windows rather than pIBS>0 subset as recommended depletion calibration.")

    ap.add_argument("--use-all-windows-for-excess", action="store_true",
                    help="Use all windows rather than pIBS>0 subset as recommended excess calibration. Usually not recommended.")

    ap.add_argument("--write-neutral-pvalues", action="store_true",
                    help="Write per-window neutral p-values. Large file.")

    ap.add_argument("--real-window-stats", default=None,
                    help="Optional real *.window_stats.tsv file to annotate with neutral p-value flags.")

    args = ap.parse_args()

    neutral_counts = load_neutral_counts(
        args.sim_dir,
        window_size=args.window_size,
        min_eu_bg_count=args.min_eu_bg_count,
    )
    neutral_pvals = compute_neutral_pvalues(neutral_counts)

    if args.write_neutral_pvalues:
        neutral_pvals.to_csv(f"{args.out_prefix}.neutral_pvalues_by_window.tsv", sep="\t", index=False)

    tbl, selected = threshold_table(
        neutral_pvals,
        quantiles=args.quantiles,
        out_prefix=args.out_prefix,
        excess_require_pibs_positive=(not args.use_all_windows_for_excess),
        depletion_require_pibs_positive=(not args.use_all_windows_for_depletion),
    )

    print("\n[SELECTED THRESHOLDS]", flush=True)
    print(selected.to_string(index=False), flush=True)

    if args.real_window_stats:
        out_file, summary = apply_to_real(
            args.real_window_stats,
            selected=selected,
            out_prefix=args.out_prefix,
            min_eu_bg_count=args.min_eu_bg_count,
        )
        print("\n[REAL WINDOW FLAGS]", flush=True)
        print(summary.to_string(index=False), flush=True)
        print(f"Annotated real windows: {out_file}", flush=True)

    print("\n[DONE]", flush=True)
    print(f"Threshold table: {args.out_prefix}.neutral_pvalue_quantile_thresholds.tsv")
    print(f"Selected table:  {args.out_prefix}.selected_neutral_pvalue_thresholds.tsv")


if __name__ == "__main__":
    main()
