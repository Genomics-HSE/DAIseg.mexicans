from pathlib import Path

import numpy as np
import pandas as pd


STATE_ORDER = ["EU", "ND_EU", "NA", "ND_NA", "AF"]


def load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def ensure_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CHR"] = out["CHR"].astype(int)
    out["Sample"] = out["Sample"].astype(str)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    out["State"] = out["State"].astype(str)
    return out


def ensure_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "CHROM" not in out.columns and "CHR" in out.columns:
        out["CHROM"] = out["CHR"]

    out["CHROM"] = out["CHROM"].astype(int)
    out["Sample"] = out["Sample"].astype(str)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    out["State"] = out["State"].astype(str)

    return out


def calculate_confusion_bp(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    state_order=STATE_ORDER,
) -> np.ndarray:
    state_map = {state: i for i, state in enumerate(state_order)}
    n_states = len(state_order)

    def clean(df: pd.DataFrame, chrom_col: str) -> pd.DataFrame:
        out = df[df["State"].isin(state_map)].copy()
        out["Sample"] = out["Sample"].astype(str)
        out["CHROM"] = out[chrom_col].astype(int)
        out["Start"] = out["Start"].astype(int)
        out["End"] = out["End"].astype(int)
        out["state_id"] = out["State"].map(state_map).astype(int)
        return out[["Sample", "CHROM", "Start", "End", "state_id"]]

    truth = clean(truth, "CHR")
    pred = clean(pred, "CHROM")

    def intervals(df: pd.DataFrame, sample: str, chrom: int):
        sub = df[(df["Sample"] == sample) & (df["CHROM"] == chrom)]
        sub = sub.sort_values("Start", kind="mergesort")

        return list(
            zip(
                sub["Start"].to_numpy(),
                sub["End"].to_numpy(),
                sub["state_id"].to_numpy(),
            )
        )

    def overlap_confusion(truth_intervals, pred_intervals):
        conf = np.zeros((n_states, n_states), dtype=np.int64)
        i = j = 0

        while i < len(truth_intervals) and j < len(pred_intervals):
            ts, te, tk = truth_intervals[i]
            ps, pe, pk = pred_intervals[j]

            left = max(ts, ps)
            right = min(te, pe)

            if left < right:
                conf[tk, pk] += right - left

            if te <= pe:
                i += 1
            else:
                j += 1

        return conf

    samples = sorted(set(truth["Sample"]).intersection(pred["Sample"]))
    chroms = sorted(set(truth["CHROM"]).intersection(pred["CHROM"]))

    conf = np.zeros((n_states, n_states), dtype=np.int64)

    for sample in samples:
        for chrom in chroms:
            truth_intervals = intervals(truth, sample, chrom)
            pred_intervals = intervals(pred, sample, chrom)

            if truth_intervals and pred_intervals:
                conf += overlap_confusion(truth_intervals, pred_intervals)

    return conf


def row_normalize(conf: np.ndarray) -> np.ndarray:
    conf = conf.astype(float)
    row_sums = conf.sum(axis=1, keepdims=True)

    return np.divide(conf, row_sums, out=np.zeros_like(conf), where=row_sums != 0)


def collapse_to_binary(conf5: np.ndarray, state_order=STATE_ORDER) -> np.ndarray:
    idx_arch = [state_order.index("ND_EU"), state_order.index("ND_NA")]
    idx_non = [state_order.index("EU"), state_order.index("NA"), state_order.index("AF")]

    tp = conf5[np.ix_(idx_arch, idx_arch)].sum()
    fn = conf5[np.ix_(idx_arch, idx_non)].sum()
    fp = conf5[np.ix_(idx_non, idx_arch)].sum()
    tn = conf5[np.ix_(idx_non, idx_non)].sum()

    return np.array([[tp, fn], [fp, tn]], dtype=np.int64)


def binary_metrics(conf2: np.ndarray) -> dict:
    tp, fn = conf2[0, 0], conf2[0, 1]
    fp, tn = conf2[1, 0], conf2[1, 1]

    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
    }
