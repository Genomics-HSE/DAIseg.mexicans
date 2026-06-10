#!/usr/bin/env python3
"""
Evaluate combined mex_compare predictions against 5-state truth.

Supported modes:
  - daiseg_mexicans
  - rfmix_hmmix       (posterior threshold OR viterbi)
  - rfmix_daiseg_simple
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = "."
STATE_ORDER = ("EU", "ND_EU", "NA", "ND_NA", "AF")


def get_project_dirs(base_dir: str, sim_name: str) -> Dict[str, Path]:
    root = Path(base_dir) / sim_name
    return {
        "root": root,
        "raw": root / "raw",
        "prepared": root / "prepared",
        "runs": root / "runs",
        "metrics": root / "metrics",
    }


def get_metrics_dir(base_dir: str, sim_name: str, mode: str) -> Path:
    return get_project_dirs(base_dir, sim_name)["metrics"] / mode


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def ensure_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"CHR", "Sample", "Start", "End", "State"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Truth file is missing columns: {missing}")

    out = df.copy()
    out["CHR"] = out["CHR"].astype(int)
    out["Sample"] = out["Sample"].astype(str)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    out["State"] = out["State"].astype(str)
    return out


def ensure_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "CHR" in out.columns and "CHROM" not in out.columns:
        out["CHROM"] = out["CHR"]

    needed = {"CHROM", "Sample", "Start", "End", "State"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    out["CHROM"] = out["CHROM"].astype(int)
    out["Sample"] = out["Sample"].astype(str)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    out["State"] = out["State"].astype(str)

    if "Length" not in out.columns:
        out["Length"] = out["End"] - out["Start"]
    out["Length"] = out["Length"].astype(int)
    return out


def calculate_aggregate_metrics_by_intersections(
    gt_full_df: pd.DataFrame,
    pred_full_df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    file_confusion: str = "confusion.matrix.txt",
    file_classification: str = "classification.report.txt",
    state_order: Sequence[str] = STATE_ORDER,
    make_plot: bool = True,
) -> Tuple[np.ndarray, Dict]:
    state_map = {s: i for i, s in enumerate(state_order)}
    n_states = len(state_order)

    def _cleanup_state_col(df: pd.DataFrame, col: str = "State") -> pd.DataFrame:
        df = df.copy()
        return df[df[col].isin(state_map.keys())]

    def _prep_gt(df: pd.DataFrame) -> pd.DataFrame:
        df = _cleanup_state_col(df, "State").copy()
        df["CHROM"] = df["CHR"].astype(int)
        df["Sample"] = df["Sample"].astype(str)
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
        df["state_id"] = df["State"].map(state_map).astype(int)
        return df[["Sample", "CHROM", "Start", "End", "state_id"]]

    def _prep_pred(df: pd.DataFrame) -> pd.DataFrame:
        df = _cleanup_state_col(df, "State").copy()
        df["CHROM"] = df["CHROM"].astype(int)
        df["Sample"] = df["Sample"].astype(str)
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
        df["state_id"] = df["State"].map(state_map).astype(int)
        return df[["Sample", "CHROM", "Start", "End", "state_id"]]

    def _df_to_intervals(df: pd.DataFrame, sample: str, chrom: int) -> List[Tuple[int, int, int]]:
        sub = df[(df["Sample"] == sample) & (df["CHROM"] == chrom)].sort_values("Start", kind="mergesort")
        if sub.empty:
            return []
        return list(zip(sub["Start"].to_numpy(), sub["End"].to_numpy(), sub["state_id"].to_numpy()))

    def _confmat_sweepline(gt_int: List[Tuple[int, int, int]], pr_int: List[Tuple[int, int, int]]) -> np.ndarray:
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

    def _report_from_conf(conf: np.ndarray) -> Dict:
        rep: Dict = {"state_order": list(state_order)}
        total = int(conf.sum())
        rep["total_bp_scored"] = total
        rep["accuracy"] = float(conf.trace() / total) if total > 0 else float("nan")

        for k, name in enumerate(state_order):
            tp = int(conf[k, k])
            fp = int(conf[:, k].sum() - tp)
            fn = int(conf[k, :].sum() - tp)

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
    if not samples or not chroms:
        conf_zero = np.zeros((n_states, n_states), dtype=np.int64)
        report_zero = _report_from_conf(conf_zero)
        return conf_zero, report_zero

    conf_total = np.zeros((n_states, n_states), dtype=np.int64)
    for sample in samples:
        for chrom in chroms:
            gt_int = _df_to_intervals(gt, sample, chrom)
            pr_int = _df_to_intervals(pr, sample, chrom)
            if gt_int and pr_int:
                conf_total += _confmat_sweepline(gt_int, pr_int)

    report = _report_from_conf(conf_total)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / file_confusion, conf_total, fmt="%d")
        with open(output_dir / file_classification, "w") as f:
            json.dump(report, f, indent=2)

        if make_plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(conf_total, aspect="auto")
                ax.set_xticks(range(n_states), labels=state_order, rotation=45, ha="right")
                ax.set_yticks(range(n_states), labels=state_order)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
                plt.close(fig)
            except Exception:
                pass

    return conf_total, report


def resolve_prediction_path(
    mode: str,
    *,
    base_dir: str,
    sim_name: str,
    rfmix_eu_ref: int | None = None,
    rfmix_na_ref: int | None = None,
    rfmix_af_ref: int | None = None,
    hmmix_af_ref: int | None = None,
    hmmix_threshold: float | None = None,
    viterbi: bool = False,
    simple_af_ref: int | None = None,
    simple_nd_ref: int | None = None,
    mexicans_eu_ref: int | None = None,
    mexicans_na_ref: int | None = None,
    mexicans_af_ref: int | None = None,
    mexicans_nd_ref: int | None = None,
) -> Tuple[Path, str]:
    runs_dir = get_project_dirs(base_dir, sim_name)["runs"]

    if mode == "rfmix_hmmix":
        if None in (rfmix_eu_ref, rfmix_na_ref, rfmix_af_ref, hmmix_af_ref):
            raise ValueError("rfmix_hmmix requires --rfmix-*-ref and --hmmix-af-ref")

        if viterbi:
            tag = f"rfmix.eu{rfmix_eu_ref}.na{rfmix_na_ref}.af{rfmix_af_ref}__hmmix.af{hmmix_af_ref}.viterbi"
            return runs_dir / "rfmix_hmmix" / tag / "combined.predictions.tsv", tag

        if hmmix_threshold is None:
            raise ValueError("rfmix_hmmix posterior mode requires --hmmix-threshold")

        thr_tag = f"{float(hmmix_threshold):.2f}".replace(".", "_")
        tag = f"rfmix.eu{rfmix_eu_ref}.na{rfmix_na_ref}.af{rfmix_af_ref}__hmmix.af{hmmix_af_ref}.thr{thr_tag}"
        return runs_dir / "rfmix_hmmix" / tag / "combined.predictions.tsv", tag

    if mode == "rfmix_daiseg_simple":
        if None in (rfmix_eu_ref, rfmix_na_ref, rfmix_af_ref, simple_af_ref, simple_nd_ref):
            raise ValueError("rfmix_daiseg_simple requires --rfmix-*-ref, --simple-af-ref, --simple-nd-ref")
        tag = f"rfmix.eu{rfmix_eu_ref}.na{rfmix_na_ref}.af{rfmix_af_ref}__simple.af{simple_af_ref}.nd{simple_nd_ref}"
        return runs_dir / "rfmix_daiseg_simple" / tag / "combined.predictions.tsv", tag

    if mode == "daiseg_mexicans":
        if None in (mexicans_eu_ref, mexicans_na_ref, mexicans_af_ref, mexicans_nd_ref):
            raise ValueError(
                "daiseg_mexicans requires --mexicans-eu-ref --mexicans-na-ref --mexicans-af-ref --mexicans-nd-ref"
            )
        tag = f"ref.eu{mexicans_eu_ref}.na{mexicans_na_ref}.af{mexicans_af_ref}.nd{mexicans_nd_ref}"
        return runs_dir / "daiseg_mexicans" / tag / "all.inferred.daiseg_mexicans.em.tsv", tag

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_mode(
    mode: str,
    *,
    base_dir: str,
    sim_name: str,
    prediction_path: Path,
    tag: str,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    truth_path = dirs["raw"] / "truth.all.tsv"
    manifest_path = dirs["raw"] / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing raw manifest: {manifest_path}")

    # Read manifest to fail early if raw layout is broken.
    json.loads(manifest_path.read_text())

    gt_df = ensure_truth_columns(load_tsv(truth_path))
    pred_df = ensure_pred_columns(load_tsv(prediction_path))

    metrics_dir = get_metrics_dir(base_dir, sim_name, mode)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = tag.replace("/", "__")
    confusion_name = f"confusion_matrix.{safe_tag}.txt"
    report_name = f"classification_report.{safe_tag}.json"

    _, report = calculate_aggregate_metrics_by_intersections(
        gt_df,
        pred_df,
        output_dir=metrics_dir,
        file_confusion=confusion_name,
        file_classification=report_name,
    )

    summary = {
        "mode": mode,
        "sim_name": sim_name,
        "prediction_file": str(prediction_path.resolve()),
        "metrics_dir": str(metrics_dir.resolve()),
        "tag": tag,
        "accuracy": report["accuracy"],
        "total_bp_scored": report["total_bp_scored"],
        "per_state": {s: report[s] for s in STATE_ORDER},
    }
    summary_path = metrics_dir / f"summary.{safe_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Evaluation completed: {mode}")
    print(f"Prediction file: {prediction_path}")
    print(f"Metrics dir:     {metrics_dir}")
    print(f"Accuracy:        {report['accuracy']}")
    print(f"Summary:         {summary_path}")
    print("=" * 72)

    return metrics_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mex_compare method outputs against truth")
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("rfmix_hmmix")
    p1.add_argument("--sim-name", type=str, required=True)
    p1.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p1.add_argument("--rfmix-eu-ref", type=int, required=True)
    p1.add_argument("--rfmix-na-ref", type=int, required=True)
    p1.add_argument("--rfmix-af-ref", type=int, required=True)
    p1.add_argument("--hmmix-af-ref", type=int, required=True)
    p1.add_argument("--hmmix-threshold", type=float, default=None)
    p1.add_argument("--viterbi", action="store_true")

    p2 = sub.add_parser("rfmix_daiseg_simple")
    p2.add_argument("--sim-name", type=str, required=True)
    p2.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p2.add_argument("--rfmix-eu-ref", type=int, required=True)
    p2.add_argument("--rfmix-na-ref", type=int, required=True)
    p2.add_argument("--rfmix-af-ref", type=int, required=True)
    p2.add_argument("--simple-af-ref", type=int, required=True)
    p2.add_argument("--simple-nd-ref", type=int, required=True)

    p3 = sub.add_parser("daiseg_mexicans")
    p3.add_argument("--sim-name", type=str, required=True)
    p3.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p3.add_argument("--mexicans-eu-ref", type=int, required=True)
    p3.add_argument("--mexicans-na-ref", type=int, required=True)
    p3.add_argument("--mexicans-af-ref", type=int, required=True)
    p3.add_argument("--mexicans-nd-ref", type=int, required=True)

    return parser


def main() -> None:
    args = make_parser().parse_args()

    if args.mode == "rfmix_hmmix":
        pred_path, tag = resolve_prediction_path(
            "rfmix_hmmix",
            base_dir=args.base_dir,
            sim_name=args.sim_name,
            rfmix_eu_ref=args.rfmix_eu_ref,
            rfmix_na_ref=args.rfmix_na_ref,
            rfmix_af_ref=args.rfmix_af_ref,
            hmmix_af_ref=args.hmmix_af_ref,
            hmmix_threshold=args.hmmix_threshold,
            viterbi=args.viterbi,
        )
        evaluate_mode(
            "rfmix_hmmix",
            base_dir=args.base_dir,
            sim_name=args.sim_name,
            prediction_path=pred_path,
            tag=tag,
        )
        return

    if args.mode == "rfmix_daiseg_simple":
        pred_path, tag = resolve_prediction_path(
            "rfmix_daiseg_simple",
            base_dir=args.base_dir,
            sim_name=args.sim_name,
            rfmix_eu_ref=args.rfmix_eu_ref,
            rfmix_na_ref=args.rfmix_na_ref,
            rfmix_af_ref=args.rfmix_af_ref,
            simple_af_ref=args.simple_af_ref,
            simple_nd_ref=args.simple_nd_ref,
        )
        evaluate_mode(
            "rfmix_daiseg_simple",
            base_dir=args.base_dir,
            sim_name=args.sim_name,
            prediction_path=pred_path,
            tag=tag,
        )
        return

    if args.mode == "daiseg_mexicans":
        pred_path, tag = resolve_prediction_path(
            "daiseg_mexicans",
            base_dir=args.base_dir,
            sim_name=args.sim_name,
            mexicans_eu_ref=args.mexicans_eu_ref,
            mexicans_na_ref=args.mexicans_na_ref,
            mexicans_af_ref=args.mexicans_af_ref,
            mexicans_nd_ref=args.mexicans_nd_ref,
        )
        evaluate_mode(
            "daiseg_mexicans",
            base_dir=args.base_dir,
            sim_name=args.sim_name,
            prediction_path=pred_path,
            tag=tag,
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
