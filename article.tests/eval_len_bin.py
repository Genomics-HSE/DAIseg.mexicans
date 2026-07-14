#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_utils import (
    STATE_ORDER,
    load_tsv,
    ensure_truth_columns,
    ensure_pred_columns,
    calculate_confusion_bp,
    row_normalize,
    collapse_to_binary,
    binary_metrics,
)


BASE_DIR = Path(".")
RUN_PREFIX = "2d.daiseg.seed"

SEED_START = 1
SEED_END = 50

MODERN_REF = 250
ND_REF = 3

OUT_DIR = Path("length_bin_analysis.ref250.nd3")
OUT_PDF = OUT_DIR / "length_bin_confusion.mean_across_runs.pdf"

LENGTH_BINS = [
    ("0_10kb", 0, 10_000),
    ("10_20kb", 10_000, 20_000),
    ("20_25kb", 20_000, 25_000),
    ("25_30kb", 25_000, 30_000),
    ("30_40kb", 30_000, 40_000),
    ("40_50kb", 40_000, 50_000),
    ("50_70kb", 50_000, 70_000),
    ("70_100kb", 70_000, 100_000),
    ("100kb_plus", 100_000, None),
]


def add_length_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Length"] = out["End"] - out["Start"]
    return out


def filter_truth_bin(
    df: pd.DataFrame,
    start_bp: int,
    end_bp: int | None,
) -> pd.DataFrame:
    if end_bp is None:
        return df[df["Length"] >= start_bp].copy()

    return df[
        (df["Length"] >= start_bp)
        & (df["Length"] < end_bp)
    ].copy()


def format_bin_title(
    bin_name: str,
    recall: float,
    precision: float,
) -> str:
    return (
        f"{bin_name}\n"
        f"R={recall:.3f}, P={precision:.3f}"
    )


def plot_results(
    mean_rownorm_5x5: dict,
    bin_summary: dict,
    out_pdf: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    for ax, (bin_name, _, _) in zip(
        axes.flat,
        LENGTH_BINS,
    ):
        mat = mean_rownorm_5x5[bin_name]
        stats = bin_summary[bin_name]

        ax.imshow(
            mat,
            cmap="OrRd",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_xticklabels(
            STATE_ORDER,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(range(len(STATE_ORDER)))
        ax.set_yticklabels(STATE_ORDER)

        ax.set_title(
            format_bin_title(
                bin_name,
                stats["binary_recall_mean"],
                stats["binary_precision_mean"],
            ),
            fontsize=9,
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val > 0.5 else "black"

                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    fig.suptitle(
        (
            "Length-stratified confusion analysis "
            f"(pooled across runs; ref={MODERN_REF}, nd={ND_REF})"
        ),
        y=0.98,
    )

    fig.tight_layout()
    fig.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bin_conf5_row = {
        name: []
        for name, _, _ in LENGTH_BINS
    }

    bin_binary_metrics = {
        name: []
        for name, _, _ in LENGTH_BINS
    }

    completed_runs = 0

    for run in range(
        SEED_START,
        SEED_END + 1,
    ):
        run_dir = BASE_DIR / f"{RUN_PREFIX}{run}"

        truth_path = (
            run_dir
            / "raw"
            / "truth.all.tsv"
        )

        pred_path = (
            run_dir
            / "runs"
            / "daiseg_mexicans"
            / (
                f"ref.eu{MODERN_REF}."
                f"na{MODERN_REF}."
                f"af{MODERN_REF}."
                f"nd{ND_REF}"
            )
            / "all.inferred.daiseg_mexicans.em.tsv"
        )

        if not truth_path.exists():
            print(f"[skip] missing truth: {truth_path}")
            continue

        if not pred_path.exists():
            print(f"[skip] missing prediction: {pred_path}")
            continue

        truth = ensure_truth_columns(
            load_tsv(truth_path)
        )
        truth = add_length_column(truth)

        pred = ensure_pred_columns(
            load_tsv(pred_path)
        )

        for bin_name, start_bp, end_bp in LENGTH_BINS:
            gt_bin = filter_truth_bin(
                truth,
                start_bp,
                end_bp,
            )

            conf5 = calculate_confusion_bp(
                gt_bin,
                pred,
            )
            conf2 = collapse_to_binary(conf5)

            bin_conf5_row[bin_name].append(
                conf5
            )

            bin_binary_metrics[bin_name].append(conf2)

        completed_runs += 1
        print(f"[ok] run={run}")

    if completed_runs == 0:
        raise SystemExit("No completed runs found.")

    summary = {}
    mean_rownorm_5x5 = {}

    for bin_name, _, _ in LENGTH_BINS:
        row_stack = np.stack(
            bin_conf5_row[bin_name],
            axis=0,
        )

        binary_stack = np.stack(
            bin_binary_metrics[bin_name],
            axis=0,
        )

        binary_stats = binary_metrics(
            binary_stack.sum(axis=0)
        )

        summary[bin_name] = {
            "binary_recall_mean": float(
                binary_stats["recall"]
            ),
            "binary_precision_mean": float(
                binary_stats["precision"]
            ),
        }

        mean_rownorm_5x5[bin_name] = (
            row_normalize(row_stack.sum(axis=0))
        )

    plot_results(
        mean_rownorm_5x5=mean_rownorm_5x5,
        bin_summary=summary,
        out_pdf=OUT_PDF,
    )

    print(f"Saved plot to {OUT_PDF}")


if __name__ == "__main__":
    main()
