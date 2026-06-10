#!/usr/bin/env python3

"""
Combine grid metrics from multiple simulation seeds and compute mean and
standard deviation of precision and recall for each parameter combination.
Generate summary tile plots for archaic and modern ancestry states.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TSV = Path("all_runs.long.tsv")

ARCHAIC_OUT = Path("archaic.tileplot.pdf")
MODERN_OUT = Path("modern.tileplot.pdf")

ARCHAIC_STATES = ["ND_EU", "ND_NA"]
MODERN_STATES = ["EU", "NA", "AF"]
METRICS = ["precision", "recall"]

METRICS_PATTERN = "2d.daiseg.seed*/metrics/daiseg_mexicans/grid_metrics.long.tsv"


def seed_number(path: Path) -> int:
    m = re.search(r"2d\.daiseg\.seed(\d+)", str(path))
    return int(m.group(1))


def collect_runs() -> pd.DataFrame:
    files = sorted(Path(".").glob(METRICS_PATTERN), key=seed_number)

    dfs = []
    for path in files:
        seed = seed_number(path)
        df = pd.read_csv(path, sep="\t", keep_default_na=False)
        df.insert(0, "seed", seed)
        dfs.append(df)

    all_runs = pd.concat(dfs, ignore_index=True)
    all_runs.to_csv(TSV, sep="\t", index=False)

    print(f"Collected {len(files)} runs into {TSV}")
    return all_runs


def build_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["state", "nd_ref", "modern_ref"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
    )


def plot_tile_grid(
    agg: pd.DataFrame,
    states: list[str],
    out_path: Path,
    title: str = "",
) -> None:
    modern_refs = sorted(agg["modern_ref"].unique())
    nd_refs = sorted(agg["nd_ref"].unique())

    p_vmin = np.nanmin(agg["precision_mean"])
    p_vmax = np.nanmax(agg["precision_mean"])
    r_vmin = np.nanmin(agg["recall_mean"])
    r_vmax = np.nanmax(agg["recall_mean"])

    fig, axes = plt.subplots(2, len(states), figsize=(4.3 * len(states), 8.2))

    if len(states) == 1:
        axes = np.array(axes).reshape(2, 1)

    ims = {}

    for col, state in enumerate(states):
        sub = agg[agg["state"] == state]

        for row, metric in enumerate(METRICS):
            ax = axes[row, col]

            mean_mat = (
                sub.pivot(index="nd_ref", columns="modern_ref", values=f"{metric}_mean")
                .reindex(index=nd_refs, columns=modern_refs)
            )
            std_mat = (
                sub.pivot(index="nd_ref", columns="modern_ref", values=f"{metric}_std")
                .reindex(index=nd_refs, columns=modern_refs)
                .fillna(0.0)
            )

            if metric == "precision":
                vmin, vmax = p_vmin, p_vmax
            else:
                vmin, vmax = r_vmin, r_vmax

            im = ax.imshow(mean_mat.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax)
            ims[metric] = im

            ax.set_xticks(range(len(modern_refs)))
            ax.set_yticks(range(len(nd_refs)))

            if row == 1:
                ax.set_xticklabels(modern_refs)
                ax.set_xlabel("Modern reference size")
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_yticklabels(nd_refs)
                ax.set_ylabel("Number of Neanderthals")
            else:
                ax.set_yticklabels([])

            if row == 0:
                ax.set_title(state)

            midpoint = (vmin + vmax) / 2.0

            for r in range(len(nd_refs)):
                for c in range(len(modern_refs)):
                    mean = mean_mat.iloc[r, c]
                    std = std_mat.iloc[r, c]
                    color = "white" if mean < midpoint else "black"

                    ax.text(
                        c,
                        r,
                        f"{mean:.3f}\n±{std:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )

    fig.colorbar(ims["precision"], ax=axes[0, :], fraction=0.03, pad=0.04)
    fig.colorbar(ims["recall"], ax=axes[1, :], fraction=0.03, pad=0.04)

    fig.text(0.015, 0.73, "Precision", rotation=90, va="center", ha="center", fontsize=12)
    fig.text(0.015, 0.29, "Recall", rotation=90, va="center", ha="center", fontsize=12)

    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(
        left=0.10,
        right=0.84,
        bottom=0.08,
        top=0.90,
        wspace=0.22,
        hspace=0.22,
    )

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {out_path}")


df = collect_runs()

archaic_agg = build_agg(df[df["state"].isin(ARCHAIC_STATES)].copy())
modern_agg = build_agg(df[df["state"].isin(MODERN_STATES)].copy())

plot_tile_grid(archaic_agg, ARCHAIC_STATES, ARCHAIC_OUT)
plot_tile_grid(modern_agg, MODERN_STATES, MODERN_OUT)
