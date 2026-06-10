#!/usr/bin/env python3

"""
Plot row-normalized 5-state confusion matrices for selected DAIseg grid points.
The script compares truth and inferred ancestry states across available
simulation seeds and saves the summary figure to confusion.selected.grid.pdf.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import (
    STATE_ORDER,
    load_tsv,
    ensure_truth_columns,
    ensure_pred_columns,
    calculate_confusion_bp,
    row_normalize,
)


BASE_DIR = Path(".")
RUN_PREFIX = "2d.daiseg.seed"

SEED_START = 1
SEED_END = 50

ND_REFS = [0, 1, 3]
MODERN_REFS = [25, 100, 250]

OUT = Path("confusion.selected.grid.pdf")


def collect_confusion(modern_ref: int, nd_ref: int) -> np.ndarray:
    conf_sum = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=np.int64)
    n_runs = 0

    for run in range(SEED_START, SEED_END + 1):
        run_dir = BASE_DIR / f"{RUN_PREFIX}{run}"

        truth_path = run_dir / "raw" / "truth.all.tsv"
        pred_path = (
            run_dir
            / "runs"
            / "daiseg_mexicans"
            / f"ref.eu{modern_ref}.na{modern_ref}.af{modern_ref}.nd{nd_ref}"
            / "all.inferred.daiseg_mexicans.em.tsv"
        )

        if not truth_path.exists():
            print(f"[skip] missing truth: {truth_path}")
            continue

        if not pred_path.exists():
            print(f"[skip] missing prediction: {pred_path}")
            continue

        truth = ensure_truth_columns(load_tsv(truth_path))
        pred = ensure_pred_columns(load_tsv(pred_path))

        conf_sum += calculate_confusion_bp(truth, pred)
        n_runs += 1

        print(f"[ok] run={run}, ref={modern_ref}, nd={nd_ref}")

    if n_runs == 0:
        raise RuntimeError(f"No completed runs found for ref={modern_ref}, nd={nd_ref}")

    print(f"[done] ref={modern_ref}, nd={nd_ref}, runs={n_runs}")
    return conf_sum


def plot_grid(confs: dict[tuple[int, int], np.ndarray]) -> None:
    fig, axes = plt.subplots(len(ND_REFS), len(MODERN_REFS), figsize=(14, 13))

    for i, nd_ref in enumerate(ND_REFS):
        for j, modern_ref in enumerate(MODERN_REFS):
            ax = axes[i, j]
            mat = confs[(nd_ref, modern_ref)]

            im = ax.imshow(mat, cmap="OrRd", vmin=0, vmax=1)

            ax.set_xticks(range(len(STATE_ORDER)))
            ax.set_yticks(range(len(STATE_ORDER)))

            if i == len(ND_REFS) - 1:
                ax.set_xticklabels(STATE_ORDER, rotation=45, ha="right")
                ax.set_xlabel("Predicted state")
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_yticklabels(STATE_ORDER)
                ax.set_ylabel("True state")
            else:
                ax.set_yticklabels([])

            ax.set_title(f"ref={modern_ref}, nd={nd_ref}")

            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    color = "white" if val > 0.5 else "black"
                    ax.text(
                        c,
                        r,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=color,
                    )

    fig.subplots_adjust(right=0.88, wspace=0.20, hspace=0.20)

    cax = fig.add_axes([0.90, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized fraction")

    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {OUT}")


def main() -> None:
    confs = {}

    for nd_ref in ND_REFS:
        for modern_ref in MODERN_REFS:
            conf = collect_confusion(modern_ref, nd_ref)
            confs[(nd_ref, modern_ref)] = row_normalize(conf)

    plot_grid(confs)


if __name__ == "__main__":
    main()
