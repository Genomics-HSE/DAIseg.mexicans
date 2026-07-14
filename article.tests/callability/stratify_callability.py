#!/usr/bin/env python3

import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# paths and settings
# =========================

CHROM_LENGTHS = "/home/share/human.data/ref.fa/hg19.chr.lengths/hg19.chrom.len"

MODERN_DIR = "/home/share/human.data/1000GP/1000GP.grch37/bed"
GAPS_FILE = "/home/share/human.data/ref.fa/gaps.grch37/gap.renamed.txt"

VINDIJA_DIR = "/home/share/human.data/neand/33.19/bed"
ALTAI_DIR = "/home/share/human.data/neand/altai/bed"
CHAGYR_DIR = "/home/share/human.data/neand/Chagyrskaya/bed"

OUT_DIR = "masked_matrix_all"
PRED_FILE = "masked_matrix_all/predictions.pooled_union.tsv"

OUT_PNG = "archaic_precision_recall_by_callability.png"
OUT_PDF = "archaic_precision_recall_by_callability.pdf"

BASE_SEED = 1234567
CHROMS = [str(i) for i in range(1, 23)]

WINDOW_LEN = 1000
MIN_ARCH_FRACTION = 0.5

ARCH = {"ND_EU", "ND_NA"}


# =========================
# files and intervals
# =========================

def open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def norm_chr(x):
    x = str(x)
    return x[3:] if x.startswith("chr") else x


def merge(xs):
    xs = sorted((int(a), int(b)) for a, b in xs if int(b) > int(a))
    if not xs:
        return []

    out = [list(xs[0])]

    for a, b in xs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])

    return [tuple(x) for x in out]


def read_lengths(path):
    d = {}

    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            p = line.split()
            if len(p) < 2:
                continue

            chrom = norm_chr(p[0])
            if chrom.lower() in {"chrom", "#chrom"}:
                continue

            d[chrom] = int(p[1])

    return d


def read_bed(path, chrom, chrom_len, gaps=False):
    chrom = norm_chr(chrom)
    out = []

    with open_text(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            p = line.split()

            if gaps:
                if len(p) < 4:
                    continue
                c = norm_chr(p[1])
                a = int(p[2])
                b = int(p[3])
            else:
                if len(p) < 3:
                    continue
                c = norm_chr(p[0])
                a = int(p[1])
                b = int(p[2])

            if c != chrom:
                continue

            a = max(0, a)
            b = min(chrom_len, b)

            if b > a:
                out.append((a, b))

    return merge(out)


def subtract(a, b):
    out = []
    j = 0

    for x1, x2 in a:
        cur = x1

        while j < len(b) and b[j][1] <= x1:
            j += 1

        k = j
        while k < len(b) and b[k][0] < x2:
            y1, y2 = b[k]

            if y1 > cur:
                out.append((cur, min(y1, x2)))

            cur = max(cur, y2)

            if cur >= x2:
                break

            k += 1

        if cur < x2:
            out.append((cur, x2))

    return out


def intersect(a, b):
    out = []
    i = 0
    j = 0

    while i < len(a) and j < len(b):
        x1 = max(a[i][0], b[j][0])
        x2 = min(a[i][1], b[j][1])

        if x2 > x1:
            out.append((x1, x2))

        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

    return out


def union_many(xs):
    z = []

    for x in xs:
        z.extend(x)

    return merge(z)


def window_fraction(intervals, chrom_len):
    n = math.ceil(chrom_len / WINDOW_LEN)
    x = np.zeros(n)

    for a, b in intervals:
        w1 = a // WINDOW_LEN
        w2 = (b - 1) // WINDOW_LEN

        for w in range(w1, w2 + 1):
            s = w * WINDOW_LEN
            e = min((w + 1) * WINDOW_LEN, chrom_len)
            x[w] += max(0, min(b, e) - max(a, s))

    for w in range(n):
        s = w * WINDOW_LEN
        e = min((w + 1) * WINDOW_LEN, chrom_len)
        x[w] /= e - s

    return x


def modern_path(chrom):
    p = Path(MODERN_DIR) / f"chr{chrom}.renamed.bed"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def nd_path(nd_dir, chrom):
    p1 = Path(nd_dir) / f"chr{chrom}_mask.bed.gz"
    p2 = Path(nd_dir) / f"chr{chrom}_mask.bed"

    if p1.exists():
        return p1

    if p2.exists():
        return p2

    raise FileNotFoundError(f"{p1} or {p2}")


def callability(chrom, chrom_len):
    modern = read_bed(modern_path(chrom), chrom, chrom_len)
    gaps = read_bed(GAPS_FILE, chrom, chrom_len, gaps=True)

    modern = subtract(modern, gaps)

    nd = union_many([
        read_bed(nd_path(VINDIJA_DIR, chrom), chrom, chrom_len),
        read_bed(nd_path(ALTAI_DIR, chrom), chrom, chrom_len),
        read_bed(nd_path(CHAGYR_DIR, chrom), chrom, chrom_len),
    ])

    arch = intersect(modern, nd)

    l_mod = window_fraction(modern, chrom_len)
    l_arch = window_fraction(arch, chrom_len)

    u_nd = np.zeros_like(l_mod)
    ok = l_mod > 0
    u_nd[ok] = l_arch[ok] / l_mod[ok]

    return l_mod, u_nd


# =========================
# truth and predictions
# =========================

def archaic_cov(df, sample, chrom_key, chrom_len, end_inclusive):
    n = math.ceil(chrom_len / WINDOW_LEN)
    cov = np.zeros(n)

    sub = df[
        (df["Sample"].astype(str) == str(sample))
        & (df["CHROM"].astype(str) == str(chrom_key))
        & (df["State"].isin(ARCH))
    ]

    for _, r in sub.iterrows():
        a = int(r["Start"])
        b = int(r["End"])

        if end_inclusive:
            b += 1

        a = max(0, a)
        b = min(chrom_len, b)

        if b <= a:
            continue

        w1 = a // WINDOW_LEN
        w2 = (b - 1) // WINDOW_LEN

        for w in range(w1, w2 + 1):
            s = w * WINDOW_LEN
            e = min((w + 1) * WINDOW_LEN, chrom_len)
            cov[w] += max(0, min(b, e) - max(a, s))

    return cov


def bin_index(x, edges):
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = len(edges) - 2
    idx[(x < edges[0]) | (x > edges[-1])] = -1
    return idx


def bin_labels(edges):
    out = []

    for i in range(len(edges) - 1):
        a = edges[i]
        b = edges[i + 1]

        if i == len(edges) - 2:
            out.append(f"[{a:.1f},{b:.1f}]")
        else:
            out.append(f"[{a:.1f},{b:.1f})")

    return out


# =========================
# plot
# =========================

def draw(ax, mat, n, labels, title, vmin, vmax):
    im = ax.imshow(mat, origin="lower", vmin=vmin, vmax=vmax)

    ax.set_title(title)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel(r"Modern callability, $\rho_{\mathrm{mod},t}$")
    ax.set_ylabel(r"Neanderthal union coverage, $U_{\mathrm{ND},t}$")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j, i + 0.08,
                    f"{mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax.text(
                    j, i - 0.20,
                    f"W={int(n[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=5,
                )
            else:
                ax.text(j, i + 0.08, "NA", ha="center", va="center", fontsize=8)
                ax.text(j, i - 0.20, "W=0", ha="center", va="center", fontsize=5)

    return im


# =========================
# main
# =========================

def main():
    out_dir = Path(OUT_DIR)
    lengths = read_lengths(CHROM_LENGTHS)

    gt_parts = []
    chrom_info = []

    for i, chrom in enumerate(CHROMS):
        chrom = norm_chr(chrom)
        seed = BASE_SEED + i

        if chrom not in lengths:
            continue

        gt_file = out_dir / f"ground_truth_5state_chr{chrom}_seed_{seed}.tsv"

        if not gt_file.exists():
            print(f"missing {gt_file}")
            continue

        gt_parts.append(pd.read_csv(gt_file, sep="\t", keep_default_na=False))
        chrom_info.append((chrom, str(seed), lengths[chrom]))

    if not gt_parts:
        raise SystemExit("no ground truth files")

    gt = pd.concat(gt_parts, ignore_index=True)
    pred = pd.read_csv(PRED_FILE, sep="\t", keep_default_na=False)

    gt["CHROM"] = gt["CHR"].astype(str)
    pred["CHROM"] = pred["CHROM"].astype(str)

    gt["Sample"] = gt["Sample"].astype(str)
    pred["Sample"] = pred["Sample"].astype(str)

    samples = sorted(set(gt["Sample"]).intersection(set(pred["Sample"])))

    edges = np.linspace(0, 1, 6)
    labels = bin_labels(edges)
    nb = len(labels)

    true_n = np.zeros((nb, nb), dtype=int)
    pred_n = np.zeros((nb, nb), dtype=int)
    tp_n = np.zeros((nb, nb), dtype=int)

    for chrom, seed, chrom_len in chrom_info:
        print(f"chr{chrom}")

        l_mod, u_nd = callability(chrom, chrom_len)

        l_bin = bin_index(l_mod, edges)
        u_bin = bin_index(u_nd, edges)

        valid = (l_bin >= 0) & (u_bin >= 0)
        cell = u_bin * nb + l_bin

        nwin = math.ceil(chrom_len / WINDOW_LEN)
        denom = np.full(nwin, WINDOW_LEN, dtype=float)
        denom[-1] = chrom_len - (nwin - 1) * WINDOW_LEN

        for sample in samples:
            true_cov = archaic_cov(
                gt,
                sample,
                seed,
                chrom_len,
                end_inclusive=False,
            )

            pred_cov = archaic_cov(
                pred,
                sample,
                seed,
                chrom_len,
                end_inclusive=True,
            )

            true_arch = (true_cov / denom) >= MIN_ARCH_FRACTION
            pred_arch = (pred_cov / denom) >= MIN_ARCH_FRACTION

            use_true = valid & true_arch
            use_pred = valid & pred_arch
            use_tp = valid & true_arch & pred_arch

            true_n += np.bincount(cell[use_true], minlength=nb * nb).reshape(nb, nb)
            pred_n += np.bincount(cell[use_pred], minlength=nb * nb).reshape(nb, nb)
            tp_n += np.bincount(cell[use_tp], minlength=nb * nb).reshape(nb, nb)

    precision = np.divide(
        tp_n,
        pred_n,
        out=np.full_like(tp_n, np.nan, dtype=float),
        where=pred_n > 0,
    )

    recall = np.divide(
        tp_n,
        true_n,
        out=np.full_like(tp_n, np.nan, dtype=float),
        where=true_n > 0,
    )

    print("\nBottom row: U_ND in [0.0,0.2)")
    for j, lab in enumerate(labels):
        tp = tp_n[0, j]
        pred = pred_n[0, j]
        true = true_n[0, j]
        fp = pred - tp
        fn = true - tp
        prec = tp / pred if pred > 0 else float("nan")
        rec = tp / true if true > 0 else float("nan")
        print(
            lab,
            "TP=", tp,
            "FP=", fp,
            "FN=", fn,
            "pred_n=", pred,
            "true_n=", true,
            "precision=", prec,
            "recall=", rec,
        )

    precision_min = np.nanmin(precision)
    precision_max = np.nanmax(precision)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    im1 = draw(
        axes[0],
        precision,
        pred_n,
        labels,
        "Precision",
        vmin=precision_min,
        vmax=precision_max,
    )

    im2 = draw(
        axes[1],
        recall,
        true_n,
        labels,
        "Recall",
        vmin=0,
        vmax=1,
    )

    cb1 = fig.colorbar(im1, ax=axes[0])
    cb1.set_label("Precision (zoomed scale)")

    cb2 = fig.colorbar(im2, ax=axes[1])
    cb2.set_label("Recall")

    fig.tight_layout()

    out_png = out_dir / OUT_PNG
    out_pdf = out_dir / OUT_PDF

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    plt.close(fig)

    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
