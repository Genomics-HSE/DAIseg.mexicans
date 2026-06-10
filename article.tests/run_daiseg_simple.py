#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


DEFAULT_BASE_DIR = "."
DEFAULT_THREADS = 4
DAISEG_SIMPLE_BIN = "/home/ailina/DAIseg.28.03/daiseg.py"

def get_project_dirs(base_dir: str, sim_name: str) -> Dict[str, Path]:
    root = Path(base_dir) / sim_name
    return {
        "root": root,
        "raw": root / "raw",
        "prepared": root / "prepared",
        "runs": root / "runs",
        "metrics": root / "metrics",
    }


def get_daiseg_simple_prepare_dir(
    base_dir: str,
    sim_name: str,
    n_af_ref: int,
    n_nd_ref: int,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["prepared"] / "daiseg_simple" / f"ref.af{n_af_ref}.nd{n_nd_ref}"


def get_daiseg_simple_run_dir(
    base_dir: str,
    sim_name: str,
    n_af_ref: int,
    n_nd_ref: int,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["runs"] / "daiseg_simple" / f"ref.af{n_af_ref}.nd{n_nd_ref}"


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def check_requirements() -> None:
    if not os.path.exists(DAISEG_SIMPLE_BIN):
        raise EnvironmentError(f"DAIseg.simple not found at {DAISEG_SIMPLE_BIN}")


def run_command(cmd: List[str], step_name: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


# ==============================================================================
# Metrics
# ==============================================================================

def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    if not cleaned:
        return []
    merged = [list(cleaned[0])]
    for start, end in cleaned[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(s), int(e)) for s, e in merged]


def interval_total_length(intervals: Iterable[Tuple[int, int]]) -> int:
    return sum(int(e) - int(s) for s, e in intervals)


def interval_overlap_length(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> int:
    i = 0
    j = 0
    overlap = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        start = max(s1, s2)
        end = min(e1, e2)
        if start < end:
            overlap += end - start
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return overlap


def _build_interval_dict(df: pd.DataFrame) -> Dict[Tuple[int, str], List[Tuple[int, int]]]:
    if df is None or len(df) == 0:
        return {}
    out = {}
    for key, group in df.groupby(["CHR", "Sample"]):
        intervals = list(zip(group["Start"].astype(int), group["End"].astype(int)))
        out[(int(key[0]), str(key[1]))] = merge_intervals(intervals)
    return out


def calculate_binary_metrics(true_df: pd.DataFrame, inf_df: pd.DataFrame, chrom_length: int) -> Dict:
    true_arch = true_df[true_df["State"].isin(["ND_EU", "ND_NA"])].copy()

    true_dict = _build_interval_dict(true_arch)
    inf_dict = _build_interval_dict(inf_df)

    pairs = set(true_dict.keys()) | set(inf_dict.keys())
    if not pairs and len(true_df) > 0:
        pairs = set(
            (int(r.CHR), str(r.Sample))
            for r in true_df[["CHR", "Sample"]].drop_duplicates().itertuples(index=False)
        )

    total_bp = len(pairs) * int(chrom_length)
    tp = 0
    total_true = 0
    total_pred = 0

    for key in pairs:
        t_int = true_dict.get(key, [])
        i_int = inf_dict.get(key, [])
        total_true += interval_total_length(t_int)
        total_pred += interval_total_length(i_int)
        tp += interval_overlap_length(t_int, i_int)

    fp = max(total_pred - tp, 0)
    fn = max(total_true - tp, 0)
    tn = max(total_bp - (tp + fp + fn), 0)

    def _stats(tp_: int, fp_: int, fn_: int) -> Tuple[float, float, float]:
        precision = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        recall = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    pa, ra, fa = _stats(tp, fp, fn)
    pm, rm, fm = _stats(tn, fn, fp)

    return {
        "Total_BP": int(total_bp),
        "Archaic": {"Precision": pa, "Recall": ra, "F1": fa, "TP": int(tp), "FP": int(fp), "FN": int(fn)},
        "Modern": {"Precision": pm, "Recall": rm, "F1": fm, "TP": int(tn), "FP": int(fn), "FN": int(fp)},
    }


# ==============================================================================
# DAIseg.simple running / parsing
# ==============================================================================

def collect_config_files(prepared_dir: Path) -> List[Path]:
    cfgs = sorted(prepared_dir.glob("config.simple_seed_*.json"))
    if not cfgs:
        raise FileNotFoundError(f"No config files found in {prepared_dir}")
    return cfgs


def run_daiseg_simple_em_v2(
    prepared_dir: Path,
    run_dir: Path,
    threads: int,
) -> Path:
    cfgs = collect_config_files(prepared_dir)
    out_file = run_dir / "all.inferred.daiseg_simple.em_v2.tsv"

    cmd = [
        "python",
        DAISEG_SIMPLE_BIN,
        "run.EM.v2",
        "-jsons",
        *[str(x) for x in cfgs],
        "-out",
        str(out_file),
        "-threads",
        str(int(threads)),
    ]
    run_command(cmd, "DAIseg.simple EM v2")
    return out_file


def load_daiseg_simple_output(output_file: Path) -> pd.DataFrame:
    df = pd.read_csv(output_file, sep="\t", keep_default_na=False)

    if "CHR" not in df.columns:
        if "CHROM" in df.columns:
            df["CHR"] = df["CHROM"]
        elif "Chrom" in df.columns:
            df["CHR"] = df["Chrom"]

    if "Sample" in df.columns:
        df["Sample"] = df["Sample"].astype(str).apply(
            lambda x: x if x.startswith("MX_") else f"MX_{x}"
        )

    if not df.empty and "State" not in df.columns:
        df["State"] = "NEAND"

    return df


def run_daiseg_simple_pipeline(
    sim_name: str,
    *,
    base_dir: str,
    n_af_ref: int,
    n_nd_ref: int,
    threads: int,
    skip_run: bool,
    force: bool,
) -> Path:
    check_requirements()

    dirs = get_project_dirs(base_dir, sim_name)
    raw_dir = dirs["raw"]
    raw_manifest = load_json(raw_dir / "manifest.json")
    chrom_length = int(raw_manifest["simulation"]["params"]["chrom_length"])

    prepared_dir = get_daiseg_simple_prepare_dir(base_dir, sim_name, n_af_ref, n_nd_ref)
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared DAIseg.simple dir not found: {prepared_dir}")
    if not (prepared_dir / "manifest.daiseg_simple.json").exists():
        raise FileNotFoundError(f"Prepared DAIseg.simple manifest not found in {prepared_dir}")

    run_dir = get_daiseg_simple_run_dir(base_dir, sim_name, n_af_ref, n_nd_ref)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    out_file = run_dir / "all.inferred.daiseg_simple.em_v2.tsv"
    if not skip_run:
        out_file = run_daiseg_simple_em_v2(prepared_dir, run_dir, threads)
    else:
        if not out_file.exists():
            raise FileNotFoundError(f"--skip-run was requested, but output not found: {out_file}")

    pred_df = load_daiseg_simple_output(out_file)
    truth_df = pd.read_csv(raw_dir / "truth.all.tsv", sep="\t")
    metrics = calculate_binary_metrics(truth_df, pred_df, chrom_length)

    run_manifest = {
        "method": "daiseg_simple",
        "sim_name": sim_name,
        "prepared_dir": str(prepared_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "reference_panel": {
            "n_af_ref": int(n_af_ref),
            "n_nd_ref": int(n_nd_ref),
        },
        "settings": {
            "threads": int(threads),
            "skip_run": bool(skip_run),
            "mode": "EM_v2",
        },
        "output_files": {
            "combined_segments": out_file.name,
        },
        "evaluation": {
            "precision": metrics["Archaic"]["Precision"],
            "recall": metrics["Archaic"]["Recall"],
            "f1": metrics["Archaic"]["F1"],
            "metrics": metrics,
        },
    }
    with open(run_dir / "manifest.run_daiseg_simple.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 72)
    print("DAIseg.simple run completed")
    print(f"Run dir:      {run_dir}")
    print(f"Output TSV:   {out_file}")
    print(f"Precision:    {metrics['Archaic']['Precision']}")
    print(f"Recall:       {metrics['Archaic']['Recall']}")
    print(f"F1:           {metrics['Archaic']['F1']}")
    print("=" * 72)

    return run_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DAIseg.simple for prepared mex_compare inputs")
    parser.add_argument("--sim-name", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--n-af-ref", type=int, required=True)
    parser.add_argument("--n-nd-ref", type=int, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    run_daiseg_simple_pipeline(
        sim_name=args.sim_name,
        base_dir=args.base_dir,
        n_af_ref=args.n_af_ref,
        n_nd_ref=args.n_nd_ref,
        threads=args.threads,
        skip_run=args.skip_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
