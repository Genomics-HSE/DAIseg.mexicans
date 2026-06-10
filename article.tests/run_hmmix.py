#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_BASE_DIR = "."
DEFAULT_THRESHOLDS = "0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99"


def get_project_dirs(base_dir: str, sim_name: str) -> Dict[str, Path]:
    root = Path(base_dir) / sim_name
    return {
        "root": root,
        "raw": root / "raw",
        "prepared": root / "prepared",
        "runs": root / "runs",
        "metrics": root / "metrics",
    }


def get_hmmix_prepare_dir(base_dir: str, sim_name: str, n_af_ref: int) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["prepared"] / "hmmix" / f"ref.af{n_af_ref}"


def get_hmmix_run_dir(base_dir: str, sim_name: str, n_af_ref: int) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["runs"] / "hmmix" / f"ref.af{n_af_ref}"


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def check_requirements() -> None:
    if shutil.which("hmmix") is None:
        raise EnvironmentError("Missing required tool: hmmix")


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
    """
    Evaluate archaic vs modern only.

    True archaic states:
      ND_EU, ND_NA
    Pred archaic:
      any interval in inf_df
    """
    if true_df is None:
        true_df = pd.DataFrame(columns=["CHR", "Sample", "Start", "End", "State"])
    if inf_df is None:
        inf_df = pd.DataFrame(columns=["CHR", "Sample", "Start", "End"])

    true_arch = true_df[true_df["State"].isin(["ND_EU", "ND_NA"])].copy()
    inf_arch = inf_df.copy()

    true_dict = _build_interval_dict(true_arch)
    inf_dict = _build_interval_dict(inf_arch)

    pairs = set(true_dict.keys()) | set(inf_dict.keys())
    if not pairs and true_df is not None and len(true_df) > 0:
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
# HMMix parsing / running
# ==============================================================================

def decode_filename_to_sample_name(decoded_file: Path, suffix: str) -> str:
    base = decoded_file.name
    sample_part = base.replace("decoded.", "").replace(".txt", "")
    suffix_token = f".{suffix}"
    if suffix_token in sample_part:
        sample_part = sample_part.replace(suffix_token, "")
    if ".hap" in sample_part:
        prefix, hap = sample_part.split(".hap")
        ind_id = prefix.replace("tsk_", "")
        return f"MX_{ind_id}_{hap}"
    ind_id = sample_part.replace("tsk_", "")
    return f"MX_{ind_id}"


def parse_hmmix_decoded(
    decoded_file: Path,
    sample_name: str,
    min_prob: float = 0.9,
    use_viterbi: bool = False,
) -> pd.DataFrame:
    if not decoded_file.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(decoded_file, sep="\t")
    except Exception:
        return pd.DataFrame()
    if df.empty or "state" not in df.columns:
        return pd.DataFrame()

    if use_viterbi:
        archaic = df[df["state"] == "Archaic"].copy()
    else:
        if "mean_prob" in df.columns:
            archaic = df[(df["state"] == "Archaic") & (df["mean_prob"] >= min_prob)].copy()
        else:
            archaic = df[df["state"] == "Archaic"].copy()

    if archaic.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "CHR": archaic["chrom"].astype(int),
            "Sample": sample_name,
            "Start": archaic["start"].astype(int),
            "End": archaic["end"].astype(int),
            "Length": (archaic["end"] - archaic["start"]).astype(int),
        }
    )
    if "mean_prob" in archaic.columns:
        out["mean_prob"] = archaic["mean_prob"].astype(float)
    return out


def run_hmmix_inference(
    prepared_dir: Path,
    run_dir: Path,
    suffix: str,
    *,
    haploid: bool = True,
    use_viterbi: bool = False,
) -> None:
    prep_manifest = load_json(prepared_dir / "manifest.hmmix.json")
    shared = prep_manifest["shared_files"]

    individuals_json = prepared_dir / shared["individuals_json"]
    weights_bed = prepared_dir / shared["weights_bed"]
    outgroup_file = prepared_dir / shared["outgroup_file"]
    obs_prefix = prepared_dir / shared["obs_prefix"]
    mutrate_name = shared.get("mutationrate_file")
    mutrate_file = prepared_dir / mutrate_name if mutrate_name else None

    with open(individuals_json) as f:
        individuals = json.load(f)
    ingroup_names = individuals.get("ingroup", [])
    if not ingroup_names:
        raise ValueError(f"No ingroup individuals found in {individuals_json}")

    for ind_name in ingroup_names:
        obs_file = Path(f"{obs_prefix}.{ind_name}.txt")
        if not obs_file.exists():
            continue

        trained_file = run_dir / f"trained.{ind_name}.{suffix}.json"
        decoded_prefix = run_dir / f"decoded.{ind_name}.{suffix}"

        train_cmd = [
            "hmmix", "train",
            f"-obs={obs_file}",
            f"-weights={weights_bed}",
            f"-param=InitialGuess.json",
            f"-out={trained_file}",
        ]
        if mutrate_file is not None and mutrate_file.exists():
            train_cmd.append(f"-mutrates={mutrate_file}")
        if haploid:
            train_cmd.append("-haploid")
        run_command(train_cmd, f"hmmix train {ind_name}")

        decode_cmd = [
            "hmmix", "decode",
            f"-obs={obs_file}",
            f"-weights={weights_bed}",
            f"-param={trained_file}",
            f"-out={decoded_prefix}",
        ]
        if mutrate_file is not None and mutrate_file.exists():
            decode_cmd.append(f"-mutrates={mutrate_file}")
        if haploid:
            decode_cmd.append("-haploid")
        if use_viterbi:
            decode_cmd.append("-viterbi")
        run_command(decode_cmd, f"hmmix decode {ind_name}")


def evaluate_hmmix_outputs(
    *,
    raw_dir: Path,
    run_dir: Path,
    suffix: str,
    chrom_length: int,
    thresholds: List[float],
    use_viterbi: bool = False,
) -> Dict:
    truth_path = raw_dir / "truth.all.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file missing: {truth_path}")

    true_df = pd.read_csv(truth_path, sep="\t")
    decoded_files = sorted(run_dir.glob(f"decoded.*.{suffix}.hap*.txt"))
    if not decoded_files:
        decoded_files = sorted(run_dir.glob(f"decoded.*.{suffix}.txt"))
    if not decoded_files:
        raise FileNotFoundError(f"No decoded files found in {run_dir}")

    if use_viterbi:
        all_segments = []
        for df_file in decoded_files:
            sample_name = decode_filename_to_sample_name(df_file, suffix=suffix)
            df = parse_hmmix_decoded(df_file, sample_name, min_prob=0.0, use_viterbi=True)
            if not df.empty:
                all_segments.append(df)

        if all_segments:
            df_inf = pd.concat(all_segments, ignore_index=True)
        else:
            df_inf = pd.DataFrame(columns=["CHR", "Sample", "Start", "End", "Length"])

        out_file = run_dir / f"all.inferred.hmmix.{suffix}.viterbi.tsv"
        df_inf.to_csv(out_file, sep="\t", index=False)
        metrics = calculate_binary_metrics(true_df, df_inf, chrom_length)

        return {
            "method": "viterbi",
            "segments": int(len(df_inf)),
            "precision": metrics["Archaic"]["Precision"],
            "recall": metrics["Archaic"]["Recall"],
            "f1": metrics["Archaic"]["F1"],
            "metrics": metrics,
            "output_file": out_file.name,
        }

    all_raw = []
    for df_file in decoded_files:
        sample_name = decode_filename_to_sample_name(df_file, suffix=suffix)
        df = parse_hmmix_decoded(df_file, sample_name, min_prob=0.0, use_viterbi=False)
        if not df.empty:
            all_raw.append(df)

    if all_raw:
        df_raw = pd.concat(all_raw, ignore_index=True)
    else:
        df_raw = pd.DataFrame(columns=["CHR", "Sample", "Start", "End", "Length", "mean_prob"])

    results = {}
    pr_rows = []

    for thresh in thresholds:
        if "mean_prob" in df_raw.columns and len(df_raw) > 0:
            df_filt = df_raw[df_raw["mean_prob"] >= thresh].copy()
        else:
            df_filt = df_raw.copy()

        thr_str = f"{thresh:.2f}".replace(".", "_")
        out_file = run_dir / f"all.inferred.hmmix.{suffix}.thr{thr_str}.tsv"
        df_filt.to_csv(out_file, sep="\t", index=False)

        metrics = calculate_binary_metrics(true_df, df_filt, chrom_length)
        pr_rows.append(
            {
                "threshold": float(thresh),
                "precision": metrics["Archaic"]["Precision"],
                "recall": metrics["Archaic"]["Recall"],
                "f1": metrics["Archaic"]["F1"],
                "segments": int(len(df_filt)),
            }
        )
        results[str(thresh)] = {
            "segments": int(len(df_filt)),
            "precision": metrics["Archaic"]["Precision"],
            "recall": metrics["Archaic"]["Recall"],
            "f1": metrics["Archaic"]["F1"],
            "output_file": out_file.name,
        }

    pr_df = pd.DataFrame(pr_rows)
    pr_file = run_dir / f"hmmix_pr_curve.{suffix}.tsv"
    pr_df.to_csv(pr_file, sep="\t", index=False)

    return {
        "method": "posterior",
        "thresholds": thresholds,
        "results": results,
        "pr_curve_file": pr_file.name,
    }


def run_hmmix_pipeline(
    sim_name: str,
    *,
    base_dir: str,
    n_af_ref: int,
    thresholds: List[float],
    haploid: bool,
    use_viterbi: bool,
    skip_run: bool,
    force: bool,
) -> Path:
    check_requirements()

    dirs = get_project_dirs(base_dir, sim_name)
    raw_dir = dirs["raw"]
    raw_manifest = load_json(raw_dir / "manifest.json")
    chrom_length = int(raw_manifest["simulation"]["params"]["chrom_length"])

    prepared_dir = get_hmmix_prepare_dir(base_dir, sim_name, n_af_ref)
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared HMMix directory not found: {prepared_dir}")
    if not (prepared_dir / "manifest.hmmix.json").exists():
        raise FileNotFoundError(f"Prepared HMMix manifest not found in {prepared_dir}")

    run_dir = get_hmmix_run_dir(base_dir, sim_name, n_af_ref)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"ref.{n_af_ref}"

    if not skip_run:
        run_hmmix_inference(
            prepared_dir=prepared_dir,
            run_dir=run_dir,
            suffix=suffix,
            haploid=haploid,
            use_viterbi=use_viterbi,
        )

    eval_summary = evaluate_hmmix_outputs(
        raw_dir=raw_dir,
        run_dir=run_dir,
        suffix=suffix,
        chrom_length=chrom_length,
        thresholds=thresholds,
        use_viterbi=use_viterbi,
    )

    run_manifest = {
        "method": "hmmix",
        "sim_name": sim_name,
        "prepared_dir": str(prepared_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "reference_panel": {"n_af_ref": int(n_af_ref)},
        "settings": {
            "haploid": bool(haploid),
            "viterbi": bool(use_viterbi),
            "skip_run": bool(skip_run),
            "thresholds": list(thresholds),
        },
        "evaluation": eval_summary,
    }
    with open(run_dir / "manifest.run_hmmix.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 72)
    print("HMMix run completed")
    print(f"Run dir:     {run_dir}")
    print(f"Mode:        {'viterbi' if use_viterbi else 'posterior'}")
    if use_viterbi:
        print(f"F1:          {eval_summary['f1']}")
    else:
        print(f"PR curve:    {run_dir / eval_summary['pr_curve_file']}")
    print("=" * 72)

    return run_dir


def parse_thresholds(raw: str) -> List[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("No thresholds provided")
    return [float(x) for x in vals]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HMMix for prepared mex_compare inputs")
    parser.add_argument("--sim-name", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--n-af-ref", type=int, required=True)
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--viterbi", action="store_true")
    parser.add_argument("--diploid", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    thresholds = parse_thresholds(args.thresholds)

    run_hmmix_pipeline(
        sim_name=args.sim_name,
        base_dir=args.base_dir,
        n_af_ref=args.n_af_ref,
        thresholds=thresholds,
        haploid=not args.diploid,
        use_viterbi=args.viterbi,
        skip_run=args.skip_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
