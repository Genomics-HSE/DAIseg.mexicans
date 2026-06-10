#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_BASE_DIR = "."
DEFAULT_THREADS = 4



DAISEG_MEXICANS_BIN = "../daiseg.py"

def get_project_dirs(base_dir: str, sim_name: str) -> Dict[str, Path]:
    root = Path(base_dir) / sim_name
    return {
        "root": root,
        "raw": root / "raw",
        "prepared": root / "prepared",
        "runs": root / "runs",
        "metrics": root / "metrics",
    }


def get_daiseg_mexicans_prepare_dir(
    base_dir: str,
    sim_name: str,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    n_nd_ref: int,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    ref_tag = f"ref.eu{n_eu_ref}.na{n_na_ref}.af{n_af_ref}.nd{n_nd_ref}"
    return dirs["prepared"] / "daiseg_mexicans" / ref_tag


def get_daiseg_mexicans_run_dir(
    base_dir: str,
    sim_name: str,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    n_nd_ref: int,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    ref_tag = f"ref.eu{n_eu_ref}.na{n_na_ref}.af{n_af_ref}.nd{n_nd_ref}"
    return dirs["runs"] / "daiseg_mexicans" / ref_tag


def collect_config_files(prepared_dir: Path) -> List[Path]:
    cfgs = sorted(prepared_dir.glob("config.mexicans_seed_*.json"))
    if not cfgs:
        raise FileNotFoundError(f"No config files found in {prepared_dir}")
    return cfgs


def check_requirements() -> None:
    if not os.path.exists(DAISEG_MEXICANS_BIN):
        raise EnvironmentError(f"DAIseg.mexicans binary not found: {DAISEG_MEXICANS_BIN}")

def run_command(cmd: List[str], step_name: str, cwd: str | None = None) -> None:
    result = subprocess.run(cmd, 
                            #capture_output=True, 
                            text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"CWD: {cwd}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def normalize_output_df(output_file: Path) -> pd.DataFrame:
    df = pd.read_csv(output_file, sep="\t", keep_default_na=False)

    if "CHROM" not in df.columns:
        if "CHR" in df.columns:
            df["CHROM"] = df["CHR"]
        elif "Chrom" in df.columns:
            df["CHROM"] = df["Chrom"]

    if "Sample" in df.columns:
        df["Sample"] = df["Sample"].astype(str)
        df["Sample"] = df["Sample"].apply(lambda x: x if x.startswith("MX_") else f"MX_{x}")

    keep = [c for c in ["Sample", "CHROM", "Start", "End", "Length", "State"] if c in df.columns]
    df = df[keep].copy()
    df.to_csv(output_file, sep="\t", index=False)
    return df


def run_daiseg_mexicans_em(prepared_dir: Path, run_dir: Path, threads: int) -> Path:
    cfgs = collect_config_files(prepared_dir)
    out_file = run_dir / "all.inferred.daiseg_mexicans.em.tsv"

    cmd = [
        "python",
        DAISEG_MEXICANS_BIN,
        "run.with.EM",
        "-jsons",
        *[str(x) for x in cfgs],
        "-out",
        str(out_file),
    ]
    if threads is not None and int(threads) > 0:
        cmd.extend(["-threads", str(int(threads))])

    run_command(cmd, "DAIseg.mexicans run.with.EM")
    return out_file


def run_daiseg_mexicans_pipeline(
    sim_name: str,
    *,
    base_dir: str,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    n_nd_ref: int,
    threads: int,
    skip_run: bool,
    force: bool,
) -> Path:
    check_requirements()

    prepared_dir = get_daiseg_mexicans_prepare_dir(
        base_dir, sim_name, n_eu_ref, n_na_ref, n_af_ref, n_nd_ref
    )
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared DAIseg.mexicans dir not found: {prepared_dir}")
    if not (prepared_dir / "manifest.daiseg_mexicans.json").exists():
        raise FileNotFoundError(f"Prepared DAIseg.mexicans manifest not found in {prepared_dir}")

    run_dir = get_daiseg_mexicans_run_dir(
        base_dir, sim_name, n_eu_ref, n_na_ref, n_af_ref, n_nd_ref
    )
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    out_file = run_dir / "all.inferred.daiseg_mexicans.em.tsv"
    if not skip_run:
        out_file = run_daiseg_mexicans_em(prepared_dir, run_dir, threads)
    else:
        if not out_file.exists():
            raise FileNotFoundError(f"--skip-run was requested, but output not found: {out_file}")

    df = normalize_output_df(out_file)

    run_manifest = {
        "method": "daiseg_mexicans",
        "sim_name": sim_name,
        "prepared_dir": str(prepared_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "reference_panel": {
            "n_eu_ref": int(n_eu_ref),
            "n_na_ref": int(n_na_ref),
            "n_af_ref": int(n_af_ref),
            "n_nd_ref": int(n_nd_ref),
        },
        "settings": {
            "threads": int(threads),
            "skip_run": bool(skip_run),
            "mode": "run.with.EM",
            "daiseg_bin": DAISEG_MEXICANS_BIN,
        },
        "output_files": {
            "combined_segments": out_file.name,
        },
        "rows": int(len(df)),
    }
    with open(run_dir / "manifest.run_daiseg_mexicans.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 72)
    print("DAIseg.mexicans run completed")
    print(f"Run dir:      {run_dir}")
    print(f"Output TSV:   {out_file}")
    print(f"Rows:         {len(df)}")
    print("=" * 72)

    return run_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DAIseg.mexicans for prepared mex_compare inputs")
    parser.add_argument("--sim-name", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--n-eu-ref", type=int, required=True)
    parser.add_argument("--n-na-ref", type=int, required=True)
    parser.add_argument("--n-af-ref", type=int, required=True)
    parser.add_argument("--n-nd-ref", type=int, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    run_daiseg_mexicans_pipeline(
        sim_name=args.sim_name,
        base_dir=args.base_dir,
        n_eu_ref=args.n_eu_ref,
        n_na_ref=args.n_na_ref,
        n_af_ref=args.n_af_ref,
        n_nd_ref=args.n_nd_ref,
        threads=args.threads,
        skip_run=args.skip_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
