#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ==============================================================================
# 0. DEFAULTS / CONFIG
# ==============================================================================

DEFAULT_BASE_DIR = "simulations.new"
DEFAULT_THREADS = 1
DEFAULT_JOBS = 4
DEFAULT_GENERATIONS = 15
RFMIX_BIN = os.path.expanduser("~/packages/rfmix_source/rfmix")


# ==============================================================================
# 1. PATHS
# ==============================================================================

def get_project_dirs(base_dir: str, sim_name: str) -> Dict[str, Path]:
    root = Path(base_dir) / sim_name
    return {
        "root": root,
        "raw": root / "raw",
        "prepared": root / "prepared",
        "runs": root / "runs",
        "metrics": root / "metrics",
    }


def get_rfmix_prepare_dir(
    base_dir: str,
    sim_name: str,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    ref_tag = f"ref.eu{n_eu_ref}.na{n_na_ref}.af{n_af_ref}"
    return dirs["prepared"] / "rfmix" / ref_tag


def get_rfmix_run_dir(
    base_dir: str,
    sim_name: str,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    ref_tag = f"ref.eu{n_eu_ref}.na{n_na_ref}.af{n_af_ref}"
    return dirs["runs"] / "rfmix" / ref_tag


# ==============================================================================
# 2. HELPERS
# ==============================================================================

def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def check_requirements() -> None:
    missing = []
    if not os.path.exists(RFMIX_BIN):
        missing.append(f"rfmix binary not found at {RFMIX_BIN}")
    if not shutil.which("bcftools"):
        missing.append("bcftools")
    if missing:
        raise EnvironmentError("Missing required tools for run_rfmix.py: " + ", ".join(missing))


def vcf_gz_to_bcf(vcf_gz: Path, out_bcf: Path) -> Path:
    subprocess.run(
        ["bcftools", "view", "-O", "b", str(vcf_gz), "-o", str(out_bcf)],
        check=True,
    )
    subprocess.run(
        ["bcftools", "index", "-f", str(out_bcf)],
        check=True,
    )
    return out_bcf


def run_rfmix_command(
    *,
    query_bcf: Path,
    ref_bcf: Path,
    sample_map: Path,
    genetic_map: Path,
    out_prefix: Path,
    generations: int,
    threads: int,
) -> Path:
    msp_output = Path(str(out_prefix) + ".msp.tsv")
    if msp_output.exists():
        msp_output.unlink()

    cmd = [
        RFMIX_BIN,
        "-f", str(query_bcf),
        "-r", str(ref_bcf),
        "-m", str(sample_map),
        "-g", str(genetic_map),
        "-o", str(out_prefix),
        "--chromosome=1",
        "-G", str(generations),
        "--n-threads=" + str(threads),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"RFMix failed for prefix {out_prefix} with exit code {result.returncode}")

    if not msp_output.exists():
        raise FileNotFoundError(f"RFMix finished but MSP output not found: {msp_output}")

    return msp_output


def parse_msp_file(msp_file: Path, chrom_seed: int) -> pd.DataFrame:
    """
    Parse one .msp.tsv into standard segment table.

    Output columns:
      CHROM, Sample, Start, End, Length, State
    """
    try:
        df = pd.read_csv(msp_file, sep="\t", skiprows=1)
        with open(msp_file, "r") as f:
            header = f.readline().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read MSP file {msp_file}: {e}") from e

    if ": " not in header:
        raise ValueError(f"Unexpected MSP header format in {msp_file}: {header}")

    pop_order = header.split(": ", 1)[1].split("\t")
    pop_order_clean = [p.split("=")[0] for p in pop_order]

    hap_cols = list(df.columns[6:])
    if not hap_cols:
        return pd.DataFrame(columns=["CHROM", "Sample", "Start", "End", "Length", "State"])

    out_rows = []
    starts = df["spos"].astype(int).values
    ends = df["epos"].astype(int).values

    for col in hap_cols:
        raw_name, hap_idx = col.split(".")
        ind_id = raw_name.replace("tsk_", "")
        sample_name = f"MX_{ind_id}_{int(hap_idx) + 1}"

        states = df[col].astype(int).values
        curr_state = states[0]
        curr_start = int(starts[0])

        for i in range(1, len(states)):
            if states[i] != curr_state:
                seg_end = int(ends[i - 1])
                out_rows.append(
                    {
                        "CHROM": int(chrom_seed),
                        "Sample": sample_name,
                        "Start": curr_start,
                        "End": seg_end,
                        "Length": seg_end - curr_start,
                        "State": pop_order_clean[curr_state],
                    }
                )
                curr_state = states[i]
                curr_start = int(starts[i])

        final_end = int(ends[-1])
        out_rows.append(
            {
                "CHROM": int(chrom_seed),
                "Sample": sample_name,
                "Start": curr_start,
                "End": final_end,
                "Length": final_end - curr_start,
                "State": pop_order_clean[curr_state],
            }
        )

    result = pd.DataFrame(out_rows)
    if not result.empty:
        result = result.sort_values(["Sample", "Start", "End"]).reset_index(drop=True)
    return result


# ==============================================================================
# 3. ONE-CHROMOSOME WORKER
# ==============================================================================

def run_one_chromosome(
    *,
    chrom_seed: int,
    query_vcf_gz: str,
    ref_vcf_gz: str,
    sample_map: str,
    genetic_map: str,
    run_dir: str,
    generations: int,
    threads: int,
) -> Dict:
    """
    Worker for one chromosome.
    """
    run_dir_p = Path(run_dir)

    query_bcf = run_dir_p / f"chr{chrom_seed}.query.bcf"
    ref_bcf = run_dir_p / f"chr{chrom_seed}.ref.bcf"

    vcf_gz_to_bcf(Path(query_vcf_gz), query_bcf)
    vcf_gz_to_bcf(Path(ref_vcf_gz), ref_bcf)

    out_prefix = run_dir_p / f"rfmix_{chrom_seed}"
    msp_file = run_rfmix_command(
        query_bcf=query_bcf,
        ref_bcf=ref_bcf,
        sample_map=Path(sample_map),
        genetic_map=Path(genetic_map),
        out_prefix=out_prefix,
        generations=generations,
        threads=threads,
    )

    df = parse_msp_file(msp_file, chrom_seed=chrom_seed)
    per_chr_out = run_dir_p / f"rfmix_{chrom_seed}.tsv"
    df.to_csv(per_chr_out, sep="\t", index=False)

    return {
        "chrom_seed": int(chrom_seed),
        "msp_file": msp_file.name,
        "segments_file": per_chr_out.name,
        "n_segments": int(len(df)),
        "df": df,
    }


# ==============================================================================
# 4. MAIN RUN LOGIC
# ==============================================================================

def run_rfmix_for_prepared_inputs(
    sim_name: str,
    *,
    base_dir: str,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    generations: int,
    threads: int,
    jobs: int,
    force: bool,
) -> Path:
    check_requirements()

    prepare_dir = get_rfmix_prepare_dir(base_dir, sim_name, n_eu_ref, n_na_ref, n_af_ref)
    if not prepare_dir.exists():
        raise FileNotFoundError(f"Prepared RFMix directory not found: {prepare_dir}")

    prep_manifest_path = prepare_dir / "manifest.rfmix.json"
    if not prep_manifest_path.exists():
        raise FileNotFoundError(f"Prepared RFMix manifest not found: {prep_manifest_path}")

    prep_manifest = load_json(prep_manifest_path)
    chromosomes = prep_manifest.get("chromosomes", [])
    if not chromosomes:
        raise ValueError(f"No chromosome entries in {prep_manifest_path}")

    run_dir = get_rfmix_run_dir(base_dir, sim_name, n_eu_ref, n_na_ref, n_af_ref)
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    genetic_map = str(prepare_dir / prep_manifest["shared_files"]["genetic_map"])
    sample_map = str(prepare_dir / prep_manifest["shared_files"]["sample_map"])

    all_segments = []
    run_records = []

    if int(jobs) <= 1:
        for entry in chromosomes:
            result = run_one_chromosome(
                chrom_seed=int(entry["chrom_seed"]),
                query_vcf_gz=str(prepare_dir / entry["query_vcf"]),
                ref_vcf_gz=str(prepare_dir / entry["ref_vcf"]),
                sample_map=sample_map,
                genetic_map=genetic_map,
                run_dir=str(run_dir),
                generations=int(generations),
                threads=int(threads),
            )
            all_segments.append(result.pop("df"))
            run_records.append(result)
    else:
        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            futures = []
            for entry in chromosomes:
                futures.append(
                    ex.submit(
                        run_one_chromosome,
                        chrom_seed=int(entry["chrom_seed"]),
                        query_vcf_gz=str(prepare_dir / entry["query_vcf"]),
                        ref_vcf_gz=str(prepare_dir / entry["ref_vcf"]),
                        sample_map=sample_map,
                        genetic_map=genetic_map,
                        run_dir=str(run_dir),
                        generations=int(generations),
                        threads=int(threads),
                    )
                )

            for fut in as_completed(futures):
                result = fut.result()
                all_segments.append(result.pop("df"))
                run_records.append(result)

        run_records.sort(key=lambda x: x["chrom_seed"])

    if all_segments:
        full_df = pd.concat(all_segments, ignore_index=True)
        full_df = full_df.sort_values(["CHROM", "Sample", "Start", "End"]).reset_index(drop=True)
    else:
        full_df = pd.DataFrame(columns=["CHROM", "Sample", "Start", "End", "Length", "State"])

    full_out = run_dir / "rfmix.all.tsv"
    full_df.to_csv(full_out, sep="\t", index=False)

    run_manifest = {
        "method": "rfmix",
        "sim_name": sim_name,
        "prepared_dir": str(prepare_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "reference_panel": {
            "n_eu_ref": int(n_eu_ref),
            "n_na_ref": int(n_na_ref),
            "n_af_ref": int(n_af_ref),
        },
        "run_settings": {
            "generations": int(generations),
            "threads_per_rfmix_run": int(threads),
            "jobs": int(jobs),
        },
        "output_files": {
            "combined_segments": full_out.name,
        },
        "chromosomes": run_records,
    }

    with open(run_dir / "manifest.run_rfmix.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 72)
    print("RFMix run completed")
    print(f"Run dir:      {run_dir}")
    print(f"Chromosomes:  {len(run_records)}")
    print(f"Jobs:         {jobs}")
    print(f"Threads/run:  {threads}")
    print(f"Combined TSV: {full_out}")
    print("=" * 72)

    return run_dir


# ==============================================================================
# 5. CLI
# ==============================================================================

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RFMix for prepared mex_compare inputs")
    parser.add_argument("--sim-name", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)

    parser.add_argument("--n-eu-ref", type=int, required=True)
    parser.add_argument("--n-na-ref", type=int, required=True)
    parser.add_argument("--n-af-ref", type=int, required=True)

    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="How many chromosomes to run in parallel")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="Threads per single RFMix run")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    run_rfmix_for_prepared_inputs(
        sim_name=args.sim_name,
        base_dir=args.base_dir,
        n_eu_ref=args.n_eu_ref,
        n_na_ref=args.n_na_ref,
        n_af_ref=args.n_af_ref,
        generations=args.generations,
        threads=args.threads,
        jobs=args.jobs,
        force=args.force,
    )


if __name__ == "__main__":
    main()
