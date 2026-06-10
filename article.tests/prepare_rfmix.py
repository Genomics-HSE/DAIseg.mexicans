#!/usr/bin/env python3
"""
Prepare RFMix input files from existing mex_compare raw simulations.

Expected layout
---------------
simulations.new/<sim_name>/
├── raw/
│   ├── manifest.json
│   ├── sim_seed_1234.trees
│   └── ...
├── prepared/
│   └── rfmix/
│       └── ref.eu50.na50.af50/
│           ├── genetic_map.txt
│           ├── manifest.rfmix.json
│           ├── classes.txt
│           ├── chr1234.query.vcf.gz
│           ├── chr1234.ref.vcf.gz
│           └── ...
├── runs/
└── metrics/

This script only prepares inputs.
It does NOT run RFMix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tskit


# ==============================================================================
# 0. DEFAULTS
# ==============================================================================

DEFAULT_BASE_DIR = "simulations.new"
DEFAULT_THREADS = 4
DEFAULT_QUERY_POP = "MX"
DEFAULT_REF_POPS = ("EU", "NA", "AF")


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


# ==============================================================================
# 2. HELPERS
# ==============================================================================

def load_manifest(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def check_requirements() -> None:
    missing = []
    if not shutil.which("bcftools"):
        missing.append("bcftools")
    if not shutil.which("bgzip"):
        missing.append("bgzip")
    if missing:
        raise EnvironmentError(
            "Missing required tools for prepare_rfmix.py: " + ", ".join(missing)
        )


def get_population_id(ts: tskit.TreeSequence, pop_name: str) -> int:
    for pop in ts.populations():
        if pop.metadata.get("name") == pop_name:
            return pop.id
    raise ValueError(f"Population '{pop_name}' not found in tree sequence")


def get_individual_ids(ts: tskit.TreeSequence, pop_name: str) -> List[int]:
    pop_id = get_population_id(ts, pop_name)
    nodes = ts.samples(population=pop_id)
    inds = np.unique(ts.nodes_individual[nodes])
    inds = inds[inds != -1]
    return sorted(int(x) for x in inds)


def select_individual_ids(ts: tskit.TreeSequence, pop_name: str, n_keep: int) -> List[int]:
    ids = get_individual_ids(ts, pop_name)
    if n_keep < 1:
        raise ValueError(f"Requested {n_keep} individuals for {pop_name}, but n_keep must be >= 1")
    if n_keep > len(ids):
        raise ValueError(
            f"Requested {n_keep} individuals for {pop_name}, but only {len(ids)} are available"
        )
    return ids[:n_keep]


def create_genetic_map(
    chrom_len: int,
    recomb_rate: float,
    out_path: Path,
    step: int = 10_000,
) -> Path:
    with open(out_path, "w") as f:
        for pos in range(0, chrom_len + step, step):
            cm_pos = pos * recomb_rate * 100.0
            f.write(f"1\t{pos}\t{cm_pos:.6f}\n")
        final_pos = chrom_len + 1_000_000
        f.write(f"1\t{final_pos}\t{final_pos * recomb_rate * 100.0:.6f}\n")
    return out_path


def create_sample_map_file(
    eu_ids: Sequence[int],
    na_ids: Sequence[int],
    af_ids: Sequence[int],
    out_path: Path,
) -> Path:
    lines = []
    for ind in eu_ids:
        lines.append(f"tsk_{ind}\tEU")
    for ind in na_ids:
        lines.append(f"tsk_{ind}\tNA")
    for ind in af_ids:
        lines.append(f"tsk_{ind}\tAF")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return out_path


def write_vcf(
    ts: tskit.TreeSequence,
    out_vcf: Path,
    individual_ids: Sequence[int],
    chrom_id: int,
) -> Path:
    with open(out_vcf, "w") as f:
        ts.write_vcf(
            f,
            individuals=list(individual_ids),
            contig_id="1",
            position_transform=lambda x: np.array(x, dtype=int) + 1,
        )
    return out_vcf


def bgzip_and_index(vcf_path: Path) -> Path:
    gz_path = Path(str(vcf_path) + ".gz")
    subprocess.run(["bgzip", "-f", str(vcf_path)], check=True)
    subprocess.run(["bcftools", "index", "-f", str(gz_path)], check=True)
    return gz_path


def make_query_and_ref_vcfs(
    ts: tskit.TreeSequence,
    out_dir: Path,
    chrom_seed: int,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    query_pop: str = "MX",
) -> Tuple[Path, Path, Path, Dict]:
    query_ids = get_individual_ids(ts, query_pop)
    eu_ids = select_individual_ids(ts, "EU", n_eu_ref)
    na_ids = select_individual_ids(ts, "NA", n_na_ref)
    af_ids = select_individual_ids(ts, "AF", n_af_ref)

    if not query_ids:
        raise ValueError(f"No query individuals found for population '{query_pop}'")

    ref_ids = sorted(set(eu_ids + na_ids + af_ids))

    query_vcf = out_dir / f"chr{chrom_seed}.query.vcf"
    ref_vcf = out_dir / f"chr{chrom_seed}.ref.vcf"

    write_vcf(ts, query_vcf, query_ids, chrom_seed)
    write_vcf(ts, ref_vcf, ref_ids, chrom_seed)

    query_vcf_gz = bgzip_and_index(query_vcf)
    ref_vcf_gz = bgzip_and_index(ref_vcf)

    sample_map = out_dir / "classes.txt"
    create_sample_map_file(eu_ids, na_ids, af_ids, sample_map)

    meta = {
        "chrom_seed": int(chrom_seed),
        "query_pop": query_pop,
        "n_query": len(query_ids),
        "n_eu_ref": len(eu_ids),
        "n_na_ref": len(na_ids),
        "n_af_ref": len(af_ids),
        "query_vcf": query_vcf_gz.name,
        "ref_vcf": ref_vcf_gz.name,
    }
    return query_vcf_gz, ref_vcf_gz, sample_map, meta


# ==============================================================================
# 3. MAIN PREPARE LOGIC
# ==============================================================================

def prepare_rfmix_inputs(
    sim_name: str,
    *,
    base_dir: str = DEFAULT_BASE_DIR,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    threads: int = DEFAULT_THREADS,
    force: bool = False,
) -> Path:
    check_requirements()

    dirs = get_project_dirs(base_dir, sim_name)
    raw_dir = dirs["raw"]
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Raw manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    sim_params = manifest["simulation"]["params"]
    execution = manifest["simulation"]["execution"]
    chrom_len = int(sim_params["chrom_length"])
    recomb_rate = float(sim_params["recomb_rate"])
    seeds = [int(x) for x in execution["seeds"]]

    out_dir = get_rfmix_prepare_dir(base_dir, sim_name, n_eu_ref, n_na_ref, n_af_ref)
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genetic_map = out_dir / "genetic_map.txt"
    create_genetic_map(chrom_len, recomb_rate, genetic_map)

    chromosome_entries = []
    for seed in seeds:
        ts_path = raw_dir / f"sim_seed_{seed}.trees"
        if not ts_path.exists():
            raise FileNotFoundError(f"Missing tree sequence: {ts_path}")

        ts = tskit.load(str(ts_path))
        query_vcf_gz, ref_vcf_gz, sample_map, meta = make_query_and_ref_vcfs(
            ts=ts,
            out_dir=out_dir,
            chrom_seed=seed,
            n_eu_ref=n_eu_ref,
            n_na_ref=n_na_ref,
            n_af_ref=n_af_ref,
            query_pop=DEFAULT_QUERY_POP,
        )
        chromosome_entries.append(meta)

    prep_manifest = {
        "method": "rfmix",
        "sim_name": sim_name,
        "source_raw_dir": str(raw_dir.resolve()),
        "prepared_dir": str(out_dir.resolve()),
        "reference_panel": {
            "n_eu_ref": int(n_eu_ref),
            "n_na_ref": int(n_na_ref),
            "n_af_ref": int(n_af_ref),
        },
        "query_population": DEFAULT_QUERY_POP,
        "reference_populations": list(DEFAULT_REF_POPS),
        "simulation_params": {
            "chrom_length": chrom_len,
            "recomb_rate": recomb_rate,
        },
        "execution_defaults": {
            "threads": int(threads),
            "generations": 15,
        },
        "shared_files": {
            "genetic_map": genetic_map.name,
            "sample_map": sample_map.name,
        },
        "chromosomes": chromosome_entries,
    }

    manifest_out = out_dir / "manifest.rfmix.json"
    with open(manifest_out, "w") as f:
        json.dump(prep_manifest, f, indent=2)

    print("=" * 72)
    print("RFMix preparation completed")
    print(f"Prepared dir: {out_dir}")
    print(f"Genetic map:  {genetic_map}")
    print(f"Sample map:   {sample_map}")
    print(f"Chromosomes:  {len(chromosome_entries)}")
    print("=" * 72)

    return out_dir


# ==============================================================================
# 4. CLI
# ==============================================================================

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare RFMix inputs from mex_compare raw simulations"
    )
    parser.add_argument("--sim-name", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)

    parser.add_argument("--n-eu-ref", type=int, required=True)
    parser.add_argument("--n-na-ref", type=int, required=True)
    parser.add_argument("--n-af-ref", type=int, required=True)

    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    prepare_rfmix_inputs(
        sim_name=args.sim_name,
        base_dir=args.base_dir,
        n_eu_ref=args.n_eu_ref,
        n_na_ref=args.n_na_ref,
        n_af_ref=args.n_af_ref,
        threads=args.threads,
        force=args.force,
    )


if __name__ == "__main__":
    main()
