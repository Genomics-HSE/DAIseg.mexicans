#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import tskit


DEFAULT_BASE_DIR = "."
DEFAULT_THREADS = 4


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


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def get_population_id(ts: tskit.TreeSequence, pop_name: str) -> int:
    for pop in ts.populations():
        if pop.metadata.get("name") == pop_name:
            return pop.id
    raise ValueError(f"Population '{pop_name}' not found")


def get_individual_ids_by_pop(ts: tskit.TreeSequence, pop_name: str) -> List[int]:
    pop_id = get_population_id(ts, pop_name)
    nodes = ts.samples(population=pop_id)
    inds = np.unique(ts.nodes_individual[nodes])
    inds = inds[inds != -1]
    return sorted(int(x) for x in inds)


def create_mask(out_dir: Path, chrom_name: str, chrom_len: int, file_name: str, value: float, w: int = 1000) -> Path:
    path = out_dir / file_name
    with open(path, "w") as f:
        for i in range(int(chrom_len / w)):
            f.write(f"{chrom_name}\t{i*w}\t{(i+1)*w}\t{value}\n")
    return path


def normalize_json_prefixes(json_paths: Sequence[Path], abs_prefix: Path) -> None:
    for j in json_paths:
        with open(j, "r") as f:
            d = json.load(f)
        if d.get("prefix") != str(abs_prefix):
            d["prefix"] = str(abs_prefix)
            if "gaps" in d:
                d["gaps"] = str(abs_prefix / Path(d["gaps"]).name)
            with open(j, "w") as f:
                json.dump(d, f, indent=4)


def generate_haplotype_table_mexicans(
    ts: tskit.TreeSequence,
    chrom_name: str,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    n_nd_ref: int,
) -> pd.DataFrame:
    """
    Create TSV for DAIseg.mexicans using selected modern and archaic references.

    Output columns:
      #CHROM, POS, REF, ALT, Ancestral, ND, EU, NA, AF, MX_0_1, MX_0_2, ...
    """
    G = ts.genotype_matrix()
    pos = ts.tables.sites.position.astype(int)
    samples = ts.samples()
    pop_map = {p.metadata["name"]: p.id for p in ts.populations()}
    '''
    def get_samples_for_pop(pop_name: str, n_keep: int | None = None) -> np.ndarray:
        pid = pop_map.get(pop_name)
        if pid is None:
            return np.array([], dtype=int)
        pop_samples = ts.samples(population=pid)
        if n_keep is not None and len(pop_samples) > n_keep:
            pop_samples = pop_samples[:n_keep]
        return pop_samples
    '''

    def get_samples_for_pop(pop_name: str, n_keep: int | None = None) -> np.ndarray:
        pid = pop_map.get(pop_name)
        if pid is None:
            return np.array([], dtype=int)
        pop_samples = ts.samples(population=pid)
        if n_keep is not None:
            n_keep_nodes = 2 * n_keep
            if len(pop_samples) > n_keep_nodes:
                pop_samples = pop_samples[:n_keep_nodes]
        return pop_samples

    i_mx = get_samples_for_pop("MX")
    i_eu = get_samples_for_pop("EU", n_eu_ref)
    i_na = get_samples_for_pop("NA", n_na_ref)
    i_af = get_samples_for_pop("AF", n_af_ref)
    i_nd = get_samples_for_pop("NEAND", n_nd_ref)

    if len(i_mx) == 0:
        return pd.DataFrame(columns=["#CHROM", "POS", "REF", "ALT", "Ancestral", "ND", "EU", "NA", "AF"])

    def has_val(indices: np.ndarray, val: int) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros(G.shape[0], dtype=bool)
        mask = np.isin(samples, indices)
        return np.any(G[:, mask] == val, axis=1)

    mx_has_0 = has_val(i_mx, 0)
    mx_has_1 = has_val(i_mx, 1)
    eu_has_0 = has_val(i_eu, 0)
    eu_has_1 = has_val(i_eu, 1)
    na_has_0 = has_val(i_na, 0)
    na_has_1 = has_val(i_na, 1)
    af_has_0 = has_val(i_af, 0)
    af_has_1 = has_val(i_af, 1)
    nd_has_0 = has_val(i_nd, 0)
    nd_has_1 = has_val(i_nd, 1)

    keep_1 = mx_has_1 & ((~eu_has_1) | (~na_has_1) | (~af_has_1) | (~nd_has_1))
    keep_0 = mx_has_0 & ((~eu_has_0) | (~na_has_0) | (~af_has_0) | (~nd_has_0))
    mask = keep_1 | keep_0

    G = G[mask]
    pos = pos[mask]
    n_vars = G.shape[0]

    if n_vars == 0:
        return pd.DataFrame(columns=["#CHROM", "POS", "REF", "ALT", "Ancestral", "ND", "EU", "NA", "AF"])

    bases = np.array(["A", "C", "G", "T"])
    ref_idx = np.random.randint(0, 4, size=n_vars)
    refs = bases[ref_idx]
    alts = bases[(ref_idx + np.random.randint(1, 4, size=n_vars)) % 4]

    def get_indices(subset_samples: np.ndarray) -> List[int]:
        return [int(np.where(samples == s)[0][0]) for s in subset_samples]

    idx_eu = get_indices(i_eu)
    idx_na = get_indices(i_na)
    idx_af = get_indices(i_af)
    idx_nd = get_indices(i_nd)

    def fmt_col(indices: List[int]) -> List[str]:
        if len(indices) == 0:
            return ["{}"] * n_vars
        sub = G[:, indices]
        h0 = np.any(sub == 0, axis=1)
        h1 = np.any(sub == 1, axis=1)
        res = []
        for i in range(n_vars):
            alleles = []
            if h0[i]:
                alleles.append(refs[i])
            if h1[i]:
                alleles.append(alts[i])
            alleles.sort()
            res.append("{" + ",".join(alleles) + "}")
        return res

    data = {
        "#CHROM": [str(chrom_name)] * n_vars,
        "POS": pos,
        "REF": refs,
        "ALT": alts,
        "Ancestral": refs,
        "ND": fmt_col(idx_nd),
        "EU": fmt_col(idx_eu),
        "NA": fmt_col(idx_na),
        "AF": fmt_col(idx_af),
    }

    n_ind = ts.nodes_individual
    all_s = ts.samples()
    mx_map: Dict[int, List[int]] = {}
    for n in i_mx:
        ind = int(n_ind[n])
        if ind != -1:
            mx_map.setdefault(ind, []).append(int(n))

    for ind in sorted(mx_map.keys()):
        nodes = mx_map[ind]
        c1 = int(np.where(all_s == nodes[0])[0][0])
        data[f"MX_{ind}_1"] = np.where(G[:, c1] == 0, refs, alts)
        if len(nodes) > 1:
            c2 = int(np.where(all_s == nodes[1])[0][0])
            data[f"MX_{ind}_2"] = np.where(G[:, c2] == 0, refs, alts)
        else:
            data[f"MX_{ind}_2"] = ["."] * n_vars

    return pd.DataFrame(data)


def make_config_mexicans(
    ts_path: str,
    seed: int,
    out_dir: str,
    chrom_len: int,
    gen_time: float,
    recomb_rate: float,
    mut_rate: float,
    t_nd_migration: float,
    t_neanderthal_split: float,
    t_africa_split: float,
    t_eu_asia_split: float,
    t_mexican_admixture: float,
    admixture_nd: float,
    admixture_modern: Sequence[float],
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    n_nd_ref: int,
) -> str:
    ts = tskit.load(ts_path)
    out_dir_p = Path(out_dir)

    postfix = f"{seed}_eu{n_eu_ref}_na{n_na_ref}_af{n_af_ref}_nd{n_nd_ref}"
    tsv_name = f"variants.mexicans_seed_{postfix}.tsv"
    gaps_name = f"gaps.mexicans_seed_{postfix}.txt"
    mask_1kg_name = f"mask.1kg.mexicans_seed_{postfix}.bed"
    mask_nd_name = f"mask.nd.mexicans_seed_{postfix}.bed"
    out_base = f"inferred.mexicans_seed_{postfix}"

    df_v = generate_haplotype_table_mexicans(
        ts,
        str(seed),
        n_eu_ref=n_eu_ref,
        n_na_ref=n_na_ref,
        n_af_ref=n_af_ref,
        n_nd_ref=n_nd_ref,
    )
    df_v.to_csv(out_dir_p / tsv_name, sep="\t", index=False)

    (out_dir_p / gaps_name).write_text("", encoding="utf-8")
    create_mask(out_dir_p, str(seed), chrom_len, mask_1kg_name, 1.0)
    create_mask(out_dir_p, str(seed), chrom_len, mask_nd_name, 1.0 if n_nd_ref > 0 else 0.0)

    raw_af_ids = get_individual_ids_by_pop(ts, "AF")[:n_af_ref]
    raw_na_ids = get_individual_ids_by_pop(ts, "NA")[:n_na_ref]
    raw_eu_ids = get_individual_ids_by_pop(ts, "EU")[:n_eu_ref]
    raw_mx_ids = get_individual_ids_by_pop(ts, "MX")
    raw_nd_ids = get_individual_ids_by_pop(ts, "NEAND")[:n_nd_ref]

    cfg = {
        "data": tsv_name,
        "description": "Simulated Data for DAIseg.mexicans",
        "CHROM": str(seed),
        "prefix": str(out_dir_p),
        "output": out_base,
        "gaps": str(out_dir_p / gaps_name),
        "window_callability": {
            "Thousand_genomes": mask_1kg_name,
            "Nd_1k_genomes": mask_nd_name,
        },
        "samples": {
            "Africans": [f"AF_{i}" for i in raw_af_ids],
            "Americans": [f"NA_{i}" for i in raw_na_ids],
            "Europeans": [f"EU_{i}" for i in raw_eu_ids],
            "Mexicans": [f"MX_{i}" for i in raw_mx_ids],
            "neand": [f"NEAND_{i}" for i in raw_nd_ids],
        },
        "parameters_initial": {
            "admixture_nd": float(admixture_nd),
            "admixture_modern": list(admixture_modern),
            "introgression_time": int(t_nd_migration * gen_time),
            "rr": float(recomb_rate),
            "mutation": float(mut_rate),
            "window_length": 1000,
            "generation_time": int(gen_time),
            "t_n_c": int(t_neanderthal_split * gen_time),
            "t_af_c": int(t_africa_split * gen_time),
            "t_introgression_c": int(t_nd_migration * gen_time),
            "t_ea_c": int(t_eu_asia_split * gen_time),
            "t_mexicans_c": int(t_mexican_admixture * gen_time),
            "t_introgression": int(t_nd_migration * gen_time),
            "t_mexicans": int(t_mexican_admixture * gen_time),
        },
    }

    cfg_path = out_dir_p / f"config.mexicans_seed_{postfix}.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    return str(cfg_path)


def prepare_daiseg_mexicans_inputs(
    sim_name: str,
    *,
    base_dir: str = DEFAULT_BASE_DIR,
    n_eu_ref: int,
    n_na_ref: int,
    n_af_ref: int,
    n_nd_ref: int,
    threads: int = DEFAULT_THREADS,
    force: bool = False,
) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    raw_dir = dirs["raw"]
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Raw manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    sim_params = manifest["simulation"]["params"]
    sim_times = manifest["simulation"]["times"]
    execution = manifest["simulation"]["execution"]
    seeds = [int(x) for x in execution["seeds"]]

    out_dir = get_daiseg_mexicans_prepare_dir(
        base_dir,
        sim_name,
        n_eu_ref,
        n_na_ref,
        n_af_ref,
        n_nd_ref,
    )
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in seeds:
        ts_path = raw_dir / f"sim_seed_{seed}.trees"
        if not ts_path.exists():
            raise FileNotFoundError(f"Missing tree sequence: {ts_path}")
        jobs.append(
            {
                "ts_path": str(ts_path),
                "seed": int(seed),
                "out_dir": str(out_dir),
                "chrom_len": int(sim_params["chrom_length"]),
                "gen_time": float(sim_params["gen_time"]),
                "recomb_rate": float(sim_params["recomb_rate"]),
                "mut_rate": float(sim_params["mut_rate"]),
                "t_nd_migration": float(sim_times["t_nd_migration"]),
                "t_neanderthal_split": float(sim_times["t_neanderthal_split"]),
                "t_africa_split": float(sim_times["t_africa_split"]),
                "t_eu_asia_split": float(sim_times["t_eu_asia_split"]),
                "t_mexican_admixture": float(sim_times["t_mexican_admixture"]),
                "admixture_nd": float(sim_params["admixture_nd"]),
                "admixture_modern": list(sim_params["admixture_modern"]),
                "n_eu_ref": int(n_eu_ref),
                "n_na_ref": int(n_na_ref),
                "n_af_ref": int(n_af_ref),
                "n_nd_ref": int(n_nd_ref),
            }
        )

    json_paths: List[Path] = []
    if int(threads) <= 1:
        for job in jobs:
            cfg_path = make_config_mexicans(**job)
            json_paths.append(Path(cfg_path))
    else:
        with ProcessPoolExecutor(max_workers=int(threads)) as ex:
            futures = [ex.submit(make_config_mexicans, **job) for job in jobs]
            for fut in as_completed(futures):
                json_paths.append(Path(fut.result()))

    json_paths = sorted(json_paths)
    normalize_json_prefixes(json_paths, out_dir.resolve())

    prep_manifest = {
        "method": "daiseg_mexicans",
        "sim_name": sim_name,
        "source_raw_dir": str(raw_dir.resolve()),
        "prepared_dir": str(out_dir.resolve()),
        "reference_panel": {
            "n_eu_ref": int(n_eu_ref),
            "n_na_ref": int(n_na_ref),
            "n_af_ref": int(n_af_ref),
            "n_nd_ref": int(n_nd_ref),
        },
        "query_population": "MX",
        "settings": {
            "threads": int(threads),
        },
        "chromosomes": [
            {
                "chrom_seed": int(Path(p).stem.split("_seed_")[1].split("_")[0]),
                "config_json": Path(p).name,
            }
            for p in json_paths
        ],
    }

    with open(out_dir / "manifest.daiseg_mexicans.json", "w") as f:
        json.dump(prep_manifest, f, indent=2)

    print("=" * 72)
    print("DAIseg.mexicans preparation completed")
    print(f"Prepared dir:   {out_dir}")
    print(f"Reference EU:   {n_eu_ref}")
    print(f"Reference NA:   {n_na_ref}")
    print(f"Reference AF:   {n_af_ref}")
    print(f"Reference ND:   {n_nd_ref}")
    print(f"Chromosomes:    {len(json_paths)}")
    print("=" * 72)

    return out_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare DAIseg.mexicans inputs from mex_compare raw simulations"
    )
    parser.add_argument("--sim-name", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--n-eu-ref", type=int, required=True)
    parser.add_argument("--n-na-ref", type=int, required=True)
    parser.add_argument("--n-af-ref", type=int, required=True)
    parser.add_argument("--n-nd-ref", type=int, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    prepare_daiseg_mexicans_inputs(
        sim_name=args.sim_name,
        base_dir=args.base_dir,
        n_eu_ref=args.n_eu_ref,
        n_na_ref=args.n_na_ref,
        n_af_ref=args.n_af_ref,
        n_nd_ref=args.n_nd_ref,
        threads=args.threads,
        force=args.force,
    )


if __name__ == "__main__":
    main()
