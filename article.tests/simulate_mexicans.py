#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import demes
import msprime
import numpy as np
import pandas as pd
import tskit
#from joblib import Parallel, delayed

from concurrent.futures import ProcessPoolExecutor, as_completed


DEFAULT_SIM_CONFIG: Dict = {
    # paths
    "base_dir": ".",
    "yaml": "mexicans.demography.yml",

    # simulation model
    "chrom_length": 2e7,
    "recomb_rate": 1e-8,
    "mut_rate": 1.25e-8,
    "ploidy": 2,
    "n_mexicans": 1,
    "n_eu": 250,
    "n_na": 250,
    "n_af": 250,
    "n_nd": 3,

    # execution
    "n_chr": 10,
    "base_seed": 12345,
    "n_jobs": 10,
}


# ==============================================================================
# 1. PATHS / LAYOUT
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


def ensure_project_layout(base_dir: str, sim_name: str, clean: bool = False) -> Dict[str, Path]:
    dirs = get_project_dirs(base_dir, sim_name)

    if clean and dirs["root"].exists():
        shutil.rmtree(dirs["root"])

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ==============================================================================
# 2. DEMES -> PARAMETER EXTRACTION
# ==============================================================================

def extract_parameters_from_demes(yaml_file: str) -> Tuple[Dict, Dict]:
    """
    Extract stable model values from demes YAML.

    """
    graph = demes.load(yaml_file)
    gen_time = float(graph.generation_time)

    def get_start_time_generations(deme_name: str) -> float:
        if deme_name not in graph:
            return 0.0
        return float(graph[deme_name].start_time) / gen_time

    times = {
        "t_mexican_admixture": get_start_time_generations("MX"),
        "t_neanderthal_split": get_start_time_generations("NEAND"),
        "t_africa_split": get_start_time_generations("OOA"),
        "t_eu_asia_split": get_start_time_generations("ASIA"),
    }

    pulse_time_years = None
    pulse_prop = None
    for pulse in graph.pulses:
        sources = [s if isinstance(s, str) else s.name for s in pulse.sources]
        if "NEAND" in sources:
            pulse_time_years = float(pulse.time)
            pulse_prop = float(pulse.proportions[0])
            break

    if pulse_time_years is None or pulse_prop is None:
        raise ValueError("Could not find NEAND pulse in demes YAML")

    times["t_nd_migration"] = pulse_time_years / gen_time
    times["t_nd_samples"] = pulse_time_years / gen_time

    yaml_params = {
        "gen_time": gen_time,
        "admixture_nd": pulse_prop,
        "admixture_modern": list(graph["MX"].proportions) if "MX" in graph else [],
    }
    return yaml_params, times


def build_simulation_params(
    yaml_file: str,
    *,
    chrom_length: float,
    recomb_rate: float,
    mut_rate: float,
    ploidy: int,
    n_mexicans: int,
    n_eu: int,
    n_na: int,
    n_af: int,
    n_nd: int,
) -> Tuple[Dict, Dict]:
    """
    Build simulation parameters and event times.
    """
    yaml_params, times = extract_parameters_from_demes(yaml_file)

    params = {
        "yaml_file": str(Path(yaml_file).resolve()),
        "chrom_length": float(chrom_length),
        "recomb_rate": float(recomb_rate),
        "mut_rate": float(mut_rate),
        "ploidy": int(ploidy),
        "n_mexicans": int(n_mexicans),
        "n_eu": int(n_eu),
        "n_na": int(n_na),
        "n_af": int(n_af),
        "n_nd": int(n_nd),
        "gen_time": float(yaml_params["gen_time"]),
        "admixture_nd": float(yaml_params["admixture_nd"]),
        "admixture_modern": list(yaml_params["admixture_modern"]),
    }
    return params, times


def build_execution_config(
    *,
    n_chr: int,
    base_seed: int,
    n_jobs: int,
) -> Dict:
    seeds = [int(base_seed) + i for i in range(int(n_chr))]
    return {
        "n_chr": int(n_chr),
        "base_seed": int(base_seed),
        "n_jobs": int(n_jobs),
        "seeds": seeds,
    }


# ==============================================================================
# 3. SIMULATION
# ==============================================================================

def simulate_one_tree_sequence(
    yaml_file: str,
    params: Dict,
    times: Dict,
    seed: int,
) -> tskit.TreeSequence:
    """
    Simulate one chromosome and add mutations.
    """
    graph = demes.load(yaml_file)
    demography = msprime.Demography.from_demes(graph)
    
    print(params["n_nd"])
    ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(params["n_mexicans"], ploidy=params["ploidy"], population="MX"),
            msprime.SampleSet(params["n_eu"], ploidy=params["ploidy"], population="EU"),
            msprime.SampleSet(params["n_na"], ploidy=params["ploidy"], population="NA"),
            msprime.SampleSet(params["n_af"], ploidy=params["ploidy"], population="AF"),
            msprime.SampleSet(
                params["n_nd"],
                ploidy=params["ploidy"],
                population="NEAND",
                time=times["t_nd_samples"],
            ),
        ],
        sequence_length=params["chrom_length"],
        recombination_rate=params["recomb_rate"],
        demography=demography,
        random_seed=int(seed),
        record_migrations=True,
    )

    ts = msprime.sim_mutations(
        ts,
        rate=params["mut_rate"],
        random_seed=int(seed),
    )
    return ts


# ==============================================================================
# 4. TRUTH EXTRACTION
# ==============================================================================

def _get_population_id(ts: tskit.TreeSequence, pop_name: str) -> int:
    for pop in ts.populations():
        if pop.metadata.get("name") == pop_name:
            return pop.id
    raise ValueError(f"Population '{pop_name}' not found")


def intersect_and_subtract(
    base_tracts: List[List[float]],
    mask_tracts: List[List[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Returns:
      intersection(base, mask), base \\ mask
    """
    intersection: List[List[float]] = []
    for b in base_tracts:
        for m in mask_tracts:
            start = max(b[0], m[0])
            end = min(b[1], m[1])
            if start < end:
                intersection.append([start, end])

    difference = [list(x) for x in base_tracts]
    for m_start, m_end in mask_tracts:
        next_fragments = []
        for f_start, f_end in difference:
            if m_end <= f_start or m_start >= f_end:
                next_fragments.append([f_start, f_end])
            else:
                if f_start < m_start:
                    next_fragments.append([f_start, m_start])
                if f_end > m_end:
                    next_fragments.append([m_end, f_end])
        difference = next_fragments

    intersection.sort()
    difference.sort()
    return intersection, difference


def get_migrating_tracts_ind(
    ts: tskit.TreeSequence,
    pop_name: str,
    ind_node: int,
    migration_time: float,
    eps: float = 1e-5,
) -> List[List[float]]:
    """
    Get migration tracts for a single haploid node at one target migration time.
    """
    pop_id = _get_population_id(ts, pop_name)

    tables = ts.tables
    mask = (
        (tables.migrations.dest == pop_id)
        & (tables.migrations.time >= migration_time - eps)
        & (tables.migrations.time <= migration_time + eps)
    )
    idx = np.where(mask)[0]

    mig_lookup: Dict[int, List[Tuple[float, float]]] = {}
    for i in idx:
        node_id = int(tables.migrations.node[i])
        left = float(tables.migrations.left[i])
        right = float(tables.migrations.right[i])
        mig_lookup.setdefault(node_id, []).append((left, right))

    for node_id in mig_lookup:
        mig_lookup[node_id].sort()

    node_times = ts.nodes_time
    tracts: List[List[float]] = []

    for tree in ts.trees():
        if tree.interval.left == tree.interval.right:
            continue

        anc = ind_node
        if node_times[anc] > migration_time + eps:
            continue

        parent = tree.parent(anc)
        while parent != tskit.NULL:
            if anc in mig_lookup:
                for mig_left, mig_right in mig_lookup[anc]:
                    start = max(tree.interval.left, mig_left)
                    end = min(tree.interval.right, mig_right)
                    if start < end:
                        if tracts and tracts[-1][1] == start:
                            tracts[-1][1] = end
                        else:
                            tracts.append([start, end])

            if node_times[anc] >= migration_time - eps:
                break

            anc = parent
            parent = tree.parent(anc)

    return tracts


def get_5state_truth_dataframe(
    ts: tskit.TreeSequence,
    times: Dict,
    *,
    mx_pop: str = "MX",
    neand_pop: str = "NEAND",
) -> pd.DataFrame:
    """
    Build 5-state truth with labels:
      EU, ND_EU, NA, ND_NA, AF
    """
    mx_pop_id = _get_population_id(ts, mx_pop)
    nodes = ts.samples(population=mx_pop_id)

    if len(nodes) == 0:
        return pd.DataFrame(columns=["Sample", "Start", "End", "Length", "State"])

    ind_ids = np.unique(ts.nodes_individual[nodes])
    ind_ids = ind_ids[ind_ids != -1]

    rows = []
    for ind in ind_ids:
        individual = ts.individual(ind)
        for hap_index, node in enumerate(individual.nodes, start=1):
            sample_name = f"{mx_pop}_{ind}_{hap_index}"

            raw_eu = get_migrating_tracts_ind(ts, "EU", node, times["t_mexican_admixture"])
            raw_na = get_migrating_tracts_ind(ts, "NA", node, times["t_mexican_admixture"])
            raw_af = get_migrating_tracts_ind(ts, "AF", node, times["t_mexican_admixture"])
            raw_nd = get_migrating_tracts_ind(ts, neand_pop, node, times["t_nd_migration"])

            nd_eu, clean_eu = intersect_and_subtract(raw_eu, raw_nd)
            nd_na, clean_na = intersect_and_subtract(raw_na, raw_nd)
            clean_af = raw_af

            def add_rows(tracts: List[List[float]], label: str) -> None:
                for start, end in tracts:
                    rows.append(
                        {
                            "Sample": sample_name,
                            "Start": int(start),
                            "End": int(end),
                            "Length": int(end - start),
                            "State": label,
                        }
                    )

            add_rows(clean_eu, "EU")
            add_rows(nd_eu, "ND_EU")
            add_rows(clean_na, "NA")
            add_rows(nd_na, "ND_NA")
            add_rows(clean_af, "AF")

    if not rows:
        return pd.DataFrame(columns=["Sample", "Start", "End", "Length", "State"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["Sample", "Start", "End"]).reset_index(drop=True)
    return df


# ==============================================================================
# 5. MANIFEST DEFAULTS FOR DOWNSTREAM STAGES
# ==============================================================================

def build_default_stage_settings(params: Dict) -> Dict:
    """
    Default settings for future prepare/run stages.
    Stored in manifest.json for convenience.
    """
    return {
        "prepare": {
            "rfmix": {
                "default": {
                    "n_eu_ref": min(50, int(params["n_eu"])),
                    "n_na_ref": min(50, int(params["n_na"])),
                    "n_af_ref": min(50, int(params["n_af"])),
                    "threads": 4,
                }
            },
            "hmmix": {
                "default": {
                    "n_af_ref": min(250, int(params["n_af"])),
                    "threads": 4,
                    "use_mutrates_file": True,
                    "mutrate_window_size": 100000,
                    "keep_multiallelic": False,
                }
            },
            "daiseg_simple": {
                "default": {
                    "n_af_ref": min(250, int(params["n_af"])),
                    "n_nd_ref": min(3, int(params["n_nd"])),
                    "threads": 4,
                }
            },
            "daiseg_mex": {
                "default": {
                    "n_eu_ref": min(50, int(params["n_eu"])),
                    "n_na_ref": min(50, int(params["n_na"])),
                    "n_af_ref": min(50, int(params["n_af"])),
                    "n_nd_ref": min(3, int(params["n_nd"])),
                    "threads": 4,
                }
            },
        },
        "run": {
            "rfmix": {
                "default": {
                    "threads": 4,
                    "generations": 15,
                }
            },
            "hmmix": {
                "default": {
                    "haploid": True,
                    "viterbi": False,
                    "thresholds": [0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92,  0.95, 0.99],
                }
            },
            "daiseg_simple": {
                "default": {
                    "mode": "EM_v2",
                    "threads": 4,
                }
            },
            "daiseg_mex": {
                "default": {
                    "mode": "EM_v2",
                    "threads": 4,
                }
            },
        },
    }


def build_manifest(
    *,
    sim_name: str,
    raw_dir: Path,
    params: Dict,
    times: Dict,
    execution: Dict,
) -> Dict:
    return {
        "sim_name": sim_name,
        "project_type": "mex_compare",
        "layout": {
            "root": str(raw_dir.parent.resolve()),
            "raw": str(raw_dir.resolve()),
            "prepared": str((raw_dir.parent / "prepared").resolve()),
            "runs": str((raw_dir.parent / "runs").resolve()),
            "metrics": str((raw_dir.parent / "metrics").resolve()),
        },
        "files": {
            "manifest": "manifest.json",
            "truth_all": "truth.all.tsv",
        },
        "simulation": {
            "params": params,
            "times": times,
            "execution": execution,
        },
        "defaults": build_default_stage_settings(params),
    }


def save_manifest(manifest: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


# ==============================================================================
# 6. RAW STAGE
# ==============================================================================

def simulate_and_save_one(
    yaml_file: str,
    params: Dict,
    times: Dict,
    seed: int,
    raw_dir: Path,
) -> Dict:
    """
    Simulate one chromosome and save:
      - sim_seed_<seed>.trees
      - truth_seed_<seed>.tsv
    """
    ts = simulate_one_tree_sequence(yaml_file, params, times, seed)

    ts_path = raw_dir / f"sim_seed_{seed}.trees"
    ts.dump(str(ts_path))

    truth_df = get_5state_truth_dataframe(ts, times)
    if truth_df.empty:
        truth_df = pd.DataFrame(columns=["CHR", "Sample", "Start", "End", "Length", "State"])
    else:
        truth_df.insert(0, "CHR", int(seed))

    truth_path = raw_dir / f"truth_seed_{seed}.tsv"
    truth_df.to_csv(truth_path, sep="\t", index=False)

    return {
        "seed": int(seed),
        "ts_path": str(ts_path),
        "truth_path": str(truth_path),
        "truth_df": truth_df,
    }


def run_raw_stage(
    *,
    sim_name: str,
    yaml_file: str,
    params: Dict,
    execution: Dict,
    base_dir: str,
    clean: bool = False,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Execute fully independent raw stage.

    """
    dirs = ensure_project_layout(base_dir, sim_name, clean=clean)
    raw_dir = dirs["raw"]

    # Times are reconstructed from YAML to keep raw stage self-contained.
    _, times = extract_parameters_from_demes(yaml_file)

    seeds = list(execution["seeds"])
    print("=" * 72)
    print(f"mex_compare raw stage: {sim_name}")
    print(f"Raw directory: {raw_dir}")
    print(f"Seeds: {seeds}")
    print("=" * 72)



    '''
    results = Parallel(
        n_jobs=int(execution["n_jobs"]),
        prefer="threads",
    )(
        delayed(simulate_and_save_one)(
            yaml_file=yaml_file,
            params=params,
            times=times,
            seed=seed,
            raw_dir=raw_dir,
        )
        for seed in seeds
    )

   '''


    results = []

    with ProcessPoolExecutor(max_workers=int(execution["n_jobs"])) as ex:
        futures = [
            ex.submit(
                simulate_and_save_one,
            yaml_file,
            params,
            times,
            seed,
            raw_dir,
            )
            for seed in seeds
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x["seed"])


    truth_tables = [r["truth_df"] for r in results]
    if truth_tables:
        truth_all = pd.concat(truth_tables, ignore_index=True)
    else:
        truth_all = pd.DataFrame(columns=["CHR", "Sample", "Start", "End", "Length", "State"])

    truth_all_path = raw_dir / "truth.all.tsv"
    truth_all.to_csv(truth_all_path, sep="\t", index=False)

    manifest = build_manifest(
        sim_name=sim_name,
        raw_dir=raw_dir,
        params=params,
        times=times,
        execution=execution,
    )
    save_manifest(manifest, raw_dir / "manifest.json")

    print("\nDone.")
    print(f"Saved trees: {len(results)}")
    print(f"Saved truth per seed: {len(results)}")
    print(f"Combined truth: {truth_all_path}")
    print(f"Manifest: {raw_dir / 'manifest.json'}")

    return manifest, truth_all


# ==============================================================================
# 7. REBUILD TRUTH
# ==============================================================================

def rebuild_truth_from_existing_trees(
    *,
    sim_name: str,
    base_dir: str,
) -> pd.DataFrame:
    """
    Recompute truth_seed_*.tsv and truth.all.tsv from existing .trees files.
    """
    dirs = get_project_dirs(base_dir, sim_name)
    raw_dir = dirs["raw"]
    manifest_path = raw_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    times = manifest["simulation"]["times"]

    tree_paths = sorted(raw_dir.glob("sim_seed_*.trees"))
    if not tree_paths:
        raise FileNotFoundError(f"No .trees files found in {raw_dir}")

    truth_tables = []
    for ts_path in tree_paths:
        seed = int(ts_path.stem.replace("sim_seed_", ""))
        ts = tskit.load(str(ts_path))

        df = get_5state_truth_dataframe(ts, times)
        if df.empty:
            df = pd.DataFrame(columns=["CHR", "Sample", "Start", "End", "Length", "State"])
        else:
            df.insert(0, "CHR", int(seed))

        out_path = raw_dir / f"truth_seed_{seed}.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        truth_tables.append(df)

    truth_all = (
        pd.concat(truth_tables, ignore_index=True)
        if truth_tables
        else pd.DataFrame(columns=["CHR", "Sample", "Start", "End", "Length", "State"])
    )
    truth_all.to_csv(raw_dir / "truth.all.tsv", sep="\t", index=False)

    print(f"Rebuilt truth files in: {raw_dir}")
    return truth_all


# ==============================================================================
# 8. CLI
# ==============================================================================

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-contained raw-stage simulator for mex_compare"
    )

    parser.add_argument("--sim-name", type=str, required=True, help="Simulation name")
    parser.add_argument("--yaml", type=str, default=DEFAULT_SIM_CONFIG["yaml"])
    parser.add_argument("--base-dir", type=str, default=DEFAULT_SIM_CONFIG["base_dir"])

    # execution
    parser.add_argument("--n-chr", type=int, default=DEFAULT_SIM_CONFIG["n_chr"])
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SIM_CONFIG["base_seed"])
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_SIM_CONFIG["n_jobs"])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--rebuild-truth", action="store_true")

    # model
    parser.add_argument("--chr-length", type=float, default=DEFAULT_SIM_CONFIG["chrom_length"])
    parser.add_argument("--recomb-rate", type=float, default=DEFAULT_SIM_CONFIG["recomb_rate"])
    parser.add_argument("--mut-rate", type=float, default=DEFAULT_SIM_CONFIG["mut_rate"])
    parser.add_argument("--ploidy", type=int, default=DEFAULT_SIM_CONFIG["ploidy"])

    parser.add_argument("--n-mexicans", type=int, default=DEFAULT_SIM_CONFIG["n_mexicans"])
    parser.add_argument("--n-eu", type=int, default=DEFAULT_SIM_CONFIG["n_eu"])
    parser.add_argument("--n-na", type=int, default=DEFAULT_SIM_CONFIG["n_na"])
    parser.add_argument("--n-af", type=int, default=DEFAULT_SIM_CONFIG["n_af"])
    parser.add_argument("--n-nd", type=int, default=DEFAULT_SIM_CONFIG["n_nd"])

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.rebuild_truth:
        rebuild_truth_from_existing_trees(
            sim_name=args.sim_name,
            base_dir=args.base_dir,
        )
        return

    params, _ = build_simulation_params(
        args.yaml,
        chrom_length=args.chr_length,
        recomb_rate=args.recomb_rate,
        mut_rate=args.mut_rate,
        ploidy=args.ploidy,
        n_mexicans=args.n_mexicans,
        n_eu=args.n_eu,
        n_na=args.n_na,
        n_af=args.n_af,
        n_nd=args.n_nd,
    )

    execution = build_execution_config(
        n_chr=args.n_chr,
        base_seed=args.base_seed,
        n_jobs=args.n_jobs,
    )

    run_raw_stage(
        sim_name=args.sim_name,
        yaml_file=args.yaml,
        params=params,
        execution=execution,
        base_dir=args.base_dir,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
