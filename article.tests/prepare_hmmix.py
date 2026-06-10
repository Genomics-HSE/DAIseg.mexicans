#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tskit


DEFAULT_BASE_DIR = "."
DEFAULT_THREADS = 4
DEFAULT_INGROUP_POP = "MX"
DEFAULT_OUTGROUP_POP = "AF"


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
    ref_tag = f"ref.af{n_af_ref}"
    return dirs["prepared"] / "hmmix" / ref_tag


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def check_requirements() -> None:
    missing = []
    if shutil.which("hmmix") is None:
        missing.append("hmmix")
    if shutil.which("bcftools") is None:
        missing.append("bcftools")
    if shutil.which("bgzip") is None:
        missing.append("bgzip")
    if missing:
        raise EnvironmentError("Missing required tools for prepare_hmmix.py: " + ", ".join(missing))


def run_command(cmd: List[str], step_name: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def get_population_id(ts: tskit.TreeSequence, pop_name: str) -> int:
    for pop in ts.populations():
        if pop.metadata.get("name") == pop_name:
            return pop.id
    raise ValueError(f"Population '{pop_name}' not found")


def get_individual_ids(ts: tskit.TreeSequence, pop_name: str) -> List[int]:
    pop_id = get_population_id(ts, pop_name)
    nodes = ts.samples(population=pop_id)
    inds = np.unique(ts.nodes_individual[nodes])
    inds = inds[inds != -1]
    return sorted(int(x) for x in inds)


def select_individual_ids(ts: tskit.TreeSequence, pop_name: str, n_keep: Optional[int]) -> List[int]:
    ids = get_individual_ids(ts, pop_name)
    if n_keep is None:
        return ids
    if n_keep < 1:
        raise ValueError(f"Requested {n_keep} individuals for {pop_name}, but n_keep must be >= 1")
    if n_keep > len(ids):
        raise ValueError(
            f"Requested {n_keep} individuals for {pop_name}, but only {len(ids)} are available"
        )
    return ids[:n_keep]


def create_weights_bed(chrom_ids: Sequence[int], chrom_len: int, output_file: Path) -> Path:
    with open(output_file, "w") as f:
        for chrom in chrom_ids:
            f.write(f"{int(chrom)}\t0\t{int(chrom_len)}\n")
    return output_file


def create_individuals_json(
    ingroup_names: Sequence[str],
    outgroup_names: Sequence[str],
    output_file: Path,
) -> Path:
    data = {"ingroup": list(ingroup_names), "outgroup": list(outgroup_names)}
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    return output_file


def create_ancestral_fasta(
    ts: tskit.TreeSequence,
    chrom_id: int,
    chrom_len: int,
    output_file: Path,
    default_base: str = "A",
) -> Path:
    seq = np.full(int(chrom_len), ord(default_base), dtype=np.uint8)
    for site in ts.sites():
        pos = int(site.position)
        anc = site.ancestral_state
        if len(anc) == 1 and 0 <= pos < chrom_len:
            seq[pos] = ord(anc)

    with open(output_file, "w") as f:
        f.write(f">{chrom_id}\n")
        seq_str = seq.tobytes().decode("ascii")
        for i in range(0, len(seq_str), 60):
            f.write(seq_str[i:i + 60] + "\n")
    return output_file


def create_vcf_for_hmmix(
    ts: tskit.TreeSequence,
    vcf_path: Path,
    individual_ids: Sequence[int],
    chrom_id: int,
    keep_only_biallelic_snps: bool = True,
) -> Path:
    with open(vcf_path, "w") as f:
        ts.write_vcf(
            f,
            individuals=list(individual_ids),
            contig_id=str(chrom_id),
            position_transform=lambda x: np.array(x, dtype=int) + 1,
        )

    raw_vcf_gz = Path(str(vcf_path) + ".gz")
    run_command(["bgzip", "-f", str(vcf_path)], "bgzip raw VCF")

    if keep_only_biallelic_snps:
        filt_vcf_gz = Path(str(raw_vcf_gz).replace(".vcf.gz", ".biallelic.vcf.gz"))
        run_command(
            [
                "bcftools", "view",
                "-v", "snps",
                "-m2",
                "-M2",
                str(raw_vcf_gz),
                "-Oz",
                "-o", str(filt_vcf_gz),
            ],
            "bcftools view biallelic SNPs",
        )
        run_command(["bcftools", "index", "-f", str(filt_vcf_gz)], "bcftools index filtered VCF")
        return filt_vcf_gz

    run_command(["bcftools", "index", "-f", str(raw_vcf_gz)], "bcftools index raw VCF")
    return raw_vcf_gz


def create_files_for_chromosome(
    seed: int,
    ts_path: str,
    hmmix_dir: str,
    chrom_len: int,
    all_individual_ids: Sequence[int],
    suffix: str,
    keep_only_biallelic_snps: bool,
) -> Tuple[int, str, str]:
    ts = tskit.load(ts_path)
    out_dir = Path(hmmix_dir)

    vcf_path = out_dir / f"chr{seed}.{suffix}.vcf"
    vcf_gz = create_vcf_for_hmmix(
        ts=ts,
        vcf_path=vcf_path,
        individual_ids=all_individual_ids,
        chrom_id=seed,
        keep_only_biallelic_snps=keep_only_biallelic_snps,
    )

    anc_fa = out_dir / f"ancestral_chr{seed}.{suffix}.fa"
    create_ancestral_fasta(ts, seed, chrom_len, anc_fa)
    return seed, str(vcf_gz), str(anc_fa)


def prepare_hmmix_inputs(
    sim_name: str,
    *,
    base_dir: str = DEFAULT_BASE_DIR,
    ingroup_pop: str = DEFAULT_INGROUP_POP,
    outgroup_pop: str = DEFAULT_OUTGROUP_POP,
    n_af_ref: Optional[int] = None,
    threads: int = DEFAULT_THREADS,
    force: bool = False,
    use_mutrates_file: bool = False,
    mutrate_window_size: int = 100000,
    keep_only_biallelic_snps: bool = True,
) -> Path:
    check_requirements()

    dirs = get_project_dirs(base_dir, sim_name)
    raw_dir = dirs["raw"]
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Raw manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    sim_params = manifest["simulation"]["params"]
    execution = manifest["simulation"]["execution"]
    seeds = [int(x) for x in execution["seeds"]]
    chrom_len = int(sim_params["chrom_length"])

    effective_n_af_ref = int(n_af_ref) if n_af_ref is not None else int(sim_params["n_af"])
    out_dir = get_hmmix_prepare_dir(base_dir, sim_name, effective_n_af_ref)
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(seeds) < 2:
        raise ValueError("HMMix preparation requires at least 2 chromosomes")

    ts_first_path = raw_dir / f"sim_seed_{seeds[0]}.trees"
    if not ts_first_path.exists():
        raise FileNotFoundError(f"Missing tree sequence: {ts_first_path}")

    ts_first = tskit.load(str(ts_first_path))
    ingroup_ids = get_individual_ids(ts_first, ingroup_pop)
    outgroup_ids = select_individual_ids(ts_first, outgroup_pop, effective_n_af_ref)

    if not ingroup_ids:
        raise ValueError(f"No ingroup individuals found for population '{ingroup_pop}'")
    if not outgroup_ids:
        raise ValueError(f"No outgroup individuals found for population '{outgroup_pop}'")

    n_ref_used = len(outgroup_ids)
    suffix = f"ref.{n_ref_used}"

    ingroup_names = [f"tsk_{i}" for i in ingroup_ids]
    outgroup_names = [f"tsk_{i}" for i in outgroup_ids]

    individuals_json = out_dir / f"individuals.{suffix}.json"
    create_individuals_json(ingroup_names, outgroup_names, individuals_json)

    weights_bed = out_dir / f"weights.{suffix}.bed"
    create_weights_bed(seeds, chrom_len, weights_bed)

    all_individual_ids = sorted(set(ingroup_ids + outgroup_ids))
    file_triplets: List[Tuple[int, str, str]] = []

    if int(threads) <= 1:
        for seed in seeds:
            ts_path = raw_dir / f"sim_seed_{seed}.trees"
            file_triplets.append(
                create_files_for_chromosome(
                    seed=seed,
                    ts_path=str(ts_path),
                    hmmix_dir=str(out_dir),
                    chrom_len=chrom_len,
                    all_individual_ids=all_individual_ids,
                    suffix=suffix,
                    keep_only_biallelic_snps=keep_only_biallelic_snps,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=int(threads)) as ex:
            futures = []
            for seed in seeds:
                ts_path = raw_dir / f"sim_seed_{seed}.trees"
                futures.append(
                    ex.submit(
                        create_files_for_chromosome,
                        seed,
                        str(ts_path),
                        str(out_dir),
                        chrom_len,
                        all_individual_ids,
                        suffix,
                        keep_only_biallelic_snps,
                    )
                )
            for fut in as_completed(futures):
                file_triplets.append(fut.result())

    file_triplets.sort(key=lambda x: x[0])
    vcf_files = [x[1] for x in file_triplets]
    ancestral_files = [x[2] for x in file_triplets]

    vcf_arg = ",".join(vcf_files)
    ancestral_arg = ",".join(ancestral_files)

    outgroup_file = out_dir / f"outgroup.{suffix}.txt"
    run_command(
        [
            "hmmix", "create_outgroup",
            f"-ind={individuals_json}",
            f"-vcf={vcf_arg}",
            f"-weights={weights_bed}",
            f"-ancestral={ancestral_arg}",
            f"-out={outgroup_file}",
        ],
        "hmmix create_outgroup",
    )

    mutrate_file = None
    if use_mutrates_file:
        mutrate_file = out_dir / f"mutationrate.{suffix}.bed"
        run_command(
            [
                "hmmix", "mutation_rate",
                f"-outgroup={outgroup_file}",
                f"-weights={weights_bed}",
                f"-window_size={int(mutrate_window_size)}",
                f"-out={mutrate_file}",
            ],
            "hmmix mutation_rate",
        )

    obs_prefix = out_dir / f"obs.{suffix}"
    run_command(
        [
            "hmmix", "create_ingroup",
            f"-ind={individuals_json}",
            f"-vcf={vcf_arg}",
            f"-outgroup={outgroup_file}",
            f"-weights={weights_bed}",
            f"-ancestral={ancestral_arg}",
            f"-out={obs_prefix}",
        ],
        "hmmix create_ingroup",
    )

    prep_manifest = {
        "method": "hmmix",
        "sim_name": sim_name,
        "source_raw_dir": str(raw_dir.resolve()),
        "prepared_dir": str(out_dir.resolve()),
        "reference_panel": {"n_af_ref": int(n_ref_used)},
        "query_population": ingroup_pop,
        "outgroup_population": outgroup_pop,
        "simulation_params": {"chrom_length": chrom_len},
        "settings": {
            "threads": int(threads),
            "use_mutrates_file": bool(use_mutrates_file),
            "mutrate_window_size": int(mutrate_window_size),
            "keep_only_biallelic_snps": bool(keep_only_biallelic_snps),
        },
        "shared_files": {
            "individuals_json": individuals_json.name,
            "weights_bed": weights_bed.name,
            "outgroup_file": outgroup_file.name,
            "mutationrate_file": None if mutrate_file is None else mutrate_file.name,
            "obs_prefix": obs_prefix.name,
        },
        "chromosomes": [
            {
                "chrom_seed": int(seed),
                "vcf_file": Path(vcf_file).name,
                "ancestral_file": Path(anc_file).name,
            }
            for seed, vcf_file, anc_file in file_triplets
        ],
    }

    with open(out_dir / "manifest.hmmix.json", "w") as f:
        json.dump(prep_manifest, f, indent=2)

    print("=" * 72)
    print("HMMix preparation completed")
    print(f"Prepared dir:   {out_dir}")
    print(f"Reference AF:   {n_ref_used}")
    print(f"Chromosomes:    {len(file_triplets)}")
    print(f"Use mutrates:   {use_mutrates_file}")
    print("=" * 72)

    return out_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare HMMix inputs from mex_compare raw simulations"
    )
    parser.add_argument("--sim-name", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--n-af-ref", type=int, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--use-mutrates-file", action="store_true")
    parser.add_argument("--mutrate-window-size", type=int, default=100000)
    parser.add_argument("--keep-multiallelic", action="store_true", help="Do not filter to biallelic SNPs")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    prepare_hmmix_inputs(
        sim_name=args.sim_name,
        base_dir=args.base_dir,
        n_af_ref=args.n_af_ref,
        threads=args.threads,
        force=args.force,
        use_mutrates_file=args.use_mutrates_file,
        mutrate_window_size=args.mutrate_window_size,
        keep_only_biallelic_snps=not args.keep_multiallelic,
    )


if __name__ == "__main__":
    main()
