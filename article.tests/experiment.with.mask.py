from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import demes
import msprime
import numpy as np
import pandas as pd
import tskit

import hmm2 as hmm
import em_alg2 as em_alg


def open_maybe_gzip(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def normalize_chrom(chrom: str | int) -> str:
    chrom = str(chrom)
    return chrom[3:] if chrom.startswith("chr") else chrom


def load_chrom_lengths(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            chrom = normalize_chrom(parts[0])
            if chrom.lower() in {"chrom", "#chrom"}:
                continue
            out[chrom] = int(parts[1])
    return out


# -----------------------------------------------------------------------------
# Interval helpers
# -----------------------------------------------------------------------------

def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    xs = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    if not xs:
        return []
    merged: List[List[int]] = [[xs[0][0], xs[0][1]]]
    for s, e in xs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def clip_intervals(intervals: Iterable[Tuple[int, int]], chrom_len: int) -> List[Tuple[int, int]]:
    return merge_intervals((max(0, s), min(chrom_len, e)) for s, e in intervals)


def intersect_two(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    i = j = 0
    out: List[Tuple[int, int]] = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def intersect_many(lists: Sequence[Sequence[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    if not lists:
        return []
    cur = list(lists[0])
    for x in lists[1:]:
        cur = intersect_two(cur, x)
        if not cur:
            break
    return cur


def union_intervals(lists: Sequence[Sequence[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    all_ints: List[Tuple[int, int]] = []
    for x in lists:
        all_ints.extend(x)
    return merge_intervals(all_ints)


def subtract_intervals(base: Sequence[Tuple[int, int]], sub: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    j = 0
    for bs, be in base:
        cur = bs
        while j < len(sub) and sub[j][1] <= bs:
            j += 1
        k = j
        while k < len(sub) and sub[k][0] < be:
            ss, se = sub[k]
            if ss > cur:
                out.append((cur, min(ss, be)))
            cur = max(cur, se)
            if cur >= be:
                break
            k += 1
        if cur < be:
            out.append((cur, be))
    return out


def intervals_to_window_fractions(intervals: Sequence[Tuple[int, int]], chrom_len: int, window_len: int = 1000) -> np.ndarray:
    nwin = int(math.ceil(chrom_len / window_len))
    arr = np.zeros(nwin, dtype=float)
    for s, e in intervals:
        w0 = s // window_len
        w1 = (e - 1) // window_len
        for w in range(w0, w1 + 1):
            ws = w * window_len
            we = min((w + 1) * window_len, chrom_len)
            ov = max(0, min(e, we) - max(s, ws))
            if ov > 0:
                arr[w] += ov
    for w in range(nwin):
        ws = w * window_len
        we = min((w + 1) * window_len, chrom_len)
        arr[w] /= max(1, we - ws)
    return arr


def intervals_from_window_vector(lengths: Sequence[float], window_len: int = 1000, min_fraction: float = 0.5) -> List[Tuple[int, int]]:
    ints = [(i * window_len, (i + 1) * window_len) for i, x in enumerate(lengths) if x > min_fraction]
    return merge_intervals(ints)


def positions_in_intervals(positions: np.ndarray, intervals: Sequence[Tuple[int, int]]) -> np.ndarray:
    out = np.zeros(len(positions), dtype=bool)
    j = 0
    for i, pos in enumerate(positions):
        while j < len(intervals) and intervals[j][1] <= pos:
            j += 1
        if j < len(intervals) and intervals[j][0] <= pos < intervals[j][1]:
            out[i] = True
    return out




def modern_bed_path(modern_dir: str, chrom: str) -> str:
    path = os.path.join(modern_dir, f"chr{chrom}.renamed.bed")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def nd_bed_path(nd_dir: str, chrom: str) -> str:
    gz = os.path.join(nd_dir, f"chr{chrom}_mask.bed.gz")
    plain = os.path.join(nd_dir, f"chr{chrom}_mask.bed")
    if os.path.exists(gz):
        return gz
    if os.path.exists(plain):
        return plain
    raise FileNotFoundError(f"No ND bed for chr{chrom} in {nd_dir}")


def load_standard_bed_intervals(path: str, chrom: str, chrom_len: int) -> List[Tuple[int, int]]:
    target = normalize_chrom(chrom)
    out: List[Tuple[int, int]] = []
    with open_maybe_gzip(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            c = normalize_chrom(parts[0])
            if c != target:
                continue
            s, e = int(parts[1]), int(parts[2])
            if e > s:
                out.append((s, e))
    return clip_intervals(out, chrom_len)


def load_gap_intervals_ucsc_like(path: str, chrom: str, chrom_len: int) -> List[Tuple[int, int]]:
    target = normalize_chrom(chrom)
    out: List[Tuple[int, int]] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            c = normalize_chrom(parts[1])
            if c != target:
                continue
            s, e = int(parts[2]), int(parts[3])
            if e > s:
                out.append((s, e))
    return clip_intervals(out, chrom_len)


def build_masks_for_chrom(chrom: str, chrom_len: int, modern_dir: str, gaps_file: str,
                          nd_dirs: Sequence[str] | None = None, nd_mode: str = "union"):
    modern_callable = load_standard_bed_intervals(modern_bed_path(modern_dir, chrom), chrom, chrom_len)
    gaps = load_gap_intervals_ucsc_like(gaps_file, chrom, chrom_len)
    modern_eval = subtract_intervals(modern_callable, gaps)

    if not nd_dirs:
        archaic_eval: List[Tuple[int, int]] = []
    else:
        nd_lists = [load_standard_bed_intervals(nd_bed_path(d, chrom), chrom, chrom_len) for d in nd_dirs]
        nd_callable = union_intervals(nd_lists) if nd_mode == "union" else intersect_many(nd_lists)
        archaic_eval = intersect_two(modern_eval, nd_callable)

    return modern_eval, archaic_eval, gaps


# -----------------------------------------------------------------------------
# Demography and simulation
# -----------------------------------------------------------------------------

def extract_parameters_from_demes(yaml_file: str):
    graph = demes.load(yaml_file)
    gen_time = float(graph.generation_time)

    def get_t(name: str) -> float:
        if name not in graph:
            return 0.0
        return float(graph[name].start_time) / gen_time

    times = {
        "t_mexican_admixture": get_t("MX"),
        "t_neanderthal_split": get_t("NEAND"),
        "t_africa_split": get_t("OOA"),
        "t_eu_asia_split": get_t("ASIA"),
    }
    pulse_time_years = None
    pulse_prop = None
    for pulse in graph.pulses:
        sources = [s if isinstance(s, str) else s.name for s in pulse.sources]
        if "NEAND" in sources:
            pulse_time_years = float(pulse.time)
            pulse_prop = float(pulse.proportions[0])
            break
    if pulse_time_years is None:
        raise ValueError("No NEAND pulse found")
    times["t_nd_migration"] = pulse_time_years / gen_time
    times["t_nd_samples"] = pulse_time_years / gen_time
    params = {
        "gen_time": gen_time,
        "admixture_nd": pulse_prop,
        "admixture_modern": list(graph["MX"].proportions) if "MX" in graph else [0.5, 0.4, 0.1],
    }
    return params, times


def simulate_one_tree_sequence(yaml_file: str, chrom_len: int, seed: int, args, time_dict) -> tskit.TreeSequence:
    graph = demes.load(yaml_file)
    demography = msprime.Demography.from_demes(graph)
    ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(args.n_mexicans, ploidy=args.ploidy, population="MX"),
            msprime.SampleSet(args.n_eu, ploidy=args.ploidy, population="EU"),
            msprime.SampleSet(args.n_na, ploidy=args.ploidy, population="NA"),
            msprime.SampleSet(args.n_af, ploidy=args.ploidy, population="AF"),
            msprime.SampleSet(args.n_nd, ploidy=args.ploidy, population="NEAND", time=time_dict["t_nd_samples"]),
        ],
        sequence_length=chrom_len,
        recombination_rate=args.recomb_rate,
        demography=demography,
        random_seed=int(seed),
        record_migrations=True,
    )
    ts = msprime.sim_mutations(ts, rate=args.mut_rate, random_seed=int(seed))
    return ts


def _get_population_id(ts: tskit.TreeSequence, pop_name: str) -> int:
    for pop in ts.populations():
        if pop.metadata.get("name") == pop_name:
            return pop.id
    raise ValueError(pop_name)


def intersect_and_subtract(base_tracts: List[List[float]], mask_tracts: List[List[float]]):
    inter: List[List[float]] = []
    for b in base_tracts:
        for m in mask_tracts:
            s = max(b[0], m[0])
            e = min(b[1], m[1])
            if s < e:
                inter.append([s, e])
    diff = [list(x) for x in base_tracts]
    for ms, me in mask_tracts:
        nxt = []
        for fs, fe in diff:
            if me <= fs or ms >= fe:
                nxt.append([fs, fe])
            else:
                if fs < ms:
                    nxt.append([fs, ms])
                if fe > me:
                    nxt.append([me, fe])
        diff = nxt
    return sorted(inter), sorted(diff)


def get_migrating_tracts_ind(ts: tskit.TreeSequence, pop_name: str, ind_node: int, migration_time: float, eps: float = 1e-5):
    pop_id = _get_population_id(ts, pop_name)
    tables = ts.tables
    mask = ((tables.migrations.dest == pop_id)
            & (tables.migrations.time >= migration_time - eps)
            & (tables.migrations.time <= migration_time + eps))
    idx = np.where(mask)[0]
    mig_lookup: Dict[int, List[Tuple[float, float]]] = {}
    for i in idx:
        node_id = int(tables.migrations.node[i])
        mig_lookup.setdefault(node_id, []).append((float(tables.migrations.left[i]), float(tables.migrations.right[i])))
    for k in mig_lookup:
        mig_lookup[k].sort()

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
                for ml, mr in mig_lookup[anc]:
                    s = max(tree.interval.left, ml)
                    e = min(tree.interval.right, mr)
                    if s < e:
                        if tracts and tracts[-1][1] == s:
                            tracts[-1][1] = e
                        else:
                            tracts.append([s, e])
            if node_times[anc] >= migration_time - eps:
                break
            anc = parent
            parent = tree.parent(anc)
    return tracts


def get_5state_truth_dataframe(ts: tskit.TreeSequence, times: Dict, mx_pop: str = "MX", neand_pop: str = "NEAND") -> pd.DataFrame:
    mx_pop_id = _get_population_id(ts, mx_pop)
    nodes = ts.samples(population=mx_pop_id)
    ind_ids = np.unique(ts.nodes_individual[nodes])
    ind_ids = ind_ids[ind_ids != -1]
    rows: List[Dict] = []
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
            def add_rows(tracts, label):
                for s, e in tracts:
                    rows.append({"Sample": sample_name, "Start": int(s), "End": int(e), "Length": int(e - s), "State": label})
            add_rows(clean_eu, "EU")
            add_rows(nd_eu, "ND_EU")
            add_rows(clean_na, "NA")
            add_rows(nd_na, "ND_NA")
            add_rows(raw_af, "AF")
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Sample", "Start", "End", "Length", "State"])
    return df.sort_values(["Sample", "Start", "End"]).reset_index(drop=True)


def simulate_or_load_one_chrom(yaml_file: str, chrom: str, seed: int, chrom_len: int, out_dir: str, resimulate: bool, args, time_dict):
    ts_file = os.path.join(out_dir, f"sim_chr{chrom}_seed_{seed}.trees")
    gt_file = os.path.join(out_dir, f"ground_truth_5state_chr{chrom}_seed_{seed}.tsv")
    if resimulate or (not os.path.exists(ts_file)) or (not os.path.exists(gt_file)):
        ts = simulate_one_tree_sequence(yaml_file, chrom_len, seed, args, time_dict)
        ts.dump(ts_file)
        gt = get_5state_truth_dataframe(ts, time_dict)
        if not gt.empty:
            gt.insert(0, "CHR", seed)
        gt.to_csv(gt_file, sep="\t", index=False)
    else:
        ts = tskit.load(ts_file)
        gt = pd.read_csv(gt_file, sep="\t", keep_default_na=False)
    return {"chrom": chrom, "seed": seed, "chrom_len": chrom_len, "ts": ts, "gt_df": gt}




def get_population_individual_ids(ts: tskit.TreeSequence, pop_name: str) -> List[int]:
    pop_id = _get_population_id(ts, pop_name)
    nodes = ts.samples(population=pop_id)
    inds = np.unique(ts.nodes_individual[nodes])
    inds = inds[inds != -1]
    return sorted(int(x) for x in inds)


def get_nodes_for_diploids(ts: tskit.TreeSequence, pop_name: str, limit_diploids: int | None = None) -> np.ndarray:
    inds = get_population_individual_ids(ts, pop_name)
    if limit_diploids is not None:
        inds = inds[: int(limit_diploids)]
    nodes: List[int] = []
    for ind in inds:
        nodes.extend(ts.individual(ind).nodes)
    return np.array(nodes, dtype=int)


def get_mexican_haps(ts: tskit.TreeSequence, all_samples: np.ndarray, mx_nodes: np.ndarray):
    node_to_col = {n: i for i, n in enumerate(all_samples)}
    n_ind = ts.nodes_individual
    mx_map: Dict[int, List[int]] = defaultdict(list)
    for n in mx_nodes:
        ind = int(n_ind[n])
        if ind != -1:
            mx_map[ind].append(int(n))
    hap_names: List[str] = []
    mx_cols: List[int] = []
    for ind in sorted(mx_map):
        nodes = sorted(mx_map[ind])
        for k, node in enumerate(nodes, start=1):
            hap_names.append(f"MX_{ind}_{k}")
            mx_cols.append(node_to_col[node])
    return hap_names, np.array(mx_cols, dtype=int)


def build_sitewise_observations_from_ts(ts: tskit.TreeSequence, chrom_len: int,
                                        modern_eval_intervals: Sequence[Tuple[int, int]],
                                        archaic_eval_intervals: Sequence[Tuple[int, int]],
                                        limits_diploid: Dict[str, int], window_len: int = 1000):
    G = ts.genotype_matrix()
    positions = ts.tables.sites.position.astype(int)
    all_samples = ts.samples()

    eu_nodes = get_nodes_for_diploids(ts, "EU", limits_diploid.get("EU"))
    na_nodes = get_nodes_for_diploids(ts, "NA", limits_diploid.get("NA"))
    af_nodes = get_nodes_for_diploids(ts, "AF", limits_diploid.get("AF"))
    nd_nodes = get_nodes_for_diploids(ts, "NEAND", limits_diploid.get("NEAND"))
    mx_nodes = get_nodes_for_diploids(ts, "MX", None)
    if len(mx_nodes) == 0:
        raise ValueError("No MX nodes found")

    node_to_col = {n: i for i, n in enumerate(all_samples)}
    eu_cols = np.array([node_to_col[n] for n in eu_nodes], dtype=int) if len(eu_nodes) else np.array([], dtype=int)
    na_cols = np.array([node_to_col[n] for n in na_nodes], dtype=int) if len(na_nodes) else np.array([], dtype=int)
    af_cols = np.array([node_to_col[n] for n in af_nodes], dtype=int) if len(af_nodes) else np.array([], dtype=int)
    nd_cols = np.array([node_to_col[n] for n in nd_nodes], dtype=int) if len(nd_nodes) else np.array([], dtype=int)

    hap_names, mx_cols = get_mexican_haps(ts, all_samples, mx_nodes)
    n_haps = len(mx_cols)
    n_win = int(math.ceil(chrom_len / window_len))

    L_mod_vec = intervals_to_window_fractions(modern_eval_intervals, chrom_len, window_len)
    L_anc_vec = intervals_to_window_fractions(archaic_eval_intervals, chrom_len, window_len)
    in_mod = positions_in_intervals(positions, modern_eval_intervals)
    in_anc = positions_in_intervals(positions, archaic_eval_intervals)

    eu_has_derived = np.any(G[:, eu_cols] == 1, axis=1) if len(eu_cols) else np.zeros(len(positions), dtype=bool)
    na_has_derived = np.any(G[:, na_cols] == 1, axis=1) if len(na_cols) else np.zeros(len(positions), dtype=bool)
    af_has_derived = np.any(G[:, af_cols] == 1, axis=1) if len(af_cols) else np.zeros(len(positions), dtype=bool)
    nd_has_derived = np.any(G[:, nd_cols] == 1, axis=1) if len(nd_cols) else np.zeros(len(positions), dtype=bool)

    O_EU = np.zeros((n_haps, n_win), dtype=np.int32)
    O_NA = np.zeros((n_haps, n_win), dtype=np.int32)
    O_AF = np.zeros((n_haps, n_win), dtype=np.int32)
    O_ND = np.zeros((n_haps, n_win), dtype=np.int32)

    for h, mx_col in enumerate(mx_cols):
        mx_derived = G[:, mx_col] == 1
        active = np.where(mx_derived)[0]
        if len(active) == 0:
            continue
        active_win = positions[active] // window_len
        idx_mod = active[in_mod[active]]
        win_mod = active_win[in_mod[active]]
        for var_idx, w in zip(idx_mod, win_mod):
            if not eu_has_derived[var_idx]:
                O_EU[h, w] += 1
            if not na_has_derived[var_idx]:
                O_NA[h, w] += 1
            if not af_has_derived[var_idx]:
                O_AF[h, w] += 1
        idx_anc = active[in_anc[active]]
        win_anc = active_win[in_anc[active]]
        for var_idx, w in zip(idx_anc, win_anc):
            if not nd_has_derived[var_idx]:
                O_ND[h, w] += 1

    L_mod = np.tile(L_mod_vec, (n_haps, 1))
    L_anc = np.tile(L_anc_vec, (n_haps, 1))
    return O_EU, O_NA, O_AF, O_ND, L_mod, L_anc, hap_names




def make_init_from_times(params: Dict, times: Dict) -> Tuple[np.ndarray, Dict]:
    config = {
        "parameters_initial": {
            "admixture_nd": float(params["admixture_nd"]),
            "admixture_modern": list(params["admixture_modern"]),
            "introgression_time": int(times["t_nd_migration"] * params["gen_time"]),
            "rr": float(args_global.recomb_rate),
            "mutation": float(args_global.mut_rate),
            "window_length": int(args_global.window_len),
            "generation_time": int(params["gen_time"]),
            "t_n_c": int(times["t_neanderthal_split"] * params["gen_time"]),
            "t_af_c": int(times["t_africa_split"] * params["gen_time"]),
            "t_introgression_c": int(times["t_nd_migration"] * params["gen_time"]),
            "t_ea_c": int(times["t_eu_asia_split"] * params["gen_time"]),
            "t_mexicans_c": int(times["t_mexican_admixture"] * params["gen_time"]),
            "t_introgression": int(times["t_nd_migration"] * params["gen_time"]),
            "t_mexicans": int(times["t_mexican_admixture"] * params["gen_time"]),
        }
    }
    return hmm.init_params_from_json(config)


def run_batch_em_on_chunks(chunks: Sequence[Dict], init_lmbd: np.ndarray, trans_params: Dict, max_iter: int = 20, tol: float = 1e-8):
    curr_lmbd = init_lmbd.copy()
    prev_ll = -np.inf
    log_A = hmm.get_log_A_5x5(
        trans_params["Ti"], trans_params["Tmex"], trans_params["rr"],
        trans_params["w_len"], trans_params["a"], trans_params["b"],
        trans_params["c1"], trans_params["c2"]
    )
    trans_linear = np.exp(log_A)
    start_linear = np.ones(5) / 5.0

    for it in range(max_iter):
        total_nums = np.zeros(5)
        total_dens = np.zeros(5)
        total_ll = 0.0
        for ch in chunks:
            O_EU, O_NA, O_AF, O_ND = ch["obs"]
            L_mod, L_anc = ch["cov"]
            log_emit = hmm.compute_emissions_unified(O_EU, O_NA, O_AF, O_ND, L_mod[0, :], L_anc[0, :], curr_lmbd)
            emit_linear = np.exp(log_emit)
            nums, dens, ll = em_alg.e_step_unified(emit_linear, trans_linear, start_linear, O_EU, O_NA, O_AF, O_ND, L_mod, L_anc)
            total_nums += np.sum(nums, axis=0)
            total_dens += np.sum(dens, axis=0)
            total_ll += ll
        new_lmbd = curr_lmbd.copy()
        for k in range(5):
            if total_dens[k] > 1e-8:
                new_lmbd[k] = total_nums[k] / (total_dens[k] + 1e-20)
        curr_lmbd = new_lmbd
        if it > 0 and abs(total_ll - prev_ll) < tol:
            break
        prev_ll = total_ll

    rows: List[Dict] = []
    for ch in chunks:
        O_EU, O_NA, O_AF, O_ND = ch["obs"]
        L_mod, L_anc = ch["cov"]
        hap_names = ch["meta"]["hap_names"]
        chrom_label = ch["meta"]["chrom_label"]
        paths, _ = hmm.run_hmm(
            O_EU, O_NA, O_AF, O_ND,
            L_mod[0, :], L_anc[0, :], curr_lmbd,
            trans_params["rr"], trans_params["Ti"], trans_params["Tmex"],
            trans_params["a"], trans_params["b"], trans_params["c1"], trans_params["c2"],
            window_len=trans_params["w_len"], mark_uninformative=True
        )
        dct_paths = {k: v for k, v in zip(hap_names, paths)}
        for nm in hap_names:
            tracts = hmm.get_tracts_5states(dct_paths[nm], window_len=trans_params["w_len"], split_on_uninformative=True)
            for state, intervals in tracts.items():
                if state == "UNINFORMATIVE":
                    continue
                for s, e in intervals:
                    rows.append({"Sample": nm, "CHROM": str(chrom_label), "Start": int(s), "End": int(e), "Length": int(e - s + 1), "State": state})
    return curr_lmbd, pd.DataFrame(rows)



def calculate_masked_metrics_by_intersections(gt_full_df: pd.DataFrame, pred_full_df: pd.DataFrame, eval_map: Dict[str, List[Tuple[int, int]]],
                                              state_order=("EU", "ND_EU", "NA", "ND_NA", "AF")):
    state_map = {s: i for i, s in enumerate(state_order)}
    n_states = len(state_order)

    def _cleanup(df, col="State"):
        df = df.copy()
        return df[df[col].isin(state_map)]

    def _prep_gt(df):
        df = _cleanup(df, "State").copy()
        df["CHROM"] = df["CHR"].astype(str)
        df["state_id"] = df["State"].map(state_map).astype(int)
        return df[["Sample", "CHROM", "Start", "End", "state_id"]]

    def _prep_pred(df):
        df = _cleanup(df, "State").copy()
        df["CHROM"] = df["CHROM"].astype(str)
        df["End"] = df["End"].astype(int) + 1
        df["state_id"] = df["State"].map(state_map).astype(int)
        return df[["Sample", "CHROM", "Start", "End", "state_id"]]

    def _df_to_intervals(df, sample, chrom):
        sub = df[(df["Sample"] == sample) & (df["CHROM"] == chrom)].sort_values("Start", kind="mergesort")
        if sub.empty:
            return []
        return list(zip(sub["Start"].to_numpy(), sub["End"].to_numpy(), sub["state_id"].to_numpy()))

    def _clip(intervals, mask_intervals):
        out = []
        i = j = 0
        while i < len(intervals) and j < len(mask_intervals):
            s1, e1, st = intervals[i]
            s2, e2 = mask_intervals[j]
            left = max(s1, s2)
            right = min(e1, e2)
            if left < right:
                out.append((left, right, st))
            if e1 <= e2:
                i += 1
            else:
                j += 1
        return out

    def _conf(gt_int, pr_int):
        conf = np.zeros((n_states, n_states), dtype=np.int64)
        i = j = 0
        while i < len(gt_int) and j < len(pr_int):
            gs, ge, gk = gt_int[i]
            ps, pe, pk = pr_int[j]
            left = max(gs, ps)
            right = min(ge, pe)
            if left < right:
                conf[gk, pk] += right - left
            if ge <= pe:
                i += 1
            else:
                j += 1
        return conf

    gt = _prep_gt(gt_full_df)
    pr = _prep_pred(pred_full_df)
    samples = sorted(set(gt["Sample"]).intersection(set(pr["Sample"])))
    chroms = sorted(set(gt["CHROM"]).intersection(set(pr["CHROM"])))
    conf_total = np.zeros((n_states, n_states), dtype=np.int64)
    for sample in samples:
        for chrom in chroms:
            if chrom not in eval_map or not eval_map[chrom]:
                continue
            gt_int = _df_to_intervals(gt, sample, chrom)
            pr_int = _df_to_intervals(pr, sample, chrom)
            if not gt_int or not pr_int:
                continue
            gt_clip = _clip(gt_int, eval_map[chrom])
            pr_clip = _clip(pr_int, eval_map[chrom])
            if gt_clip and pr_clip:
                conf_total += _conf(gt_clip, pr_clip)

    total = int(conf_total.sum())
    report = {"state_order": list(state_order), "total_bp_scored": total, "accuracy": float(conf_total.trace() / total) if total > 0 else float("nan")}
    for k, name in enumerate(state_order):
        tp = int(conf_total[k, k])
        fp = int(conf_total[:, k].sum() - tp)
        fn = int(conf_total[k, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")
        report[name] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support_bp": int(conf_total[k, :].sum())}
    return conf_total, report


def collapse_confusion_to_arch_binary(conf_5x5: np.ndarray, state_order=("EU", "ND_EU", "NA", "ND_NA", "AF")):
    idx = {s: i for i, s in enumerate(state_order)}
    archaic = [idx["ND_EU"], idx["ND_NA"]]
    non_arch = [idx["EU"], idx["NA"], idx["AF"]]
    TP = conf_5x5[np.ix_(archaic, archaic)].sum()
    FN = conf_5x5[np.ix_(archaic, non_arch)].sum()
    FP = conf_5x5[np.ix_(non_arch, archaic)].sum()
    TN = conf_5x5[np.ix_(non_arch, non_arch)].sum()
    prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else np.nan
    return {"TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


args_global = None


def _simulate_worker(payload):
    yaml_file, chrom, seed, chrom_len, out_dir, resimulate, args_dict, time_dict = payload
    class A: pass
    a = A()
    for k, v in args_dict.items():
        setattr(a, k, v)
    return simulate_or_load_one_chrom(yaml_file, chrom, seed, chrom_len, out_dir, resimulate, a, time_dict)


def _build_chunk_worker(payload):
    item, regime_name, nd_dirs, args_dict = payload
    chrom, seed, chrom_len, ts = item["chrom"], item["seed"], item["chrom_len"], item["ts"]
    nd_mode = "intersection" if regime_name == "pooled_intersection" else "union"
    modern_eval, archaic_eval, _ = build_masks_for_chrom(chrom, chrom_len, args_dict["modern_dir"], args_dict["gaps_file"], nd_dirs if nd_dirs else None, nd_mode)
    limits = {"EU": args_dict["ref_eu_diploids"], "NA": args_dict["ref_na_diploids"], "AF": args_dict["ref_af_diploids"], "NEAND": args_dict["ref_nd_diploids"]}
    O_EU, O_NA, O_AF, O_ND, L_mod, L_anc, hap_names = build_sitewise_observations_from_ts(ts, chrom_len, modern_eval, archaic_eval, limits, args_dict["window_len"])
    return {
        "chunk": {"obs": (O_EU, O_NA, O_AF, O_ND), "cov": (L_mod, L_anc), "meta": {"chrom": chrom, "chrom_label": seed, "seed": seed, "hap_names": hap_names}},
        "callable_row": {"regime": regime_name, "chrom": chrom, "seed": seed, "modern_callable_fraction": float(np.mean(L_mod[0])), "archaic_callable_fraction": float(np.mean(L_anc[0])), "modern_callable_bp": float(np.sum(L_mod[0]) * args_dict["window_len"]), "archaic_callable_bp": float(np.sum(L_anc[0]) * args_dict["window_len"]), "chrom_len": chrom_len},
    }


def main():
    global args_global
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True)
    p.add_argument("--chrom_lengths", required=True)
    p.add_argument("--modern_dir", required=True)
    p.add_argument("--gaps_file", required=True)
    p.add_argument("--vindija_dir", default=None)
    p.add_argument("--altai_dir", default=None)
    p.add_argument("--chagyr_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--chroms", nargs="*", default=[str(i) for i in range(1, 23)])
    p.add_argument("--base_seed", type=int, default=1234567)
    p.add_argument("--n_threads", type=int, default=1)
    p.add_argument("--resimulate", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--window_len", type=int, default=1000)
    p.add_argument("--lmod_threshold", type=float, default=0.5)
    p.add_argument("--max_iter", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--ploidy", type=int, default=2)
    p.add_argument("--n_mexicans", type=int, default=1)
    p.add_argument("--n_eu", type=int, default=250)
    p.add_argument("--n_na", type=int, default=250)
    p.add_argument("--n_af", type=int, default=250)
    p.add_argument("--n_nd", type=int, default=10)
    p.add_argument("--ref_eu_diploids", type=int, default=250)
    p.add_argument("--ref_na_diploids", type=int, default=250)
    p.add_argument("--ref_af_diploids", type=int, default=250)
    p.add_argument("--ref_nd_diploids", type=int, default=3)
    p.add_argument("--recomb_rate", type=float, default=1e-8)
    p.add_argument("--mut_rate", type=float, default=1.25e-8)
    args = p.parse_args()
    args_global = args

    os.makedirs(args.out_dir, exist_ok=True)
    chrom_lengths = load_chrom_lengths(args.chrom_lengths)
    yaml_params, time_dict = extract_parameters_from_demes(args.yaml)
    init_lmbd, trans_params = make_init_from_times(yaml_params, time_dict)

    chrom_info = []
    for i, chrom in enumerate(args.chroms):
        chrom = normalize_chrom(chrom)
        if chrom not in chrom_lengths:
            continue
        chrom_info.append({"chrom": chrom, "seed": args.base_seed + i, "chrom_len": chrom_lengths[chrom]})

    regimes = {"none": []}
    if args.vindija_dir:
        regimes["vindija"] = [args.vindija_dir]
    if args.altai_dir:
        regimes["altai"] = [args.altai_dir]
    if args.chagyr_dir:
        regimes["chagyr"] = [args.chagyr_dir]
    pooled = [x for x in [args.vindija_dir, args.altai_dir, args.chagyr_dir] if x]
    if pooled:
        regimes["pooled_union"] = pooled
        if len(pooled) >= 2:
            regimes["pooled_intersection"] = pooled

    sim_payloads = [(args.yaml, x["chrom"], x["seed"], x["chrom_len"], args.out_dir, args.resimulate, vars(args), time_dict) for x in chrom_info]
    results = []
    if args.n_threads > 1:
        with ProcessPoolExecutor(max_workers=args.n_threads) as ex:
            futs = [ex.submit(_simulate_worker, pl) for pl in sim_payloads]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for pl in sim_payloads:
            results.append(_simulate_worker(pl))
    results.sort(key=lambda x: x["seed"])

    gt_all = pd.concat([r["gt_df"] for r in results], ignore_index=True) if results else pd.DataFrame()
    gt_all.to_csv(os.path.join(args.out_dir, "ground_truth_5state.all.tsv"), sep="\t", index=False)

    # Lmod hist using modern callable only
    lmod_rows = []
    pooled_counts = np.zeros(10, dtype=np.int64)
    pooled_total = 0
    edges = np.round(np.arange(0.0, 1.0 + 0.1, 0.1), 10)
    labels = [f"[{edges[i]:.1f},{edges[i+1]:.1f})" for i in range(len(edges) - 2)] + ["[0.9,1.0]"]
    for item in results:
        modern_eval, _, _ = build_masks_for_chrom(item["chrom"], item["chrom_len"], args.modern_dir, args.gaps_file, None, "union")
        L_mod_vec = intervals_to_window_fractions(modern_eval, item["chrom_len"], args.window_len)
        idx = np.digitize(L_mod_vec, edges, right=False) - 1
        idx = np.clip(idx, 0, len(labels) - 1)
        counts = np.bincount(idx, minlength=len(labels))
        pooled_counts += counts
        pooled_total += len(L_mod_vec)
        for i, lab in enumerate(labels):
            lmod_rows.append({"chrom": item["chrom"], "window_len": args.window_len, "lmod_bin": lab, "n_windows": int(counts[i]), "fraction_windows": float(counts[i] / len(L_mod_vec)) if len(L_mod_vec) else np.nan, "bp_equivalent": int(counts[i] * args.window_len)})
    pd.DataFrame(lmod_rows).to_csv(os.path.join(args.out_dir, "lmod_distribution_by_chrom.tsv"), sep="\t", index=False)
    pd.DataFrame([{"window_len": args.window_len, "lmod_bin": labels[i], "n_windows": int(pooled_counts[i]), "fraction_windows": float(pooled_counts[i] / pooled_total) if pooled_total else np.nan, "bp_equivalent": int(pooled_counts[i] * args.window_len)} for i in range(len(labels))]).to_csv(os.path.join(args.out_dir, "lmod_distribution_pooled.tsv"), sep="\t", index=False)

    summary_rows = []
    args_dict = vars(args)
    for regime_name, nd_dirs in regimes.items():
        payloads = [(item, regime_name, nd_dirs, args_dict) for item in results]
        chunk_results = []
        if args.n_threads > 1:
            with ProcessPoolExecutor(max_workers=args.n_threads) as ex:
                futs = [ex.submit(_build_chunk_worker, pl) for pl in payloads]
                for fut in as_completed(futs):
                    chunk_results.append(fut.result())
        else:
            for pl in payloads:
                chunk_results.append(_build_chunk_worker(pl))
        chunks = [x["chunk"] for x in chunk_results]
        callable_df = pd.DataFrame([x["callable_row"] for x in chunk_results]).sort_values(["chrom", "seed"])
        callable_df.to_csv(os.path.join(args.out_dir, f"callable_space.{regime_name}.tsv"), sep="\t", index=False)

        final_lmbd, pred_df = run_batch_em_on_chunks(chunks, init_lmbd.copy(), trans_params, max_iter=args.max_iter, tol=args.tol)
        pred_path = os.path.join(args.out_dir, f"predictions.{regime_name}.tsv")
        pred_df.to_csv(pred_path, sep="\t", index=False)

        eval_map = {}
        for item in results:
            modern_eval, _, _ = build_masks_for_chrom(item["chrom"], item["chrom_len"], args.modern_dir, args.gaps_file, nd_dirs if nd_dirs else None, "intersection" if regime_name == "pooled_intersection" else "union")
            L_mod_vec = intervals_to_window_fractions(modern_eval, item["chrom_len"], args.window_len)
            good_windows = intervals_from_window_vector(L_mod_vec, args.window_len, args.lmod_threshold)
            eval_map[str(item["seed"])] = intersect_two(modern_eval, good_windows)

        conf, report = calculate_masked_metrics_by_intersections(gt_all, pred_df, eval_map)
        np.savetxt(os.path.join(args.out_dir, f"confusion.{regime_name}.txt"), conf, fmt="%d")
        with open(os.path.join(args.out_dir, f"class_report.{regime_name}.json"), "w") as f:
            json.dump(report, f, indent=2)
        binary = collapse_confusion_to_arch_binary(conf)
        with open(os.path.join(args.out_dir, f"binary_arch_report.{regime_name}.json"), "w") as f:
            json.dump(binary, f, indent=2)

        row = {
            "regime": regime_name,
            "accuracy": report.get("accuracy", np.nan),
            "total_bp_scored": report.get("total_bp_scored", np.nan),
            "archaic_precision": binary["precision"],
            "archaic_recall": binary["recall"],
            "archaic_f1": binary["f1"],
            "lambda_i": float(final_lmbd[0]),
            "lambda_n": float(final_lmbd[1]),
            "lambda_af": float(final_lmbd[2]),
            "lambda_ea": float(final_lmbd[3]),
            "lambda_mex": float(final_lmbd[4]),
        }
        for state in ["EU", "ND_EU", "NA", "ND_NA", "AF"]:
            row[f"{state}_precision"] = report.get(state, {}).get("precision", np.nan)
            row[f"{state}_recall"] = report.get(state, {}).get("recall", np.nan)
            row[f"{state}_f1"] = report.get(state, {}).get("f1", np.nan)
            row[f"{state}_support_bp"] = report.get(state, {}).get("support_bp", np.nan)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
