#!/usr/bin/env python3
import os
import gc
import glob
import multiprocessing as mp

import demes
import msprime
import tskit
import numpy as np
import pandas as pd


YAML_FILE = None
OUT_DIR = "validation_batch_run"

BASE_SEED = 12345
NUM_THREADS = 20

CHROM_LEN = 30_000_000
TOTAL_LEN = 3_000_000_000
NUM_CHROMOSOMES = TOTAL_LEN // CHROM_LEN

WINDOW = 1000
MIN_EU_HAPS = 10
MIN_ND_TRACT = 0  # no archaic-tract length filter for truth-level calibration

# Reuse previously generated simulations/truth tables whenever possible.
# Priority: ground_truth_5state_all.tsv -> per-seed ground_truth_5state_*.tsv -> sim_seed_*.trees -> new msprime simulation.
REUSE_EXISTING_TRUTH = True
REUSE_EXISTING_TREES = True
SAVE_TREES = False
SIMULATE_MUTATIONS = False  # mutations are not needed for truth-level tract extraction

N_MEXICANS = 98
N_IBS = 108

RECOMB_RATE = 1e-8
MUT_RATE = 1.25e-8
PLOIDY = 2


def intersect_and_subtract(base_tracts, mask_tracts):
    intersection = []

    for b in base_tracts:
        for m in mask_tracts:
            start = max(b[0], m[0])
            end = min(b[1], m[1])
            if start < end:
                intersection.append([start, end])

    intersection.sort()

    difference = list(base_tracts)

    for m in mask_tracts:
        next_fragments = []
        m_start, m_end = m[0], m[1]

        for frag in difference:
            f_start, f_end = frag[0], frag[1]

            if m_end <= f_start or m_start >= f_end:
                next_fragments.append(frag)
            else:
                if f_start < m_start:
                    next_fragments.append([f_start, m_start])
                if f_end > m_end:
                    next_fragments.append([m_end, f_end])

        difference = next_fragments

    difference.sort()
    return intersection, difference



def filter_min_length(tracts, min_len):
    if min_len is None or min_len <= 0:
        return list(tracts)
    return [x for x in tracts if int(x[1]) - int(x[0]) >= min_len]


def extract_parameters_from_demes(yaml_file):
    graph = demes.load(yaml_file)
    gen_time = graph.generation_time

    def get_time(deme_name):
        return graph[deme_name].start_time / gen_time if deme_name in graph else 0

    times = {
        "t_mexican_admixture": get_time("MX"),
        "t_neanderthal_split": get_time("NEAND"),
        "t_africa_split": get_time("OOA"),
        "t_east_asian_split": get_time("EU"),
    }

    pulse_time_years = 0
    pulse_prop = 0.0

    for pulse in graph.pulses:
        sources = [s if isinstance(s, str) else s.name for s in pulse.sources]
        if "NEAND" in sources:
            pulse_time_years = pulse.time
            pulse_prop = pulse.proportions[0]
            break

    times["t_nd_migration"] = pulse_time_years / gen_time
    times["t_nd_samples"] = pulse_time_years / gen_time

    params_extracted = {
        "gen_time": gen_time,
        "admixture_nd": pulse_prop,
        "admixture_modern": graph["MX"].proportions if "MX" in graph else [],
    }

    return params_extracted, times


def history_archaic(file_yml, prms, t, seed):
    graph = demes.load(file_yml)
    demography = msprime.Demography.from_demes(graph)

    samples = [
        msprime.SampleSet(prms["n_mexicans"], ploidy=prms["ploidy"], population="MX"),
        msprime.SampleSet(prms["n_eu"], ploidy=prms["ploidy"], population="EU"),
    ]

    ts = msprime.sim_ancestry(
        samples=samples,
        sequence_length=prms["chrom_length"],
        recombination_rate=prms["recomb_rate"],
        demography=demography,
        random_seed=seed,
        record_migrations=True,
    )

    if prms.get("simulate_mutations", False):
        ts = msprime.sim_mutations(
            ts,
            rate=prms["mut_rate"],
            random_seed=seed,
        )

    return ts


def get_migrating_tracts_ind(ts, pop_name, ind, T_anc):
    pop_id = -1

    for p in ts.populations():
        if p.metadata.get("name") == pop_name:
            pop_id = p.id
            break

    if pop_id == -1:
        return []

    eps = 1e-5
    tables = ts.tables

    mask = (
        (tables.migrations.dest == pop_id)
        & (tables.migrations.time >= T_anc - eps)
        & (tables.migrations.time <= T_anc + eps)
    )

    idx = np.where(mask)[0]

    mig_lookup = {}

    if len(idx) > 0:
        nodes = tables.migrations.node[idx]
        lefts = tables.migrations.left[idx]
        rights = tables.migrations.right[idx]

        for i, n in enumerate(nodes):
            mig_lookup.setdefault(n, []).append((lefts[i], rights[i]))

    for n in mig_lookup:
        mig_lookup[n].sort()

    tracts = []
    nodes_time = ts.nodes_time

    for tree in ts.trees():
        if tree.interval.left == tree.interval.right:
            continue

        anc = ind

        if nodes_time[anc] > T_anc + eps:
            continue

        parent = tree.parent(anc)

        while parent != tskit.NULL:
            if anc in mig_lookup:
                for m_left, m_right in mig_lookup[anc]:
                    start = max(tree.interval.left, m_left)
                    end = min(tree.interval.right, m_right)

                    if start < end:
                        if tracts and tracts[-1][1] == start:
                            tracts[-1][1] = end
                        else:
                            tracts.append([start, end])

            if nodes_time[anc] >= T_anc - eps:
                break

            anc = parent
            parent = tree.parent(anc)

    return tracts


def get_5state_tracts_dataframe(ts, times, mx_pop="MX", neand_pop="NEAND"):
    t_id = -1

    for p in ts.populations():
        if p.metadata.get("name") == mx_pop:
            t_id = p.id
            break

    if t_id == -1:
        return pd.DataFrame()

    nodes = ts.samples(population=t_id)

    if len(nodes) == 0:
        return pd.DataFrame()

    ind_ids = np.unique(ts.nodes_individual[nodes])
    ind_ids = ind_ids[ind_ids != -1]

    data = []

    for ind in ind_ids:
        individual = ts.individual(ind)

        for i, node in enumerate(individual.nodes):
            name = f"{mx_pop}_{ind}_{i + 1}"

            raw_eu = get_migrating_tracts_ind(ts, "EU", node, times["recent"])
            raw_na = get_migrating_tracts_ind(ts, "NA", node, times["recent"])
            raw_af = get_migrating_tracts_ind(ts, "AF", node, times["recent"])
            raw_nd = get_migrating_tracts_ind(ts, neand_pop, node, times["ancient"])

            nd_eu, clean_eu = intersect_and_subtract(raw_eu, raw_nd)
            nd_na, clean_na = intersect_and_subtract(raw_na, raw_nd)

            # No length filter by default. If MIN_ND_TRACT is set above 0, this becomes active.
            nd_eu = filter_min_length(nd_eu, MIN_ND_TRACT)
            nd_na = filter_min_length(nd_na, MIN_ND_TRACT)
            clean_af = raw_af

            def add_entries(tracts, label):
                for s, e in tracts:
                    data.append(
                        {
                            "Sample": name,
                            "Start": int(s),
                            "End": int(e),
                            "Length": int(e - s),
                            "State": label,
                        }
                    )

            add_entries(clean_eu, "EU")
            add_entries(nd_eu, "ND_EU")
            add_entries(clean_na, "NA")
            add_entries(nd_na, "ND_NA")
            add_entries(clean_af, "AF")

    df = pd.DataFrame(data)

    if not df.empty:
        df = df.sort_values(by=["Sample", "Start"])
        df = df[["Sample", "Start", "End", "Length", "State"]]

    return df


def get_ibs_tracts_dataframe(ts, times, ibs_pop="EU", neand_pop="NEAND", chrom_length=30_000_000):
    t_id = -1

    for p in ts.populations():
        if p.metadata.get("name") == ibs_pop:
            t_id = p.id
            break

    if t_id == -1:
        return pd.DataFrame()

    nodes = ts.samples(population=t_id)

    if len(nodes) == 0:
        return pd.DataFrame()

    ind_ids = np.unique(ts.nodes_individual[nodes])
    ind_ids = ind_ids[ind_ids != -1]

    data = []
    whole = [[0, chrom_length]]

    for ind in ind_ids:
        individual = ts.individual(ind)

        for i, node in enumerate(individual.nodes):
            name = f"{ibs_pop}_{ind}_{i + 1}"

            raw_nd = get_migrating_tracts_ind(
                ts,
                neand_pop,
                node,
                times["ancient"],
            )
            raw_nd = filter_min_length(raw_nd, MIN_ND_TRACT)

            nd_eu, clean_eu = intersect_and_subtract(whole, raw_nd)

            def add_entries(tracts, label):
                for s, e in tracts:
                    data.append(
                        {
                            "Sample": name,
                            "Start": int(s),
                            "End": int(e),
                            "Length": int(e - s),
                            "State": label,
                        }
                    )

            add_entries(clean_eu, "EU")
            add_entries(nd_eu, "ND_EU")

    df = pd.DataFrame(data)

    if not df.empty:
        df = df.sort_values(by=["Sample", "Start"])
        df = df[["Sample", "Start", "End", "Length", "State"]]

    return df


def build_truth_dataframe_from_ts(ts, times, prms):
    times_config = {
        "recent": times["t_mexican_admixture"],
        "ancient": times["t_nd_migration"],
    }

    df_mx = get_5state_tracts_dataframe(
        ts,
        times_config,
        mx_pop="MX",
        neand_pop="NEAND",
    )

    df_ibs = get_ibs_tracts_dataframe(
        ts,
        times_config,
        ibs_pop="EU",
        neand_pop="NEAND",
        chrom_length=prms["chrom_length"],
    )

    if not df_mx.empty:
        df_mx.insert(0, "Group", "MX")

    if not df_ibs.empty:
        df_ibs.insert(0, "Group", "IBS")

    return pd.concat([df_mx, df_ibs], ignore_index=True)


def process_one_chromosome(file_yml, seed, prms, t, out_dir):
    truth_path = os.path.join(out_dir, f"ground_truth_5state_{seed}.tsv")
    tree_path = os.path.join(out_dir, f"sim_seed_{seed}.trees")

    if REUSE_EXISTING_TRUTH and os.path.exists(truth_path):
        print(f"[seed {seed}] reuse existing truth TSV", flush=True)
        return pd.read_csv(truth_path, sep="\t")

    print(f"[seed {seed}] start", flush=True)

    simulated_new_ts = False
    if REUSE_EXISTING_TREES and os.path.exists(tree_path):
        print(f"[seed {seed}] reuse existing tree sequence", flush=True)
        ts = tskit.load(tree_path)
    else:
        ts = history_archaic(file_yml, prms, t, seed)
        simulated_new_ts = True

    df_t = build_truth_dataframe_from_ts(ts, t, prms)

    if not df_t.empty:
        df_t.insert(0, "CHR", seed)

    df_t.to_csv(
        truth_path,
        sep="\t",
        index=False,
    )

    if SAVE_TREES and simulated_new_ts:
        ts.dump(tree_path)

    del ts
    gc.collect()

    print(f"[seed {seed}] done rows={len(df_t)}", flush=True)

    return df_t

def truth_to_window_counts_fast(truth_df, window=1000, chrom_len=30_000_000):
    n_windows = int(np.ceil(chrom_len / window))
    out_rows = []

    for (chrom, group), sub in truth_df.groupby(["CHR", "Group"], sort=False):
        eu = np.zeros(n_windows, dtype=np.int64)
        nd = np.zeros(n_windows, dtype=np.int64)

        sub = sub[sub["State"].isin(["EU", "ND_EU"])]

        for _, r in sub.iterrows():
            start = int(r["Start"])
            end = int(r["End"])
            state = r["State"]

            if end <= start:
                continue

            start = max(0, start)
            end = min(chrom_len, end)

            w0 = start // window
            w1 = (end - 1) // window

            for w in range(w0, w1 + 1):
                ws = w * window
                we = min((w + 1) * window, chrom_len)

                ov_start = max(start, ws)
                ov_end = min(end, we)

                if ov_start >= ov_end:
                    continue

                bp = ov_end - ov_start

                if state == "EU":
                    eu[w] += bp
                elif state == "ND_EU":
                    nd[w] += bp

        window_start = np.arange(n_windows) * window
        window_end = np.minimum(window_start + window, chrom_len)

        df = pd.DataFrame(
            {
                "CHR": chrom,
                "Group": group,
                "Window": np.arange(n_windows),
                "WindowStart": window_start,
                "WindowEnd": window_end,
                "EU": eu,
                "ND_EU": nd,
            }
        )

        df["EU_background_bp"] = df["EU"] + df["ND_EU"]
        df["p_ND_EU"] = np.where(
            df["EU_background_bp"] > 0,
            df["ND_EU"] / df["EU_background_bp"],
            np.nan,
        )

        out_rows.append(df)

    return pd.concat(out_rows, ignore_index=True)


def compute_effect_table_fast(truth_df, window=1000, chrom_len=30_000_000, min_eu_bp=0):
    win = truth_to_window_counts_fast(
        truth_df,
        window=window,
        chrom_len=chrom_len,
    )

    mx = win[win["Group"] == "MX"].copy()
    ibs = win[win["Group"] == "IBS"].copy()

    mx = mx.rename(
        columns={
            "EU": "EU_MX",
            "ND_EU": "ND_EU_MX",
            "EU_background_bp": "EU_background_bp_MX",
            "p_ND_EU": "p_MX",
        }
    )

    ibs = ibs.rename(
        columns={
            "EU": "EU_IBS",
            "ND_EU": "ND_EU_IBS",
            "EU_background_bp": "EU_background_bp_IBS",
            "p_ND_EU": "p_IBS",
        }
    )

    out = mx.merge(
        ibs[
            [
                "CHR",
                "Window",
                "WindowStart",
                "WindowEnd",
                "EU_IBS",
                "ND_EU_IBS",
                "EU_background_bp_IBS",
                "p_IBS",
            ]
        ],
        on=["CHR", "Window", "WindowStart", "WindowEnd"],
        how="inner",
    )

    out = out[
        (out["EU_background_bp_MX"] >= min_eu_bp)
        & (out["EU_background_bp_IBS"] >= min_eu_bp)
    ].copy()

    out["effect_delta"] = out["p_MX"] - out["p_IBS"]
    out["effect_log2_ratio"] = np.log2(
        (out["p_MX"] + 1e-9) / (out["p_IBS"] + 1e-9)
    )

    return out


def make_quantile_table(delta, label):
    delta = pd.Series(delta).dropna()
    delta = delta[np.isfinite(delta)]

    quantile_levels = [
        0.0001,
        0.001,
        0.005,
        0.01,
        0.05,
        0.50,
        0.95,
        0.99,
        0.995,
        0.999,
        0.9999,
    ]

    quantile_table = pd.DataFrame(
        {
            "calibration_set": label,
            "quantile": quantile_levels,
            "effect_delta_threshold": [delta.quantile(q) for q in quantile_levels],
        }
    )

    quantile_table["tail"] = np.where(
        quantile_table["quantile"] < 0.5,
        "depletion",
        np.where(quantile_table["quantile"] > 0.5, "excess", "median"),
    )

    quantile_table["one_sided_p_level"] = np.where(
        quantile_table["quantile"] < 0.5,
        quantile_table["quantile"],
        1 - quantile_table["quantile"],
    )

    quantile_table["rule"] = np.where(
        quantile_table["tail"] == "depletion",
        "effect_delta < " + quantile_table["effect_delta_threshold"].round(6).astype(str),
        np.where(
            quantile_table["tail"] == "excess",
            "effect_delta > " + quantile_table["effect_delta_threshold"].round(6).astype(str),
            "median",
        ),
    )

    summary = delta.describe().reset_index()
    summary.columns = ["statistic", "value"]
    summary.insert(0, "calibration_set", label)

    return quantile_table, summary


def make_selected_thresholds(quantile_tables):
    rows = []

    configs = [
        ("excess", "suggestive", "excess_pIBS_positive", 0.999),
        ("excess", "strong", "excess_pIBS_positive", 0.9999),
        ("depletion", "suggestive", "all_windows", 0.001),
        ("depletion", "strong", "all_windows", 0.0001),
    ]

    for direction, strength, calibration_set, q in configs:
        sub = quantile_tables[
            (quantile_tables["calibration_set"] == calibration_set)
            & (quantile_tables["quantile"] == q)
        ]

        if sub.empty:
            threshold = np.nan
            rule = "unavailable"
        else:
            threshold = float(sub.iloc[0]["effect_delta_threshold"])
            if direction == "excess":
                rule = f"effect_delta > {threshold:.6f}"
            else:
                rule = f"effect_delta < {threshold:.6f}"

        rows.append(
            {
                "direction": direction,
                "strength": strength,
                "calibration_set": calibration_set,
                "quantile": q,
                "effect_delta_threshold": threshold,
                "rule": rule,
            }
        )

    return pd.DataFrame(rows)


def write_threshold_tables(neutral_effects, out_dir):
    delta_all = neutral_effects["effect_delta"].dropna()
    delta_all = delta_all[np.isfinite(delta_all)]

    # For excess, p_IBS = 0 makes the one-sided binomial test degenerate:
    # any MXL ND_EU observation has probability 0 under Binomial(N, 0).
    # We therefore provide a separate excess-eligible calibration set.
    delta_excess_eligible = neutral_effects.loc[
        (neutral_effects["p_IBS"] > 0) & np.isfinite(neutral_effects["effect_delta"]),
        "effect_delta",
    ].dropna()

    q_all, summary_all = make_quantile_table(delta_all, "all_windows")
    q_excess, summary_excess = make_quantile_table(
        delta_excess_eligible,
        "excess_pIBS_positive",
    )

    quantile_tables = pd.concat([q_all, q_excess], ignore_index=True)
    summary = pd.concat([summary_all, summary_excess], ignore_index=True)
    selected_thresholds = make_selected_thresholds(quantile_tables)

    settings = pd.DataFrame(
        [
            {
                "calibration_type": "truth_level",
                "inference_used": False,
                "window": WINDOW,
                "min_nd_tract": MIN_ND_TRACT,
                "chrom_len": CHROM_LEN,
                "total_len": TOTAL_LEN,
                "num_chromosomes": NUM_CHROMOSOMES,
                "n_mexicans": N_MEXICANS,
                "n_ibs": N_IBS,
                "min_eu_haps": MIN_EU_HAPS,
                "min_eu_bp": MIN_EU_HAPS * WINDOW,
                "n_windows": len(neutral_effects),
                "n_finite_effects_all_windows": len(delta_all),
                "n_finite_effects_excess_pIBS_positive": len(delta_excess_eligible),
                "reuse_existing_truth": REUSE_EXISTING_TRUTH,
                "reuse_existing_trees": REUSE_EXISTING_TREES,
                "save_trees": SAVE_TREES,
                "simulate_mutations": SIMULATE_MUTATIONS,
                "note": (
                    "Truth-level neutral calibration. Thresholds use true simulated "
                    "tracts, not DAIseg-inferred labels. No archaic-tract length filter "
                    "is applied when MIN_ND_TRACT=0. Excess thresholds should be taken "
                    "from the p_IBS>0 calibration set; depletion thresholds use all windows."
                ),
            }
        ]
    )

    neutral_effects.to_csv(
        os.path.join(out_dir, "neutral_effects_by_window.tsv"),
        sep="\t",
        index=False,
    )

    summary.to_csv(
        os.path.join(out_dir, "neutral_effect_delta_summary.tsv"),
        sep="\t",
        index=False,
    )

    quantile_tables.to_csv(
        os.path.join(out_dir, "neutral_effect_delta_quantile_thresholds.tsv"),
        sep="\t",
        index=False,
    )

    selected_thresholds.to_csv(
        os.path.join(out_dir, "selected_truth_level_delta_thresholds.tsv"),
        sep="\t",
        index=False,
    )

    settings.to_csv(
        os.path.join(out_dir, "threshold_settings.tsv"),
        sep="\t",
        index=False,
    )

def load_existing_all_truth_if_available(out_dir):
    """Load combined truth table if it already exists.

    This is the fastest reuse mode. It assumes the file was generated with the
    tract-filtering choice you want. If an older filtered truth table is present
    and you want unfiltered tracts, remove it or use a different OUT_DIR.
    """
    path = os.path.join(out_dir, "ground_truth_5state_all.tsv")
    if REUSE_EXISTING_TRUTH and os.path.exists(path):
        print(f"[reuse] loading combined truth table: {path}", flush=True)
        return pd.read_csv(path, sep="\t")
    return None


def warn_if_existing_settings_look_filtered(out_dir):
    settings_path = os.path.join(out_dir, "threshold_settings.tsv")
    if not os.path.exists(settings_path):
        return
    try:
        settings = pd.read_csv(settings_path, sep="\t")
    except Exception:
        return
    if "min_nd_tract" in settings.columns:
        old_min = pd.to_numeric(settings["min_nd_tract"], errors="coerce").dropna()
        if len(old_min) and old_min.iloc[0] > 0 and MIN_ND_TRACT == 0:
            print(
                "[warning] Existing threshold_settings.tsv reports min_nd_tract > 0, "
                "but this run uses MIN_ND_TRACT=0. If ground_truth_5state_all.tsv "
                "was generated after filtering short tracts, it cannot be unfiltered. "
                "Delete the old truth TSVs or use a clean OUT_DIR if needed.",
                flush=True,
            )


def main():
    yaml_file = YAML_FILE
    if yaml_file is None:
        yaml_candidates = glob.glob("*.yml") + glob.glob("*.yaml")
        if not yaml_candidates:
            raise FileNotFoundError("No .yml/.yaml demography file found and YAML_FILE is None.")
        yaml_file = yaml_candidates[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    warn_if_existing_settings_look_filtered(OUT_DIR)

    gt_all = load_existing_all_truth_if_available(OUT_DIR)

    if gt_all is None:
        yml_params, time_dict = extract_parameters_from_demes(yaml_file)

        params = {
            "n_mexicans": N_MEXICANS,
            "n_eu": N_IBS,
            "n_na": 0,
            "n_af": 0,
            "n_nd": 0,
            "ploidy": PLOIDY,
            "chrom_length": CHROM_LEN,
            "recomb_rate": RECOMB_RATE,
            "mut_rate": MUT_RATE,
            "simulate_mutations": SIMULATE_MUTATIONS,
            "gen_time": yml_params["gen_time"],
            "admixture_nd": yml_params["admixture_nd"],
            "admixture_modern": yml_params["admixture_modern"],
        }

        seeds = [BASE_SEED + i for i in range(NUM_CHROMOSOMES)]

        tasks = [
            (yaml_file, seed, params, time_dict, OUT_DIR)
            for seed in seeds
        ]

        with mp.Pool(processes=NUM_THREADS) as pool:
            gt_results = pool.starmap(process_one_chromosome, tasks)

        gt_all = pd.concat(gt_results, ignore_index=True)

        gt_all.to_csv(
            os.path.join(OUT_DIR, "ground_truth_5state_all.tsv"),
            sep="\t",
            index=False,
        )

    neutral_effects = compute_effect_table_fast(
        gt_all,
        window=WINDOW,
        chrom_len=CHROM_LEN,
        min_eu_bp=MIN_EU_HAPS * WINDOW,
    )

    write_threshold_tables(
        neutral_effects=neutral_effects,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
