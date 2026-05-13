import json
import multiprocessing
import sys
import os
import numba
import numpy as np
import pandas as pd
import math

from numba import jit, prange
from scipy.stats import poisson

import obs  


def create_observations(tsv, bed):
    try:
        result = obs.process_data(tsv, bed)
        print(' Observation sequences for HMM created successfully!')
    except Exception as e:
        print(f"!!! Critical error: {e}")
        sys.exit(1)

    # Get number of states
    max_val, max_info = obs.get_number_states(result)
    print(f'Max differences in 1000bp window: {max_val + 1}')

    return result, max_val + 1


def prepare_matrices_from_dict(data_dict):
    hap_names = list(data_dict.keys())
    M = len(hap_names)
    if M == 0:
        raise ValueError("Dictionary contains no haplotypes.")

    N = len(data_dict[hap_names[0]])

    O_EU = np.zeros((M, N), dtype=np.int32)
    O_NA = np.zeros((M, N), dtype=np.int32)
    O_AF = np.zeros((M, N), dtype=np.int32)
    O_ND = np.zeros((M, N), dtype=np.int32)

    for i, hap in enumerate(hap_names):
        hap_data = np.array(data_dict[hap])
        O_EU[i, :] = hap_data[:, 0]
        O_NA[i, :] = hap_data[:, 1]
        O_AF[i, :] = hap_data[:, 2]
        O_ND[i, :] = hap_data[:, 3]

    return O_EU, O_NA, O_AF, O_ND, hap_names





def get_log_A_5x5(Ti, Tmex, r, L, a, b, c1, c2) -> np.ndarray:
    A = np.zeros((5, 5), dtype=float)

    d = 1.0 - a - b

    e_i  = math.exp(-Ti   * r * L)
    e_i_m = math.exp(-(Ti-Tmex) * r * L)
    e_m  = math.exp(-Tmex * r * L)
    e_ip = 1.0 - e_i
    e_mp = 1.0 - e_m
    e_i_mp = 1.0 - e_i_m

    f_a = e_i_mp * e_m + e_mp * a
    f_b = e_i_mp * e_m + e_mp * b

    c1p = 1.0 - c1
    c2p = 1.0 - c2

    # EU
    A[0, 0] = e_i  + f_a * c1p
    A[0, 1] = f_a * c1
    A[0, 2] = e_mp * b * c2p
    A[0, 3] = e_mp * b * c2
    A[0, 4] = e_mp * d

    # ND_EU
    A[1, 0] = f_a * c1p
    A[1, 1] = e_i  + f_a * c1
    A[1, 2] = e_mp * b * c2p
    A[1, 3] = e_mp * b * c2
    A[1, 4] = e_mp * d

    # NA
    A[2, 0] = e_mp * a * c1p
    A[2, 1] = e_mp * a * c1
    A[2, 2] = e_i  + f_b * c2p
    A[2, 3] = f_b * c2
    A[2, 4] = e_mp * d

    # ND_NA
    A[3, 0] = e_mp * a * c1p
    A[3, 1] = e_mp * a * c1
    A[3, 2] = f_b * c2p
    A[3, 3] = e_i  + f_b * c2
    A[3, 4] = e_mp * d

    # AF
    A[4, 0] = e_mp * a * c1p
    A[4, 1] = e_mp * a * c1
    A[4, 2] = e_mp * b * c2p
    A[4, 3] = e_mp * b * c2
    A[4, 4] = e_m + e_mp * d

    return np.log(np.maximum(A, 1e-300))




def init_params_from_json(data):
    p = data["parameters_initial"]
    gen_time = p["generation_time"]
    mu = p["mutation"]
    w_len = p["window_length"]
    rr = p["rr"]
    theta = (mu * w_len) / gen_time

            # начальные λ (из демографии)
    init_lmbd = np.array([
                p["t_introgression_c"] * theta,  # 0: i
                p["t_n_c"]             * theta,  # 1: n
                p["t_af_c"]            * theta,  # 2: af
                p["t_ea_c"]            * theta,  # 3: ea
                p["t_mexicans_c"]      * theta   # 4: mex
            ])

            # параметры переходов
    trans_params = {
                "Ti":   p["t_introgression"] / gen_time,
                "Tmex": p["t_mexicans"]      / gen_time,
                "rr":   rr,
                "a":    p["admixture_modern"][0],
                "b":    p["admixture_modern"][1],
                "c1":   p["admixture_nd"],
                "c2":   p["admixture_nd"],
                "w_len": w_len,
            }
    return init_lmbd, trans_params



def compute_emissions_unified(O_EU, O_NA, O_AF, O_ND, L_modern, L_ancient, lmbd):
    M, N = O_EU.shape
    n_states = 5
    log_emit = np.zeros((M, N, n_states))

    # Распаковываем rates
    r_i, r_n, r_af, r_ea, r_mex = lmbd
    eps = 1e-300  # защита от log(0)

    # Предвычисляем логарифмы rates
    ln_i   = np.log(r_i   + eps)
    ln_n   = np.log(r_n   + eps)
    ln_af  = np.log(r_af  + eps)
    ln_ea  = np.log(r_ea  + eps)
    ln_mex = np.log(r_mex + eps)


    # has_modern[t] = True если в окне t есть данные по современным панелям
    has_modern = (L_modern > 0)   # shape (N,)

    # has_ancient[t] = True если в окне t есть данные по неандертальцу
    has_ancient = (L_ancient > 0)  # shape (N,)

    # Если L=0, но O>0 — это ошибка в данных: откуда различия без покрытия?
    invalid_modern = ~has_modern & (
        (O_EU.sum(axis=0) > 0) |
        (O_NA.sum(axis=0) > 0) |
        (O_AF.sum(axis=0) > 0)
    )
    invalid_ancient = ~has_ancient & (O_ND.sum(axis=0) > 0)

    if np.any(invalid_modern):
        n_invalid = np.sum(invalid_modern)
        print(f"  {n_invalid} окон: L_modern=0, но наблюдения>0")

    if np.any(invalid_ancient):
        n_invalid = np.sum(invalid_ancient)
        print(f"  {n_invalid} окон: L_ancient=0, но O_ND>0")

    # Функция для безопасного вычисления Poisson log-score

    def safe_score(Obs, Length, ln_rate, rate_val, valid_mask):
        """
        log P(O=k | λ, L) ∝ k * log(λ) - λ * L
        1. valid_mask[t] = True, L[t] > 0:
           → Нормальный score: k * log(λ) - λ * L
        2. valid_mask[t] = False (L[t] = 0):
           → Score = 0 (нет информации, нейтральный вклад)
        """
        # Базовый Poisson score: k * log(λ) - λ * L
        score = (Obs * ln_rate) - (rate_val * Length)  # shape (M, N)
        # Зануляем невалидные окна (нет данных → нет информации)
        score[:, ~valid_mask] = 0.0
        return score


    eu_mex = safe_score(O_EU, L_modern, ln_mex, r_mex, has_modern)  # EU близко
    eu_ea  = safe_score(O_EU, L_modern, ln_ea,  r_ea,  has_modern)  # EU далеко (OoA)
    eu_af  = safe_score(O_EU, L_modern, ln_af,  r_af,  has_modern)  # EU очень далеко

    na_mex = safe_score(O_NA, L_modern, ln_mex, r_mex, has_modern)  # NA близко
    na_ea  = safe_score(O_NA, L_modern, ln_ea,  r_ea,  has_modern)  # NA далеко
    na_af  = safe_score(O_NA, L_modern, ln_af,  r_af,  has_modern)  # NA очень далеко

    af_mex = safe_score(O_AF, L_modern, ln_mex, r_mex, has_modern)  # AF близко
    af_af  = safe_score(O_AF, L_modern, ln_af,  r_af,  has_modern)  # AF "своё" расстояние
    af_n   = safe_score(O_AF, L_modern, ln_n,   r_n,   has_modern)  # AF нейтрально

    # Использует L_ancient и маску has_ancient
    nd_n = safe_score(O_ND, L_ancient, ln_n, r_n, has_ancient)  # Нет интрогрессии
    nd_i = safe_score(O_ND, L_ancient, ln_i, r_i, has_ancient)  # Есть интрогрессия

    log_emit[:, :, 0] = eu_mex + na_ea + af_af + nd_n
    log_emit[:, :, 1] = eu_mex + na_ea + af_n + nd_i
    log_emit[:, :, 2] = eu_ea + na_mex + af_af + nd_n
    log_emit[:, :, 3] = eu_ea + na_mex + af_n + nd_i
    log_emit[:, :, 4] = eu_af + na_af + af_mex + nd_n

    return log_emit






@jit(nopython=True, parallel=True)
def viterbi_fast(log_emit, log_trans, log_start):

    M, N, n_states = log_emit.shape
    paths = np.zeros((M, N), dtype=np.int32)

    for m in prange(M):
        viterbi = np.zeros((N, n_states))
        backpointer = np.zeros((N, n_states), dtype=np.int32)

        for s in range(n_states):
            viterbi[0, s] = log_start[s] + log_emit[m, 0, s]

        for i in range(1, N):
            for s in range(n_states):
                max_val = -1e200
                best_prev = 0
                for p in range(n_states):
                    val = viterbi[i - 1, p] + log_trans[p, s]
                    if val > max_val:
                        max_val = val
                        best_prev = p
                viterbi[i, s] = max_val + log_emit[m, i, s]
                backpointer[i, s] = best_prev

        best_final_state = 0
        max_final_val = -1e200
        for s in range(n_states):
            if viterbi[N - 1, s] > max_final_val:
                max_final_val = viterbi[N - 1, s]
                best_final_state = s
        paths[m, N - 1] = best_final_state

        for i in range(N - 2, -1, -1):
            paths[m, i] = backpointer[i + 1, paths[m, i + 1]]

    return paths





def run_hmm(O_EU, O_NA, O_AF, O_ND, L_mod, L_anc, 
            lmbd, rr, Ti, Tmex, a, b, c1, c2, 
            window_len=1000,
            mark_uninformative=False):  # Новый параметр!
    """
    mark_uninformative : bool
        Если False (по умолчанию): возвращает только paths 
        Если True: возвращает (paths, uninformative_mask)
    """

    log_emissions = compute_emissions_unified(
        O_EU, O_NA, O_AF, O_ND, L_mod, L_anc, lmbd
    )

    print("Calculating transition matrix...")
    log_A = get_log_A_5x5(Ti, Tmex, rr, window_len, a, b, c1, c2)

    n_states = 5
    log_start = np.full(n_states, -np.log(n_states))

    print("Running Viterbi...")
    paths = viterbi_fast(log_emissions, log_A, log_start)

    if mark_uninformative:
        uninformative_mask = (L_mod == 0) & (L_anc == 0)
        paths[:, uninformative_mask] = -1
        return paths, uninformative_mask
    else:
        return paths  







def clean_gaps(dct, gap_file, target_chrom):
    """Filters gaps from results."""
    print(' [post-process] Processing gaps...')
    raw_gaps = []
    target_c = target_chrom if target_chrom.startswith('chr') else 'chr' + target_chrom

    try:
        with open(gap_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3 or parts[0].startswith('#') or parts[0] == 'bin': continue
                curr_c, idx_s, idx_e = parts[0], 1, 2
                if curr_c.isdigit() and parts[1].startswith('chr'): 
                    curr_c, idx_s, idx_e = parts[1], 2, 3
                if not curr_c.startswith('chr'): curr_c = 'chr' + curr_c

                if curr_c == target_c:
                    try:
                        raw_gaps.append((int(parts[idx_s]), int(parts[idx_e]) - 1))
                    except: continue
    except FileNotFoundError:
        print(f" !!! Gap file not found: {gap_file}. Skipping.")
        return dct

    merged_gaps = []
    if raw_gaps:
        raw_gaps.sort()
        merged_gaps = [raw_gaps[0]]
        for curr in raw_gaps[1:]:
            prev = merged_gaps[-1]
            if curr[0] <= prev[1] + 1: merged_gaps[-1] = (prev[0], max(prev[1], curr[1]))
            else: merged_gaps.append(curr)

    def subtract(interval, gaps):
        start, end = interval
        res = []
        curr = start
        for g_s, g_e in gaps:
            if g_e < curr: continue
            if g_s > end: break
            if curr < g_s: res.append((curr, g_s - 1))
            curr = max(curr, g_e + 1)
        if curr <= end: res.append((curr, end))
        return res

    new_dct = {}
    for sample, categories in dct.items():
        new_dct[sample] = {}
        for cat, intervals in categories.items():
            cleaned = []
            for interval in intervals:
                if not merged_gaps: cleaned.append(interval)
                else: cleaned.extend(subtract(interval, merged_gaps))
            new_dct[sample][cat] = cleaned
    return new_dct


def get_tracts_5states(path_array, window_len, bed_windows=None,
                       split_on_uninformative=True):
    """
    Конвертирует массив состояний в интервалы
        {state_name: [(start, end), ...]}.
    """
    state_map = {
        0: "EU",
        1: "ND_EU",
        2: "NA",
        3: "ND_NA",
        4: "AF",
        -1: "UNINFORMATIVE"
    }

    # Инициализируем словарь для всех возможных состояний
    tracts = {name: [] for name in state_map.values()}

    if len(path_array) == 0:
        return tracts

    def get_coords(idx):
        """Получить геномные координаты для окна idx."""
        if bed_windows is not None:
            return bed_windows[idx]['s'], bed_windows[idx]['e']
        else:
            return idx * window_len, (idx + 1) * window_len - 1

    if split_on_uninformative:

        curr_state = path_array[0]
        tract_start_idx = 0

        for i in range(1, len(path_array)):
            if path_array[i] != curr_state:

                start_coord, _ = get_coords(tract_start_idx)
                _, end_coord = get_coords(i - 1)

                state_name = state_map.get(curr_state, f"State_{curr_state}")
                tracts[state_name].append((start_coord, end_coord))

                # Начинаем новый тракт
                curr_state = path_array[i]
                tract_start_idx = i

        # Последний тракт
        start_coord, _ = get_coords(tract_start_idx)
        _, end_coord = get_coords(len(path_array) - 1)
        state_name = state_map.get(curr_state, f"State_{curr_state}")
        tracts[state_name].append((start_coord, end_coord))
    
    else:
        # =================================================================
        # Режим 2: Игнорировать -1, включать в соседние тракты
        # =================================================================
        # Находим первое информативное окно
        first_informative = 0
        while first_informative < len(path_array) and path_array[first_informative] == -1:
            first_informative += 1
        
        if first_informative == len(path_array):
            # Все окна неинформативны
            start_coord, _ = get_coords(0)
            _, end_coord = get_coords(len(path_array) - 1)
            tracts["UNINFORMATIVE"].append((start_coord, end_coord))
            return tracts
        
        curr_state = path_array[first_informative]
        tract_start_idx = 0  # Включаем начальные uninformative
        
        for i in range(first_informative + 1, len(path_array)):
            if path_array[i] == -1:
                # Пропускаем неинформативные
                continue
            
            if path_array[i] != curr_state:
                # Информативное окно с другим состоянием
                start_coord, _ = get_coords(tract_start_idx)
                _, end_coord = get_coords(i - 1)
                
                state_name = state_map.get(curr_state, f"State_{curr_state}")
                tracts[state_name].append((start_coord, end_coord))
                
                curr_state = path_array[i]
                tract_start_idx = i
        
        # Последний тракт
        start_coord, _ = get_coords(tract_start_idx)
        _, end_coord = get_coords(len(path_array) - 1)
        state_name = state_map.get(curr_state, f"State_{curr_state}")
        tracts[state_name].append((start_coord, end_coord))
    
    return tracts


def run_daiseg_logic(json_file):
    """Main pipeline logic."""
    with open(json_file, 'r') as f: data = json.load(f)
    prefix = data.get("prefix", "")

    # Files
    tsv_path = os.path.join(prefix, data["data"])
    bed_path = os.path.join(prefix, data["window_callability"]["Thousand_genomes"])

    print(f" [Pipeline] Processing data from {tsv_path}...")
    obs_seq = obs.process_data(tsv_path, bed_path)
    if not obs_seq: raise ValueError("No observations found!")

    O_EU, O_NA, O_AF, O_ND, names = prepare_matrices_from_dict(obs_seq)



    np.savez_compressed(
        "obs_debug_project1.npz",
    O_EU=O_EU,
    O_NA=O_NA,
    O_AF=O_AF,
    O_ND=O_ND,
    names=np.array(names, dtype=object),
    )


    # Params
    p = data["parameters_initial"]
    theta = (p['mutation'] * p['window_length']) / p['generation_time']
    lmbd = [
        p['t_introgression_c'] * theta,
        p['t_n_c']             * theta,
        p['t_af_c']            * theta,
        p['t_ea_c']            * theta,
        p['t_mexicans_c']      * theta
    ]
    # Callability
    try:
        cal_1 = np.loadtxt(os.path.join(prefix, data["window_callability"]["Thousand_genomes"]), usecols=-1)
        cal_nd = np.loadtxt(os.path.join(prefix, data["window_callability"]["Nd_1k_genomes"]), usecols=-1)
    except:
        cal_1 = np.ones(O_EU.shape[1])
        cal_nd = np.ones(O_EU.shape[1])

    min_l = min(O_EU.shape[1], len(cal_1), len(cal_nd))
    O_EU, O_NA, O_AF, O_ND = O_EU[:,:min_l], O_NA[:,:min_l], O_AF[:,:min_l], O_ND[:,:min_l]
    cal_1, cal_nd = cal_1[:min_l], cal_nd[:min_l]

    # Run
    print(" Running HMM...")
    Ti_gen = p['t_introgression'] / p['generation_time']
    Tmex_gen = p['t_mexicans'] / p['generation_time'] 

    paths = run_hmm(
        O_EU, O_NA, O_AF, O_ND, cal_1, cal_nd, lmbd, p['rr'],
        Ti_gen, Tmex_gen,         p['admixture_modern'][0], p['admixture_modern'][1],
        p['admixture_nd'], p['admixture_nd'],
        window_len=p['window_length']
    )
    # Post-process
    dct = {k: v for k, v in zip(names, paths)}
    out_dict = {name: get_tracts_5states(dct[name], p['window_length']) for name in names}

    chrom = data["data"].split('.')[1] if '.' in data["data"] else "unknown"
    if "gaps" in data and data["gaps"]:
        gp = data["gaps"]
        if not os.path.exists(gp) and not gp.startswith('/'): gp = os.path.join(prefix, gp)
        out_dict = clean_gaps(out_dict, gp, chrom)

    # Save
    out_tsv = os.path.join(prefix, f"{data.get('output', 'daiseg_output')}.tsv")
    print(f" Saving to: {out_tsv}")
    rows = []
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("Sample\tCHROM\tStart\tEnd\tLength\tState\n")
        for samp, tracks in out_dict.items():
            for st, intervals in tracks.items():
                for s, e in intervals:
                    l = e - s + 1
                    f.write(f"{samp}\t{chrom}\t{s}\t{e}\t{l}\t{st}\n")
                    rows.append({"Sample": samp, "CHROM": chrom, "Start": s, "End": e, "Length": l, "State": st})
    
    return pd.DataFrame(rows), out_dict


# FOR PARALLELIZATION
_original_logic = run_daiseg_logic

def _worker_proxy(filepath):

    # Safety for Numba in multiprocessing
    try: numba.set_num_threads(1)
    except: pass 
    return _original_logic(filepath)

def run_daiseg(json_input):
    """Wrapper"""

    # Case 1: Single file string -> Standard
    if not isinstance(json_input, list):
        print(f" [Wrapper] Processing single file...")
        return _original_logic(json_input)

    # Case 2: Single item list -> Sequential
    if len(json_input) == 1:
        print(f" [Wrapper] Processing single file from list...")
        return [_original_logic(json_input[0])]

    # Case 3: Daemon -> No recursion
    if multiprocessing.current_process().daemon:
        return [_original_logic(f) for f in json_input]

    # Case 4: Multiple files -> Parallel
    MAX_WORKERS = 64
    cpu = multiprocessing.cpu_count()
    pool_size = max(1, min(cpu - 1, MAX_WORKERS, len(json_input))) if cpu > 4 else max(1, min(cpu, len(json_input)))

    print(f" [Wrapper] Parallelizing {len(json_input)} files on {pool_size} cores...")
    with multiprocessing.Pool(processes=pool_size) as pool:
        return pool.map(_worker_proxy, json_input)


