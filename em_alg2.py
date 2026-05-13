import numpy as np
from numba import jit, prange, config, set_num_threads
import json
import hmm2 as hmm
import gc
import sys, os
import pandas as pd

import math

if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = '32'

@jit(nopython=True)
def forward_backward_normalized(emit, trans, start):
    """
    Standard Forward-Backward with scaling.
    """
    N, n_states = emit.shape

    alpha = np.zeros((N, n_states))
    scales = np.zeros(N)

    # Init 
    for s in range(n_states):
        alpha[0, s] = start[s] * emit[0, s]

    scales[0] = 1.0 / (np.sum(alpha[0]) + 1e-300)
    alpha[0] *= scales[0]

    for t in range(1, N):
        for s in range(n_states):
            acc = 0.0
            for p in range(n_states):
                acc += alpha[t-1, p] * trans[p, s]
            alpha[t, s] = acc * emit[t, s]

        scales[t] = 1.0 / (np.sum(alpha[t]) + 1e-300)
        alpha[t] *= scales[t]

    log_lik = -np.sum(np.log(scales + 1e-300))

    #  BACKWARD
    beta = np.zeros((N, n_states))
    beta[N-1, :] = scales[N-1]

    for t in range(N-2, -1, -1):
        for s in range(n_states):
            acc = 0.0
            for next_s in range(n_states):
                acc += trans[s, next_s] * emit[t+1, next_s] * beta[t+1, next_s]
            beta[t, s] = acc * scales[t]

    # GAMMA 
    gamma = alpha * beta
    for t in range(N):
        norm = np.sum(gamma[t]) + 1e-300
        gamma[t] /= norm

    return gamma, log_lik

# ==========================================
# 2. E-STEP 
# ==========================================

@jit(nopython=True, parallel=True)
def e_step_unified(emit, trans, start, O_EU, O_NA, O_AF, O_ND, L_mod, L_anc):

    M, N, n_states = emit.shape

    numerators = np.zeros((M, 5))
    denominators = np.zeros((M, 5))
    total_log_lik = 0.0

    for m in prange(M):
        # Forward-Backward для гаплотипа m
        gamma, log_lik = forward_backward_normalized(emit[m], trans, start)
        total_log_lik += log_lik

        # Наблюдения для гаплотипа m
        o_eu = O_EU[m]
        o_na = O_NA[m]
        o_af = O_AF[m]
        o_nd = O_ND[m]
        
        # Callability для гаплотипа m
        l_mod = L_mod[m]
        l_anc = L_anc[m]

        num_i = 0.0;   den_i = 0.0    # λ_i: rate интрогрессии (ND)
        num_n = 0.0;   den_n = 0.0    # λ_n: rate не-интрогрессии
        num_af = 0.0;  den_af = 0.0   # λ_af: rate африканский
        num_ea = 0.0;  den_ea = 0.0   # λ_ea: rate евразийский
        num_mex = 0.0; den_mex = 0.0  # λ_mex: rate мексиканский

        for t in range(N):
            g0 = gamma[t, 0]  # P(state=EU | observations)
            g1 = gamma[t, 1]  # P(state=ND_EU | observations)
            g2 = gamma[t, 2]  # P(state=NA | observations)
            g3 = gamma[t, 3]  # P(state=ND_NA | observations)
            g4 = gamma[t, 4]  # P(state=AF | observations)

            if l_anc[t] > 0:
                num_i += (g1 + g3) * o_nd[t]
                den_i += (g1 + g3) * l_anc[t]

            if l_anc[t] > 0:
                num_n += (g0 + g2 + g4) * o_nd[t]
                den_n += (g0 + g2 + g4) * l_anc[t]
            
            if l_mod[t] > 0:
                num_n += (g1 + g3) * o_af[t]
                den_n += (g1 + g3) * l_mod[t]

            if l_mod[t] > 0:

                num_af += (g0 + g2) * o_af[t]
                den_af += (g0 + g2) * l_mod[t]
                

                num_af += g4 * (o_eu[t] + o_na[t])
                den_af += g4 * l_mod[t] * 2  # два источника: EU и NA

            if l_mod[t] > 0:
                num_ea += (g0 + g1) * o_na[t]
                num_ea += (g2 + g3) * o_eu[t]
                den_ea += (g0 + g1 + g2 + g3) * l_mod[t]

            if l_mod[t] > 0:
                num_mex += (g0 + g1) * o_eu[t]
                num_mex += (g2 + g3) * o_na[t]
                num_mex += g4 * o_af[t]
                den_mex += (g0 + g1 + g2 + g3 + g4) * l_mod[t]

        numerators[m, 0] = num_i
        numerators[m, 1] = num_n
        numerators[m, 2] = num_af
        numerators[m, 3] = num_ea
        numerators[m, 4] = num_mex

        denominators[m, 0] = den_i
        denominators[m, 1] = den_n
        denominators[m, 2] = den_af
        denominators[m, 3] = den_ea
        denominators[m, 4] = den_mex

    return numerators, denominators, total_log_lik





# GLOBAL EM
def run_batch_em_pipeline(json_files_list, output_combined_file=None, max_iter=20, tol=1e-8):
    """
    Batch EM с unified обработкой missing data.
    """

    batch_data = []
    print(f"[EM] Loading {len(json_files_list)} files...")


    init_lmbd = None
    trans_params = None

    for j_file in json_files_list:
        with open(j_file, "r") as f:
            data = json.load(f)
        prefix = data.get("prefix", "")

        # Загрузка наблюдений
        tsv_path = os.path.join(prefix, data["data"])
        bed_path_1k = os.path.join(prefix, data["window_callability"]["Thousand_genomes"])
        bed_path_nd = os.path.join(prefix, data["window_callability"]["Nd_1k_genomes"])

        obs_seq = hmm.obs.process_data(tsv_path, bed_path_1k)
        O_EU, O_NA, O_AF, O_ND, names = hmm.prepare_matrices_from_dict(obs_seq)
        chrom_id = str(data.get("CHROM", "unknown"))
        debug_obs_path = f"/home/ailina/mx.vs.mx/obs_debug_project2_{chrom_id}.npz"
        print("[DEBUG] saving obs to", debug_obs_path)

        np.savez_compressed(
        debug_obs_path,
        O_EU=O_EU,
        O_NA=O_NA,
        O_AF=O_AF,
        O_ND=O_ND,
        names=np.array(names, dtype=object),
        )

        M, N = O_EU.shape

        # Загрузка параметров
        if init_lmbd is None:
            init_lmbd, trans_params = hmm.init_params_from_json(data)

        # Загрузка callability
        c1 = np.loadtxt(bed_path_1k, usecols=-1)
        
        # Обработка отсутствующего ND файла
        if not bed_path_nd or not os.path.exists(bed_path_nd):
            cn = np.zeros(N)
            print(f"  [INFO] {j_file}: нет ND callability, cn=0")
        else:
            cn = np.loadtxt(bed_path_nd, usecols=-1)

        # Выравнивание длин
        min_len = min(N, len(c1), len(cn))
        c1 = c1[:min_len]
        cn = cn[:min_len]
        O_EU = O_EU[:, :min_len]
        O_NA = O_NA[:, :min_len]
        O_AF = O_AF[:, :min_len]
        O_ND = O_ND[:, :min_len]

        # >>> НОРМАЛИЗАЦИЯ: L=0 → O=0 <<<
        no_modern = (c1 == 0)
        no_ancient = (cn == 0)
        
        if np.any(no_modern):
            n_fixed = (np.sum(O_EU[:, no_modern] > 0) + 
                       np.sum(O_NA[:, no_modern] > 0) + 
                       np.sum(O_AF[:, no_modern] > 0))
            if n_fixed > 0:
                print(f"  [NORM] Обнулено {n_fixed} modern наблюдений (L_mod=0)")
            O_EU[:, no_modern] = 0
            O_NA[:, no_modern] = 0
            O_AF[:, no_modern] = 0
        
        if np.any(no_ancient):
            n_fixed = np.sum(O_ND[:, no_ancient] > 0)
            if n_fixed > 0:
                print(f"  [NORM] Обнулено {n_fixed} ancient наблюдений (L_anc=0)")
            O_ND[:, no_ancient] = 0

        # Расширяем callability до (M, N)
        L_mod = np.tile(c1, (M, 1))
        L_anc = np.tile(cn, (M, 1))


        # Статистика по окнам
        n_both = np.sum((c1 > 0) & (cn > 0))
        n_mod_only = np.sum((c1 > 0) & (cn == 0))
        n_anc_only = np.sum((c1 == 0) & (cn > 0))
        n_neither = np.sum((c1 == 0) & (cn == 0))
        
        print(f"  [INFO] {j_file}: {min_len} окон")
        print(f"         Both: {n_both}, ModOnly: {n_mod_only}, AncOnly: {n_anc_only}, Neither: {n_neither}")

        batch_data.append({
            "obs": (O_EU, O_NA, O_AF, O_ND),
            "cov": (L_mod, L_anc),
            "meta": (names, data, min_len),
        })
        gc.collect()

    if not batch_data or init_lmbd is None:
        print("[EM] No data loaded, exiting.")
        return pd.DataFrame()

    print(f"[EM] Loaded {len(batch_data)} chunks. Starting Optimization...")

    # EM
    curr_lmbd = init_lmbd.copy()
    prev_ll = -np.inf

    print(f'[EM] Initial rates: {curr_lmbd}')

    # Матрица переходов (фиксирована)
    log_A = hmm.get_log_A_5x5(
        trans_params["Ti"], trans_params["Tmex"], trans_params["rr"],
        trans_params["w_len"], trans_params["a"], trans_params["b"],
        trans_params["c1"], trans_params["c2"],
    )
    trans_linear = np.exp(log_A)
    
    # Начальные вероятности
    n_states = 5
    start_linear = np.ones(n_states) / n_states

    for it in range(max_iter):
        total_nums = np.zeros(5)
        total_dens = np.zeros(5)
        iter_ll = 0.0

        # E-STEP: собираем статистики со всех батчей
        for batch in batch_data:
            O_EU, O_NA, O_AF, O_ND = batch["obs"]
            L_mod, L_anc = batch["cov"]

            # Эмиссии с unified функцией (автоматически обрабатывает L=0)
            log_emit = hmm.compute_emissions_unified(
                O_EU, O_NA, O_AF, O_ND,
                L_mod[0, :], L_anc[0, :],
                curr_lmbd
            )
            emit_linear = np.exp(log_emit)

            # E-step с unified функцией
            nums, dens, ll = e_step_unified(
                emit_linear, trans_linear, start_linear,
                O_EU, O_NA, O_AF, O_ND, L_mod, L_anc
            )

            total_nums += np.sum(nums, axis=0)
            total_dens += np.sum(dens, axis=0)
            iter_ll += ll

        # M-STEP
        new_lmbd = curr_lmbd.copy()
        min_exposure = 1e-8

        for k in range(5):
            if total_dens[k] > min_exposure:
                new_lmbd[k] = total_nums[k] / (total_dens[k] + 1e-20)
            else:
                # Нет данных для этого rate — не обновляем
                rate_names = ['λ_i', 'λ_n', 'λ_af', 'λ_ea', 'λ_mex']
                print(f"   [WARN] {rate_names[k]} not updated: denominator={total_dens[k]:.2e}")

        # Convergence
        diff = iter_ll - prev_ll
        print(f"Iter {it+1}: LL={iter_ll:.2f} | Delta={diff:.4f}")
        print(f"   Rates [I, N, AF, EA, MX]: {np.round(new_lmbd, 5)}")

        if abs(diff) < tol and it > 0:
            print("[EM] Converged.")
            curr_lmbd = new_lmbd
            break

        prev_ll = iter_ll
        curr_lmbd = new_lmbd

    print(f"[EM] Final Optimized Rates: {curr_lmbd}")

    # Viterbi
    print("[EM] Running Final Viterbi on all files...")
    all_rows = []

    for batch in batch_data:
        O_EU, O_NA, O_AF, O_ND = batch["obs"]
        L_mod, L_anc = batch["cov"]
        names, j_data, N = batch["meta"]

        L_vec_mod = L_mod[0, :]
        L_vec_anc = L_anc[0, :]

        # Unified run_hmm 
        paths, uninformative_mask = hmm.run_hmm(
            O_EU, O_NA, O_AF, O_ND, 
            L_vec_mod, L_vec_anc,
            curr_lmbd, 
            trans_params["rr"], trans_params["Ti"], trans_params["Tmex"],
            trans_params["a"], trans_params["b"], 
            trans_params["c1"], trans_params["c2"],
            window_len=trans_params["w_len"],
            mark_uninformative=True
        )

        # Формирование трактов
        dct_paths = {k: v for k, v in zip(names, paths)}
        out_dict = {}
        for nm in names:
            out_dict[nm] = hmm.get_tracts_5states(
                dct_paths[nm], 
                trans_params["w_len"],
                split_on_uninformative=False
            )

        # Обработка gaps
        chrom = j_data.get("CHROM", "unknown")
        if "gaps" in j_data and j_data["gaps"]:
            gp = j_data["gaps"]
            prefix = j_data.get("prefix", "")
            if not os.path.exists(gp) and not gp.startswith("/"):
                gp = os.path.join(prefix, gp)
            out_dict = hmm.clean_gaps(out_dict, gp, chrom)

        # Сохранение
        out_tsv = os.path.join(
            j_data.get("prefix", ""), 
            f"{j_data.get('output', 'daiseg')}.em.tsv"
        )
        with open(out_tsv, "w") as f:
            f.write("Sample\tCHROM\tStart\tEnd\tLength\tState\n")
            for samp, tracks in out_dict.items():
                for st_lbl, intervals in tracks.items():
                    for s, e in intervals:
                        length = e - s + 1
                        f.write(f"{samp}\t{chrom}\t{s}\t{e}\t{length}\t{st_lbl}\n")
                        all_rows.append({
                            "Sample": samp,
                            "CHROM": chrom,
                            "Start": s,
                            "End": e,
                            "Length": length,
                            "State": st_lbl,
                        })

    # Сохранение результатов
    if output_combined_file:
        print(f"[EM] Saving merged results to {output_combined_file}")
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df.to_csv(output_combined_file, sep="\t", index=False)

    return pd.DataFrame(all_rows)












'''

#========================================================================



from scipy.optimize import minimize



if 'NUMBA_NUM_THREADS' not in os.environ:
    os.environ['NUMBA_NUM_THREADS'] = '32'


@jit(nopython=True)
def forward_backward_xi_normalized(emit, trans, start):
    """
    Scaled Forward-Backward.

    Returns
    -------
    gamma : (N, K)
        Posterior state probabilities.
    xi_sum : (K, K)
        Sum of expected transitions over all adjacent windows.
    log_lik : float
        Log-likelihood of the sequence.
    """
    N, n_states = emit.shape

    # --- Forward ---
    alpha = np.zeros((N, n_states))
    scales = np.zeros(N)

    for s in range(n_states):
        alpha[0, s] = start[s] * emit[0, s]

    scales[0] = 1.0 / (np.sum(alpha[0]) + 1e-300)
    alpha[0] *= scales[0]

    for t in range(1, N):
        for s in range(n_states):
            acc = 0.0
            for p in range(n_states):
                acc += alpha[t - 1, p] * trans[p, s]
            alpha[t, s] = acc * emit[t, s]

        scales[t] = 1.0 / (np.sum(alpha[t]) + 1e-300)
        alpha[t] *= scales[t]

    log_lik = -np.sum(np.log(scales + 1e-300))

    # --- Backward ---
    beta = np.zeros((N, n_states))
    beta[N - 1, :] = scales[N - 1]

    for t in range(N - 2, -1, -1):
        for s in range(n_states):
            acc = 0.0
            for next_s in range(n_states):
                acc += trans[s, next_s] * emit[t + 1, next_s] * beta[t + 1, next_s]
            beta[t, s] = acc * scales[t]

    # --- Gamma ---
    gamma = np.zeros((N, n_states))
    for t in range(N):
        denom = 0.0
        for s in range(n_states):
            gamma[t, s] = alpha[t, s] * beta[t, s]
            denom += gamma[t, s]

        inv_denom = 1.0 / (denom + 1e-300)
        for s in range(n_states):
            gamma[t, s] *= inv_denom

    # --- Xi sum ---
    xi_sum = np.zeros((n_states, n_states))

    for t in range(N - 1):
        denom = 0.0

        for i in range(n_states):
            for j in range(n_states):
                denom += alpha[t, i] * trans[i, j] * emit[t + 1, j] * beta[t + 1, j]

        inv_denom = 1.0 / (denom + 1e-300)

        for i in range(n_states):
            for j in range(n_states):
                val = alpha[t, i] * trans[i, j] * emit[t + 1, j] * beta[t + 1, j]
                xi_sum[i, j] += val * inv_denom

    return gamma, xi_sum, log_lik


@jit(nopython=True)
def e_step_unified_with_xi(emit, trans, start, O_EU, O_NA, O_AF, O_ND, L_mod, L_anc):
    """
    Unified E-step with missing-data handling.

    Returns
    -------
    numerators : (M, 5)
    denominators : (M, 5)
    xi_totals : (M, K, K)
    total_log_lik : float
    """
    M, N, n_states = emit.shape

    numerators = np.zeros((M, 5))
    denominators = np.zeros((M, 5))
    xi_totals = np.zeros((M, n_states, n_states))
    total_log_lik = 0.0

    for m in range(M):
        gamma, xi_sum, log_lik = forward_backward_xi_normalized(emit[m], trans, start)
        total_log_lik += log_lik
        xi_totals[m, :, :] = xi_sum

        o_eu = O_EU[m]
        o_na = O_NA[m]
        o_af = O_AF[m]
        o_nd = O_ND[m]

        l_mod = L_mod[m]
        l_anc = L_anc[m]

        num_i = 0.0
        den_i = 0.0

        num_n = 0.0
        den_n = 0.0

        num_af = 0.0
        den_af = 0.0

        num_ea = 0.0
        den_ea = 0.0

        num_mex = 0.0
        den_mex = 0.0

        for t in range(N):
            g0 = gamma[t, 0]  # EU
            g1 = gamma[t, 1]  # ND_EU
            g2 = gamma[t, 2]  # NA
            g3 = gamma[t, 3]  # ND_NA
            g4 = gamma[t, 4]  # AF

            if l_anc[t] > 0:
                num_i += (g1 + g3) * o_nd[t]
                den_i += (g1 + g3) * l_anc[t]

                num_n += (g0 + g2 + g4) * o_nd[t]
                den_n += (g0 + g2 + g4) * l_anc[t]

            if l_mod[t] > 0:
                num_n += (g1 + g3) * o_af[t]
                den_n += (g1 + g3) * l_mod[t]

                num_af += (g0 + g2) * o_af[t]
                den_af += (g0 + g2) * l_mod[t]

                num_af += g4 * (o_eu[t] + o_na[t])
                den_af += g4 * l_mod[t] * 2.0

                num_ea += (g0 + g1) * o_na[t]
                num_ea += (g2 + g3) * o_eu[t]
                den_ea += (g0 + g1 + g2 + g3) * l_mod[t]

                num_mex += (g0 + g1) * o_eu[t]
                num_mex += (g2 + g3) * o_na[t]
                num_mex += g4 * o_af[t]
                den_mex += (g0 + g1 + g2 + g3 + g4) * l_mod[t]

        numerators[m, 0] = num_i
        numerators[m, 1] = num_n
        numerators[m, 2] = num_af
        numerators[m, 3] = num_ea
        numerators[m, 4] = num_mex

        denominators[m, 0] = den_i
        denominators[m, 1] = den_n
        denominators[m, 2] = den_af
        denominators[m, 3] = den_ea
        denominators[m, 4] = den_mex

    return numerators, denominators, xi_totals, total_log_lik


# =========================================================================
# Transition GEM helpers
# =========================================================================

def _softmax3_from_uv(u: float, v: float):
    eu = math.exp(u)
    ev = math.exp(v)
    denom = eu + ev + 1.0
    a = eu / denom
    b = ev / denom
    d = 1.0 / denom
    return a, b, d



def _sigmoid(x: float):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)



def unpack_transition_params_abc(raw_params, c_max=0.2):
    """
    raw_params = [u, v, w] -> (a, b, c, d)
    with:
        a, b, d > 0, a+b+d = 1
        c in (0, c_max)
    """
    u, v, w = raw_params
    a, b, d = _softmax3_from_uv(u, v)
    c = c_max * _sigmoid(w)
    return a, b, c, d



def pack_transition_params_abc(a: float, b: float, c: float, c_max=0.2):
    eps = 1e-12

    a = min(max(a, eps), 1.0 - eps)
    b = min(max(b, eps), 1.0 - eps)
    d = 1.0 - a - b
    d = min(max(d, eps), 1.0 - eps)

    u = math.log(a / d)
    v = math.log(b / d)

    c = min(max(c, eps), c_max - eps)
    x = c / c_max
    w = math.log(x / (1.0 - x))

    return np.array([u, v, w], dtype=float)



def transition_objective_me_abc(raw_params, xi_sum_total, trans_params, c_max=0.2, l2_penalty=0.0, prior=None):
    """
    Negative expected complete-data log-likelihood for transitions.
    Assumes c1 = c2 = c and biological matrix is hmm.get_log_A_5x5(...).
    """
    a, b, c, d = unpack_transition_params_abc(raw_params, c_max=c_max)

    log_A = hmm.get_log_A_5x5(
        trans_params["Ti"],
        trans_params["Tmex"],
        trans_params["rr"],
        trans_params["w_len"],
        a,
        b,
        c,
        c,
    )

    obj = -np.sum(xi_sum_total * log_A)

    if l2_penalty > 0.0 and prior is not None:
        diff = np.asarray(raw_params) - np.asarray(prior)
        obj += l2_penalty * np.sum(diff * diff)

    return float(obj)



def optimize_transition_params_me_abc(
    xi_sum_total,
    trans_params,
    init_a,
    init_b,
    init_c,
    c_max=0.2,
    l2_penalty=0.0,
    maxiter=200,
):
    """
    Numerical M-step for biological transition parameters:
        a, b, c with c1 = c2 = c and d = 1-a-b.
    """
    x0 = pack_transition_params_abc(init_a, init_b, init_c, c_max=c_max)

    res = minimize(
        transition_objective_me_abc,
        x0=x0,
        args=(xi_sum_total, trans_params, c_max, l2_penalty, x0.copy()),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )

    a, b, c, d = unpack_transition_params_abc(res.x, c_max=c_max)

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "d": float(d),
        "raw": np.asarray(res.x, dtype=float),
        "success": bool(res.success),
        "fun": float(res.fun),
        "message": str(res.message),
    }



def m_step_update_emissions(total_nums, total_dens, old_lmbd, min_rate=1e-12, min_exposure=1e-8):
    new_lmbd = np.asarray(old_lmbd, dtype=float).copy()

    for k in range(len(new_lmbd)):
        if total_dens[k] > min_exposure:
            new_lmbd[k] = max(total_nums[k] / (total_dens[k] + 1e-20), min_rate)
        else:
            new_lmbd[k] = max(old_lmbd[k], min_rate)

    return new_lmbd


# =========================================================================
# GLOBAL EM / GEM
# =========================================================================

def run_batch_em_pipeline(json_files_list, output_combined_file=None, max_iter=20, tol=1e-8):
    """
    Batch GEM with unified missing-data handling.

    - Emissions are updated in closed form.
    - Biological transition parameters (a, b, c) are updated numerically.
    - Assumes the biologically motivated matrix is named hmm.get_log_A_5x5(...).
    - Uses c1 = c2 = c during GEM updates.
    """
    batch_data = []
    print(f"[EM] Loading {len(json_files_list)} files...")

    init_lmbd = None
    trans_params = None

    for j_file in json_files_list:
        with open(j_file, "r") as f:
            data = json.load(f)
        prefix = data.get("prefix", "")

        # Observations
        tsv_path = os.path.join(prefix, data["data"])
        bed_path_1k = os.path.join(prefix, data["window_callability"]["Thousand_genomes"])
        bed_path_nd = os.path.join(prefix, data["window_callability"]["Nd_1k_genomes"])

        obs_seq = hmm.obs.process_data(tsv_path, bed_path_1k)
        O_EU, O_NA, O_AF, O_ND, names = hmm.prepare_matrices_from_dict(obs_seq)
        M, N = O_EU.shape

        # Parameters
        if init_lmbd is None:
            init_lmbd, trans_params = hmm.init_params_from_json(data)

        # Callability
        c1 = np.loadtxt(bed_path_1k, usecols=-1)

        if not bed_path_nd or not os.path.exists(bed_path_nd):
            cn = np.zeros(N)
            print(f"  [INFO] {j_file}: нет ND callability, cn=0")
        else:
            cn = np.loadtxt(bed_path_nd, usecols=-1)

        # Align lengths
        min_len = min(N, len(c1), len(cn))
        c1 = c1[:min_len]
        cn = cn[:min_len]
        O_EU = O_EU[:, :min_len]
        O_NA = O_NA[:, :min_len]
        O_AF = O_AF[:, :min_len]
        O_ND = O_ND[:, :min_len]

        # Normalize impossible observations away
        no_modern = (c1 == 0)
        no_ancient = (cn == 0)

        if np.any(no_modern):
            n_fixed = (
                np.sum(O_EU[:, no_modern] > 0)
                + np.sum(O_NA[:, no_modern] > 0)
                + np.sum(O_AF[:, no_modern] > 0)
            )
            if n_fixed > 0:
                print(f"  [NORM] Обнулено {n_fixed} modern наблюдений (L_mod=0)")
            O_EU[:, no_modern] = 0
            O_NA[:, no_modern] = 0
            O_AF[:, no_modern] = 0

        if np.any(no_ancient):
            n_fixed = np.sum(O_ND[:, no_ancient] > 0)
            if n_fixed > 0:
                print(f"  [NORM] Обнулено {n_fixed} ancient наблюдений (L_anc=0)")
            O_ND[:, no_ancient] = 0

        L_mod = np.tile(c1, (M, 1))
        L_anc = np.tile(cn, (M, 1))

        n_both = np.sum((c1 > 0) & (cn > 0))
        n_mod_only = np.sum((c1 > 0) & (cn == 0))
        n_anc_only = np.sum((c1 == 0) & (cn > 0))
        n_neither = np.sum((c1 == 0) & (cn == 0))

        print(f"  [INFO] {j_file}: {min_len} окон")
        print(f"         Both: {n_both}, ModOnly: {n_mod_only}, AncOnly: {n_anc_only}, Neither: {n_neither}")

        batch_data.append({
            "obs": (O_EU, O_NA, O_AF, O_ND),
            "cov": (L_mod, L_anc),
            "meta": (names, data, min_len),
        })
        gc.collect()

    if not batch_data or init_lmbd is None:
        print("[EM] No data loaded, exiting.")
        return pd.DataFrame()

    print(f"[EM] Loaded {len(batch_data)} chunks. Starting Optimization...")

    # Initial parameters
    curr_lmbd = init_lmbd.copy()
    curr_a = float(trans_params["a"])
    curr_b = float(trans_params["b"])
    curr_c = float(trans_params["c1"])

    prev_ll = -np.inf

    print(f"[EM] Initial rates: {curr_lmbd}")
    print(f"[EM] Initial transitions: a={curr_a:.6f}, b={curr_b:.6f}, c={curr_c:.6f}")

    n_states = 5
    start_linear = np.ones(n_states) / n_states

    for it in range(max_iter):
        # Build current transition matrix
        log_A = hmm.get_log_A_5x5(
            trans_params["Ti"],
            trans_params["Tmex"],
            trans_params["rr"],
            trans_params["w_len"],
            curr_a,
            curr_b,
            curr_c,
            curr_c,
        )
        trans_linear = np.exp(log_A)

        total_nums = np.zeros(5)
        total_dens = np.zeros(5)
        total_xi_sum = np.zeros((n_states, n_states))
        iter_ll = 0.0

        # E-step
        for batch in batch_data:
            O_EU, O_NA, O_AF, O_ND = batch["obs"]
            L_mod, L_anc = batch["cov"]

            log_emit = hmm.compute_emissions_unified(
                O_EU, O_NA, O_AF, O_ND,
                L_mod[0, :],
                L_anc[0, :],
                curr_lmbd,
            )
            emit_linear = np.exp(log_emit)

            nums, dens, xi_totals, ll = e_step_unified_with_xi(
                emit_linear, trans_linear, start_linear,
                O_EU, O_NA, O_AF, O_ND, L_mod, L_anc,
            )

            total_nums += np.sum(nums, axis=0)
            total_dens += np.sum(dens, axis=0)
            total_xi_sum += np.sum(xi_totals, axis=0)
            iter_ll += ll

        # M-step: emissions
        new_lmbd = m_step_update_emissions(total_nums, total_dens, curr_lmbd)

        # M-step: biological transitions
        opt = optimize_transition_params_me_abc(
            xi_sum_total=total_xi_sum,
            trans_params=trans_params,
            init_a=curr_a,
            init_b=curr_b,
            init_c=curr_c,
            c_max=0.2,
            l2_penalty=1e-4,
            maxiter=200,
        )

        new_a = opt["a"]
        new_b = opt["b"]
        new_c = opt["c"]

        diff = iter_ll - prev_ll
        print(f"Iter {it + 1}: LL={iter_ll:.2f} | Delta={diff:.6f}")
        print(f"   Rates [I, N, AF, EA, MX]: {np.round(new_lmbd, 6)}")
        print(
            f"   Transitions: a={new_a:.6f}, b={new_b:.6f}, c={new_c:.6f}, "
            f"d={opt['d']:.6f} | success={opt['success']}"
        )
        if not opt["success"]:
            print(f"   [WARN] Transition optimizer: {opt['message']}")

        curr_lmbd = new_lmbd
        curr_a = new_a
        curr_b = new_b
        curr_c = new_c

        if abs(diff) < tol and it > 0:
            print("[EM] Converged.")
            break

        prev_ll = iter_ll

    print(f"[EM] Final Optimized Rates: {curr_lmbd}")
    print(f"[EM] Final Transitions: a={curr_a:.6f}, b={curr_b:.6f}, c={curr_c:.6f}")

    # Final Viterbi
    print("[EM] Running Final Viterbi on all files...")
    all_rows = []

    for batch in batch_data:
        O_EU, O_NA, O_AF, O_ND = batch["obs"]
        L_mod, L_anc = batch["cov"]
        names, j_data, N = batch["meta"]

        L_vec_mod = L_mod[0, :]
        L_vec_anc = L_anc[0, :]

        paths, uninformative_mask = hmm.run_hmm(
            O_EU, O_NA, O_AF, O_ND,
            L_vec_mod, L_vec_anc,
            curr_lmbd,
            trans_params["rr"], trans_params["Ti"], trans_params["Tmex"],
            curr_a, curr_b,
            curr_c, curr_c,
            window_len=trans_params["w_len"],
            mark_uninformative=True,
        )

        dct_paths = {k: v for k, v in zip(names, paths)}
        out_dict = {}
        for nm in names:
            out_dict[nm] = hmm.get_tracts_5states(
                dct_paths[nm],
                trans_params["w_len"],
                split_on_uninformative=True,
            )

        chrom = j_data.get("CHROM", "unknown")
        if "gaps" in j_data and j_data["gaps"]:
            gp = j_data["gaps"]
            prefix = j_data.get("prefix", "")
            if not os.path.exists(gp) and not gp.startswith("/"):
                gp = os.path.join(prefix, gp)
            out_dict = hmm.clean_gaps(out_dict, gp, chrom)

        out_tsv = os.path.join(
            j_data.get("prefix", ""),
            f"{j_data.get('output', 'daiseg')}.em.tsv",
        )
        with open(out_tsv, "w") as f:
            f.write("Sample\tCHROM\tStart\tEnd\tLength\tState\n")
            for samp, tracks in out_dict.items():
                for st_lbl, intervals in tracks.items():
                    for s, e in intervals:
                        length = e - s + 1
                        f.write(f"{samp}\t{chrom}\t{s}\t{e}\t{length}\t{st_lbl}\n")
                        all_rows.append({
                            "Sample": samp,
                            "CHROM": chrom,
                            "Start": s,
                            "End": e,
                            "Length": length,
                            "State": st_lbl,
                        })

    if output_combined_file:
        print(f"[EM] Saving merged results to {output_combined_file}")
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df.to_csv(output_combined_file, sep="\t", index=False)

    return pd.DataFrame(all_rows)

'''


