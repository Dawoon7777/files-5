#!/usr/bin/env python3
"""
Phase 4 RERUN: TWA Bias Mapping (CORRECTED)
=============================================
Standalone script — does NOT rely on __pycache__.

Key corrections:
  1. fit_kd=True → SD gets its own fitted kd
  2. SD prediction uses kd_SD_hat for separate D trajectory
  3. Multiple prediction concentrations (1.0x, 1.5x LC50)
  4. lc50_approx accounts for kd*T equilibrium fraction
"""
import sys, os, importlib

if 'core' in sys.modules:
    del sys.modules['core']
os.makedirs('results', exist_ok=True)

import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count

import core
importlib.reload(core)
from core import (generate_lc50_data, design_concentrations,
                  sim_damage_1c, sim_damage_1c_pulsed,
                  survival_classB, survival_SD,
                  fit_classB, fit_SD,
                  make_pulse_profile, profile_to_func,
                  compute_twa, twa_predict_survival)

# ── Configuration ────────────────────────────────────────────────────────────
KR_VALUES = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
BETA_TRUE = 5.0
K_TRUE = 0.45
N_REPS = 200
N_STARTS = 15
N_PER_CONC = 20
HB = 0.001
TH_MED_TRUE = 3.0

EXPOSURE_PATTERNS = {
    'gentle_1':   {'pulse_dur': 5.0, 'off_dur': 2.0,  'n_pulses': 2},
    'gentle_2':   {'pulse_dur': 3.0, 'off_dur': 4.0,  'n_pulses': 2},
    'gentle_3':   {'pulse_dur': 2.0, 'off_dur': 5.0,  'n_pulses': 2},
    'moderate_1': {'pulse_dur': 1.0, 'off_dur': 6.0,  'n_pulses': 2},
    'moderate_2': {'pulse_dur': 1.0, 'off_dur': 6.0,  'n_pulses': 3},
    'moderate_3': {'pulse_dur': 0.5, 'off_dur': 6.5,  'n_pulses': 2},
    'sharp_1':    {'pulse_dur': 0.5, 'off_dur': 13.5, 'n_pulses': 2},
    'sharp_2':    {'pulse_dur': 0.25,'off_dur': 6.75, 'n_pulses': 2},
    'sharp_3':    {'pulse_dur': 0.25,'off_dur': 13.75,'n_pulses': 2},
    'spike_1':    {'pulse_dur': 0.1, 'off_dur': 6.9,  'n_pulses': 2},
    'spike_2':    {'pulse_dur': 0.1, 'off_dur': 13.9, 'n_pulses': 2},
    'spike_3':    {'pulse_dur': 0.05,'off_dur': 13.95,'n_pulses': 2},
}


def build_exposure_profile(pattern_def, conc):
    """Build pulse profile and compute PECmax/TWA ratio."""
    pd = pattern_def['pulse_dur']
    od = pattern_def['off_dur']
    np_ = pattern_def['n_pulses']
    interval = pd + od
    
    profile = []
    for i in range(np_):
        t_start = i * interval
        t_end = t_start + pd
        profile.append((t_start, t_end, conc))
    
    tmax = np_ * interval + 2.0
    c_func = profile_to_func(profile)
    
    pec_max = conc
    twa_7d = compute_twa(c_func, tmax, window=7.0)
    peak_twa_ratio = pec_max / max(twa_7d, 1e-15)
    
    return profile, tmax, c_func, peak_twa_ratio, twa_7d


# ── Worker ───────────────────────────────────────────────────────────────────

def run_twa_point(args):
    """Run TWA bias for one (kr, pattern) — kd FREE, SD uses fitted kd."""
    kr, pattern_name, pattern_def, seed = args
    rng = np.random.default_rng(seed)
    
    true_params = {
        'kd': kr, 'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE, 'hb': HB
    }
    concs = design_concentrations(TH_MED_TRUE)
    
    frac_eq = 1.0 - np.exp(-kr * 4.0)
    lc50_approx = TH_MED_TRUE / max(frac_eq, 0.05)
    
    # PECmax/TWA ratio (concentration-independent)
    ref_profile, _, _, ref_peak_twa_ratio, _ = build_exposure_profile(pattern_def, lc50_approx)
    
    pred_conc_mults = [1.0, 1.5]
    
    S_true_list, S_sd_list, S_twa_list = [], [], []
    
    for rep in range(N_REPS):
        datasets = generate_lc50_data(concs, true_params, N_PER_CONC, rng=rng)
        
        # ── CRITICAL: fit_kd=True ──
        res_SD = fit_SD(datasets, kd=kr, hb=HB,
                       n_starts=N_STARTS, fit_kd=True, rng=rng)
        
        if (not np.isfinite(res_SD['z_med']) or res_SD['z_med'] <= 0 or
            not np.isfinite(res_SD['kd']) or res_SD['kd'] <= 0):
            continue
        
        kd_SD_hat = res_SD['kd']
        
        for mult in pred_conc_mults:
            pred_conc = mult * lc50_approx
            profile, tmax, c_func, _, twa_7d = build_exposure_profile(pattern_def, pred_conc)
            
            # True: Class B with TRUE kd
            t_true, D_true = sim_damage_1c_pulsed(profile, kr, tmax=tmax)
            S_true = survival_classB(D_true, t_true, K_TRUE, TH_MED_TRUE, BETA_TRUE, HB)
            
            # SD: with FITTED kd_SD_hat
            t_sd, D_sd = sim_damage_1c_pulsed(profile, kd_SD_hat, tmax=tmax)
            S_sd = survival_SD(D_sd, t_sd, res_SD['bw'], res_SD['z_med'], BETA_TRUE, HB)
            
            # TWA
            S_twa = twa_predict_survival(twa_7d, lc50_approx, slope=BETA_TRUE)
            
            S_true_list.append(float(S_true[-1]))
            S_sd_list.append(float(S_sd[-1]))
            S_twa_list.append(S_twa)
    
    # ── Compute bias ──
    S_true_arr = np.array(S_true_list)
    S_sd_arr = np.array(S_sd_list)
    S_twa_arr = np.array(S_twa_list)
    
    mort_true = 1 - S_true_arr
    mort_sd = 1 - S_sd_arr
    mort_twa = 1 - S_twa_arr
    
    valid_twa = (mort_twa > 0.01) & (mort_true > 0.01) & np.isfinite(mort_true)
    valid_sd = (mort_sd > 0.01) & (mort_true > 0.01) & np.isfinite(mort_true)
    
    def rstats(num, den, vmask):
        if vmask.sum() < 5: return np.nan, np.nan, np.nan
        r = num[vmask] / den[vmask]
        return float(np.median(r)), float(np.percentile(r, 10)), float(np.percentile(r, 90))
    
    twa_m, twa_p10, twa_p90 = rstats(mort_true, mort_twa, valid_twa)
    sd_m, sd_p10, sd_p90 = rstats(mort_true, mort_sd, valid_sd)
    
    valid_s = (S_true_arr > 0.01) & (S_true_arr < 0.99)
    surv_sd = float(np.median(S_sd_arr[valid_s] / S_true_arr[valid_s])) if valid_s.sum() > 5 else np.nan
    surv_twa = float(np.median(S_twa_arr[valid_s] / S_true_arr[valid_s])) if valid_s.sum() > 5 else np.nan
    
    return {
        'kr': kr, 'pattern': pattern_name,
        'peak_twa_ratio': float(ref_peak_twa_ratio),
        'lc50_approx': float(lc50_approx),
        'n_valid': len(S_true_list),
        'twa_mort_bias_median': twa_m, 'twa_mort_bias_p90': twa_p90,
        'sd_mort_bias_median': sd_m, 'sd_mort_bias_p90': sd_p90,
        'sd_surv_bias': surv_sd, 'twa_surv_bias': surv_twa,
        'median_S_true': float(np.median(S_true_arr)) if len(S_true_arr) > 0 else np.nan,
        'median_S_sd': float(np.median(S_sd_arr)) if len(S_sd_arr) > 0 else np.nan,
        'median_S_twa': float(np.median(S_twa_arr)) if len(S_twa_arr) > 0 else np.nan,
    }


# ── TWA-UF Matrix derivation ────────────────────────────────────────────────

def derive_twa_uf_matrix(results):
    kr_bins = [(0, 0.1), (0.1, 1.0), (1.0, 100)]
    kr_labels = ['slow (kr<0.1)', 'moderate (0.1-1)', 'fast (kr>1)']
    peak_bins = [(1, 3), (3, 10), (10, 100)]
    peak_labels = ['PECmax/TWA<3', '3-10', '>10']
    
    matrix = {}
    for i, (kr_lo, kr_hi) in enumerate(kr_bins):
        for j, (pk_lo, pk_hi) in enumerate(peak_bins):
            matching = [r for r in results
                       if kr_lo <= r['kr'] < kr_hi
                       and pk_lo <= r['peak_twa_ratio'] < pk_hi
                       and np.isfinite(r['twa_mort_bias_p90'])]
            if matching:
                uf = max(max(r['twa_mort_bias_p90'] for r in matching), 1.0)
            else:
                uf = np.nan
            matrix[f"{kr_labels[i]}|{peak_labels[j]}"] = round(uf, 1) if np.isfinite(uf) else 'N/A'
    
    return matrix, kr_labels, peak_labels


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    n_workers = max(1, cpu_count() - 1)
    
    tasks = []
    seed_counter = 4000
    for kr in KR_VALUES:
        for pname, pdef in EXPOSURE_PATTERNS.items():
            tasks.append((kr, pname, pdef, seed_counter))
            seed_counter += 1
    
    print("=" * 60)
    print(f"PHASE 4 RERUN: TWA Bias Mapping (CORRECTED)")
    print(f"  k_r values: {KR_VALUES}")
    print(f"  Exposure patterns: {len(EXPOSURE_PATTERNS)}")
    print(f"  Total conditions: {len(tasks)}")
    print(f"  Reps/condition: {N_REPS}")
    print(f"  fit_kd: TRUE")
    print(f"  SD uses fitted kd: YES")
    print(f"  Workers: {n_workers}")
    print("=" * 60)
    
    t0 = time.time()
    
    if n_workers == 1:
        results = [run_twa_point(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(run_twa_point, tasks)
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    uf_matrix, kr_labels, peak_labels = derive_twa_uf_matrix(results)
    
    output = {
        'results': results,
        'twa_uf_matrix': uf_matrix,
        'kr_labels': kr_labels,
        'peak_labels': peak_labels,
    }
    
    with open('results/phase4_twa_bias.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("Results saved to results/phase4_twa_bias.json")
    
    # ── Print TWA-UF matrix ──
    print("\n" + "=" * 60)
    print("TWA-UF MATRIX (simulation-derived)")
    print("=" * 60)
    print(f"{'':>25}", end="")
    for pl in peak_labels: print(f"{pl:>15}", end="")
    print()
    print("-" * 70)
    for kl in kr_labels:
        print(f"{kl:>25}", end="")
        for pl in peak_labels:
            val = uf_matrix.get(f"{kl}|{pl}", 'N/A')
            print(f"{val:>15}", end="")
        print()
    
    # ── Print bias summary ──
    print("\n" + "=" * 60)
    print("TWA & SD BIAS SUMMARY")
    print("=" * 60)
    for kr in KR_VALUES:
        kr_results = sorted([r for r in results if r['kr'] == kr],
                           key=lambda r: r['peak_twa_ratio'])
        print(f"\n  k_r = {kr} (lc50_approx={kr_results[0]['lc50_approx']:.2f}):")
        for r in kr_results:
            twa_b = r['twa_mort_bias_median']
            sd_b = r['sd_mort_bias_median']
            twa_s = f"{twa_b:.2f}" if np.isfinite(twa_b) else "nan"
            sd_s = f"{sd_b:.2f}" if np.isfinite(sd_b) else "nan"
            print(f"    {r['pattern']:>12} (PEC/TWA={r['peak_twa_ratio']:5.1f})"
                  f"  TWA_bias={twa_s:>6}  SD_bias={sd_s:>6}"
                  f"  n={r['n_valid']}")
