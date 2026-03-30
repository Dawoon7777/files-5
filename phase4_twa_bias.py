"""
Phase 4: TWA Bias Mapping and TWA-UF Matrix Derivation
========================================================
Maps TWA prediction bias across (k_r, PECmax/TWA) space.
Derives correction factor matrix for Tier-1/2A regulatory use.
"""
import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count
from core import (generate_lc50_data, design_concentrations,
                  sim_damage_1c, sim_damage_1c_pulsed,
                  survival_classB, survival_SD,
                  fit_classB, fit_SD,
                  make_pulse_profile, profile_to_func,
                  compute_twa, twa_predict_survival)

# ── Configuration ────────────────────────────────────────────────────────────
KR_VALUES = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]   # 7 k_r values
BETA_TRUE = 5.0
K_TRUE = 0.45
N_REPS = 200
N_STARTS = 10
N_PER_CONC = 20

HB = 0.001
TH_MED_TRUE = 3.0

# ── Exposure patterns spanning PECmax/TWA range ──────────────────────────────
# Each defined by (pulse_dur, total_period) → PECmax/TWA ≈ total_period/pulse_dur

EXPOSURE_PATTERNS = {
    # Gentle fluctuation: PECmax/TWA ~ 1.5–3
    'gentle_1':  {'pulse_dur': 5.0, 'off_dur': 2.0,  'n_pulses': 2, 'label': 'gentle_1'},
    'gentle_2':  {'pulse_dur': 3.0, 'off_dur': 4.0,  'n_pulses': 2, 'label': 'gentle_2'},
    'gentle_3':  {'pulse_dur': 2.0, 'off_dur': 5.0,  'n_pulses': 2, 'label': 'gentle_3'},
    # Moderate pulses: PECmax/TWA ~ 3–10
    'moderate_1': {'pulse_dur': 1.0, 'off_dur': 6.0,  'n_pulses': 2, 'label': 'moderate_1'},
    'moderate_2': {'pulse_dur': 1.0, 'off_dur': 6.0,  'n_pulses': 3, 'label': 'moderate_2'},
    'moderate_3': {'pulse_dur': 0.5, 'off_dur': 6.5,  'n_pulses': 2, 'label': 'moderate_3'},
    # Sharp pulses: PECmax/TWA ~ 10–30
    'sharp_1':   {'pulse_dur': 0.5, 'off_dur': 13.5, 'n_pulses': 2, 'label': 'sharp_1'},
    'sharp_2':   {'pulse_dur': 0.25,'off_dur': 6.75, 'n_pulses': 2, 'label': 'sharp_2'},
    'sharp_3':   {'pulse_dur': 0.25,'off_dur': 13.75,'n_pulses': 2, 'label': 'sharp_3'},
    # Extreme spikes: PECmax/TWA ~ 30–100
    'spike_1':   {'pulse_dur': 0.1, 'off_dur': 6.9,  'n_pulses': 2, 'label': 'spike_1'},
    'spike_2':   {'pulse_dur': 0.1, 'off_dur': 13.9, 'n_pulses': 2, 'label': 'spike_2'},
    'spike_3':   {'pulse_dur': 0.05,'off_dur': 13.95,'n_pulses': 2, 'label': 'spike_3'},
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
    
    # Compute PECmax / TWA
    pec_max = conc
    twa_7d = compute_twa(c_func, tmax, window=7.0)
    twa_total = compute_twa(c_func, tmax, window=tmax - 1.0)
    
    peak_twa_ratio = pec_max / max(twa_7d, 1e-15)
    
    return profile, tmax, c_func, peak_twa_ratio, twa_7d


# ── Worker: one (k_r, pattern) combination ───────────────────────────────────

def run_twa_point(args):
    """Run TWA bias analysis for one (k_r, pattern) combination."""
    kr, pattern_name, pattern_def, seed = args
    rng = np.random.default_rng(seed)
    
    true_params = {
        'kd': kr,
        'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE, 'hb': HB
    }
    concs = design_concentrations(TH_MED_TRUE)
    lc50_approx = TH_MED_TRUE  # GUTS-RED: D_ss = C_ext at equilibrium
    pred_conc = 0.8 * lc50_approx  # concentration for prediction
    
    # Build exposure profile
    profile, tmax, c_func, peak_twa_ratio, twa_7d = build_exposure_profile(
        pattern_def, pred_conc)
    
    true_B_params = {'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE}
    
    S_true_list = []
    S_sd_list = []
    S_twa_list = []
    
    for rep in range(N_REPS):
        # ── Calibrate ──
        datasets = generate_lc50_data(concs, true_params, N_PER_CONC, rng=rng)
        
        res_SD = fit_SD(datasets, kd=kr, hb=HB,
                       n_starts=N_STARTS, fit_kd=False, rng=rng)
        
        if not np.isfinite(res_SD['z_med']) or res_SD['z_med'] <= 0:
            continue
        
        fit_SD_params = {'bw': res_SD['bw'], 'z_med': res_SD['z_med'], 'beta': BETA_TRUE}
        
        # ── Predict: True (Class B, true params) ──
        t_sim, D_true = sim_damage_1c_pulsed(profile, kr, tmax=tmax)
        S_true = survival_classB(D_true, t_sim, K_TRUE, TH_MED_TRUE, BETA_TRUE, HB)
        
        # ── Predict: GUTS-SD (fitted params) ──
        S_sd = survival_SD(D_true, t_sim, res_SD['bw'], res_SD['z_med'], BETA_TRUE, HB)
        # Note: using same D trajectory but SD hazard structure with fitted params
        # More correctly, we should use SD's own k_d, but here k_d is fixed = kr
        
        # ── Predict: TWA ──
        S_twa = twa_predict_survival(twa_7d, lc50_approx, slope=BETA_TRUE)
        
        S_true_list.append(float(S_true[-1]))
        S_sd_list.append(float(S_sd[-1]))
        S_twa_list.append(S_twa)
    
    # ── Compute bias ratios ──
    S_true_arr = np.array(S_true_list)
    S_sd_arr = np.array(S_sd_list)
    S_twa_arr = np.array(S_twa_list)
    
    # Mortality ratio: true_mortality / predicted_mortality
    # mortality = 1 - S. Higher ratio = more underestimation of toxicity
    mort_true = 1 - S_true_arr
    mort_sd = 1 - S_sd_arr
    mort_twa = 1 - S_twa_arr
    
    # For TWA bias: ratio of true to TWA-predicted mortality
    valid_twa = (mort_twa > 0.01) & (mort_true > 0.01) & np.isfinite(mort_true)
    valid_sd = (mort_sd > 0.01) & (mort_true > 0.01) & np.isfinite(mort_true)
    
    def ratio_stats(num, den, valid_mask):
        if valid_mask.sum() < 5:
            return np.nan, np.nan, np.nan
        r = num[valid_mask] / den[valid_mask]
        return float(np.median(r)), float(np.percentile(r, 10)), float(np.percentile(r, 90))
    
    twa_bias_med, twa_bias_p10, twa_bias_p90 = ratio_stats(mort_true, mort_twa, valid_twa)
    sd_bias_med, sd_bias_p10, sd_bias_p90 = ratio_stats(mort_true, mort_sd, valid_sd)
    
    # Also compute survival-based bias
    valid_s = (S_true_arr > 0.01) & (S_true_arr < 0.99)
    surv_sd = np.nan
    surv_twa = np.nan
    if valid_s.sum() > 5:
        surv_sd = float(np.median(S_sd_arr[valid_s] / S_true_arr[valid_s]))
        surv_twa = float(np.median(S_twa_arr[valid_s] / S_true_arr[valid_s]))
    
    return {
        'kr': kr,
        'pattern': pattern_name,
        'peak_twa_ratio': float(peak_twa_ratio),
        'twa_7d': float(twa_7d),
        'pred_conc': float(pred_conc),
        'n_valid': len(S_true_list),
        # Mortality-based bias (true_mort / predicted_mort; >1 = underestimation)
        'twa_mort_bias_median': twa_bias_med,
        'twa_mort_bias_p90': twa_bias_p90,
        'sd_mort_bias_median': sd_bias_med,
        'sd_mort_bias_p90': sd_bias_p90,
        # Survival-based bias (pred_S / true_S; >1 = overestimation of survival)
        'sd_surv_bias': surv_sd,
        'twa_surv_bias': surv_twa,
        # Raw medians
        'median_S_true': float(np.median(S_true_arr)) if len(S_true_arr) > 0 else np.nan,
        'median_S_sd': float(np.median(S_sd_arr)) if len(S_sd_arr) > 0 else np.nan,
        'median_S_twa': float(np.median(S_twa_arr)) if len(S_twa_arr) > 0 else np.nan,
    }


# ── TWA-UF Matrix derivation ────────────────────────────────────────────────

def derive_twa_uf_matrix(results):
    """Derive TWA-UF correction factor matrix from Phase 4 results.
    
    TWA-UF = 90th percentile of (true_mortality / TWA_mortality)
    indexed by (k_r bin, PECmax/TWA bin).
    """
    # Define bins
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
                # Take the maximum p90 across matching conditions
                uf = max(r['twa_mort_bias_p90'] for r in matching)
                uf = max(uf, 1.0)  # UF >= 1
            else:
                uf = np.nan
            
            matrix[f"{kr_labels[i]}|{peak_labels[j]}"] = round(uf, 1) if np.isfinite(uf) else 'N/A'
    
    return matrix, kr_labels, peak_labels


# ── Main execution ───────────────────────────────────────────────────────────

def run_phase4(n_workers=None, output_file='results/phase4_twa_bias.json'):
    """Run Phase 4: TWA bias mapping."""
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Build task list
    tasks = []
    seed_counter = 4000
    for kr in KR_VALUES:
        for pname, pdef in EXPOSURE_PATTERNS.items():
            tasks.append((kr, pname, pdef, seed_counter))
            seed_counter += 1
    
    print("=" * 60)
    print(f"PHASE 4: TWA Bias Mapping")
    print(f"  k_r values: {KR_VALUES}")
    print(f"  Exposure patterns: {len(EXPOSURE_PATTERNS)}")
    print(f"  Total conditions: {len(tasks)}")
    print(f"  Reps/condition: {N_REPS}")
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
    
    # Derive TWA-UF matrix
    uf_matrix, kr_labels, peak_labels = derive_twa_uf_matrix(results)
    
    output = {
        'results': results,
        'twa_uf_matrix': uf_matrix,
        'kr_labels': kr_labels,
        'peak_labels': peak_labels,
    }
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Results saved to {output_file}")
    
    # ── Print TWA-UF matrix ──
    print("\n" + "=" * 60)
    print("TWA-UF MATRIX (simulation-derived)")
    print("=" * 60)
    print(f"{'':>25}", end="")
    for pl in peak_labels:
        print(f"{pl:>15}", end="")
    print()
    print("-" * 70)
    for kl in kr_labels:
        print(f"{kl:>25}", end="")
        for pl in peak_labels:
            key = f"{kl}|{pl}"
            val = uf_matrix.get(key, 'N/A')
            print(f"{val:>15}", end="")
        print()
    
    # ── Print bias summary by kr ──
    print("\n" + "=" * 60)
    print("TWA BIAS SUMMARY (mortality-based, median)")
    print("=" * 60)
    for kr in KR_VALUES:
        kr_results = [r for r in results if r['kr'] == kr]
        kr_results.sort(key=lambda r: r['peak_twa_ratio'])
        print(f"\n  k_r = {kr}:")
        for r in kr_results:
            print(f"    {r['pattern']:>15} (PECmax/TWA={r['peak_twa_ratio']:5.1f})"
                  f"  TWA_bias={r['twa_mort_bias_median']:5.2f}"
                  f"  SD_bias={r['sd_mort_bias_median']:5.2f}")
    
    return output


if __name__ == '__main__':
    run_phase4()
