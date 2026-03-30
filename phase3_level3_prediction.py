"""
Phase 3: Level 3 (R-bias) + Chronic Prediction Crossover
==========================================================
3A: Quantifies prediction-domain bias from SD's structural R=1
    by comparing calibration-fitted models on pulsed scenarios.
3B: Demonstrates SD's chronic crossover — non-conservative at
    intermediate t, over-conservative at long t.
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
KR_VALUES = [0.1, 0.5, 1.0, 2.0]   # 4 representative k_r
BETA_TRUE = 5.0
K_TRUE = 0.45
N_REPS = 200
N_STARTS = 10
N_PER_CONC = 20

HB = 0.001
TH_MED_TRUE = 3.0

# Pulsed prediction scenarios
PULSE_SCENARIOS = {
    'P1_2pulse_3d':  {'n_pulses': 2, 'pulse_dur': 1.0, 'interval': 3.0,  'label': '2-pulse, 3d interval'},
    'P2_2pulse_7d':  {'n_pulses': 2, 'pulse_dur': 1.0, 'interval': 7.0,  'label': '2-pulse, 7d interval'},
    'P3_2pulse_28d': {'n_pulses': 2, 'pulse_dur': 1.0, 'interval': 28.0, 'label': '2-pulse, 28d interval'},
    'P4_5pulse_3d':  {'n_pulses': 5, 'pulse_dur': 1.0, 'interval': 3.0,  'label': '5-pulse, 3d interval'},
    'P5_5pulse_7d':  {'n_pulses': 5, 'pulse_dur': 1.0, 'interval': 7.0,  'label': '5-pulse, 7d interval'},
}

# Chronic prediction time points
CHRONIC_TIMES = [1, 2, 4, 7, 14, 21, 28, 56, 96]


# ── Helper: predict survival for a given scenario ────────────────────────────

def predict_pulsed_survival(pulse_profile, kd, hb, model, params, tmax=None):
    """Predict final survival under a pulsed scenario.
    
    Args:
        pulse_profile: list of (t_start, t_end, conc)
        kd, hb: TK/TD parameters
        model: 'classB' or 'SD'
        params: dict with model-specific keys
        tmax: end time
    
    Returns:
        S_final (float), S_trajectory (array), t_array
    """
    if tmax is None:
        tmax = max(p[1] for p in pulse_profile) + 5.0
    
    t, D = sim_damage_1c_pulsed(pulse_profile, kd, tmax=tmax)
    
    if model == 'classB':
        S = survival_classB(D, t, params['k'], params['th_med'], params.get('beta', 5.0), hb)
    elif model == 'SD':
        S = survival_SD(D, t, params['bw'], params['z_med'], params.get('beta', 5.0), hb)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return float(S[-1]), S, t


def predict_chronic_survival(c0, kd, hb, model, params, t_eval):
    """Predict survival at multiple time points under constant exposure.
    
    Args:
        c0: constant external concentration
        kd, hb: model parameters
        t_eval: list of evaluation times (days)
    
    Returns:
        dict {t: S(t)}
    """
    tmax = max(t_eval) + 1.0
    t, D = sim_damage_1c(c0, kd, tmax=tmax)
    
    if model == 'classB':
        S = survival_classB(D, t, params['k'], params['th_med'], params.get('beta', 5.0), hb)
    elif model == 'SD':
        S = survival_SD(D, t, params['bw'], params['z_med'], params.get('beta', 5.0), hb)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return {te: float(np.interp(te, t, S)) for te in t_eval}


# ── Worker: one k_r condition ────────────────────────────────────────────────

def run_phase3_point(args):
    """Run Phase 3 for one k_r value."""
    kr, seed = args
    rng = np.random.default_rng(seed)
    
    true_params = {
        'kd': kr,
        'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE, 'hb': HB
    }
    concs = design_concentrations(TH_MED_TRUE)
    
    # Approximate LC50 for prediction scenarios
    lc50_approx = TH_MED_TRUE  # GUTS-RED: D_ss = C_ext at equilibrium
    pred_conc = 0.7 * lc50_approx   # sub-LC50 for prediction
    chronic_conc = 0.5 * lc50_approx  # half-LC50 for chronic
    
    # Storage for results
    pulsed_results = {sc: {'true': [], 'sd': [], 'twa': []} for sc in PULSE_SCENARIOS}
    chronic_results = {t: {'true': [], 'sd': []} for t in CHRONIC_TIMES}
    calibration_bias = {'th_B': [], 'z_SD': []}
    
    for rep in range(N_REPS):
        # ── Step 1: Calibrate on 96h LC50 data ──
        datasets = generate_lc50_data(concs, true_params, N_PER_CONC, rng=rng)
        
        res_B = fit_classB(datasets, kd=kr, hb=HB,
                          n_starts=N_STARTS, fit_kd=False, rng=rng)
        res_SD = fit_SD(datasets, kd=kr, hb=HB,
                       n_starts=N_STARTS, fit_kd=False, rng=rng)
        
        if not np.isfinite(res_B['th_med']) or not np.isfinite(res_SD['z_med']):
            continue
        if res_B['th_med'] <= 0 or res_SD['z_med'] <= 0:
            continue
        
        calibration_bias['th_B'].append(res_B['th_med'])
        calibration_bias['z_SD'].append(res_SD['z_med'])
        
        fit_B_params = {'k': res_B['k'], 'th_med': res_B['th_med'], 'beta': BETA_TRUE}
        fit_SD_params = {'bw': res_SD['bw'], 'z_med': res_SD['z_med'], 'beta': BETA_TRUE}
        true_B_params = {'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE}
        
        # ── Step 2: Pulsed prediction scenarios ──
        for sc_name, sc_def in PULSE_SCENARIOS.items():
            profile = make_pulse_profile(
                sc_def['n_pulses'], sc_def['pulse_dur'],
                sc_def['interval'], pred_conc
            )
            tmax_sc = max(p[1] for p in profile) + 5.0
            
            # True survival (Class B, true params)
            S_true, _, _ = predict_pulsed_survival(
                profile, kr, HB, 'classB', true_B_params, tmax_sc)
            
            # SD prediction (fitted params)
            S_sd, _, _ = predict_pulsed_survival(
                profile, kr, HB, 'SD', fit_SD_params, tmax_sc)
            
            # TWA prediction
            c_func = profile_to_func(profile)
            twa_val = compute_twa(c_func, tmax_sc, window=7.0)
            S_twa = twa_predict_survival(twa_val, lc50_approx, slope=BETA_TRUE)
            
            pulsed_results[sc_name]['true'].append(S_true)
            pulsed_results[sc_name]['sd'].append(S_sd)
            pulsed_results[sc_name]['twa'].append(S_twa)
        
        # ── Step 3: Chronic constant-exposure prediction ──
        S_true_chronic = predict_chronic_survival(
            chronic_conc, kr, HB, 'classB', true_B_params, CHRONIC_TIMES)
        S_sd_chronic = predict_chronic_survival(
            chronic_conc, kr, HB, 'SD', fit_SD_params, CHRONIC_TIMES)
        
        for t_ch in CHRONIC_TIMES:
            chronic_results[t_ch]['true'].append(S_true_chronic[t_ch])
            chronic_results[t_ch]['sd'].append(S_sd_chronic[t_ch])
    
    # ── Compute summaries ──
    def safe_median(arr):
        a = np.array(arr)
        v = a[np.isfinite(a)]
        return float(np.median(v)) if len(v) > 5 else np.nan
    
    def safe_ratio(num, den):
        n, d = np.array(num), np.array(den)
        valid = np.isfinite(n) & np.isfinite(d) & (d > 1e-10)
        if valid.sum() < 5:
            return np.nan, np.nan, np.nan
        r = n[valid] / d[valid]
        return float(np.median(r)), float(np.percentile(r, 25)), float(np.percentile(r, 75))
    
    # Pulsed summary: survival overestimation ratio = S_sd / S_true
    pulsed_summary = {}
    for sc_name in PULSE_SCENARIOS:
        sd_over_true = safe_ratio(pulsed_results[sc_name]['sd'],
                                   pulsed_results[sc_name]['true'])
        twa_over_true = safe_ratio(pulsed_results[sc_name]['twa'],
                                    pulsed_results[sc_name]['true'])
        pulsed_summary[sc_name] = {
            'sd_over_true_median': sd_over_true[0],
            'sd_over_true_iqr': [sd_over_true[1], sd_over_true[2]],
            'twa_over_true_median': twa_over_true[0],
            'twa_over_true_iqr': [twa_over_true[1], twa_over_true[2]],
            'median_S_true': safe_median(pulsed_results[sc_name]['true']),
            'median_S_sd': safe_median(pulsed_results[sc_name]['sd']),
            'median_S_twa': safe_median(pulsed_results[sc_name]['twa']),
        }
    
    # Chronic summary: bias ratio B(t) = S_sd(t) / S_true(t)
    chronic_summary = {}
    for t_ch in CHRONIC_TIMES:
        bias_med, bias_q25, bias_q75 = safe_ratio(
            chronic_results[t_ch]['sd'], chronic_results[t_ch]['true'])
        chronic_summary[str(t_ch)] = {
            'bias_median': bias_med,
            'bias_iqr': [bias_q25, bias_q75],
            'median_S_true': safe_median(chronic_results[t_ch]['true']),
            'median_S_sd': safe_median(chronic_results[t_ch]['sd']),
        }
    
    # Find crossover time t*
    crossover_t = np.nan
    bias_values = [chronic_summary[str(t)]['bias_median'] for t in CHRONIC_TIMES]
    for i in range(len(bias_values) - 1):
        b1, b2 = bias_values[i], bias_values[i + 1]
        if np.isfinite(b1) and np.isfinite(b2) and b1 > 1.0 and b2 < 1.0:
            # Linear interpolation for crossover
            t1, t2 = CHRONIC_TIMES[i], CHRONIC_TIMES[i + 1]
            crossover_t = t1 + (t2 - t1) * (b1 - 1.0) / (b1 - b2)
            break
    
    # Calibration bias for Level 3 isolation
    cal_ratio = safe_ratio(calibration_bias['z_SD'], calibration_bias['th_B'])
    
    return {
        'kr': kr,
        'n_valid': len(calibration_bias['th_B']),
        'calibration_bias_median': cal_ratio[0],
        'pulsed': pulsed_summary,
        'chronic': chronic_summary,
        'crossover_t_star': float(crossover_t) if np.isfinite(crossover_t) else None,
    }


# ── Main execution ───────────────────────────────────────────────────────────

def run_phase3(n_workers=None, output_file='results/phase3_level3_chronic.json'):
    """Run Phase 3: Level 3 + Chronic Crossover."""
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    tasks = [(kr, 3000 + i) for i, kr in enumerate(KR_VALUES)]
    
    print("=" * 60)
    print(f"PHASE 3: Level 3 + Chronic Crossover")
    print(f"  k_r values: {KR_VALUES}")
    print(f"  Reps: {N_REPS}")
    print(f"  Pulsed scenarios: {len(PULSE_SCENARIOS)}")
    print(f"  Chronic timepoints: {CHRONIC_TIMES}")
    print(f"  Workers: {n_workers}")
    print("=" * 60)
    
    t0 = time.time()
    
    if n_workers == 1:
        results = [run_phase3_point(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(run_phase3_point, tasks)
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # ── Print summaries ──
    print("\n" + "=" * 60)
    print("PULSED PREDICTION BIAS (S_sd / S_true, median)")
    print("=" * 60)
    header = f"{'kr':>6} " + " ".join(f"{sc[:8]:>10}" for sc in PULSE_SCENARIOS)
    print(header)
    print("-" * len(header))
    for r in results:
        vals = [r['pulsed'][sc]['sd_over_true_median'] for sc in PULSE_SCENARIOS]
        line = f"{r['kr']:6.2f} " + " ".join(f"{v:10.3f}" for v in vals)
        print(line)
    
    print("\n" + "=" * 60)
    print("CHRONIC CROSSOVER: B(t) = S_sd(t) / S_true(t)")
    print("=" * 60)
    header = f"{'kr':>6} " + " ".join(f"{'t='+str(t):>8}" for t in CHRONIC_TIMES) + f" {'t*':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        vals = [r['chronic'][str(t)]['bias_median'] for t in CHRONIC_TIMES]
        t_star = r['crossover_t_star']
        t_str = f"{t_star:.1f}" if t_star else "N/A"
        line = (f"{r['kr']:6.2f} " + " ".join(f"{v:8.3f}" for v in vals)
                + f" {t_str:>8}")
        print(line)
    
    return results


if __name__ == '__main__':
    run_phase3()
