#!/usr/bin/env python3
"""
Phase 3 RERUN: Level 3 + Chronic Crossover (CORRECTED)
=======================================================
Standalone script — does NOT rely on __pycache__.
Imports core.py directly and verifies kd-free fitting is active.

Key corrections vs previous run:
  1. fit_kd=True → SD gets its own fitted kd (inflated)
  2. SD prediction uses kd_SD_hat for damage dynamics (separate D trajectory)
  3. Multiple prediction concentrations (1.0x and 1.5x LC50)
  4. lc50_approx accounts for kd*T equilibrium fraction
"""
import sys, os, importlib

# Force reimport — bypass any __pycache__
if 'core' in sys.modules:
    del sys.modules['core']
os.makedirs('results', exist_ok=True)

import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count

# Import core fresh
import core
importlib.reload(core)
from core import (generate_lc50_data, design_concentrations,
                  sim_damage_1c, sim_damage_1c_pulsed,
                  survival_classB, survival_SD,
                  fit_classB, fit_SD,
                  make_pulse_profile, profile_to_func,
                  compute_twa, twa_predict_survival)

# ── Configuration ────────────────────────────────────────────────────────────
KR_VALUES = [0.1, 0.5, 1.0, 2.0]
BETA_TRUE = 5.0
K_TRUE = 0.45
N_REPS = 200
N_STARTS = 15
N_PER_CONC = 20
HB = 0.001
TH_MED_TRUE = 3.0

PULSE_SCENARIOS = {
    'P1_2pulse_3d':  {'n_pulses': 2, 'pulse_dur': 1.0, 'interval': 3.0},
    'P2_2pulse_7d':  {'n_pulses': 2, 'pulse_dur': 1.0, 'interval': 7.0},
    'P3_2pulse_28d': {'n_pulses': 2, 'pulse_dur': 1.0, 'interval': 28.0},
    'P4_5pulse_3d':  {'n_pulses': 5, 'pulse_dur': 1.0, 'interval': 3.0},
    'P5_5pulse_7d':  {'n_pulses': 5, 'pulse_dur': 1.0, 'interval': 7.0},
}
CHRONIC_TIMES = [1, 2, 4, 7, 14, 21, 28, 56, 96]


# ── Built-in verification ────────────────────────────────────────────────────

def verify_kd_free():
    """Verify that kd-free fitting produces different kd for SD vs Class B."""
    print("=" * 60)
    print("VERIFICATION: kd-free fitting active")
    print("=" * 60)
    rng = np.random.default_rng(999)
    params = {'kd': 2.0, 'k': 0.45, 'th_med': 3.0, 'beta': 5.0, 'hb': 0.001}
    concs = design_concentrations(3.0, kd=2.0)
    ds = generate_lc50_data(concs, params, 20, rng=rng)
    
    rB = fit_classB(ds, kd=2.0, hb=0.001, n_starts=15, fit_kd=True, rng=rng)
    rSD = fit_SD(ds, kd=2.0, hb=0.001, n_starts=15, fit_kd=True, rng=rng)
    
    print(f"  Class B: kd={rB['kd']:.3f}")
    print(f"  GUTS-SD: kd={rSD['kd']:.3f}")
    print(f"  Inflation: x{rSD['kd']/rB['kd']:.1f}")
    
    # Verify different D trajectories
    profile = make_pulse_profile(2, 1.0, 7.0, 3.6)
    _, D_true = sim_damage_1c_pulsed(profile, 2.0, tmax=20.0)
    _, D_sd = sim_damage_1c_pulsed(profile, rSD['kd'], tmax=20.0)
    
    print(f"  D_max true (kd=2.0): {D_true.max():.3f}")
    print(f"  D_max SD (kd={rSD['kd']:.1f}): {D_sd.max():.3f}")
    
    S_true = survival_classB(D_true, np.linspace(0,20,201), 0.45, 3.0, 5.0, 0.001)
    S_sd = survival_SD(D_sd, np.linspace(0,20,201), rSD['bw'], rSD['z_med'], 5.0, 0.001)
    
    ratio = S_sd[-1] / S_true[-1]
    print(f"  S_true={S_true[-1]:.4f}  S_sd={S_sd[-1]:.4f}  ratio={ratio:.3f}")
    
    if abs(ratio - 1.0) < 0.001:
        print("  *** WARNING: ratio ≈ 1.0 — kd-free fix may not be working! ***")
        return False
    else:
        print(f"  ✓ Level 3 bias detected: {ratio:.3f} ≠ 1.000")
        return True


# ── Worker ───────────────────────────────────────────────────────────────────

def run_phase3_point(args):
    """Run Phase 3 for one k_r value — kd FREE in both models."""
    kr, seed = args
    rng = np.random.default_rng(seed)
    
    true_params = {
        'kd': kr, 'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE, 'hb': HB
    }
    concs = design_concentrations(TH_MED_TRUE, kd=kr)
    
    # LC50 approximation: D(T) = C * (1 - exp(-kd*T)), so C_LC50 = th / frac_eq
    frac_eq = 1.0 - np.exp(-kr * 4.0)
    lc50_approx = TH_MED_TRUE / max(frac_eq, 0.05)
    
    pred_concs = [1.0 * lc50_approx, 1.5 * lc50_approx]
    chronic_concs = [0.7 * lc50_approx, 1.0 * lc50_approx]
    
    pulsed_results = {sc: {'true': [], 'sd': [], 'twa': []} for sc in PULSE_SCENARIOS}
    chronic_results = {t: {'true': [], 'sd': []} for t in CHRONIC_TIMES}
    cal_data = {'th_B': [], 'z_SD': [], 'kd_B': [], 'kd_SD': []}
    
    for rep in range(N_REPS):
        datasets = generate_lc50_data(concs, true_params, N_PER_CONC, rng=rng)
        
        # ── CRITICAL: fit_kd=True ──
        res_B = fit_classB(datasets, kd=kr, hb=HB,
                          n_starts=N_STARTS, fit_kd=True, rng=rng)
        res_SD = fit_SD(datasets, kd=kr, hb=HB,
                       n_starts=N_STARTS, fit_kd=True, rng=rng)
        
        # Validate
        if (not np.isfinite(res_B['th_med']) or res_B['th_med'] <= 0 or
            not np.isfinite(res_SD['z_med']) or res_SD['z_med'] <= 0 or
            not np.isfinite(res_B['kd']) or res_B['kd'] <= 0 or
            not np.isfinite(res_SD['kd']) or res_SD['kd'] <= 0):
            continue
        
        cal_data['th_B'].append(res_B['th_med'])
        cal_data['z_SD'].append(res_SD['z_med'])
        cal_data['kd_B'].append(res_B['kd'])
        cal_data['kd_SD'].append(res_SD['kd'])
        
        # ── CRITICAL: each model uses its OWN kd ──
        kd_B_hat = res_B['kd']
        kd_SD_hat = res_SD['kd']
        
        true_B_params = {'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE}
        fit_SD_params = {'bw': res_SD['bw'], 'z_med': res_SD['z_med'], 'beta': BETA_TRUE}
        
        # ── Pulsed prediction ──
        for sc_name, sc_def in PULSE_SCENARIOS.items():
            for pc in pred_concs:
                profile = make_pulse_profile(
                    sc_def['n_pulses'], sc_def['pulse_dur'], sc_def['interval'], pc)
                tmax_sc = max(p[1] for p in profile) + 5.0
                
                # True: Class B with TRUE kd
                S_true, _, _ = predict_pulsed(profile, kr, HB, 'classB', true_B_params, tmax_sc)
                
                # SD: with FITTED kd_SD_hat (inflated!)
                S_sd, _, _ = predict_pulsed(profile, kd_SD_hat, HB, 'SD', fit_SD_params, tmax_sc)
                
                # TWA
                c_func = profile_to_func(profile)
                twa_val = compute_twa(c_func, tmax_sc, window=7.0)
                S_twa = twa_predict_survival(twa_val, lc50_approx, slope=BETA_TRUE)
                
                pulsed_results[sc_name]['true'].append(S_true)
                pulsed_results[sc_name]['sd'].append(S_sd)
                pulsed_results[sc_name]['twa'].append(S_twa)
        
        # ── Chronic prediction ──
        for cc in chronic_concs:
            S_true_ch = predict_chronic(cc, kr, HB, 'classB', true_B_params, CHRONIC_TIMES)
            # SD uses kd_SD_hat for chronic prediction too!
            S_sd_ch = predict_chronic(cc, kd_SD_hat, HB, 'SD', fit_SD_params, CHRONIC_TIMES)
            
            for t_ch in CHRONIC_TIMES:
                chronic_results[t_ch]['true'].append(S_true_ch[t_ch])
                chronic_results[t_ch]['sd'].append(S_sd_ch[t_ch])
    
    # ── Summaries ──
    def smed(arr):
        a = np.array(arr); v = a[np.isfinite(a)]
        return float(np.median(v)) if len(v) > 5 else np.nan
    
    def sratio(num, den):
        n, d = np.array(num), np.array(den)
        ok = np.isfinite(n) & np.isfinite(d) & (d > 1e-10)
        if ok.sum() < 5: return np.nan, np.nan, np.nan
        r = n[ok] / d[ok]
        return float(np.median(r)), float(np.percentile(r, 25)), float(np.percentile(r, 75))
    
    pulsed_summary = {}
    for sc in PULSE_SCENARIOS:
        sd_r = sratio(pulsed_results[sc]['sd'], pulsed_results[sc]['true'])
        twa_r = sratio(pulsed_results[sc]['twa'], pulsed_results[sc]['true'])
        pulsed_summary[sc] = {
            'sd_over_true_median': sd_r[0], 'sd_over_true_iqr': [sd_r[1], sd_r[2]],
            'twa_over_true_median': twa_r[0], 'twa_over_true_iqr': [twa_r[1], twa_r[2]],
            'median_S_true': smed(pulsed_results[sc]['true']),
            'median_S_sd': smed(pulsed_results[sc]['sd']),
            'median_S_twa': smed(pulsed_results[sc]['twa']),
        }
    
    chronic_summary = {}
    for t_ch in CHRONIC_TIMES:
        bm, bq25, bq75 = sratio(chronic_results[t_ch]['sd'], chronic_results[t_ch]['true'])
        chronic_summary[str(t_ch)] = {
            'bias_median': bm, 'bias_iqr': [bq25, bq75],
            'median_S_true': smed(chronic_results[t_ch]['true']),
            'median_S_sd': smed(chronic_results[t_ch]['sd']),
        }
    
    # Crossover
    crossover_t = np.nan
    bvals = [chronic_summary[str(t)]['bias_median'] for t in CHRONIC_TIMES]
    for i in range(len(bvals) - 1):
        b1, b2 = bvals[i], bvals[i+1]
        if np.isfinite(b1) and np.isfinite(b2) and b1 > 1.0 and b2 < 1.0:
            t1, t2 = CHRONIC_TIMES[i], CHRONIC_TIMES[i+1]
            crossover_t = t1 + (t2 - t1) * (b1 - 1.0) / (b1 - b2)
            break
    
    cal_r = sratio(cal_data['z_SD'], cal_data['th_B'])
    kd_infl = sratio(cal_data['kd_SD'], cal_data['kd_B'])
    
    return {
        'kr': kr, 'n_valid': len(cal_data['th_B']),
        'calibration_bias_median': cal_r[0],
        'kd_inflation_median': kd_infl[0],
        'median_kd_B': smed(cal_data['kd_B']),
        'median_kd_SD': smed(cal_data['kd_SD']),
        'lc50_approx': lc50_approx,
        'pulsed': pulsed_summary,
        'chronic': chronic_summary,
        'crossover_t_star': float(crossover_t) if np.isfinite(crossover_t) else None,
    }


# ── Inline predict functions (avoid import issues) ───────────────────────────

def predict_pulsed(profile, kd, hb, model, params, tmax):
    t, D = sim_damage_1c_pulsed(profile, kd, tmax=tmax)
    if model == 'classB':
        S = survival_classB(D, t, params['k'], params['th_med'], params.get('beta', 5.0), hb)
    else:
        S = survival_SD(D, t, params['bw'], params['z_med'], params.get('beta', 5.0), hb)
    return float(S[-1]), S, t

def predict_chronic(c0, kd, hb, model, params, t_eval):
    tmax = max(t_eval) + 1.0
    t, D = sim_damage_1c(c0, kd, tmax=tmax)
    if model == 'classB':
        S = survival_classB(D, t, params['k'], params['th_med'], params.get('beta', 5.0), hb)
    else:
        S = survival_SD(D, t, params['bw'], params['z_med'], params.get('beta', 5.0), hb)
    return {te: float(np.interp(te, t, S)) for te in t_eval}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Step 0: Verify
    ok = verify_kd_free()
    if not ok:
        print("ABORTING: kd-free verification failed!")
        sys.exit(1)
    
    n_workers = max(1, cpu_count() - 1)
    tasks = [(kr, 3000 + i) for i, kr in enumerate(KR_VALUES)]
    
    print("\n" + "=" * 60)
    print(f"PHASE 3 RERUN: Level 3 + Chronic Crossover (CORRECTED)")
    print(f"  k_r values: {KR_VALUES}")
    print(f"  Reps: {N_REPS}")
    print(f"  fit_kd: TRUE (3 params each)")
    print(f"  SD uses fitted kd for prediction: YES")
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
    
    with open('results/phase3_level3_chronic.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/phase3_level3_chronic.json")
    
    # ── Print ──
    print("\n" + "=" * 60)
    print("PULSED PREDICTION BIAS (S_sd / S_true, median)")
    print("=" * 60)
    for r in results:
        print(f"\nkr={r['kr']}  (kd_B={r['median_kd_B']:.2f}, kd_SD={r['median_kd_SD']:.2f}, "
              f"inflation=x{r['kd_inflation_median']:.1f}):")
        for sc in PULSE_SCENARIOS:
            d = r['pulsed'][sc]
            print(f"  {sc:>15}: sd/true={d['sd_over_true_median']:.3f}  "
                  f"twa/true={d['twa_over_true_median']:.3f}  "
                  f"S_true={d['median_S_true']:.3f}  S_sd={d['median_S_sd']:.3f}")
    
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
        print(f"{r['kr']:6.2f} " + " ".join(f"{v:8.3f}" for v in vals) + f" {t_str:>8}")
