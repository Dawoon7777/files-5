"""
Phase 2: Level 2 (k_d Bias) Quantification
============================================
k_d freed as a fitting parameter in both Class B and GUTS-SD.
Compares fitted k_d values to isolate the recovery speed inflation bias.
"""
import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count
from core import (generate_lc50_data, design_concentrations,
                  fit_classB, fit_SD)

# ── Configuration ────────────────────────────────────────────────────────────
KR_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0]   # 5 representative k_r values
BETA_TRUE = 5.0
K_TRUE = 0.45
N_REPS = 500
N_STARTS = 15      # more starts for 3-param fit
N_PER_CONC = 20

HB = 0.001
TH_MED_TRUE = 3.0


def run_kd_free_point(args):
    """Run N_REPS for one k_r value with k_d free."""
    kr, seed = args
    rng = np.random.default_rng(seed)
    
    true_params = {
        'kd': kr,
        'k': K_TRUE, 'th_med': TH_MED_TRUE, 'beta': BETA_TRUE, 'hb': HB
    }
    concs = design_concentrations(TH_MED_TRUE)
    
    results = {
        'kr_true': kr,
        # k_d fixed results (for Level 1 isolation)
        'th_B_fixed': [], 'z_SD_fixed': [], 'nll_B_fixed': [], 'nll_SD_fixed': [],
        # k_d free results (Level 1 + Level 2 combined)
        'kd_B_free': [], 'kd_SD_free': [],
        'th_B_free': [], 'z_SD_free': [],
        'nll_B_free': [], 'nll_SD_free': [],
    }
    
    for rep in range(N_REPS):
        datasets = generate_lc50_data(concs, true_params, N_PER_CONC, rng=rng)
        
        # ── k_d FIXED fitting (Level 1 only) ──
        res_B_fixed = fit_classB(datasets, kd=kr, hb=HB,
                                n_starts=N_STARTS, fit_kd=False, rng=rng)
        res_SD_fixed = fit_SD(datasets, kd=kr, hb=HB,
                             n_starts=N_STARTS, fit_kd=False, rng=rng)
        
        results['th_B_fixed'].append(res_B_fixed['th_med'])
        results['z_SD_fixed'].append(res_SD_fixed['z_med'])
        results['nll_B_fixed'].append(res_B_fixed['nll'])
        results['nll_SD_fixed'].append(res_SD_fixed['nll'])
        
        # ── k_d FREE fitting (Level 1 + Level 2) ──
        res_B_free = fit_classB(datasets, kd=kr, hb=HB,
                               n_starts=N_STARTS, fit_kd=True, rng=rng)
        res_SD_free = fit_SD(datasets, kd=kr, hb=HB,
                            n_starts=N_STARTS, fit_kd=True, rng=rng)
        
        results['kd_B_free'].append(res_B_free['kd'])
        results['kd_SD_free'].append(res_SD_free['kd'])
        results['th_B_free'].append(res_B_free['th_med'])
        results['z_SD_free'].append(res_SD_free['z_med'])
        results['nll_B_free'].append(res_B_free['nll'])
        results['nll_SD_free'].append(res_SD_free['nll'])
    
    # ── Compute summaries ──
    def safe_median(arr):
        a = np.array(arr)
        valid = np.isfinite(a) & (a > 0)
        return float(np.median(a[valid])) if valid.sum() > 5 else np.nan
    
    def safe_ratio_median(num, den):
        n, d = np.array(num), np.array(den)
        valid = np.isfinite(n) & np.isfinite(d) & (n > 0) & (d > 0)
        return float(np.median(n[valid] / d[valid])) if valid.sum() > 5 else np.nan
    
    th_B_f = np.array(results['th_B_fixed'])
    z_SD_f = np.array(results['z_SD_fixed'])
    kd_B_fr = np.array(results['kd_B_free'])
    kd_SD_fr = np.array(results['kd_SD_free'])
    th_B_fr = np.array(results['th_B_free'])
    z_SD_fr = np.array(results['z_SD_free'])
    
    summary = {
        'kr_true': kr,
        'n_reps': N_REPS,
        # Level 1 only (k_d fixed)
        'level1_ratio': safe_ratio_median(results['z_SD_fixed'], results['th_B_fixed']),
        # Level 1+2 combined (k_d free)
        'level12_ratio': safe_ratio_median(results['z_SD_free'], results['th_B_free']),
        # Level 2 isolation: k_d inflation ratio
        'kd_inflation': safe_ratio_median(results['kd_SD_free'], results['kd_B_free']),
        'median_kd_B': safe_median(results['kd_B_free']),
        'median_kd_SD': safe_median(results['kd_SD_free']),
        # Level 2 = level12 / level1
        'level2_ratio': None,  # computed below
    }
    
    if (summary['level1_ratio'] and summary['level12_ratio'] and 
        summary['level1_ratio'] > 0 and not np.isnan(summary['level1_ratio'])):
        summary['level2_ratio'] = summary['level12_ratio'] / summary['level1_ratio']
    
    return summary


def run_phase2(n_workers=None, output_file='results/phase2_level2_kd.json'):
    """Run Phase 2: k_d bias quantification."""
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    tasks = [(kr, 2000 + i) for i, kr in enumerate(KR_VALUES)]
    
    print("=" * 60)
    print(f"PHASE 2: Level 2 (k_d Bias) Quantification")
    print(f"  Conditions: {len(KR_VALUES)} k_r values")
    print(f"  Reps/condition: {N_REPS}")
    print(f"  k_d: FREE in both models (3 parameters each)")
    print(f"  Workers: {n_workers}")
    print("=" * 60)
    
    t0 = time.time()
    
    if n_workers == 1:
        results = [run_kd_free_point(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(run_kd_free_point, tasks)
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\n" + "=" * 60)
    print("THREE-LEVEL DECOMPOSITION (Partial: Level 1 + Level 2)")
    print("=" * 60)
    print(f"{'kr_true':>8} {'Level1':>8} {'Level2':>8} {'L1×L2':>8} {'kd_infl':>8} {'kd_B':>8} {'kd_SD':>8}")
    print("-" * 64)
    for r in results:
        l1 = r['level1_ratio']
        l2 = r.get('level2_ratio', np.nan)
        l12 = r['level12_ratio']
        ki = r['kd_inflation']
        print(f"{r['kr_true']:8.2f} {l1:8.2f} {l2:8.2f} {l12:8.2f} {ki:8.2f} "
              f"{r['median_kd_B']:8.3f} {r['median_kd_SD']:8.3f}")
    
    return results


if __name__ == '__main__':
    run_phase2()
