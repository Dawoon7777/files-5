"""
Phase 1: Level 1 Sensitivity Map
=================================
Maps R_bias = median(z_SD / theta_B) across (k_r, beta, k) grid.
k_d fixed at true value in both models → isolates Level 1 (threshold bias).
"""
import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count
from core import (generate_lc50_data, design_concentrations,
                  fit_classB, fit_SD)

# ── Grid definition ──────────────────────────────────────────────────────────
KR_GRID = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
BETA_GRID = [2, 3, 5, 7, 10, 15]
K_GRID = [0.05, 0.1, 0.25, 0.45, 1.0, 5.0]

N_REPS = 200          # replicates per grid point
N_STARTS = 10         # optimizer starting points
N_PER_CONC = 20       # organisms per concentration

# Fixed parameters
HB = 0.001
TH_MED_TRUE = 3.0
BETA_FIT = 5.0        # beta fixed during fitting

# ── Single grid point worker ─────────────────────────────────────────────────

def run_grid_point(args):
    """Run N_REPS MC iterations for one (kr, beta, k) combination."""
    kr, beta_true, k_true, seed = args
    rng = np.random.default_rng(seed)
    
    true_params = {
        'kd': kr,
        'k': k_true, 'th_med': TH_MED_TRUE, 'beta': beta_true, 'hb': HB
    }
    
    # Design concentrations for this parameter combination
    concs = design_concentrations(TH_MED_TRUE)
    
    results = {
        'kr': kr, 'beta': beta_true, 'k': k_true,
        'th_B': [], 'z_SD': [], 'nll_B': [], 'nll_SD': [],
        'converged_B': 0, 'converged_SD': 0
    }
    
    for rep in range(N_REPS):
        # Generate data from Class B (true model)
        datasets = generate_lc50_data(concs, true_params, N_PER_CONC, rng=rng)
        
        # Fit Class B (kd fixed at true value)
        res_B = fit_classB(datasets, kd=kr, hb=HB,
                          n_starts=N_STARTS, fit_kd=False, rng=rng)
        
        # Fit GUTS-SD (kd fixed at true value)
        res_SD = fit_SD(datasets, kd=kr, hb=HB,
                       n_starts=N_STARTS, fit_kd=False, rng=rng)
        
        results['th_B'].append(res_B['th_med'])
        results['z_SD'].append(res_SD['z_med'])
        results['nll_B'].append(res_B['nll'])
        results['nll_SD'].append(res_SD['nll'])
        results['converged_B'] += int(res_B.get('converged', False))
        results['converged_SD'] += int(res_SD.get('converged', False))
    
    # Compute summary statistics
    th_B = np.array(results['th_B'])
    z_SD = np.array(results['z_SD'])
    nll_B = np.array(results['nll_B'])
    nll_SD = np.array(results['nll_SD'])
    
    valid = (th_B > 0) & (z_SD > 0) & np.isfinite(th_B) & np.isfinite(z_SD)
    
    if valid.sum() < 10:
        return {**{k: v for k, v in results.items() if k in ['kr', 'beta', 'k']},
                'n_valid': int(valid.sum()),
                'median_ratio': np.nan, 'iqr_low': np.nan, 'iqr_high': np.nan,
                'overest_frac': np.nan, 'indist_frac': np.nan}
    
    ratio = z_SD[valid] / th_B[valid]
    dnll = nll_SD[valid] - nll_B[valid]
    
    summary = {
        'kr': kr, 'beta': beta_true, 'k': k_true,
        'n_valid': int(valid.sum()),
        'median_ratio': float(np.median(ratio)),
        'iqr_low': float(np.percentile(ratio, 25)),
        'iqr_high': float(np.percentile(ratio, 75)),
        'mean_ratio': float(np.mean(ratio)),
        'overest_frac': float(np.mean(z_SD[valid] > th_B[valid])),
        'indist_frac': float(np.mean(np.abs(dnll) < 2)),
        'median_dnll': float(np.median(dnll)),
        'converged_B': results['converged_B'],
        'converged_SD': results['converged_SD'],
    }
    
    return summary


# ── Scale invariance pre-validation ──────────────────────────────────────────

def validate_scale_invariance():
    """Check that R_bias is independent of theta_med absolute value."""
    print("=" * 60)
    print("PRE-VALIDATION: Scale invariance check")
    print("=" * 60)
    
    th_values = [0.3, 1.0, 3.0, 10.0, 30.0]
    kr_test, beta_test, k_test = 0.5, 5.0, 0.45
    
    for th in th_values:
        true_params = {
            'kd': kr_test,
            'k': k_test, 'th_med': th, 'beta': beta_test, 'hb': HB
        }
        concs = design_concentrations(th)
        
        rng = np.random.default_rng(42)
        ratios = []
        for _ in range(100):
            ds = generate_lc50_data(concs, true_params, N_PER_CONC, rng=rng)
            rB = fit_classB(ds, kd=kr_test, hb=HB, n_starts=8, rng=rng)
            rSD = fit_SD(ds, kd=kr_test, hb=HB, n_starts=8, rng=rng)
            if rB['th_med'] > 0 and rSD['z_med'] > 0:
                ratios.append(rSD['z_med'] / rB['th_med'])
        
        med = np.median(ratios) if ratios else np.nan
        print(f"  theta_med = {th:6.1f}  →  median ratio = {med:.3f}  (n={len(ratios)})")
    
    print()


# ── Main execution ───────────────────────────────────────────────────────────

def run_phase1(n_workers=None, output_file='results/phase1_level1_map.json'):
    """Run full Phase 1 sensitivity map."""
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Build task list
    tasks = []
    seed_counter = 1000
    for kr in KR_GRID:
        for beta in BETA_GRID:
            for k in K_GRID:
                tasks.append((kr, beta, k, seed_counter))
                seed_counter += 1
    
    print("=" * 60)
    print(f"PHASE 1: Level 1 Sensitivity Map")
    print(f"  Grid: {len(KR_GRID)} kr × {len(BETA_GRID)} beta × {len(K_GRID)} k = {len(tasks)} points")
    print(f"  Reps/point: {N_REPS}")
    print(f"  Workers: {n_workers}")
    print(f"  Total optimizations: {len(tasks) * N_REPS * 2 * N_STARTS:,}")
    print("=" * 60)
    
    t0 = time.time()
    
    if n_workers == 1:
        results = [run_grid_point(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(run_grid_point, tasks)
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    # Save results
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in sorted(results, key=lambda x: (x['k'], x['beta'], x['kr'])):
        if np.isnan(r.get('median_ratio', np.nan)):
            continue
        print(f"  kr={r['kr']:.2f} β={r['beta']:2d} k={r['k']:.2f}"
              f"  →  R_bias={r['median_ratio']:.2f}"
              f"  overest={r['overest_frac']:.0%}"
              f"  indist={r['indist_frac']:.0%}")
    
    return results


if __name__ == '__main__':
    validate_scale_invariance()
    run_phase1()
