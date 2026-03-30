#!/usr/bin/env python3
"""
Paper B: Master Orchestration Script
======================================
Runs all 4 simulation phases + visualization.

Usage:
    python run_all.py                 # Run everything
    python run_all.py --phase 1       # Run Phase 1 only
    python run_all.py --phase 2 3     # Run Phases 2 and 3
    python run_all.py --viz           # Visualization only
    python run_all.py --workers 4     # Set parallel workers
    python run_all.py --quick         # Quick test (reduced reps)
"""
import argparse
import os
import sys
import time
from pathlib import Path

def setup_paths():
    """Ensure working directory and results/figures dirs exist."""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    Path('results').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)

def apply_quick_mode():
    """Reduce replicate counts for quick testing."""
    import phase1_level1_map as p1
    import phase2_level2_kd as p2
    import phase3_level3_prediction as p3
    import phase4_twa_bias as p4
    
    p1.N_REPS = 20
    p1.KR_GRID = [0.1, 0.5, 2.0]
    p1.BETA_GRID = [3, 5, 10]
    p1.K_GRID = [0.1, 0.45, 2.0]
    
    p2.N_REPS = 50
    p2.KR_VALUES = [0.1, 0.5, 2.0]
    
    p3.N_REPS = 30
    p3.KR_VALUES = [0.1, 0.5, 2.0]
    
    p4.N_REPS = 30
    p4.KR_VALUES = [0.1, 0.5, 2.0]
    p4.EXPOSURE_PATTERNS = {k: v for i, (k, v) in 
                            enumerate(p4.EXPOSURE_PATTERNS.items()) if i < 4}


def run_phase1(n_workers):
    """Phase 1: Level 1 Sensitivity Map."""
    from phase1_level1_map import validate_scale_invariance, run_phase1 as _run
    validate_scale_invariance()
    return _run(n_workers=n_workers)


def run_phase2(n_workers):
    """Phase 2: Level 2 k_d Bias."""
    from phase2_level2_kd import run_phase2 as _run
    return _run(n_workers=n_workers)


def run_phase3(n_workers):
    """Phase 3: Level 3 + Chronic Crossover."""
    from phase3_level3_prediction import run_phase3 as _run
    return _run(n_workers=n_workers)


def run_phase4(n_workers):
    """Phase 4: TWA Bias Mapping."""
    from phase4_twa_bias import run_phase4 as _run
    return _run(n_workers=n_workers)


def run_viz():
    """Generate all figures and tables."""
    from visualize import generate_all
    generate_all()


def main():
    parser = argparse.ArgumentParser(description='Paper B Simulation Pipeline')
    parser.add_argument('--phase', nargs='+', type=int, default=None,
                       help='Phase(s) to run (1-4). Default: all.')
    parser.add_argument('--viz', action='store_true',
                       help='Run visualization only.')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers.')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (reduced replicates).')
    args = parser.parse_args()
    
    setup_paths()
    
    if args.quick:
        print("=" * 60)
        print("QUICK TEST MODE: reduced replicates")
        print("=" * 60)
        apply_quick_mode()
    
    phases_to_run = args.phase if args.phase else [1, 2, 3, 4]
    
    t_total = time.time()
    
    if not args.viz:
        # ── Run simulation phases ──
        phase_funcs = {
            1: ('Phase 1: Level 1 Sensitivity Map', run_phase1),
            2: ('Phase 2: Level 2 k_d Bias', run_phase2),
            3: ('Phase 3: Level 3 + Chronic Crossover', run_phase3),
            4: ('Phase 4: TWA Bias Mapping', run_phase4),
        }
        
        for p in phases_to_run:
            if p not in phase_funcs:
                print(f"Unknown phase: {p}")
                continue
            
            name, func = phase_funcs[p]
            print(f"\n{'#' * 60}")
            print(f"# {name}")
            print(f"{'#' * 60}\n")
            
            t0 = time.time()
            func(args.workers)
            elapsed = time.time() - t0
            print(f"\n>>> {name} completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    # ── Visualization ──
    if args.viz or not args.phase:
        print(f"\n{'#' * 60}")
        print(f"# VISUALIZATION")
        print(f"{'#' * 60}\n")
        run_viz()
    
    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"ALL DONE. Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"\nOutputs:")
    print(f"  Results: results/phase{{1,2,3,4}}_*.json")
    print(f"  Figures: figures/Fig{{1,2,3,4}}_*.png/.tiff")


if __name__ == '__main__':
    main()
