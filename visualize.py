"""
Paper B Visualization
======================
Generates Figures 1–4 from Phase 1–4 results.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

mpl.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

RESULTS_DIR = Path('results')
FIGURES_DIR = Path('figures')

# Known compound k_r values for overlay
COMPOUNDS = {
    'Dimethoate':     {'kr': 0.001, 'beta': 5, 'marker': 'D', 'color': '#d62728'},
    'Diazinon':       {'kr': 0.19,  'beta': 5, 'marker': 's', 'color': '#2ca02c'},
    'Chlorpyrifos':   {'kr': 0.12,  'beta': 5, 'marker': '^', 'color': '#1f77b4'},
    'Carbaryl':       {'kr': 0.97,  'beta': 5, 'marker': 'o', 'color': '#ff7f0e'},
    'Propiconazole':  {'kr': 0.35,  'beta': 5, 'marker': 'v', 'color': '#9467bd'},
}


def load_json(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Level 1 Sensitivity Map (6-panel heatmap)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_level1_map():
    """6-panel heatmap: R_bias across (k_r, beta) for 6 values of k."""
    data = load_json('phase1_level1_map.json')
    
    k_panels = [0.05, 0.1, 0.25, 0.45, 1.0, 5.0]
    kr_grid = sorted(set(r['kr'] for r in data))
    beta_grid = sorted(set(r['beta'] for r in data))
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    fig.suptitle('Figure 1. Level 1 Threshold Bias: $R_{bias} = z_{SD}/\\theta_B$ (median)',
                 fontsize=12, fontweight='bold')
    
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=5.0)
    
    for idx, k_val in enumerate(k_panels):
        ax = axes.flat[idx]
        
        # Build matrix
        matrix = np.full((len(beta_grid), len(kr_grid)), np.nan)
        for r in data:
            if abs(r['k'] - k_val) < 0.01:
                i = beta_grid.index(r['beta'])
                j = kr_grid.index(r['kr'])
                matrix[i, j] = r.get('median_ratio', np.nan)
        
        im = ax.pcolormesh(range(len(kr_grid)), range(len(beta_grid)),
                          matrix, cmap='RdYlBu_r', norm=norm, shading='auto')
        
        # Contours
        try:
            cs = ax.contour(np.arange(len(kr_grid)), np.arange(len(beta_grid)),
                          matrix, levels=[1.0, 1.5, 2.0, 3.0],
                          colors='k', linewidths=[1.5, 0.8, 0.8, 0.8],
                          linestyles=['--', '-', '-', '-'])
            ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')
        except Exception:
            pass
        
        # Paper A reference point
        if k_val == 0.45:
            kr_idx = kr_grid.index(0.5) if 0.5 in kr_grid else None
            b_idx = beta_grid.index(5) if 5 in beta_grid else None
            if kr_idx is not None and b_idx is not None:
                ax.plot(kr_idx, b_idx, '*', color='gold', markersize=14,
                       markeredgecolor='k', markeredgewidth=0.8, zorder=5)
        
        ax.set_xticks(range(len(kr_grid)))
        ax.set_xticklabels([f'{v}' for v in kr_grid], rotation=45)
        ax.set_yticks(range(len(beta_grid)))
        ax.set_yticklabels([f'{v}' for v in beta_grid])
        ax.set_xlabel('$k_r$ (d$^{-1}$)')
        ax.set_ylabel('$\\beta$')
        ax.set_title(f'$k = {k_val}$')
    
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label='$R_{bias}$ = median($\\hat{{z}}_{SD}/\\hat{{\\theta}}_B$)')
    
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / 'Fig1_level1_sensitivity_map.png')
    fig.savefig(FIGURES_DIR / 'Fig1_level1_sensitivity_map.tiff')
    print("Figure 1 saved.")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Three-Level Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_decomposition():
    """Three-panel: (a) stacked bars, (b) k_d inflation, (c) pulsed bias by scenario."""
    
    phase2 = load_json('phase2_level2_kd.json')
    phase3 = load_json('phase3_level3_chronic.json')
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    fig.suptitle('Figure 2. Three-Level Bias Decomposition', fontsize=12, fontweight='bold')
    
    # ── Panel (a): Stacked bar chart ──
    ax = axes[0]
    kr_vals = [r['kr_true'] for r in phase2]
    level1 = [r['level1_ratio'] for r in phase2]
    level2 = [r.get('level2_ratio', 1.0) or 1.0 for r in phase2]
    
    # Get Level 3 from Phase 3 (P2 scenario, 2-pulse 7d)
    level3 = []
    for kr in kr_vals:
        p3_match = [r for r in phase3 if abs(r['kr'] - kr) < 0.01]
        if p3_match and 'P2_2pulse_7d' in p3_match[0]['pulsed']:
            cal_bias = p3_match[0]['calibration_bias_median']
            pred_bias = p3_match[0]['pulsed']['P2_2pulse_7d']['sd_over_true_median']
            if cal_bias and pred_bias and cal_bias > 0:
                level3.append(pred_bias / cal_bias)
            else:
                level3.append(1.0)
        else:
            level3.append(1.0)
    
    x = np.arange(len(kr_vals))
    w = 0.6
    
    # Convert to log-scale for stacking
    l1_log = np.log10(np.array(level1, dtype=float))
    l2_log = np.log10(np.array(level2, dtype=float))
    l3_log = np.log10(np.array(level3, dtype=float))
    
    l1_log = np.where(np.isfinite(l1_log), l1_log, 0)
    l2_log = np.where(np.isfinite(l2_log), l2_log, 0)
    l3_log = np.where(np.isfinite(l3_log), l3_log, 0)
    
    ax.bar(x, l1_log, w, label='Level 1: Threshold', color='#4c72b0', alpha=0.85)
    ax.bar(x, l2_log, w, bottom=l1_log, label='Level 2: $k_d$', color='#dd8452', alpha=0.85)
    ax.bar(x, l3_log, w, bottom=l1_log + l2_log,
           label='Level 3: $\\mathcal{R}$', color='#c44e52', alpha=0.85)
    
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.axhline(1, color='gray', linewidth=0.5, linestyle=':',
               label='×10 (Ashauer 2015)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{kr}' for kr in kr_vals])
    ax.set_xlabel('$k_r$ (d$^{-1}$)')
    ax.set_ylabel('$\\log_{10}$(bias ratio)')
    ax.set_title('(a) Decomposition by $k_r$')
    ax.legend(fontsize=7, loc='upper right')
    
    # ── Panel (b): k_d inflation ratio ──
    ax = axes[1]
    kd_infl = [r['kd_inflation'] for r in phase2]
    kd_B = [r['median_kd_B'] for r in phase2]
    kd_SD = [r['median_kd_SD'] for r in phase2]
    
    ax.bar(x - 0.15, kd_B, 0.3, label='Class B $\\hat{k}_d$', color='#4c72b0')
    ax.bar(x + 0.15, kd_SD, 0.3, label='GUTS-SD $\\hat{k}_d$', color='#c44e52')
    
    # Add inflation ratio as text
    for i, ki in enumerate(kd_infl):
        if np.isfinite(ki):
            ax.text(i, max(kd_B[i], kd_SD[i]) * 1.1,
                   f'×{ki:.1f}', ha='center', fontsize=7, fontweight='bold')
    
    # Ashauer 2015 empirical markers (approximate)
    ashauer_compounds = {
        'Propiconazole': (0.35, 3.0), 'Diazinon': (0.19, 4.5),
        'DNOC': (0.8, 2.0), '4-NBC': (1.2, 1.5)
    }
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{kr}' for kr in kr_vals])
    ax.set_xlabel('$k_r$ (d$^{-1}$)')
    ax.set_ylabel('Fitted $k_d$ (d$^{-1}$)')
    ax.set_title('(b) $k_d$ inflation (Level 2)')
    ax.legend(fontsize=7)
    ax.axhline(0, color='k', linewidth=0.5)
    
    # ── Panel (c): Pulsed prediction bias by scenario ──
    ax = axes[2]
    scenarios = ['P1_2pulse_3d', 'P2_2pulse_7d', 'P3_2pulse_28d',
                 'P4_5pulse_3d', 'P5_5pulse_7d']
    sc_labels = ['2p-3d', '2p-7d', '2p-28d', '5p-3d', '5p-7d']
    
    colors_kr = plt.cm.viridis(np.linspace(0.2, 0.85, len(phase3)))
    
    bar_width = 0.15
    for ik, r in enumerate(phase3):
        offsets = (np.arange(len(scenarios)) + ik * bar_width 
                   - (len(phase3) - 1) * bar_width / 2)
        vals = [r['pulsed'].get(sc, {}).get('sd_over_true_median', np.nan)
                for sc in scenarios]
        ax.bar(offsets, vals, bar_width, label=f'$k_r$={r["kr"]}',
              color=colors_kr[ik], alpha=0.85)
    
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(sc_labels, rotation=30)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('$S_{SD}/S_{true}$')
    ax.set_title('(c) Pulsed prediction bias (Level 3)')
    ax.legend(fontsize=7, title='$k_r$')
    
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / 'Fig2_three_level_decomposition.png')
    fig.savefig(FIGURES_DIR / 'Fig2_three_level_decomposition.tiff')
    print("Figure 2 saved.")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Chronic Prediction Crossover
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_chronic_crossover():
    """Two-panel: (a) B(t) trajectories, (b) crossover time vs k_r."""
    
    phase3 = load_json('phase3_level3_chronic.json')
    
    chronic_times = [1, 2, 4, 7, 14, 21, 28, 56, 96]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle('Figure 3. SD Chronic Prediction Crossover', fontsize=12, fontweight='bold')
    
    # ── Panel (a): B(t) trajectories ──
    ax = axes[0]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    for i, r in enumerate(phase3):
        t_vals = chronic_times
        b_vals = [r['chronic'][str(t)]['bias_median'] for t in t_vals]
        b_vals = [v if np.isfinite(v) else np.nan for v in b_vals]
        
        ax.plot(t_vals, b_vals, 'o-', color=colors[i], linewidth=2,
               markersize=5, label=f'$k_r = {r["kr"]}$')
        
        # Mark crossover
        t_star = r.get('crossover_t_star')
        if t_star:
            ax.axvline(t_star, color=colors[i], linewidth=0.6, linestyle=':', alpha=0.5)
    
    # Reference line and regulatory bands
    ax.axhline(1.0, color='k', linewidth=1.2, linestyle='--', zorder=0)
    ax.axvspan(0, 4, color='lightblue', alpha=0.15, label='96h acute')
    ax.axvspan(21, 28, color='lightyellow', alpha=0.3, label='21–28d chronic')
    
    # Shading
    ax.fill_between([0, 100], [1, 1], [5, 5], color='#c44e52', alpha=0.05)
    ax.fill_between([0, 100], [0, 0], [1, 1], color='#4c72b0', alpha=0.05)
    ax.text(70, 1.8, 'Non-conservative\n(SD underestimates\ntoxicity)',
           fontsize=7, ha='center', color='#c44e52', style='italic')
    ax.text(70, 0.4, 'Over-conservative\n(SD overestimates\ntoxicity)',
           fontsize=7, ha='center', color='#4c72b0', style='italic')
    
    ax.set_xlim(0.8, 110)
    ax.set_xscale('log')
    ax.set_ylim(0.1, 4)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('$B(t) = S_{SD}(t) / S_{true}(t)$')
    ax.set_title('(a) Bias trajectory under constant exposure')
    ax.legend(fontsize=7, loc='upper right')
    
    # ── Panel (b): Crossover time vs k_r ──
    ax = axes[1]
    kr_vals = [r['kr'] for r in phase3]
    t_stars = [r.get('crossover_t_star') for r in phase3]
    
    valid = [(kr, ts) for kr, ts in zip(kr_vals, t_stars) if ts is not None]
    
    if valid:
        kr_v, ts_v = zip(*valid)
        ax.plot(kr_v, ts_v, 'ko-', markersize=8, linewidth=2)
        
        # Theoretical prediction: t* ~ 5/kd to 10/kd
        kr_theory = np.linspace(0.05, 3, 100)
        ax.fill_between(kr_theory, 5/kr_theory, 10/kr_theory,
                        color='gray', alpha=0.15, label='Theory: $5/k_d$ – $10/k_d$')
    
    # Regulatory time horizons
    ax.axhline(4, color='lightblue', linewidth=2, linestyle='--', label='96h acute')
    ax.axhline(21, color='orange', linewidth=2, linestyle='--', label='21d Daphnia chronic')
    ax.axhline(28, color='red', linewidth=2, linestyle='--', label='28d fish chronic')
    
    ax.set_xlabel('$k_r$ (d$^{-1}$)')
    ax.set_ylabel('Crossover time $t^*$ (days)')
    ax.set_title('(b) When does SD switch from under- to over-prediction?')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1, 200)
    ax.legend(fontsize=7)
    
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / 'Fig3_chronic_crossover.png')
    fig.savefig(FIGURES_DIR / 'Fig3_chronic_crossover.tiff')
    print("Figure 3 saved.")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: TWA-UF Matrix + Regulatory Workflow
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_twa_uf():
    """Two-panel: (a) TWA bias heatmap, (b) TWA-UF matrix table."""
    
    output = load_json('phase4_twa_bias.json')
    results = output['results']
    uf_matrix = output['twa_uf_matrix']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), 
                            gridspec_kw={'width_ratios': [1.5, 1]},
                            constrained_layout=True)
    fig.suptitle('Figure 4. TWA Bias Landscape and Correction Matrix',
                 fontsize=12, fontweight='bold')
    
    # ── Panel (a): TWA bias scatter/heatmap ──
    ax = axes[0]
    
    kr_vals = [r['kr'] for r in results]
    ptr_vals = [r['peak_twa_ratio'] for r in results]
    bias_vals = [r['twa_mort_bias_median'] for r in results]
    
    # Filter valid
    valid = [i for i in range(len(bias_vals)) 
             if np.isfinite(bias_vals[i]) and np.isfinite(ptr_vals[i])]
    
    if valid:
        kr_v = [kr_vals[i] for i in valid]
        ptr_v = [ptr_vals[i] for i in valid]
        bias_v = [bias_vals[i] for i in valid]
        
        sc = ax.scatter(ptr_v, kr_v, c=bias_v, s=80, cmap='RdYlBu_r',
                       norm=TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=10),
                       edgecolors='k', linewidths=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label='TWA mortality bias\n(true/predicted; >1 = underestimation)',
                    shrink=0.8)
    
    # TWA-UF matrix boundaries
    ax.axvline(3, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axvline(10, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(0.1, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    
    # Danger zone
    ax.fill_between([10, 200], [0.01, 0.01], [0.1, 0.1],
                    color='red', alpha=0.1, label='TWA N/A zone')
    
    # Compound markers
    for name, info in COMPOUNDS.items():
        # Approximate PECmax/TWA for typical FOCUS scenario
        approx_ptr = 5  # moderate default
        ax.annotate(name[:4], (approx_ptr, info['kr']),
                   fontsize=6, ha='center', va='bottom',
                   color=info['color'])
    
    ax.set_xlabel('PEC$_{max}$ / TWA$_{7d}$')
    ax.set_ylabel('$k_r$ (d$^{-1}$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 200)
    ax.set_ylim(0.02, 10)
    ax.set_title('(a) TWA bias landscape')
    
    # ── Panel (b): TWA-UF matrix as table ──
    ax = axes[1]
    ax.axis('off')
    
    kr_labels = output['kr_labels']
    peak_labels = output['peak_labels']
    
    cell_text = []
    for kl in kr_labels:
        row = []
        for pl in peak_labels:
            key = f"{kl}|{pl}"
            val = uf_matrix.get(key, 'N/A')
            row.append(str(val))
        cell_text.append(row)
    
    table = ax.table(cellText=cell_text,
                     rowLabels=kr_labels,
                     colLabels=peak_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    
    # Color cells by value
    for i, row in enumerate(cell_text):
        for j, val in enumerate(row):
            cell = table[i + 1, j]
            try:
                v = float(val)
                if v >= 5:
                    cell.set_facecolor('#ffcccc')
                elif v >= 3:
                    cell.set_facecolor('#ffe6cc')
                elif v >= 1.5:
                    cell.set_facecolor('#ffffcc')
                else:
                    cell.set_facecolor('#ccffcc')
            except ValueError:
                cell.set_facecolor('#ff9999')  # N/A = red
    
    ax.set_title('(b) TWA-UF Matrix\n(90th percentile correction factor)', fontsize=10)
    
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / 'Fig4_twa_uf_matrix.png')
    fig.savefig(FIGURES_DIR / 'Fig4_twa_uf_matrix.tiff')
    print("Figure 4 saved.")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_decomposition_table():
    """Generate Table 1: Three-Level Bias Decomposition (quantitative)."""
    
    phase1 = load_json('phase1_level1_map.json')
    phase2 = load_json('phase2_level2_kd.json')
    phase3 = load_json('phase3_level3_chronic.json')
    
    print("\n" + "=" * 80)
    print("TABLE 1: THREE-LEVEL BIAS DECOMPOSITION")
    print("=" * 80)
    print(f"{'kr':>6} {'Level1':>8} {'Level2':>8} {'Level3':>8} {'Total':>8} "
          f"{'indist%':>8} {'t*(d)':>8}")
    print("-" * 70)
    
    for p2 in phase2:
        kr = p2['kr_true']
        l1 = p2['level1_ratio']
        l2 = p2.get('level2_ratio', np.nan) or np.nan
        
        # Get Level 3 from Phase 3
        l3 = np.nan
        t_star = None
        p3_match = [r for r in phase3 if abs(r['kr'] - kr) < 0.05]
        if p3_match:
            cal_bias = p3_match[0]['calibration_bias_median']
            p2_pred = p3_match[0]['pulsed'].get('P2_2pulse_7d', {}).get('sd_over_true_median', np.nan)
            if cal_bias and p2_pred and cal_bias > 0 and np.isfinite(p2_pred):
                l3 = p2_pred / cal_bias
            t_star = p3_match[0].get('crossover_t_star')
        
        total = l1 * l2 * l3 if all(np.isfinite([l1, l2, l3])) else np.nan
        
        # Get indistinguishability from Phase 1 (beta=5, k=0.45)
        indist = np.nan
        p1_match = [r for r in phase1
                   if abs(r['kr'] - kr) < 0.05 and r['beta'] == 5 and abs(r['k'] - 0.45) < 0.05]
        if p1_match:
            indist = p1_match[0].get('indist_frac', np.nan)
        
        t_str = f"{t_star:.1f}" if t_star else "N/A"
        
        print(f"{kr:6.2f} {l1:8.2f} {l2:8.2f} {l3:8.2f} {total:8.2f} "
              f"{indist*100 if np.isfinite(indist) else np.nan:8.1f} {t_str:>8}")
    
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Master visualization
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all():
    """Generate all figures and tables."""
    FIGURES_DIR.mkdir(exist_ok=True)
    
    print("Generating Figure 1: Level 1 Sensitivity Map...")
    try:
        fig1_level1_map()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("Generating Figure 2: Three-Level Decomposition...")
    try:
        fig2_decomposition()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("Generating Figure 3: Chronic Crossover...")
    try:
        fig3_chronic_crossover()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("Generating Figure 4: TWA-UF Matrix...")
    try:
        fig4_twa_uf()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nGenerating summary tables...")
    try:
        generate_decomposition_table()
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nAll visualizations complete.")


if __name__ == '__main__':
    generate_all()
