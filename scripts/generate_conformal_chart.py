"""
Generate 04_conformal_coverage.png and .svg using verified Phase 8.5 interval metrics.
Also ensures 03_predicted_vs_actual_pce.png/.svg is present.
"""
import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)

OUT_DIR = PROJECT_ROOT / "docs" / "report_assets" / "research_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Conformal Coverage Chart
csv_path = PROJECT_ROOT / "ml" / "results" / "phase_8_5" / "phase_8_5_interval_metrics.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    
    # Filter for nominal confidence levels 80%, 90%, 95%
    # We can plot grouped bar chart: Nominal vs Empirical coverage across M0, M1, M2A, M3
    # Let's average across strategies or plot Strategy A (Donor Scaffold Holdout) as representative
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot 1: Strategy A (Donor Scaffold Holdout) Empirical Coverage vs Nominal
    strat_a = df[df['strategy_id'] == 'Strategy_A'].copy()
    models = ['M0', 'M1', 'M2A', 'M3']
    conf_levels = ['80%', '90%', '95%']
    nominal_vals = [0.80, 0.90, 0.95]
    
    x = np.arange(len(models))
    width = 0.22
    
    colors = ['#3b82f6', '#10b981', '#8b5cf6']
    
    for i, (conf, nom) in enumerate(zip(conf_levels, nominal_vals)):
        sub = strat_a[strat_a['confidence_level'] == conf].set_index('model_id')
        covs = [sub.loc[m, 'empirical_coverage'] * 100 for m in models]
        rects = axes[0].bar(x + (i - 1) * width, covs, width, label=f'Nominal {conf}', color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.8)
        # Add value label
        for rect in rects:
            height = rect.get_height()
            axes[0].annotate(f'{height:.1f}%',
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    axes[0].axhline(80, color='#3b82f6', linestyle='--', linewidth=1.2, alpha=0.7)
    axes[0].axhline(90, color='#10b981', linestyle='--', linewidth=1.2, alpha=0.7)
    axes[0].axhline(95, color='#8b5cf6', linestyle='--', linewidth=1.2, alpha=0.7)
    axes[0].set_ylabel('Empirical Coverage (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Conformal Coverage by Model (Strategy A - Donor Holdout)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=11, fontweight='bold')
    axes[0].set_ylim(70, 105)
    axes[0].legend(loc='lower right', frameon=True, fontsize=9)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    
    # Subplot 2: Interval Mean Width (sharpness/efficiency)
    for i, (conf, nom) in enumerate(zip(conf_levels, nominal_vals)):
        sub = strat_a[strat_a['confidence_level'] == conf].set_index('model_id')
        widths = [sub.loc[m, 'mean_width'] for m in models]
        rects = axes[1].bar(x + (i - 1) * width, widths, width, label=f'{conf} Interval', color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.8)
        for rect in rects:
            height = rect.get_height()
            axes[1].annotate(f'{height:.1f}%',
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    axes[1].set_ylabel('Mean Interval Width (PCE %)', fontsize=11, fontweight='bold')
    axes[1].set_title('Conformal Interval Efficiency / Sharpness (Strategy A)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, fontsize=9)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    chart_png = OUT_DIR / "04_conformal_coverage.png"
    chart_svg = OUT_DIR / "04_conformal_coverage.svg"
    plt.savefig(chart_png, dpi=300)
    plt.savefig(chart_svg)
    plt.close()
    print(f"Generated {chart_png.name} and {chart_svg.name}")

# 2. Ensure 03_predicted_vs_actual_pce exists
src_pce_png = OUT_DIR / "06_predicted_vs_actual_pce.png"
dst_pce_png = OUT_DIR / "03_predicted_vs_actual_pce.png"
src_pce_svg = OUT_DIR / "06_predicted_vs_actual_pce.svg"
dst_pce_svg = OUT_DIR / "03_predicted_vs_actual_pce.svg"

if src_pce_png.exists() and not dst_pce_png.exists():
    shutil.copy(src_pce_png, dst_pce_png)
    shutil.copy(src_pce_svg, dst_pce_svg)
    print("Copied 06_predicted_vs_actual_pce to 03_predicted_vs_actual_pce")
elif dst_pce_png.exists():
    print("03_predicted_vs_actual_pce already exists.")
