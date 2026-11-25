"""
Regenerate 03_predicted_vs_actual_pce with accurate title N=2,067 matching actual Strategy C2 test split.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
OUT_DIR = PROJECT_ROOT / "docs" / "report_assets" / "research_charts"

df_test_preds = pd.read_csv(PROJECT_ROOT / "ml" / "results" / "phase_8_5" / "phase_8_5_test_predictions.csv")
c2_preds = df_test_preds[df_test_preds["strategy_id"] == "Strategy_C2"]
actual = c2_preds["pce"]
predicted_m3 = c2_preds["pred_M3"]
n_obs = len(c2_preds)

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
scatter = ax.scatter(actual, predicted_m3, alpha=0.45, color="#1d3557", edgecolors="none", s=28)
lims = [0, 18]
ax.plot(lims, lims, color="#e63946", linestyle="--", linewidth=1.8, label="Ideal Parity ($y = \\hat{y}$)")
ax.set_xlim(0, 18)
ax.set_ylim(0, 18)
ax.set_aspect("equal")
ax.set_xlabel("Experimental Observed PCE (%)", fontsize=11, fontweight="bold")
ax.set_ylabel("Predicted PCE (%) — Model M3", fontsize=11, fontweight="bold")
ax.set_title(f"Predicted vs. Actual PCE for Champion Model M3\n(Strategy C2 Scaffold-Pair Holdout, N={n_obs:,})", pad=12, fontweight="bold", fontsize=12)

mae_val = np.mean(np.abs(actual - predicted_m3))
rmse_val = np.sqrt(np.mean((actual - predicted_m3)**2))
r2_val = 1 - np.sum((actual - predicted_m3)**2) / np.sum((actual - actual.mean())**2)
stats_text = f"N = {n_obs:,}\nMAE = {mae_val:.2f}%\nRMSE = {rmse_val:.2f}%\n$R^2$ = {r2_val:.3f}"
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9), fontsize=10)
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()

for name in ["03_predicted_vs_actual_pce", "06_predicted_vs_actual_pce"]:
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.svg")
plt.close(fig)
print(f"Successfully generated 03 and 06 predicted_vs_actual_pce with verified N={n_obs}")
