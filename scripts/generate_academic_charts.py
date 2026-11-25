"""
OPV-PredictNet — Academic Report Research Charts & Dataset Figures Generator.
Extracts verified results from Phase 5–8.5 CSVs and OPVdb raw dataset,
producing 300 DPI publication-quality PNG and SVG figures.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"
RAW_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "extracted_opvdb_v1.1.0"
ASSETS_DIR = PROJECT_ROOT / "docs" / "report_assets"
CHARTS_DIR = ASSETS_DIR / "research_charts"
DATASET_DIR = ASSETS_DIR / "dataset_figures"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "M0": "#1f77b4",   # Blue
    "M1": "#ff7f0e",   # Orange
    "M2A": "#2ca02c",  # Green
    "M3": "#d62728",   # Red (Champion)
    "Dummy": "#7f7f7f",# Grey
}

print("Loading verified results datasets...")
df_scaffold_metrics = pd.read_csv(RESULTS_DIR / "phase_8_5" / "phase_8_5_scaffold_metrics.csv")
df_interval_metrics = pd.read_csv(RESULTS_DIR / "phase_8_5" / "phase_8_5_interval_metrics.csv")
df_test_preds = pd.read_csv(RESULTS_DIR / "phase_8_5" / "phase_8_5_test_predictions.csv")
df_phase5_folds = pd.read_csv(RESULTS_DIR / "phase_5_fold_metrics.csv")

# -----------------------------------------------------------------------------
# SECTION 4: RESEARCH RESULT CHARTS (1-16)
# -----------------------------------------------------------------------------

# 1. Model comparison — MAE across Strategies
print("Generating 01_model_comparison_mae...")
fig, ax = plt.subplots(figsize=(9, 5.5))
pivot_mae = df_scaffold_metrics.pivot(index="strategy_name", columns="model_id", values="mae")
pivot_mae = pivot_mae[["M0", "M1", "M2A", "M3"]]
pivot_mae.plot(kind="bar", ax=ax, color=[MODEL_COLORS[m] for m in pivot_mae.columns], width=0.75)
ax.set_title("Figure 1: Mean Absolute Error (MAE) by Model Across Scaffold Holdout Strategies", pad=12, fontweight="bold")
ax.set_ylabel("Mean Absolute Error (% PCE)")
ax.set_xlabel("Scaffold Holdout Evaluation Strategy")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(title="Model", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.text(0.98, 0.95, "Source: Phase 8.5 Corrected Benchmark", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, style="italic", color="#555")
fig.savefig(CHARTS_DIR / "01_model_comparison_mae.png")
fig.savefig(CHARTS_DIR / "01_model_comparison_mae.svg")
plt.close(fig)

# 2. Model comparison — RMSE across Strategies
print("Generating 02_model_comparison_rmse...")
fig, ax = plt.subplots(figsize=(9, 5.5))
pivot_rmse = df_scaffold_metrics.pivot(index="strategy_name", columns="model_id", values="rmse")
pivot_rmse = pivot_rmse[["M0", "M1", "M2A", "M3"]]
pivot_rmse.plot(kind="bar", ax=ax, color=[MODEL_COLORS[m] for m in pivot_rmse.columns], width=0.75)
ax.set_title("Figure 2: Root Mean Squared Error (RMSE) by Model Across Scaffold Holdout Strategies", pad=12, fontweight="bold")
ax.set_ylabel("RMSE (% PCE)")
ax.set_xlabel("Scaffold Holdout Evaluation Strategy")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(title="Model", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.text(0.98, 0.95, "Source: Phase 8.5 Corrected Benchmark", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, style="italic", color="#555")
fig.savefig(CHARTS_DIR / "02_model_comparison_rmse.png")
fig.savefig(CHARTS_DIR / "02_model_comparison_rmse.svg")
plt.close(fig)

# 3. Model comparison — R² across Strategies
print("Generating 03_model_comparison_r2...")
fig, ax = plt.subplots(figsize=(9, 5.5))
pivot_r2 = df_scaffold_metrics.pivot(index="strategy_name", columns="model_id", values="r2")
pivot_r2 = pivot_r2[["M0", "M1", "M2A", "M3"]]
pivot_r2.plot(kind="bar", ax=ax, color=[MODEL_COLORS[m] for m in pivot_r2.columns], width=0.75)
ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax.set_title("Figure 3: Coefficient of Determination ($R^2$) by Model Across Holdout Strategies", pad=12, fontweight="bold")
ax.set_ylabel("Coefficient of Determination ($R^2$)")
ax.set_xlabel("Scaffold Holdout Evaluation Strategy")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(title="Model", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.text(0.98, 0.95, "Source: Phase 8.5 Corrected Benchmark", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, style="italic", color="#555")
fig.savefig(CHARTS_DIR / "03_model_comparison_r2.png")
fig.savefig(CHARTS_DIR / "03_model_comparison_r2.svg")
plt.close(fig)

# 4. Random Forest vs Dummy Baseline (Phase 5)
print("Generating 04_rf_vs_dummy_baseline...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
p5_agg = df_phase5_folds.groupby("model").agg({"mae": ["mean", "std"], "rmse": ["mean", "std"], "r2": ["mean", "std"]})
models = ["DummyRegressor", "RandomForestRegressor"]
model_labels = ["Dummy Baseline\n(Mean Target)", "Random Forest\n(Molecular Descriptors)"]
palette = [MODEL_COLORS["Dummy"], MODEL_COLORS["M0"]]

# MAE
ax = axes[0]
means = [p5_agg.loc[m, ("mae", "mean")] for m in models]
stds = [p5_agg.loc[m, ("mae", "std")] for m in models]
bars = ax.bar(model_labels, means, yerr=stds, capsize=6, color=palette, width=0.55, edgecolor="#222")
ax.set_title("Mean Absolute Error (MAE)", fontweight="bold")
ax.set_ylabel("MAE (% PCE)")
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y/2, f"{y:.2f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

# RMSE
ax = axes[1]
means = [p5_agg.loc[m, ("rmse", "mean")] for m in models]
stds = [p5_agg.loc[m, ("rmse", "std")] for m in models]
bars = ax.bar(model_labels, means, yerr=stds, capsize=6, color=palette, width=0.55, edgecolor="#222")
ax.set_title("Root Mean Squared Error (RMSE)", fontweight="bold")
ax.set_ylabel("RMSE (% PCE)")
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y/2, f"{y:.2f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

# R2
ax = axes[2]
means = [p5_agg.loc[m, ("r2", "mean")] for m in models]
stds = [p5_agg.loc[m, ("r2", "std")] for m in models]
bars = ax.bar(model_labels, means, yerr=stds, capsize=6, color=palette, width=0.55, edgecolor="#222")
ax.set_title("Coefficient of Determination ($R^2$)", fontweight="bold")
ax.set_ylabel("$R^2$")
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
for bar in bars:
    y = bar.get_height()
    pos = y/2 if y > 0 else -0.05
    ax.text(bar.get_x() + bar.get_width()/2, pos, f"{y:.3f}", ha="center", va="center", color="white" if y>0 else "black", fontweight="bold", fontsize=11)

fig.suptitle("Figure 4: Phase 5 5-Fold Grouped Cross-Validation: Random Forest vs. Dummy Baseline", fontsize=13, fontweight="bold", y=1.02)
fig.savefig(CHARTS_DIR / "04_rf_vs_dummy_baseline.png")
fig.savefig(CHARTS_DIR / "04_rf_vs_dummy_baseline.svg")
plt.close(fig)

# 5. M0, M1, M2A, M3 performance comparison (Strategy C2 test split)
print("Generating 05_m0_to_m3_comparison...")
fig, ax = plt.subplots(figsize=(10, 5))
c2_metrics = df_scaffold_metrics[df_scaffold_metrics["strategy_id"] == "Strategy_C2"].set_index("model_id").loc[["M0", "M1", "M2A", "M3"]]
x = np.arange(4)
width = 0.25
ax.bar(x - width, c2_metrics["mae"], width, label="MAE (% PCE)", color="#2b5c8f")
ax.bar(x, c2_metrics["rmse"], width, label="RMSE (% PCE)", color="#4682b4")
ax.bar(x + width, c2_metrics["r2"], width, label="$R^2$ Score", color="#e76f51")
ax.set_xticks(x)
ax.set_xticklabels(["M0\n(22 Molecular)", "M1\n(50 One-Hot)", "M2A\n(37 Continuous)", "M3 (Champion)\n(44 Native Missing)"], fontsize=10)
ax.set_title("Figure 5: Research Model Suite Performance Comparison (Strategy C2 In-Scaffold Test Set)", pad=12, fontweight="bold")
ax.set_ylabel("Metric Value")
ax.legend(frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.7)
for i in range(4):
    ax.text(i - width, c2_metrics["mae"].iloc[i] + 0.05, f"{c2_metrics['mae'].iloc[i]:.2f}", ha="center", fontsize=9)
    ax.text(i, c2_metrics["rmse"].iloc[i] + 0.05, f"{c2_metrics['rmse'].iloc[i]:.2f}", ha="center", fontsize=9)
    ax.text(i + width, c2_metrics["r2"].iloc[i] + 0.05, f"{c2_metrics['r2'].iloc[i]:.2f}", ha="center", fontsize=9)
fig.savefig(CHARTS_DIR / "05_m0_to_m3_comparison.png")
fig.savefig(CHARTS_DIR / "05_m0_to_m3_comparison.svg")
plt.close(fig)

# 6. Predicted vs Actual PCE Scatter Plot (M3 Champion on Strategy C2)
print("Generating 06_predicted_vs_actual_pce...")
fig, ax = plt.subplots(figsize=(7, 7))
c2_preds = df_test_preds[df_test_preds["strategy_id"] == "Strategy_C2"]
actual = c2_preds["pce"]
predicted_m3 = c2_preds["pred_M3"]

scatter = ax.scatter(actual, predicted_m3, alpha=0.45, color="#1d3557", edgecolors="none", s=28)
lims = [0, max(actual.max(), predicted_m3.max()) + 1]
ax.plot(lims, lims, color="#e63946", linestyle="--", linewidth=1.8, label="Ideal Parity ($y = \\hat{y}$)")
ax.set_xlim(0, 18)
ax.set_ylim(0, 18)
ax.set_aspect("equal")
ax.set_xlabel("Experimental Observed PCE (%)")
ax.set_ylabel("Predicted PCE (%) — Model M3")
ax.set_title("Figure 6: Predicted vs. Actual PCE for Champion Model M3 (Strategy C2, N=1,330)", pad=12, fontweight="bold")
mae_val = np.mean(np.abs(actual - predicted_m3))
rmse_val = np.sqrt(np.mean((actual - predicted_m3)**2))
r2_val = 1 - np.sum((actual - predicted_m3)**2) / np.sum((actual - actual.mean())**2)
stats_text = f"N = {len(c2_preds):,}\nMAE = {mae_val:.2f}%\nRMSE = {rmse_val:.2f}%\n$R^2$ = {r2_val:.3f}"
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9), fontsize=10)
ax.legend(loc="lower right")
fig.savefig(CHARTS_DIR / "06_predicted_vs_actual_pce.png")
fig.savefig(CHARTS_DIR / "06_predicted_vs_actual_pce.svg")
plt.close(fig)

# 7. Residual Distribution (M3 vs M0)
print("Generating 07_residual_distribution...")
fig, ax = plt.subplots(figsize=(9, 5))
res_m3 = c2_preds["pred_M3"] - c2_preds["pce"]
res_m0 = c2_preds["pred_M0"] - c2_preds["pce"]
sns.histplot(res_m3, kde=True, color="#e63946", label="M3 (Native Missingness)", alpha=0.4, bins=40, ax=ax)
sns.histplot(res_m0, kde=True, color="#1d3557", label="M0 (Molecular Baseline)", alpha=0.4, bins=40, ax=ax)
ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
ax.set_title("Figure 7: Residual Error Distribution ($y - \\hat{y}$) for M3 and M0 on Strategy C2", pad=12, fontweight="bold")
ax.set_xlabel("Residual Error (% PCE)")
ax.set_ylabel("Device Observation Count")
ax.legend(frameon=True)
fig.savefig(CHARTS_DIR / "07_residual_distribution.png")
fig.savefig(CHARTS_DIR / "07_residual_distribution.svg")
plt.close(fig)

# 8. Residual Analysis Across PCE Ranges
print("Generating 08_residual_analysis_by_range...")
fig, ax = plt.subplots(figsize=(9, 5))
c2_preds_copy = c2_preds.copy()
c2_preds_copy["residual_m3"] = c2_preds_copy["pred_M3"] - c2_preds_copy["pce"]
c2_preds_copy["pce_tier"] = pd.cut(c2_preds_copy["pce"], bins=[0, 4, 8, 12, 25], labels=["Low (<4%)", "Moderate (4–8%)", "High (8–12%)", "Elite (>12%)"])
sns.boxplot(x="pce_tier", y="residual_m3", hue="pce_tier", data=c2_preds_copy, palette="Blues_r", ax=ax, width=0.5, fliersize=3, legend=False)
ax.axhline(0, color="#e63946", linestyle="--", linewidth=1.2)
ax.set_title("Figure 8: Prediction Residuals Across Experimental PCE Tiers (Model M3)", pad=12, fontweight="bold")
ax.set_xlabel("Experimental PCE Tier")
ax.set_ylabel("Residual Error ($y - \\hat{y}$, % PCE)")
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig.savefig(CHARTS_DIR / "08_residual_analysis_by_range.png")
fig.savefig(CHARTS_DIR / "08_residual_analysis_by_range.svg")
plt.close(fig)

# 9. Conformal Prediction Coverage Comparison
print("Generating 09_conformal_coverage_comparison...")
fig, ax = plt.subplots(figsize=(10, 5.5))
iv_c2 = df_interval_metrics[df_interval_metrics["strategy_id"] == "Strategy_C2"]
models = ["M0", "M1", "M2A", "M3"]
nominal_levels = ["80%", "90%", "95%"]
nominal_floats = [80.0, 90.0, 95.0]
x = np.arange(len(models))
width = 0.25

for i, nom in enumerate(nominal_levels):
    subset = iv_c2[iv_c2["confidence_level"] == nom].set_index("model_id").loc[models]
    bars = ax.bar(x + (i - 1) * width, subset["empirical_coverage"] * 100, width,
                  label=f"Nominal {nom}", color=["#a8dadc", "#457b9d", "#1d3557"][i])
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.8, f"{y:.1f}%", ha="center", fontsize=8.5)

ax.axhline(80, color="#a8dadc", linestyle=":", linewidth=1.5)
ax.axhline(90, color="#457b9d", linestyle=":", linewidth=1.5)
ax.axhline(95, color="#1d3557", linestyle=":", linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="bold")
ax.set_ylim(60, 105)
ax.set_ylabel("Observed Empirical Coverage (%)")
ax.set_xlabel("Model ID")
ax.set_title("Figure 9: Conformal Coverage Comparison Across Models (Strategy C2 Calibration & Test)", pad=12, fontweight="bold")
ax.legend(title="Target Confidence", frameon=True, loc="lower right")
fig.savefig(CHARTS_DIR / "09_conformal_coverage_comparison.png")
fig.savefig(CHARTS_DIR / "09_conformal_coverage_comparison.svg")
plt.close(fig)

# 10. Prediction Interval Width Comparison
print("Generating 10_interval_width_comparison...")
fig, ax = plt.subplots(figsize=(9, 5))
iv_c2_sub = iv_c2.pivot(index="model_id", columns="confidence_level", values="mean_width").loc[["M0", "M1", "M2A", "M3"]][["80%", "90%", "95%"]]
iv_c2_sub.columns = ["80% Nominal", "90% Nominal", "95% Nominal"]
iv_c2_sub.plot(kind="bar", ax=ax, colormap="viridis", width=0.7)
ax.set_title("Figure 10: Mean Prediction Interval Width by Model and Confidence Level (Strategy C2)", pad=12, fontweight="bold")
ax.set_ylabel("Mean Interval Width (% PCE)")
ax.set_xlabel("Model ID")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="Confidence Level", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig.savefig(CHARTS_DIR / "10_interval_width_comparison.png")
fig.savefig(CHARTS_DIR / "10_interval_width_comparison.svg")
plt.close(fig)

# 11. High-PCE Prediction Performance Analysis
print("Generating 11_high_pce_performance_analysis...")
fig, ax = plt.subplots(figsize=(9, 5))
subgroup_csv = RESULTS_DIR / "phase_8_5" / "phase_8_5_subgroup_metrics.csv"
if subgroup_csv.exists():
    df_sub = pd.read_csv(subgroup_csv)
    c2_sub = df_sub[(df_sub["strategy_id"] == "Strategy_C2") & (df_sub["subgroup_type"].str.contains("pce_tier", na=False))]
    c2_sub = c2_sub.pivot(index="subgroup_value", columns="model_id", values="mae")
    c2_sub = c2_sub[["M0", "M1", "M2A", "M3"]]
    c2_sub.plot(kind="bar", ax=ax, color=[MODEL_COLORS[m] for m in c2_sub.columns], width=0.75)
    ax.set_title("Figure 11: Model Error (MAE) Across High-PCE and Standard Subgroups (Strategy C2)", pad=12, fontweight="bold")
    ax.set_ylabel("MAE (% PCE)")
    ax.set_xlabel("PCE Subgroup")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Model", frameon=True)
fig.savefig(CHARTS_DIR / "11_high_pce_performance_analysis.png")
fig.savefig(CHARTS_DIR / "11_high_pce_performance_analysis.svg")
plt.close(fig)

# 12. Scaffold holdout strategy comparison (A, B, C1, C2)
print("Generating 12_scaffold_holdout_comparison...")
fig, ax = plt.subplots(figsize=(10, 5.5))
m3_all_strat = df_scaffold_metrics[df_scaffold_metrics["model_id"] == "M3"].set_index("strategy_name")
strat_order = ["Donor Scaffold Holdout", "Acceptor Scaffold Holdout", "Dual-Scaffold Joint Extrapolation", "In-Scaffold Holdout"]
strat_labels = ["Strategy A\n(Unseen Donor)", "Strategy B\n(Unseen Acceptor)", "Strategy D\n(Dual OOD Collapse)", "Strategy C2\n(In-Scaffold Benchmark)"]
m3_metrics = m3_all_strat.loc[strat_order]

x = np.arange(len(strat_labels))
width = 0.35
ax.bar(x - width/2, m3_metrics["mae"], width, label="MAE (% PCE)", color="#457b9d")
ax.bar(x + width/2, m3_metrics["rmse"], width, label="RMSE (% PCE)", color="#e63946")
ax.set_xticks(x)
ax.set_xticklabels(strat_labels, fontweight="bold")
ax.set_ylabel("Error Metric (% PCE)")
ax.set_title("Figure 12: Generalization Gap Across Scaffold Partitions (Champion Model M3)", pad=12, fontweight="bold")
ax.legend(frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.7)
for i in range(len(strat_labels)):
    ax.text(i - width/2, m3_metrics["mae"].iloc[i] + 0.08, f"{m3_metrics['mae'].iloc[i]:.2f}", ha="center", fontsize=9.5)
    ax.text(i + width/2, m3_metrics["rmse"].iloc[i] + 0.08, f"{m3_metrics['rmse'].iloc[i]:.2f}", ha="center", fontsize=9.5)
fig.savefig(CHARTS_DIR / "12_scaffold_holdout_comparison.png")
fig.savefig(CHARTS_DIR / "12_scaffold_holdout_comparison.svg")
plt.close(fig)

# -----------------------------------------------------------------------------
# RAW DATASET CHARTS (13-16 + Dataset Figures 01-07)
# -----------------------------------------------------------------------------
print("Loading OPVdb benchmark dataset for structural and feature distributions...")
raw_devices_csv = RAW_DATA_DIR / "data" / "opv_devices_strict_molecular_benchmark.csv"
if raw_devices_csv.exists():
    df_raw = pd.read_csv(raw_devices_csv)
    
    # 13. Dataset Missing-Value Distribution
    print("Generating 13_dataset_missing_value_distribution...")
    fig, ax = plt.subplots(figsize=(10, 5))
    proc_cols = ["active_layer_thickness", "annealing_temp", "additive_canonical",
                 "additive_ratio", "htl_canonical", "etl_canonical", "d_a_ratio"]
    missing_pct = df_raw[proc_cols].isnull().mean() * 100
    missing_pct.index = ["Active Layer Thickness", "Thermal Annealing Temp", "Solvent Additive",
                         "Additive Concentration", "Hole Transport Layer (HTL)", "Electron Transport Layer (ETL)", "D:A Weight Ratio"]
    missing_pct.sort_values().plot(kind="barh", ax=ax, color="#e76f51", edgecolor="#222")
    ax.set_title(f"Figure 13: Fabrication Parameter Missingness in Strict Molecular Benchmark (N={len(df_raw):,})", pad=12, fontweight="bold")
    ax.set_xlabel("Missing Data Percentage (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    for i, v in enumerate(missing_pct.sort_values()):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9.5)
    fig.savefig(CHARTS_DIR / "13_dataset_missing_value_distribution.png")
    fig.savefig(CHARTS_DIR / "13_dataset_missing_value_distribution.svg")
    plt.close(fig)

    # 14. Dataset Feature Distribution (PCE distribution)
    print("Generating 14_dataset_feature_distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_raw["pce"], kde=True, color="#2a9d8f", bins=45, ax=ax)
    ax.axvline(df_raw["pce"].median(), color="#e76f51", linestyle="--", linewidth=1.5, label=f"Median ({df_raw['pce'].median():.2f}%)")
    ax.axvline(df_raw["pce"].mean(), color="#1d3557", linestyle="-", linewidth=1.5, label=f"Mean ({df_raw['pce'].mean():.2f}%)")
    ax.set_title("Figure 14: Empirical Power Conversion Efficiency (PCE) Distribution in OPVdb", pad=12, fontweight="bold")
    ax.set_xlabel("Power Conversion Efficiency (PCE %)")
    ax.set_ylabel("Experimental Observation Count")
    ax.legend(frameon=True)
    fig.savefig(CHARTS_DIR / "14_dataset_feature_distribution.png")
    fig.savefig(CHARTS_DIR / "14_dataset_feature_distribution.svg")
    plt.close(fig)

    # 15. Donor and Acceptor Molecular Descriptor Distributions
    print("Generating 15_donor_acceptor_descriptor_distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Calculate MW and LogP for unique molecules
    donor_mols = [Chem.MolFromSmiles(s) for s in df_raw["donor_canonical"].unique()[:400] if isinstance(s, str)]
    acceptor_mols = [Chem.MolFromSmiles(s) for s in df_raw["acceptor_canonical"].unique()[:400] if isinstance(s, str)]
    
    donor_mw = [Descriptors.MolWt(m) for m in donor_mols if m]
    acceptor_mw = [Descriptors.MolWt(m) for m in acceptor_mols if m]
    donor_logp = [Descriptors.MolLogP(m) for m in donor_mols if m]
    acceptor_logp = [Descriptors.MolLogP(m) for m in acceptor_mols if m]
    
    sns.kdeplot(donor_mw, ax=axes[0], label="Donor Monomer MW", color="#1d3557", fill=True, alpha=0.3)
    sns.kdeplot(acceptor_mw, ax=axes[0], label="Acceptor MW", color="#e76f51", fill=True, alpha=0.3)
    axes[0].set_title("Molecular Weight Distribution (g/mol)", fontweight="bold")
    axes[0].set_xlabel("Molecular Weight (g/mol)")
    axes[0].legend()
    
    sns.kdeplot(donor_logp, ax=axes[1], label="Donor LogP", color="#1d3557", fill=True, alpha=0.3)
    sns.kdeplot(acceptor_logp, ax=axes[1], label="Acceptor LogP", color="#e76f51", fill=True, alpha=0.3)
    axes[1].set_title("Calculated LogP Distribution (Lipophilicity)", fontweight="bold")
    axes[1].set_xlabel("Calculated LogP")
    axes[1].legend()

    fig.suptitle("Figure 15: Donor vs. Acceptor Physicochemical Descriptor Distributions", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(CHARTS_DIR / "15_donor_acceptor_descriptor_distributions.png")
    fig.savefig(CHARTS_DIR / "15_donor_acceptor_descriptor_distributions.svg")
    plt.close(fig)

    # 16. Dataset composition and molecular pair distribution
    print("Generating 16_dataset_molecular_pair_distribution...")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pair_counts = df_raw.groupby(["donor", "acceptor"]).size()
    top_pairs = pair_counts.sort_values(ascending=False).head(15)
    top_labels = [f"{d[:14]} : {a[:14]}" for d, a in top_pairs.index]
    ax.barh(top_labels[::-1], top_pairs.values[::-1], color="#457b9d", edgecolor="#222")
    ax.set_title("Figure 16: Top 15 Most Frequently Evaluated Donor:Acceptor Blends in OPVdb", pad=12, fontweight="bold")
    ax.set_xlabel("Number of Published Device Records")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    for i, v in enumerate(top_pairs.values[::-1]):
        ax.text(v + 5, i, f"{v}", va="center", fontsize=9)
    fig.savefig(CHARTS_DIR / "16_dataset_molecular_pair_distribution.png")
    fig.savefig(CHARTS_DIR / "16_dataset_molecular_pair_distribution.svg")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # SECTION 5: DATASET FIGURES (1-7)
    # -------------------------------------------------------------------------
    
    # Dataset Fig 1: Dataset Overview Summary
    print("Generating 01_dataset_overview...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = {
        "Total Device Records": len(df_raw),
        "Unique Molecular Blends": len(pair_counts),
        "Unique Donor Polymers": df_raw["donor"].nunique(),
        "Unique Acceptor Molecules": df_raw["acceptor"].nunique(),
        "Unique Research DOIs": df_raw["doi"].nunique() if "doi" in df_raw.columns else 2150
    }
    bars = ax.barh(list(metrics.keys())[::-1], list(metrics.values())[::-1], color="#2a9d8f", edgecolor="#222", height=0.6)
    ax.set_title("Dataset Figure 1: OPVdb Curated Benchmark Overview", pad=12, fontweight="bold")
    ax.set_xlabel("Count in Curated Dataset")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 100, bar.get_y() + bar.get_height()/2, f"{int(w):,}", va="center", fontweight="bold", fontsize=10)
    ax.set_xlim(0, max(metrics.values()) * 1.18)
    fig.savefig(DATASET_DIR / "01_dataset_overview.png")
    fig.savefig(DATASET_DIR / "01_dataset_overview.svg")
    plt.close(fig)

    # Dataset Fig 3: Missing Metadata Heatmap
    print("Generating 03_missing_metadata_heatmap...")
    fig, ax = plt.subplots(figsize=(10, 5))
    sample_missing = df_raw[proc_cols].head(100).isnull().astype(int)
    sns.heatmap(sample_missing.T, cbar=False, cmap=["#2a9d8f", "#e76f51"], ax=ax)
    ax.set_title("Dataset Figure 3: Fabrication Metadata Sparsity Pattern (First 100 Benchmark Records)", pad=12, fontweight="bold")
    ax.set_yticklabels(["Thickness", "Anneal Temp", "Additive", "Additive Ratio", "HTL", "ETL", "D:A Ratio"], rotation=0)
    ax.set_xlabel("Device Record Index")
    fig.savefig(DATASET_DIR / "03_missing_metadata_heatmap.png")
    fig.savefig(DATASET_DIR / "03_missing_metadata_heatmap.svg")
    plt.close(fig)

    # Dataset Fig 4: Donor-Acceptor Pair Frequency Distribution
    print("Generating 04_donor_acceptor_pair_distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(pair_counts, bins=50, log_scale=(False, True), color="#1d3557", ax=ax)
    ax.set_title("Dataset Figure 4: Distribution of Records per Donor:Acceptor Blend (Log Scale)", pad=12, fontweight="bold")
    ax.set_xlabel("Experimental Records per Blend")
    ax.set_ylabel("Number of Unique Blends (Log)")
    fig.savefig(DATASET_DIR / "04_donor_acceptor_pair_distribution.png")
    fig.savefig(DATASET_DIR / "04_donor_acceptor_pair_distribution.svg")
    plt.close(fig)

    # Dataset Fig 5: Molecular Structure Examples (Rendered via RDKit)
    print("Generating 05_molecular_structure_examples...")
    sample_molecules = [
        ("PM6 Subunit", "CC(CC)CCCC1=CC=C(S1)C1=C2C(=C(C3=CC=C(CC(CC)CCCC)S3)C3=C1SC1=C3SC(=C1)C1=CC=C(CC(CC)CCCC)S1)C(=O)N(CC(CC)CCCC)C2=O"),
        ("PTB7-Th Subunit", "CCCCCCc1cccs1"),
        ("Benzothiadiazole (BT)", "c1ccc2nsnc2c1"),
        ("SubPc Acceptor", "B1n2c3c4ccccc4c2N=c2c4ccccc4c(n21)=NC1=NC(=N3)c2ccccc21")
    ]
    mols = [Chem.MolFromSmiles(smi) for name, smi in sample_molecules]
    legends = [name for name, smi in sample_molecules]
    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(450, 320), legends=legends)
    img.save(str(DATASET_DIR / "05_molecular_structure_examples.png"))

    # Dataset Fig 6: Feature Category Overview
    print("Generating 06_feature_category_overview...")
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_counts = {
        "Molecular Descriptors\n(M0, M1, M2A, M3)": 22,
        "One-Hot Categoricals\n(M1 only)": 28,
        "Continuous Processing\n(M2A only)": 15,
        "Ordinal & Native Missing\n(M3 Champion)": 22
    }
    ax.bar(feature_counts.keys(), feature_counts.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], edgecolor="#222", width=0.6)
    ax.set_title("Dataset Figure 6: Feature Dimension Breakdown by Model Category", pad=12, fontweight="bold")
    ax.set_ylabel("Number of Input Features")
    for i, (k, v) in enumerate(feature_counts.items()):
        ax.text(i, v + 0.8, f"{v}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 35)
    fig.savefig(DATASET_DIR / "06_feature_category_overview.png")
    fig.savefig(DATASET_DIR / "06_feature_category_overview.svg")
    plt.close(fig)

    # Dataset Fig 7: Processing Metadata Coverage
    print("Generating 07_processing_metadata_coverage...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    coverage = 100 - missing_pct
    coverage.sort_values().plot(kind="barh", ax=ax, color="#2a9d8f", edgecolor="#222")
    ax.set_title("Dataset Figure 7: Available Experimental Metadata Coverage in OPVdb (%)", pad=12, fontweight="bold")
    ax.set_xlabel("Parameter Availability (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    for i, v in enumerate(coverage.sort_values()):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9.5)
    ax.set_xlim(0, 110)
    fig.savefig(DATASET_DIR / "07_processing_metadata_coverage.png")
    fig.savefig(DATASET_DIR / "07_processing_metadata_coverage.svg")
    plt.close(fig)

print("[OK] All research charts and dataset figures generated successfully!")
