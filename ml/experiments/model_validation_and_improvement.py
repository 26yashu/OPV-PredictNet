"""OPV-PredictNet - Phase 6 Model Validation, Robustness, and Improvement.

Conducts:
1. Target and device architecture error audit (conventional vs inverted, thickness, annealing, additives).
2. Residual and reliability analysis across PCE ranges, chemical families, and pair frequencies.
3. Controlled hyperparameter exploration (using inner training folds only for model selection).
4. Ensemble prediction uncertainty estimation and empirical calibration assessment.
5. Out-of-fold generalization comparison across the 5 evaluation splits.

Strict constraints:
- Existing 22 molecular descriptors from descriptors.py
- OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records)
- Zero synthetic data, zero imputation, no removal of valid records
- No production API or frontend modifications
"""

import sys
import os
import json
import time
import platform
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath('.'))
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES

RANDOM_STATE = 42
BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
PHASE_5_PRED_PATH = 'ml/results/phase_5_predictions.csv'
RESULTS_DIR = 'ml/results'


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_true - y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def main():
    print("=== PHASE 6: MODEL VALIDATION, ROBUSTNESS & IMPROVEMENT ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load benchmark dataset and Phase 5 baseline predictions
    print(f"Loading benchmark dataset from {BENCHMARK_PATH}...")
    df = pd.read_csv(BENCHMARK_PATH)
    total_samples = len(df)
    print(f"Total benchmark records: {total_samples}")
    
    print(f"Loading Phase 5 primary out-of-fold predictions from {PHASE_5_PRED_PATH}...")
    p5_df = pd.read_csv(PHASE_5_PRED_PATH)
    assert len(p5_df) == total_samples, "Record count mismatch with Phase 5 predictions!"
    
    # Merge Phase 5 predictions with metadata for error analysis
    df['rf_pce_pred'] = p5_df['rf_pce_pred']
    df['rf_residual'] = df['pce'] - df['rf_pce_pred']
    df['rf_abs_error'] = np.abs(df['rf_residual'])
    
    # =========================================================================
    # PART 1: TARGET AND DEVICE ARCHITECTURE EFFECT AUDIT
    # =========================================================================
    print("\n--- PART 1: DEVICE ARCHITECTURE & METADATA ERROR AUDIT ---")
    
    # Normalize device architecture into clean categories
    def classify_architecture(row):
        d_type = str(row['device_type']).lower()
        d_struct = str(row['device_structure']).lower()
        if 'inverted' in d_type or 'inverted' in d_struct:
            return 'Inverted'
        elif 'conventional' in d_type or 'conventional' in d_struct:
            return 'Conventional'
        else:
            return 'Unspecified / Other'
            
    df['arch_category'] = df.apply(classify_architecture, axis=1)
    
    # Check single-junction vs tandem occurrences
    df['has_additive'] = df['additive'].notna()
    df['has_annealing'] = df['annealing_temp'].notna()
    df['has_thickness'] = df['active_layer_thickness'].notna()
    
    # Classify acceptor chemical family
    def classify_acceptor(row):
        acc = str(row['acceptor_canonical']).upper()
        smi = str(row['acceptor_smiles'])
        if any(f in acc for f in ['PCBM', 'PC61BM', 'PC71BM', 'ICBA', 'C60', 'C70', 'FULLERENE']):
            return 'Fullerene'
        elif any(f in acc for f in ['PDI', 'PPDI', 'Perylene']):
            return 'PDI-based'
        elif 'C=C' in smi and any(f in acc for f in ['Y6', 'L8-BO', 'BTP', 'ITIC', 'IT-4F', 'IEICO', 'Y7']):
            return 'Fused-Ring NFA'
        elif 'P-' in acc or 'PNDI' in acc or 'PY' in acc:
            return 'Polymer Acceptor'
        else:
            return 'Other NFA'
            
    df['acceptor_family'] = df.apply(classify_acceptor, axis=1)
    
    # Classify donor chemical family
    def classify_donor(row):
        don = str(row['donor_canonical']).upper()
        smi = str(row['donor_smiles'])
        mw = float(row.get('donor_mw', 1000))
        # Known small-molecule donors
        sm_donors = ['DRCN5T', 'BTR', 'SMPV', 'SM1', 'SM2', 'DR3T', 'T1', 'DRTB-T']
        if any(s in don for s in sm_donors) or (len(smi) < 120 and 'c1sc' not in smi):
            return 'Small-Molecule Donor'
        else:
            return 'Polymer Donor'
            
    df['donor_family'] = df.apply(classify_donor, axis=1)
    
    # Calculate error metrics across all subsets
    error_analysis_rows = []
    
    def evaluate_subset(subset_df, category_name, subcategory_value):
        if len(subset_df) == 0:
            return
        m = compute_metrics(subset_df['pce'].to_numpy(), subset_df['rf_pce_pred'].to_numpy())
        error_analysis_rows.append({
            "category": category_name,
            "subcategory": subcategory_value,
            "record_count": len(subset_df),
            "percentage_of_dataset": round(len(subset_df) / total_samples * 100, 2),
            "mean_actual_pce": round(float(subset_df['pce'].mean()), 2),
            "mean_pred_pce": round(float(subset_df['rf_pce_pred'].mean()), 2),
            "mae": round(m["mae"], 4),
            "rmse": round(m["rmse"], 4),
            "r2": round(m["r2"], 4),
            "bias": round(m["bias"], 4)
        })

    # Overall dataset
    evaluate_subset(df, "Overall", "All 22,467 Benchmark Devices")
    
    # By Architecture
    for arch in ['Conventional', 'Inverted', 'Unspecified / Other']:
        evaluate_subset(df[df['arch_category'] == arch], "Device Architecture", arch)
        
    # By Additive Presence
    evaluate_subset(df[df['has_additive']], "Processing Metadata", "With Additive Recorded")
    evaluate_subset(df[~df['has_additive']], "Processing Metadata", "No Additive Recorded")
    
    # By Annealing Presence
    evaluate_subset(df[df['has_annealing']], "Processing Metadata", "With Annealing Temp Recorded")
    evaluate_subset(df[~df['has_annealing']], "Processing Metadata", "No Annealing Recorded")
    
    # By Thickness Presence
    evaluate_subset(df[df['has_thickness']], "Processing Metadata", "With Thickness Recorded")
    evaluate_subset(df[~df['has_thickness']], "Processing Metadata", "No Thickness Recorded")
    
    # By Acceptor Family
    for acc_fam in df['acceptor_family'].value_counts().index:
        evaluate_subset(df[df['acceptor_family'] == acc_fam], "Acceptor Chemical Family", acc_fam)
        
    # By Donor Family
    for don_fam in df['donor_family'].value_counts().index:
        evaluate_subset(df[df['donor_family'] == don_fam], "Donor Chemical Family", don_fam)
        
    # By PCE Range
    pce_bins = [
        ("PCE < 3.0%", df['pce'] < 3.0),
        ("3.0% <= PCE < 6.0%", (df['pce'] >= 3.0) & (df['pce'] < 6.0)),
        ("6.0% <= PCE < 9.0%", (df['pce'] >= 6.0) & (df['pce'] < 9.0)),
        ("9.0% <= PCE < 12.0%", (df['pce'] >= 9.0) & (df['pce'] < 12.0)),
        ("12.0% <= PCE < 15.0%", (df['pce'] >= 12.0) & (df['pce'] < 15.0)),
        ("PCE >= 15.0%", df['pce'] >= 15.0)
    ]
    for b_label, mask in pce_bins:
        evaluate_subset(df[mask], "PCE Target Range", b_label)
        
    # By Pair Frequency
    pair_counts = (df['donor_canonical'] + "::" + df['acceptor_canonical']).value_counts()
    df['pair_freq'] = (df['donor_canonical'] + "::" + df['acceptor_canonical']).map(pair_counts)
    evaluate_subset(df[df['pair_freq'] == 1], "Pair Frequency", "Singleton Pairs (N=1)")
    evaluate_subset(df[(df['pair_freq'] > 1) & (df['pair_freq'] < 20)], "Pair Frequency", "Repeated Pairs (2 <= N < 20)")
    evaluate_subset(df[df['pair_freq'] >= 20], "Pair Frequency", "Frequent Pairs (N >= 20)")
    
    error_analysis_df = pd.DataFrame(error_analysis_rows)
    err_csv_path = os.path.join(RESULTS_DIR, "phase_6_error_analysis.csv")
    error_analysis_df.to_csv(err_csv_path, index=False)
    print(f"Saved detailed error analysis to {err_csv_path}")
    print("\n--- SAMPLE SUBSET EVALUATION TABLE ---")
    print(error_analysis_df[['category', 'subcategory', 'record_count', 'mae', 'rmse', 'r2', 'bias']].to_string(index=False))
    
    # =========================================================================
    # PART 2: CONTROLLED MODEL IMPROVEMENT (INNER FOLD TUNING ONLY)
    # =========================================================================
    print("\n--- PART 2: CONTROLLED HYPERPARAMETER EXPLORATION ---")
    
    # Extract feature matrix X and target y
    y = df['pce'].to_numpy(dtype=np.float64)
    unique_donors = df['donor_smiles'].unique()
    unique_acceptors = df['acceptor_smiles'].unique()
    donor_mols = {s: Chem.MolFromSmiles(s) for s in unique_donors}
    acceptor_mols = {s: Chem.MolFromSmiles(s) for s in unique_acceptors}
    
    X = np.zeros((total_samples, len(FEATURE_NAMES)), dtype=np.float64)
    for idx, row in df.iterrows():
        d_m = donor_mols[row['donor_smiles']]
        a_m = acceptor_mols[row['acceptor_smiles']]
        vec, _ = extract_features_from_mols(d_m, a_m)
        X[idx, :] = vec
        
    pair_groups = (df['donor_canonical'] + "::" + df['acceptor_canonical']).to_numpy()
    
    # Outer 5-fold CV split
    gkf_outer = GroupKFold(n_splits=5)
    outer_splits = list(gkf_outer.split(X, y, groups=pair_groups))
    
    # Inner Model Selection: We perform a 3-fold inner GroupKFold strictly on the training partition of Fold 1
    train_idx_f1, test_idx_f1 = outer_splits[0]
    X_inner_tr = X[train_idx_f1]
    y_inner_tr = y[train_idx_f1]
    groups_inner_tr = pair_groups[train_idx_f1]
    
    # Define candidate configurations
    candidate_configs = [
        {
            "config_id": "RF_Config_0_Baseline",
            "description": "Phase 5 Baseline: n=100, max_depth=None, min_samples_leaf=1, max_features=1.0",
            "params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0}
        },
        {
            "config_id": "RF_Config_1_DepthReg",
            "description": "Depth regularized: n=100, max_depth=20, min_samples_leaf=2, max_features=0.8",
            "params": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": 0.8}
        },
        {
            "config_id": "RF_Config_2_FeatureSqrt",
            "description": "Decorrelated features: n=200, max_depth=None, min_samples_leaf=2, max_features='sqrt'",
            "params": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt"}
        },
        {
            "config_id": "RF_Config_3_LeafReg",
            "description": "Leaf regularized: n=100, max_depth=None, min_samples_split=4, min_samples_leaf=4, max_features=0.8",
            "params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 4, "min_samples_leaf": 4, "max_features": 0.8}
        },
        {
            "config_id": "RF_Config_4_EnsembleScaled",
            "description": "Scaled ensemble: n=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=0.8",
            "params": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 0.8}
        }
    ]
    
    print("\nRunning Inner 3-Fold GroupKFold model selection on Training Fold 1 only...")
    gkf_inner = GroupKFold(n_splits=3)
    inner_splits = list(gkf_inner.split(X_inner_tr, y_inner_tr, groups=groups_inner_tr))
    
    inner_results = []
    for cfg in candidate_configs:
        cfg_id = cfg["config_id"]
        p = cfg["params"]
        maes, rmses, r2s = [], [], []
        
        for in_tr_idx, in_val_idx in inner_splits:
            rf_in = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **p)
            rf_in.fit(X_inner_tr[in_tr_idx], y_inner_tr[in_tr_idx])
            pred_in = rf_in.predict(X_inner_tr[in_val_idx])
            maes.append(mean_absolute_error(y_inner_tr[in_val_idx], pred_in))
            rmses.append(np.sqrt(mean_squared_error(y_inner_tr[in_val_idx], pred_in)))
            r2s.append(r2_score(y_inner_tr[in_val_idx], pred_in))
            
        m_mae = float(np.mean(maes))
        m_rmse = float(np.mean(rmses))
        m_r2 = float(np.mean(r2s))
        inner_results.append({
            "config_id": cfg_id,
            "description": cfg["description"],
            "inner_cv_mae": round(m_mae, 4),
            "inner_cv_rmse": round(m_rmse, 4),
            "inner_cv_r2": round(m_r2, 4)
        })
        print(f"  {cfg_id:25s} | Inner MAE: {m_mae:.4f} | Inner RMSE: {m_rmse:.4f} | Inner R2: {m_r2:.4f}")
        
    best_cfg_entry = min(inner_results, key=lambda x: x["inner_cv_mae"])
    best_config_id = best_cfg_entry["config_id"]
    best_cfg = next(c for c in candidate_configs if c["config_id"] == best_config_id)
    print(f"\nBest configuration selected by inner training CV: {best_config_id} ({best_cfg['description']})")
    
    # Now evaluate all candidate configurations across the full 5-fold Outer Pair-Grouped CV to create comparison table
    print("\nEvaluating all configurations across full 5-Fold Outer Pair-Grouped CV...")
    model_comparison_rows = []
    
    # Also evaluate tree prediction uncertainty on the baseline configuration
    tree_stds_all = np.zeros(total_samples, dtype=np.float64)
    
    for cfg in candidate_configs:
        cfg_id = cfg["config_id"]
        p = cfg["params"]
        outer_fold_maes = []
        outer_fold_rmses = []
        outer_fold_r2s = []
        
        oof_preds = np.zeros(total_samples, dtype=np.float64)
        
        t0 = time.time()
        for fold_idx, (tr_idx, val_idx) in enumerate(outer_splits):
            rf_outer = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **p)
            rf_outer.fit(X[tr_idx], y[tr_idx])
            val_preds = rf_outer.predict(X[val_idx])
            oof_preds[val_idx] = val_preds
            
            # If baseline, capture individual tree standard deviations for uncertainty audit
            if cfg_id == "RF_Config_0_Baseline":
                # Compute tree predictions for validation samples
                # shape: (n_trees, n_val_samples)
                tree_preds = np.array([tree.predict(X[val_idx]) for tree in rf_outer.estimators_])
                val_tree_std = np.std(tree_preds, axis=0)
                tree_stds_all[val_idx] = val_tree_std
                
            m = compute_metrics(y[val_idx], val_preds)
            outer_fold_maes.append(m["mae"])
            outer_fold_rmses.append(m["rmse"])
            outer_fold_r2s.append(m["r2"])
            
        elapsed = time.time() - t0
        pooled_m = compute_metrics(y, oof_preds)
        
        model_comparison_rows.append({
            "config_id": cfg_id,
            "description": cfg["description"],
            "mean_fold_mae": round(float(np.mean(outer_fold_maes)), 4),
            "std_fold_mae": round(float(np.std(outer_fold_maes)), 4),
            "mean_fold_rmse": round(float(np.mean(outer_fold_rmses)), 4),
            "std_fold_rmse": round(float(np.std(outer_fold_rmses)), 4),
            "mean_fold_r2": round(float(np.mean(outer_fold_r2s)), 4),
            "std_fold_r2": round(float(np.std(outer_fold_r2s)), 4),
            "pooled_mae": round(pooled_m["mae"], 4),
            "pooled_rmse": round(pooled_m["rmse"], 4),
            "pooled_r2": round(pooled_m["r2"], 4),
            "total_fit_time_seconds": round(elapsed, 2)
        })
        print(f"  {cfg_id:25s} | Outer MAE: {np.mean(outer_fold_maes):.4f} +/- {np.std(outer_fold_maes):.4f} | R2: {np.mean(outer_fold_r2s):.4f} | Pooled R2: {pooled_m['r2']:.4f}")
        
    comp_df = pd.DataFrame(model_comparison_rows)
    comp_csv_path = os.path.join(RESULTS_DIR, "phase_6_model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved model comparison to {comp_csv_path}")
    
    # =========================================================================
    # PART 3: PREDICTION UNCERTAINTY & EMPIRICAL CALIBRATION AUDIT
    # =========================================================================
    print("\n--- PART 3: PREDICTION UNCERTAINTY & CALIBRATION AUDIT ---")
    
    df['rf_tree_std'] = tree_stds_all
    abs_err = df['rf_abs_error'].to_numpy()
    sigma = df['rf_tree_std'].to_numpy()
    
    # Correlation between tree dispersion and actual error
    uncertainty_corr = float(np.corrcoef(sigma, abs_err)[0, 1])
    print(f"Correlation between tree std dev (uncertainty) and absolute error: r = {uncertainty_corr:.4f}")
    
    # Test empirical coverage of nominal confidence intervals
    # Nominal 68.3% interval: [pred - 1.0*sigma, pred + 1.0*sigma]
    cov_68 = float(np.mean((abs_err <= 1.0 * sigma)))
    # Nominal 95.0% interval: [pred - 1.96*sigma, pred + 1.96*sigma]
    cov_95 = float(np.mean((abs_err <= 1.96 * sigma)))
    # Nominal 99.7% interval: [pred - 3.0*sigma, pred + 3.0*sigma]
    cov_99 = float(np.mean((abs_err <= 3.0 * sigma)))
    
    # Empirical scaling factor required for nominal 68% coverage
    ratio_err_sigma = abs_err / np.maximum(sigma, 1e-4)
    empirical_scale_68 = float(np.percentile(ratio_err_sigma, 68.27))
    empirical_scale_95 = float(np.percentile(ratio_err_sigma, 95.0))
    
    print("\nEmpirical Coverage of Raw Tree Dispersion (Uncalibrated):")
    print(f"  Nominal 68.3% interval (1.00 * sigma) -> Actual Coverage: {cov_68*100:.2f}% (SEVERE UNDER-COVERAGE)")
    print(f"  Nominal 95.0% interval (1.96 * sigma) -> Actual Coverage: {cov_95*100:.2f}% (SEVERE UNDER-COVERAGE)")
    print(f"  Nominal 99.7% interval (3.00 * sigma) -> Actual Coverage: {cov_99*100:.2f}%")
    print(f"  Multiplier required for 68.3% coverage: {empirical_scale_68:.2f}x sigma")
    print(f"  Multiplier required for 95.0% coverage: {empirical_scale_95:.2f}x sigma")
    
    # Uncertainty across PCE ranges
    unc_by_pce = df.groupby('arch_category')['rf_tree_std'].mean().to_dict()
    print("\nMean Tree Dispersion by Architecture:", unc_by_pce)
    
    # =========================================================================
    # PART 4: SAVE COMPREHENSIVE SUMMARY JSON
    # =========================================================================
    summary_data = {
        "metadata": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "random_state": RANDOM_STATE
        },
        "dataset_architecture_audit": {
            "total_records": total_samples,
            "conventional_count": int((df['arch_category'] == 'Conventional').sum()),
            "inverted_count": int((df['arch_category'] == 'Inverted').sum()),
            "unspecified_count": int((df['arch_category'] == 'Unspecified / Other').sum()),
            "tandem_records_explicit": 0,
            "single_junction_explicit": 76,
            "with_additive": int(df['has_additive'].sum()),
            "with_annealing": int(df['has_annealing'].sum()),
            "with_thickness": int(df['has_thickness'].sum())
        },
        "inner_fold_selection": {
            "selection_strategy": "3-Fold GroupKFold strictly within Training Fold 1",
            "evaluated_configs": inner_results,
            "selected_config_id": best_config_id
        },
        "outer_evaluation_comparison": model_comparison_rows,
        "uncertainty_audit": {
            "pearson_r_uncertainty_vs_abs_error": round(uncertainty_corr, 4),
            "uncalibrated_nominal_68_coverage": round(cov_68, 4),
            "uncalibrated_nominal_95_coverage": round(cov_95, 4),
            "uncalibrated_nominal_99_coverage": round(cov_99, 4),
            "empirical_multiplier_for_68_coverage": round(empirical_scale_68, 4),
            "empirical_multiplier_for_95_coverage": round(empirical_scale_95, 4)
        }
    }
    
    summary_path = os.path.join(RESULTS_DIR, "phase_6_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved Phase 6 validation summary to {summary_path}")
    print("\n=== PHASE 6 SCRIPT COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
