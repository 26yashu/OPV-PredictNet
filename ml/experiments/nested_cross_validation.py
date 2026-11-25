"""OPV-PredictNet - Phase 6.1 True Nested Grouped Cross-Validation and Correction.

Implements:
1. True 5x3 Nested Grouped Cross-Validation:
   - Outer: 5-Fold GroupKFold on material pair (donor_canonical + '::' + acceptor_canonical)
   - Inner: 3-Fold GroupKFold on material pairs within the outer training partition only
   - For each outer fold independently, selects best config by inner CV MAE, refits on full outer train, predicts untouched outer val.
2. Fair Paired Model Comparison:
   - Evaluates Phase 5 Baseline RF and Nested-CV Selected RF on identical outer validation records.
   - Computes per-record absolute error differences: delta = |err_baseline| - |err_nested|.
   - Computes group-aware (material pair clustered) bootstrap 95% confidence intervals.
3. Corrected Metadata Classification:
   - Separates explicitly reported present, explicitly reported absent, and unknown/unrecorded.
   - Objective architecture classification without causal speculation.

Strict Constraints:
- 22 pure molecular descriptors from descriptors.py
- OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records)
- Zero synthetic data, zero imputation, no record removal
- Random seed: 42
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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath('.'))
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES

RANDOM_STATE = 42
BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
RESULTS_DIR = 'ml/results'

# 5 Candidate Hyperparameter Configurations (identical to Phase 6)
CANDIDATE_CONFIGS = [
    {
        "config_id": "RF_Config_0_Baseline",
        "description": "Baseline: n=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0",
        "params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0}
    },
    {
        "config_id": "RF_Config_1_DepthReg",
        "description": "Depth regularized: n=100, max_depth=20, min_samples_split=2, min_samples_leaf=2, max_features=0.8",
        "params": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": 0.8}
    },
    {
        "config_id": "RF_Config_2_FeatureSqrt",
        "description": "Decorrelated features: n=200, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features='sqrt'",
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_true - y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def main():
    print("=== PHASE 6.1: NESTED GROUPED CROSS-VALIDATION & CORRECTION ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load benchmark dataset
    print(f"Loading benchmark dataset from {BENCHMARK_PATH}...")
    df = pd.read_csv(BENCHMARK_PATH)
    total_samples = len(df)
    print(f"Total benchmark records: {total_samples}")
    
    y = df['pce'].to_numpy(dtype=np.float64)
    
    # 2. Extract feature matrix X using production 22-feature pipeline
    print("Extracting 22 molecular descriptors...")
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
        
    assert X.shape == (total_samples, 22), "Feature matrix shape mismatch!"
    assert not np.isnan(X).any(), "NaN in features!"
    assert not np.isinf(X).any(), "Inf in features!"
    print("Feature matrix verified: 22 features, zero NaNs, zero Infs.")
    
    # 3. Primary Grouping: Material Pair
    pair_series = df['donor_canonical'] + "::" + df['acceptor_canonical']
    pair_groups = pair_series.to_numpy()
    unique_pairs = np.unique(pair_groups)
    print(f"Total material pairs (groups): {len(unique_pairs)}")
    
    # 4. Initialize Outer 5-Fold GroupKFold
    gkf_outer = GroupKFold(n_splits=5)
    outer_splits = list(gkf_outer.split(X, y, groups=pair_groups))
    
    fold_metrics_records = []
    baseline_oof_preds = np.zeros(total_samples, dtype=np.float64)
    nested_oof_preds = np.zeros(total_samples, dtype=np.float64)
    dummy_oof_preds = np.zeros(total_samples, dtype=np.float64)
    
    outer_selected_configs = []
    
    # =========================================================================
    # TRUE NESTED GROUPED CROSS-VALIDATION LOOP
    # =========================================================================
    print("\nExecuting True 5x3 Nested Grouped Cross-Validation...")
    
    for outer_fold_idx, (out_train_idx, out_val_idx) in enumerate(outer_splits):
        outer_fold_num = outer_fold_idx + 1
        print(f"\n=======================================================")
        print(f"OUTER FOLD {outer_fold_num} / 5")
        print(f"=======================================================")
        
        X_out_tr, y_out_tr = X[out_train_idx], y[out_train_idx]
        X_out_va, y_out_va = X[out_val_idx], y[out_val_idx]
        groups_out_tr = pair_groups[out_train_idx]
        groups_out_va = pair_groups[out_val_idx]
        
        # Verify strict outer isolation
        leak_out = set(groups_out_tr).intersection(set(groups_out_va))
        assert len(leak_out) == 0, f"Outer group leakage in fold {outer_fold_num}!"
        print(f"Outer Train: {len(out_train_idx)} records ({len(set(groups_out_tr))} pairs)")
        print(f"Outer Val:   {len(out_val_idx)} records ({len(set(groups_out_va))} pairs) [STRICTLY HELD OUT]")
        
        # 4a. Evaluate DummyRegressor baseline on outer validation
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_out_tr, y_out_tr)
        dum_pred = dummy.predict(X_out_va)
        dummy_oof_preds[out_val_idx] = dum_pred
        dum_m = compute_metrics(y_out_va, dum_pred)
        fold_metrics_records.append({
            "outer_fold": outer_fold_num,
            "model": "DummyRegressor",
            "selected_config_id": "Dummy_Mean",
            "selected_params": "strategy=mean",
            "mae": round(dum_m["mae"], 4),
            "rmse": round(dum_m["rmse"], 4),
            "r2": round(dum_m["r2"], 4),
            "bias": round(dum_m["bias"], 4),
            "n_train_records": len(out_train_idx),
            "n_val_records": len(out_val_idx),
            "n_train_groups": len(set(groups_out_tr)),
            "n_val_groups": len(set(groups_out_va)),
            "fit_time_seconds": 0.01
        })
        
        # 4b. Evaluate Fixed Phase 5 Baseline RF on outer validation
        t0_base = time.time()
        rf_base = RandomForestRegressor(
            n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            max_features=1.0, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_base.fit(X_out_tr, y_out_tr)
        t_base = time.time() - t0_base
        base_pred = rf_base.predict(X_out_va)
        baseline_oof_preds[out_val_idx] = base_pred
        base_m = compute_metrics(y_out_va, base_pred)
        fold_metrics_records.append({
            "outer_fold": outer_fold_num,
            "model": "Phase_5_Baseline_RF",
            "selected_config_id": "RF_Config_0_Baseline",
            "selected_params": str(CANDIDATE_CONFIGS[0]["params"]),
            "mae": round(base_m["mae"], 4),
            "rmse": round(base_m["rmse"], 4),
            "r2": round(base_m["r2"], 4),
            "bias": round(base_m["bias"], 4),
            "n_train_records": len(out_train_idx),
            "n_val_records": len(out_val_idx),
            "n_train_groups": len(set(groups_out_tr)),
            "n_val_groups": len(set(groups_out_va)),
            "fit_time_seconds": round(t_base, 2)
        })
        print(f"  Phase 5 Baseline RF -> MAE: {base_m['mae']:.4f}, RMSE: {base_m['rmse']:.4f}, R2: {base_m['r2']:.4f} (Fit: {t_base:.2f}s)")
        
        # 4c. Inner 3-Fold GroupKFold Hyperparameter Selection (USING ONLY X_out_tr)
        print(f"  Running Inner 3-Fold GroupKFold on Outer Training Partition {outer_fold_num}...")
        gkf_inner = GroupKFold(n_splits=3)
        inner_splits = list(gkf_inner.split(X_out_tr, y_out_tr, groups=groups_out_tr))
        
        inner_config_scores = []
        for cfg in CANDIDATE_CONFIGS:
            cfg_id = cfg["config_id"]
            p = cfg["params"]
            in_maes = []
            
            for in_tr_idx, in_va_idx in inner_splits:
                # Assert inner isolation
                in_tr_grp = set(groups_out_tr[in_tr_idx])
                in_va_grp = set(groups_out_tr[in_va_idx])
                assert len(in_tr_grp.intersection(in_va_grp)) == 0, "Inner group leakage!"
                
                rf_in = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **p)
                rf_in.fit(X_out_tr[in_tr_idx], y_out_tr[in_tr_idx])
                p_in = rf_in.predict(X_out_tr[in_va_idx])
                in_maes.append(mean_absolute_error(y_out_tr[in_va_idx], p_in))
                
            mean_in_mae = float(np.mean(in_maes))
            inner_config_scores.append((cfg_id, mean_in_mae, p, cfg["description"]))
            print(f"    Inner Config {cfg_id:25s} -> Mean Inner MAE: {mean_in_mae:.4f}")
            
        # Select best config by inner MAE
        best_cfg_tuple = min(inner_config_scores, key=lambda x: x[1])
        sel_cfg_id, sel_in_mae, sel_params, sel_desc = best_cfg_tuple
        print(f"  -> Outer Fold {outer_fold_num} Selected: {sel_cfg_id} (Inner MAE: {sel_in_mae:.4f})")
        
        outer_selected_configs.append({
            "outer_fold": outer_fold_num,
            "selected_config_id": sel_cfg_id,
            "inner_cv_mae": round(sel_in_mae, 4),
            "params": sel_params,
            "description": sel_desc,
            "all_inner_scores": {c[0]: round(c[1], 4) for c in inner_config_scores}
        })
        
        # 4d. Refit Selected Configuration on COMPLETE Outer Training Partition
        t0_nest = time.time()
        rf_nest = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **sel_params)
        rf_nest.fit(X_out_tr, y_out_tr)
        t_nest = time.time() - t0_nest
        
        nest_pred = rf_nest.predict(X_out_va)
        nested_oof_preds[out_val_idx] = nest_pred
        nest_m = compute_metrics(y_out_va, nest_pred)
        
        fold_metrics_records.append({
            "outer_fold": outer_fold_num,
            "model": "Nested_Selected_RF",
            "selected_config_id": sel_cfg_id,
            "selected_params": str(sel_params),
            "mae": round(nest_m["mae"], 4),
            "rmse": round(nest_m["rmse"], 4),
            "r2": round(nest_m["r2"], 4),
            "bias": round(nest_m["bias"], 4),
            "n_train_records": len(out_train_idx),
            "n_val_records": len(out_val_idx),
            "n_train_groups": len(set(groups_out_tr)),
            "n_val_groups": len(set(groups_out_va)),
            "fit_time_seconds": round(t_nest, 2)
        })
        print(f"  Nested Selected RF  -> MAE: {nest_m['mae']:.4f}, RMSE: {nest_m['rmse']:.4f}, R2: {nest_m['r2']:.4f} (Fit: {t_nest:.2f}s)")
        
    # Save nested fold metrics CSV
    nested_metrics_df = pd.DataFrame(fold_metrics_records)
    metrics_csv_path = os.path.join(RESULTS_DIR, "phase_6_1_nested_fold_metrics.csv")
    nested_metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved nested fold metrics to {metrics_csv_path}")
    
    # =========================================================================
    # FAIR PAIRED MODEL COMPARISON & CLUSTERED BOOTSTRAP
    # =========================================================================
    print("\n--- FAIR PAIRED ERROR COMPARISON ---")
    
    baseline_abs_err = np.abs(y - baseline_oof_preds)
    nested_abs_err = np.abs(y - nested_oof_preds)
    delta_abs_err = baseline_abs_err - nested_abs_err  # > 0 means nested model improved
    
    paired_df = pd.DataFrame({
        "id": df['id'],
        "doi": df['doi'],
        "donor": df['donor_canonical'],
        "acceptor": df['acceptor_canonical'],
        "pair": pair_groups,
        "pce_true": y,
        "baseline_pred": baseline_oof_preds,
        "nested_pred": nested_oof_preds,
        "baseline_abs_err": baseline_abs_err,
        "nested_abs_err": nested_abs_err,
        "delta_abs_err": delta_abs_err
    })
    paired_csv_path = os.path.join(RESULTS_DIR, "phase_6_1_paired_comparison.csv")
    paired_df.to_csv(paired_csv_path, index=False)
    print(f"Saved per-record paired comparison to {paired_csv_path}")
    
    mean_delta = float(np.mean(delta_abs_err))
    median_delta = float(np.median(delta_abs_err))
    n_better = int((delta_abs_err > 0).sum())
    n_worse = int((delta_abs_err < 0).sum())
    n_equal = int((delta_abs_err == 0).sum())
    
    print(f"Paired Absolute Error Differences (Baseline - Nested):")
    print(f"  Mean Delta MAE:   {mean_delta:+.4f}% (positive indicates nested model reduced error)")
    print(f"  Median Delta MAE: {median_delta:+.4f}%")
    print(f"  Records where Nested is Better: {n_better} ({n_better/total_samples*100:.2f}%)")
    print(f"  Records where Nested is Worse:  {n_worse} ({n_worse/total_samples*100:.2f}%)")
    print(f"  Records Exactly Tied:           {n_equal} ({n_equal/total_samples*100:.2f}%)")
    
    # Material-Pair Clustered Bootstrap (1,000 resamples of material pairs)
    print("\nComputing Material-Pair Clustered Bootstrap (B=1000 resamples)...")
    np.random.seed(RANDOM_STATE)
    n_bootstraps = 1000
    boot_means = []
    
    # Precompute indices per pair for fast group-aware bootstrap
    pair_to_indices = {}
    for idx, p in enumerate(pair_groups):
        pair_to_indices.setdefault(p, []).append(idx)
        
    for b in range(n_bootstraps):
        sampled_pairs = np.random.choice(unique_pairs, size=len(unique_pairs), replace=True)
        sampled_indices = []
        for p in sampled_pairs:
            sampled_indices.extend(pair_to_indices[p])
        b_mean_delta = np.mean(delta_abs_err[sampled_indices])
        boot_means.append(b_mean_delta)
        
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    p_value_two_sided = float(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    
    print(f"Group-Aware Clustered Bootstrap Results (95% CI):")
    print(f"  Bootstrap Mean Delta: {np.mean(boot_means):+.4f}%")
    print(f"  Bootstrap 95% CI:     [{ci_low:+.4f}%, {ci_high:+.4f}%]")
    print(f"  Zero Crossing Check:  CI {'CROSSES ZERO (NOT STATISTICALLY SIGNIFICANT)' if ci_low <= 0 <= ci_high else 'DOES NOT CROSS ZERO'}")
    print(f"  Empirical p-value:    p = {p_value_two_sided:.4f}")
    
    # =========================================================================
    # PART 4: CORRECTED METADATA & OBJECTIVE ARCHITECTURE AUDIT
    # =========================================================================
    print("\n--- PART 4: CORRECTED METADATA CLASSIFICATION ---")
    
    # 4a. Additive Classification (Present vs Explicitly Absent vs Unknown)
    explicit_no_add_patterns = ['w/o', '0', 'no', 'without', 'none', '-']
    def classify_additive(row):
        val = row['additive']
        ratio = str(row['additive_ratio']).strip()
        if pd.isna(val):
            if ratio in ['0', '0%', '0.0']:
                return 'Explicitly Reported Absent'
            return 'Status Unknown (Unrecorded)'
        val_str = str(val).strip().lower()
        if val_str in explicit_no_add_patterns or ratio in ['0', '0%', '0.0']:
            return 'Explicitly Reported Absent'
        return 'Explicitly Reported Present'
        
    df['additive_status'] = df.apply(classify_additive, axis=1)
    print("Additive Status Classification:")
    print(df['additive_status'].value_counts())
    
    # 4b. Annealing Classification
    def classify_annealing(row):
        val = row['annealing_temp']
        if pd.isna(val):
            return 'Status Unknown (Unrecorded)'
        try:
            temp = float(val)
            if temp <= 25.0:
                return 'Explicitly Room Temp / Absent'
            return 'Explicitly Thermal Annealed'
        except:
            return 'Status Unknown (Unrecorded)'
            
    df['annealing_status'] = df.apply(classify_annealing, axis=1)
    print("\nAnnealing Status Classification:")
    print(df['annealing_status'].value_counts())
    
    # 4c. Device Architecture Classification (Objective rules without causal inference)
    def classify_architecture_strict(row):
        d_type = str(row['device_type']).lower()
        d_struct = str(row['device_structure']).lower()
        
        has_conv_type = 'conventional' in d_type
        has_inv_type = 'inverted' in d_type
        has_conv_struct = 'conventional' in d_struct
        has_inv_struct = 'inverted' in d_struct
        
        # Check cross-conflict or dual-reported
        is_conv = has_conv_type or has_conv_struct
        is_inv = has_inv_type or has_inv_struct
        
        if is_conv and is_inv:
            return 'Ambiguous / Dual Structure Reported'
        elif is_conv:
            return 'Conventional Architecture'
        elif is_inv:
            return 'Inverted Architecture'
        else:
            return 'Architecture Unspecified'
            
    df['architecture_strict'] = df.apply(classify_architecture_strict, axis=1)
    print("\nObjective Architecture Classification:")
    print(df['architecture_strict'].value_counts())
    
    # Sliced Performance Metrics for Corrected Categories
    df['nested_pce_pred'] = nested_oof_preds
    df['baseline_pce_pred'] = baseline_oof_preds
    
    subgroup_analysis = []
    
    def evaluate_subgroup(series_name, group_col):
        for val, grp in df.groupby(group_col):
            m_base = compute_metrics(grp['pce'].to_numpy(), grp['baseline_pce_pred'].to_numpy())
            m_nest = compute_metrics(grp['pce'].to_numpy(), grp['nested_pce_pred'].to_numpy())
            subgroup_analysis.append({
                "dimension": series_name,
                "category": val,
                "count": len(grp),
                "percentage": round(len(grp)/total_samples*100, 2),
                "actual_pce_mean": round(float(grp['pce'].mean()), 2),
                "actual_pce_std": round(float(grp['pce'].std()), 2),
                "baseline_mae": round(m_base["mae"], 4),
                "baseline_r2": round(m_base["r2"], 4),
                "nested_mae": round(m_nest["mae"], 4),
                "nested_r2": round(m_nest["r2"], 4)
            })
            
    evaluate_subgroup("Additive Classification", "additive_status")
    evaluate_subgroup("Annealing Classification", "annealing_status")
    evaluate_subgroup("Device Architecture", "architecture_strict")
    
    subgroup_df = pd.DataFrame(subgroup_analysis)
    print("\n--- SUBGROUP PERFORMANCE SUMMARY ---")
    print(subgroup_df[['dimension', 'category', 'count', 'actual_pce_mean', 'actual_pce_std', 'baseline_mae', 'baseline_r2', 'nested_mae', 'nested_r2']].to_string(index=False))
    
    # =========================================================================
    # SUMMARY JSON
    # =========================================================================
    # Compute aggregate fold metrics
    base_grp = nested_metrics_df[nested_metrics_df['model'] == 'Phase_5_Baseline_RF']
    nest_grp = nested_metrics_df[nested_metrics_df['model'] == 'Nested_Selected_RF']
    dum_grp = nested_metrics_df[nested_metrics_df['model'] == 'DummyRegressor']
    
    pooled_base = compute_metrics(y, baseline_oof_preds)
    pooled_nest = compute_metrics(y, nested_oof_preds)
    pooled_dum = compute_metrics(y, dummy_oof_preds)
    
    summary_dict = {
        "metadata": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "random_state": RANDOM_STATE
        },
        "nested_cross_validation_protocol": {
            "outer_splits": "5-Fold GroupKFold on material pair",
            "inner_splits": "3-Fold GroupKFold on material pairs within outer training partition only",
            "search_space": [c["config_id"] for c in CANDIDATE_CONFIGS],
            "selection_criterion": "inner validation MAE",
            "outer_selected_configurations": outer_selected_configs
        },
        "model_comparison": {
            "Phase_5_Baseline_RF": {
                "fold_mae_mean": round(float(base_grp["mae"].mean()), 4),
                "fold_mae_std": round(float(base_grp["mae"].std()), 4),
                "fold_rmse_mean": round(float(base_grp["rmse"].mean()), 4),
                "fold_rmse_std": round(float(base_grp["rmse"].std()), 4),
                "fold_r2_mean": round(float(base_grp["r2"].mean()), 4),
                "fold_r2_std": round(float(base_grp["r2"].std()), 4),
                "pooled_mae": round(pooled_base["mae"], 4),
                "pooled_rmse": round(pooled_base["rmse"], 4),
                "pooled_r2": round(pooled_base["r2"], 4)
            },
            "Nested_Selected_RF": {
                "fold_mae_mean": round(float(nest_grp["mae"].mean()), 4),
                "fold_mae_std": round(float(nest_grp["mae"].std()), 4),
                "fold_rmse_mean": round(float(nest_grp["rmse"].mean()), 4),
                "fold_rmse_std": round(float(nest_grp["rmse"].std()), 4),
                "fold_r2_mean": round(float(nest_grp["r2"].mean()), 4),
                "fold_r2_std": round(float(nest_grp["r2"].std()), 4),
                "pooled_mae": round(pooled_nest["mae"], 4),
                "pooled_rmse": round(pooled_nest["rmse"], 4),
                "pooled_r2": round(pooled_nest["r2"], 4)
            },
            "DummyRegressor": {
                "fold_mae_mean": round(float(dum_grp["mae"].mean()), 4),
                "fold_mae_std": round(float(dum_grp["mae"].std()), 4),
                "fold_rmse_mean": round(float(dum_grp["rmse"].mean()), 4),
                "fold_rmse_std": round(float(dum_grp["rmse"].std()), 4),
                "fold_r2_mean": round(float(dum_grp["r2"].mean()), 4),
                "fold_r2_std": round(float(dum_grp["r2"].std()), 4),
                "pooled_mae": round(pooled_dum["mae"], 4),
                "pooled_rmse": round(pooled_dum["rmse"], 4),
                "pooled_r2": round(pooled_dum["r2"], 4)
            }
        },
        "paired_difference_statistics": {
            "mean_delta_mae": round(mean_delta, 4),
            "median_delta_mae": round(median_delta, 4),
            "n_records_nested_better": n_better,
            "n_records_nested_worse": n_worse,
            "n_records_equal": n_equal,
            "clustered_bootstrap_mean_delta": round(float(np.mean(boot_means)), 4),
            "clustered_bootstrap_95_ci_low": round(ci_low, 4),
            "clustered_bootstrap_95_ci_high": round(ci_high, 4),
            "ci_crosses_zero": bool(ci_low <= 0 <= ci_high),
            "empirical_p_value": round(p_value_two_sided, 4)
        },
        "corrected_metadata_audit": {
            "additive_status_counts": df['additive_status'].value_counts().to_dict(),
            "annealing_status_counts": df['annealing_status'].value_counts().to_dict(),
            "architecture_strict_counts": df['architecture_strict'].value_counts().to_dict(),
            "tandem_identifiable": False,
            "subgroup_metrics": subgroup_analysis
        }
    }
    
    summary_json_path = os.path.join(RESULTS_DIR, "phase_6_1_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\nSaved complete summary JSON to {summary_json_path}")
    print("\n=== PHASE 6.1 NESTED CV SCRIPT COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
