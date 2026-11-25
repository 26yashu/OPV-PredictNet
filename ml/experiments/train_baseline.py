"""OPV-PredictNet - Phase 5 Controlled Baseline Model Training Experiment.

Trains and evaluates:
1. DummyRegressor (mean prediction baseline)
2. RandomForestRegressor (initial nonlinear ML baseline)

Primary evaluation: 5-Fold GroupKFold with Donor-Acceptor Pair grouping.
Secondary evaluations:
- Publication (DOI) grouped cross-validation
- Combined Pair-Scaffold grouped cross-validation
- Individual Donor Scaffold holdout cross-validation
- Individual Acceptor Scaffold holdout cross-validation

Strict constraints:
- 22 pure molecular descriptors from descriptors.py
- Target: PCE (percentage points)
- Reproducible seed: 42
- Zero imputation, no synthetic data, no incomplete device metadata
"""

import sys
import os
import json
import time
import platform
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

import rdkit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import sklearn
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split_name: str,
    n_splits: int = 5
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Execute 5-fold GroupKFold evaluation for DummyRegressor and RandomForestRegressor."""
    gkf = GroupKFold(n_splits=n_splits)
    
    dummy_oof_preds = np.zeros_like(y)
    rf_oof_preds = np.zeros_like(y)
    fold_results = []
    
    unique_groups_total = len(np.unique(groups))
    print(f"\n=======================================================")
    print(f"Running 5-Fold GroupKFold: {split_name}")
    print(f"Total samples: {len(y)}, Total groups: {unique_groups_total}")
    print(f"=======================================================")
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        
        # Rigorous programmatic leakage assertion
        leakage = train_groups.intersection(val_groups)
        assert len(leakage) == 0, f"DATA LEAKAGE DETECTED in {split_name} fold {fold_idx + 1}: {len(leakage)} overlapping groups!"
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        n_train_records = len(train_idx)
        n_val_records = len(val_idx)
        n_train_groups = len(train_groups)
        n_val_groups = len(val_groups)
        
        # 1. DummyRegressor baseline
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_val)
        dummy_oof_preds[val_idx] = dummy_pred
        dummy_metrics = compute_metrics(y_val, dummy_pred)
        
        fold_results.append({
            "split_name": split_name,
            "fold": fold_idx + 1,
            "model": "DummyRegressor",
            "mae": dummy_metrics["mae"],
            "rmse": dummy_metrics["rmse"],
            "r2": dummy_metrics["r2"],
            "train_records": n_train_records,
            "val_records": n_val_records,
            "train_groups": n_train_groups,
            "val_groups": n_val_groups
        })
        
        # 2. RandomForestRegressor baseline
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        rf_pred = rf.predict(X_val)
        rf_oof_preds[val_idx] = rf_pred
        rf_metrics = compute_metrics(y_val, rf_pred)
        
        fold_results.append({
            "split_name": split_name,
            "fold": fold_idx + 1,
            "model": "RandomForestRegressor",
            "mae": rf_metrics["mae"],
            "rmse": rf_metrics["rmse"],
            "r2": rf_metrics["r2"],
            "train_records": n_train_records,
            "val_records": n_val_records,
            "train_groups": n_train_groups,
            "val_groups": n_val_groups,
            "fit_time_seconds": round(fit_time, 2)
        })
        
        print(f"Fold {fold_idx + 1} | Train: {n_train_records} ({n_train_groups} grp), Val: {n_val_records} ({n_val_groups} grp)")
        print(f"   Dummy -> MAE: {dummy_metrics['mae']:.4f}, RMSE: {dummy_metrics['rmse']:.4f}, R2: {dummy_metrics['r2']:.4f}")
        print(f"   RF    -> MAE: {rf_metrics['mae']:.4f}, RMSE: {rf_metrics['rmse']:.4f}, R2: {rf_metrics['r2']:.4f} (Fit: {fit_time:.2f}s)")
        
    return fold_results, dummy_oof_preds, rf_oof_preds


def main():
    print("=== PHASE 5: CONTROLLED BASELINE MODEL TRAINING ===")
    
    # 1. Environment metadata
    env_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "rdkit_version": rdkit.__version__,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "random_state": RANDOM_STATE
    }
    print("Environment:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")
        
    # 2. Load dataset
    print(f"\nLoading benchmark dataset from {BENCHMARK_PATH}...")
    df = pd.read_csv(BENCHMARK_PATH)
    total_samples = len(df)
    print(f"Total benchmark records: {total_samples}")
    
    # Target
    y = df['pce'].to_numpy(dtype=np.float64)
    assert not np.isnan(y).any(), "NaN found in target variable!"
    assert not np.isinf(y).any(), "Inf found in target variable!"
    print(f"Target variable 'pce': mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.6f}, max={y.max():.4f}")
    
    # 3. Extract 22 Features using existing production pipeline
    print("\nExtracting 22 features from RDKit Mol objects...")
    unique_donors = df['donor_smiles'].unique()
    unique_acceptors = df['acceptor_smiles'].unique()
    
    donor_mols = {s: Chem.MolFromSmiles(s) for s in unique_donors}
    acceptor_mols = {s: Chem.MolFromSmiles(s) for s in unique_acceptors}
    
    # Precompute descriptors for fast deterministic vector construction
    X = np.zeros((total_samples, len(FEATURE_NAMES)), dtype=np.float64)
    for idx, row in df.iterrows():
        d_mol = donor_mols[row['donor_smiles']]
        a_mol = acceptor_mols[row['acceptor_smiles']]
        vec, _ = extract_features_from_mols(d_mol, a_mol)
        X[idx, :] = vec
        
    assert X.shape == (total_samples, 22), f"Feature matrix shape mismatch: {X.shape}"
    assert not np.isnan(X).any(), "NaN found in feature matrix!"
    assert not np.isinf(X).any(), "Inf found in feature matrix!"
    print(f"Feature matrix X verified: {X.shape}, 22 features, zero NaNs, zero Infs.")
    
    # 4. Construct Grouping Columns
    # Primary: Donor-Acceptor Pair
    pair_groups = (df['donor_canonical'] + "::" + df['acceptor_canonical']).to_numpy()
    
    # Secondary 1: Publication DOI
    doi_groups = df['doi_norm'].astype(str).to_numpy()
    
    # Secondary 2 & 3: Murcko Scaffolds
    donor_scaffolds = {s: MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or "ACF" for s, m in donor_mols.items()}
    acceptor_scaffolds = {s: MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or "ACF" for s, m in acceptor_mols.items()}
    
    d_scaff_series = df['donor_smiles'].map(donor_scaffolds).to_numpy()
    a_scaff_series = df['acceptor_smiles'].map(acceptor_scaffolds).to_numpy()
    pair_scaff_groups = (df['donor_smiles'].map(donor_scaffolds) + ":::" + df['acceptor_smiles'].map(acceptor_scaffolds)).to_numpy()
    
    all_fold_metrics = []
    
    # =========================================================================
    # PRIMARY EVALUATION: 5-Fold GroupKFold on Donor-Acceptor Pair
    # =========================================================================
    primary_metrics, dummy_oof_pce, rf_oof_pce = run_grouped_cv(
        X=X, y=y, groups=pair_groups, split_name="Pair_Grouped_5Fold", n_splits=5
    )
    all_fold_metrics.extend(primary_metrics)
    
    # Save Out-of-Fold Predictions for Primary Split
    oof_df = pd.DataFrame({
        "id": df['id'],
        "doi": df['doi'],
        "donor": df['donor_canonical'],
        "acceptor": df['acceptor_canonical'],
        "pce_true": y,
        "dummy_pce_pred": dummy_oof_pce,
        "rf_pce_pred": rf_oof_pce,
        "rf_residual": y - rf_oof_pce,
        "rf_abs_error": np.abs(y - rf_oof_pce)
    })
    oof_path = os.path.join(RESULTS_DIR, "phase_5_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved primary out-of-fold predictions to {oof_path}")
    
    # =========================================================================
    # SECONDARY EVALUATION 1: Publication (DOI) Grouped
    # =========================================================================
    sec1_metrics, _, _ = run_grouped_cv(
        X=X, y=y, groups=doi_groups, split_name="DOI_Grouped_5Fold", n_splits=5
    )
    all_fold_metrics.extend(sec1_metrics)
    
    # =========================================================================
    # SECONDARY EVALUATION 2: Combined Pair Scaffold Grouped
    # =========================================================================
    sec2_metrics, _, _ = run_grouped_cv(
        X=X, y=y, groups=pair_scaff_groups, split_name="Combined_Scaffold_Pair_5Fold", n_splits=5
    )
    all_fold_metrics.extend(sec2_metrics)
    
    # =========================================================================
    # SECONDARY EVALUATION 3A: Donor Scaffold Grouped Holdout
    # =========================================================================
    sec3a_metrics, _, _ = run_grouped_cv(
        X=X, y=y, groups=d_scaff_series, split_name="Donor_Scaffold_Holdout_5Fold", n_splits=5
    )
    all_fold_metrics.extend(sec3a_metrics)
    
    # =========================================================================
    # SECONDARY EVALUATION 3B: Acceptor Scaffold Grouped Holdout
    # =========================================================================
    sec3b_metrics, _, _ = run_grouped_cv(
        X=X, y=y, groups=a_scaff_series, split_name="Acceptor_Scaffold_Holdout_5Fold", n_splits=5
    )
    all_fold_metrics.extend(sec3b_metrics)
    
    # =========================================================================
    # SAVE METRICS & GENERATE SUMMARY
    # =========================================================================
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_csv_path = os.path.join(RESULTS_DIR, "phase_5_fold_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved all fold metrics to {metrics_csv_path}")
    
    # Aggregate summary by split_name and model
    summary_dict: Dict[str, Any] = {
        "metadata": env_info,
        "dataset": {
            "path": BENCHMARK_PATH,
            "total_records": total_samples,
            "target": "pce",
            "features": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES)
        },
        "rf_hyperparameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        },
        "splits_summary": {}
    }
    
    print("\n=======================================================")
    print("PHASE 5 AGGREGATE EVALUATION SUMMARY (Mean +/- Std)")
    print("=======================================================")
    
    for (split_name, model_name), grp in metrics_df.groupby(["split_name", "model"], sort=False):
        mae_mean = float(grp["mae"].mean())
        mae_std = float(grp["mae"].std())
        rmse_mean = float(grp["rmse"].mean())
        rmse_std = float(grp["rmse"].std())
        r2_mean = float(grp["r2"].mean())
        r2_std = float(grp["r2"].std())
        
        split_entry = summary_dict["splits_summary"].setdefault(split_name, {})
        split_entry[model_name] = {
            "mae_mean": round(mae_mean, 4),
            "mae_std": round(mae_std, 4),
            "rmse_mean": round(rmse_mean, 4),
            "rmse_std": round(rmse_std, 4),
            "r2_mean": round(r2_mean, 4),
            "r2_std": round(r2_std, 4)
        }
        
        print(f"[{split_name:30s}] {model_name:22s} | MAE: {mae_mean:.4f} +/- {mae_std:.4f} | RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f} | R2: {r2_mean:.4f} +/- {r2_std:.4f}")
        
    summary_json_path = os.path.join(RESULTS_DIR, "phase_5_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\nSaved complete summary JSON to {summary_json_path}")
    print("\n=== PHASE 5 BASELINE TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
