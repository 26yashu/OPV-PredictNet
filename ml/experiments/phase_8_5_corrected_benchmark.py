"""
OPV-PredictNet — Phase 8.5: Corrected Structure-Generalization Benchmark Reproduction

Controlled reproduction of Phase 8.2 using the authoritative Phase 8.4
UnifiedFeaturePipeline (ml/src/features/processing.py), strictly freezing:
- Dataset and SHA-256 checksum
- Frozen Phase 8.2 scaffold grouping definitions and train/cal/test partitions
- Random seeds (7701, 7702, 7703, 8888)
- Model architectures & hyperparameters (n_jobs=1)
- Disjoint split-conformal calibration (80%, 90%, 95%)

Additionally executes:
- Multi-margin separate one-way cluster-bootstrap sensitivity analyses
  (donor_scaffold, acceptor_scaffold, scaffold_pair, pair_id, doi_norm)
- Scaffold-level and scaffold-pair subgroup breakdowns (n >= 20)
- Macro-averaged per-scaffold error diagnostics
- Controlled causal attribution decomposition explaining M1, M2A, and M3 deltas
  relative to frozen Phase 8.2 historical results.

Does NOT modify any historical ml/results/phase_8_2_* files.
"""

import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure workspace root is importable
sys.path.insert(0, os.path.abspath("."))

from ml.experiments.phase_8_2_scaffold_benchmark import (
    BENCHMARK_PATH,
    CACHE_PATH,
    EXPECTED_SHA256,
    MODEL_SEED,
    SEED_A,
    SEED_B,
    SEED_C2,
    FeaturePipeline as Phase82FeaturePipeline,
    compute_conformal_quantile,
    compute_sha256,
    partition_strategy_a,
    partition_strategy_b,
    partition_strategy_c1,
    partition_strategy_c2,
)
from ml.src.features.processing import (
    ETL_WF_MAP,
    HTL_WF_MAP,
    SOLV_BP_MAP,
    UnifiedFeaturePipeline,
    _lookup_physical_constant,
    is_thermal_annealed_regime,
    parse_f_donor,
    validate_and_parse_smiles,
)

RESULTS_DIR_8_5 = "ml/results/phase_8_5"
RESULTS_DIR_8_2 = "ml/results"
ALPHAS = [0.20, 0.10, 0.05]
BOOTSTRAP_REPS = 1000
MIN_SUBGROUP_N = 20


def hash_indices(idx: pd.Index) -> str:
    payload = ",".join(str(int(i)) for i in idx).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_target(y: np.ndarray) -> Dict[str, Any]:
    return {
        "n": int(len(y)),
        "mean": round(float(np.mean(y)), 4),
        "std": round(float(np.std(y)), 4),
        "var": round(float(np.var(y)), 4),
        "min": round(float(np.min(y)), 4),
        "q25": round(float(np.percentile(y, 25)), 4),
        "median": round(float(np.median(y)), 4),
        "q75": round(float(np.percentile(y, 75)), 4),
        "max": round(float(np.max(y)), 4),
        "n_high_pce_ge_15": int(np.sum(y >= 15.0)),
        "pct_high_pce_ge_15": round(float(np.mean(y >= 15.0) * 100.0), 4),
    }


def cluster_bootstrap_diff(
    y_true: np.ndarray,
    y_pred_m: np.ndarray,
    y_pred_ref: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = BOOTSTRAP_REPS,
    seed: int = MODEL_SEED,
) -> Tuple[float, float, float, int]:
    unique_clusters = np.unique(clusters)
    n_c = len(unique_clusters)
    rng = np.random.RandomState(seed)

    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    diff = np.abs(y_true - y_pred_m) - np.abs(y_true - y_pred_ref)
    mean_diff = float(np.mean(diff))

    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_c = rng.choice(unique_clusters, size=n_c, replace=True)
        sample_idx = np.concatenate([cluster_indices[c] for c in sampled_c])
        boot_means[b] = np.mean(diff[sample_idx])

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_diff, ci_low, ci_high, n_c


def check_split_leakage(df_tr: pd.DataFrame, df_cal: pd.DataFrame, df_te: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    cols = [
        "donor_scaffold",
        "acceptor_scaffold",
        "scaffold_pair",
        "donor_smiles",
        "acceptor_smiles",
        "chem_pair",
        "pair_id",
        "doi_norm",
    ]
    report: Dict[str, Dict[str, int]] = {}
    for col in cols:
        s_tr = set(df_tr[col].dropna())
        s_cl = set(df_cal[col].dropna())
        s_ts = set(df_te[col].dropna())
        report[col] = {
            "n_train": len(s_tr),
            "n_cal": len(s_cl),
            "n_test": len(s_ts),
            "tr_cal_overlap": len(s_tr & s_cl),
            "tr_test_overlap": len(s_tr & s_ts),
            "cal_test_overlap": len(s_cl & s_ts),
        }
    return report


def macro_scaffold_mae(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    ae = np.abs(y_true - y_pred)
    s = pd.Series(ae).groupby(groups).mean()
    return float(s.mean())


def make_fresh_models() -> Dict[str, Any]:
    return {
        "M0": RandomForestRegressor(n_estimators=100, random_state=MODEL_SEED, n_jobs=1),
        "M1": RandomForestRegressor(n_estimators=100, random_state=MODEL_SEED, n_jobs=1),
        "M2A": RandomForestRegressor(n_estimators=100, random_state=MODEL_SEED, n_jobs=1),
        "M3": HistGradientBoostingRegressor(max_iter=100, random_state=MODEL_SEED),
    }


def extract_all_unified_features(
    df_train: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    mol_dict: Dict[Tuple[str, str], np.ndarray],
) -> Tuple[UnifiedFeaturePipeline, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    pipe = UnifiedFeaturePipeline()
    pipe.mol_dict = dict(mol_dict)  # Deterministic read-only copy of 22-descriptor SMILES cache
    pipe.fit_training(df_train)

    tr_m0 = pipe.extract_m0(df_train)
    cl_m0 = pipe.extract_m0(df_cal)
    te_m0 = pipe.extract_m0(df_test)

    train_data = {
        "M0": tr_m0,
        "M1": pipe.extract_m1(df_train, tr_m0),
        "M2A": pipe.extract_m2a(df_train, tr_m0),
        "M3": pipe.extract_m3(df_train, tr_m0),
    }
    cal_data = {
        "M0": cl_m0,
        "M1": pipe.extract_m1(df_cal, cl_m0),
        "M2A": pipe.extract_m2a(df_cal, cl_m0),
        "M3": pipe.extract_m3(df_cal, cl_m0),
    }
    test_data = {
        "M0": te_m0,
        "M1": pipe.extract_m1(df_test, te_m0),
        "M2A": pipe.extract_m2a(df_test, te_m0),
        "M3": pipe.extract_m3(df_test, te_m0),
    }
    return pipe, train_data, cal_data, test_data


def run_strategy(
    strategy_id: str,
    strategy_name: str,
    df_train: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    mol_dict: Dict[Tuple[str, str], np.ndarray],
    primary_cluster_col: str,
) -> Dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"Executing {strategy_id}: {strategy_name}")
    print(f"  Train: {len(df_train)} | Cal: {len(df_cal)} | Test: {len(df_test)}")

    pipe, train_data, cal_data, test_data = extract_all_unified_features(
        df_train, df_cal, df_test, mol_dict
    )

    y_train = df_train["pce"].values.astype(np.float64)
    y_cal = df_cal["pce"].values.astype(np.float64)
    y_test = df_test["pce"].values.astype(np.float64)

    models = make_fresh_models()
    fit_times: Dict[str, float] = {}
    test_preds: Dict[str, np.ndarray] = {}
    cal_quantiles: Dict[str, Dict[float, float]] = {}

    model_metrics: List[Dict[str, Any]] = []
    interval_metrics: List[Dict[str, Any]] = []

    d_groups = df_test["donor_scaffold"].values
    a_groups = df_test["acceptor_scaffold"].values
    p_groups = df_test["scaffold_pair"].values

    for m_id, model in models.items():
        t0 = time.time()
        model.fit(train_data[m_id], y_train)
        fit_times[m_id] = time.time() - t0

        y_pred_cal = model.predict(cal_data[m_id])
        scores_cal = np.abs(y_cal - y_pred_cal)
        cal_quantiles[m_id] = {alpha: compute_conformal_quantile(scores_cal, alpha) for alpha in ALPHAS}

        y_pred_te = model.predict(test_data[m_id])
        test_preds[m_id] = y_pred_te

        mae = float(mean_absolute_error(y_test, y_pred_te))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
        r2 = float(r2_score(y_test, y_pred_te))
        bias = float(np.mean(y_pred_te - y_test))

        model_metrics.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "model_id": m_id,
                "feature_count": int(train_data[m_id].shape[1]),
                "fit_time_seconds": round(fit_times[m_id], 3),
                "n_train": int(len(df_train)),
                "n_cal": int(len(df_cal)),
                "n_test": int(len(y_test)),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "bias": round(bias, 4),
                "macro_donor_scaffold_mae": round(macro_scaffold_mae(y_test, y_pred_te, d_groups), 4),
                "macro_acceptor_scaffold_mae": round(macro_scaffold_mae(y_test, y_pred_te, a_groups), 4),
                "macro_scaffold_pair_mae": round(macro_scaffold_mae(y_test, y_pred_te, p_groups), 4),
                "pce_mean": round(float(np.mean(y_test)), 4),
                "pce_std": round(float(np.std(y_test)), 4),
                "pce_median": round(float(np.median(y_test)), 4),
                "pce_iqr": round(float(np.percentile(y_test, 75) - np.percentile(y_test, 25)), 4),
            }
        )

        for alpha in ALPHAS:
            q = cal_quantiles[m_id][alpha]
            conf_str = f"{int(round((1.0 - alpha) * 100))}%"
            cov = float(np.mean((y_test >= y_pred_te - q) & (y_test <= y_pred_te + q)))
            width = 2.0 * q
            interval_metrics.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "model_id": m_id,
                    "confidence_level": conf_str,
                    "alpha": alpha,
                    "conformal_cutoff_q": round(q, 4),
                    "empirical_coverage": round(cov, 4),
                    "mean_width": round(width, 4),
                    "median_width": round(width, 4),
                }
            )

    # Multi-margin separate one-way cluster-bootstrap sensitivity analyses (vs M0)
    paired_comparisons: List[Dict[str, Any]] = []
    cluster_axes = ["donor_scaffold", "acceptor_scaffold", "scaffold_pair", "pair_id", "doi_norm"]
    y_pred_m0 = test_preds["M0"]

    for m_id in ["M1", "M2A", "M3"]:
        y_pred_m = test_preds[m_id]
        axis_results: Dict[str, Any] = {}
        all_exclude_zero = True
        for axis in cluster_axes:
            mean_diff, ci_l, ci_h, n_c = cluster_bootstrap_diff(
                y_test, y_pred_m, y_pred_m0, df_test[axis].values, n_boot=BOOTSTRAP_REPS, seed=MODEL_SEED
            )
            axis_results[axis] = (mean_diff, ci_l, ci_h, n_c)
            if not (ci_h < 0.0 or ci_l > 0.0):
                all_exclude_zero = False

        p_mean, p_cil, p_cih, p_g = axis_results[primary_cluster_col]
        paired_comparisons.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "comparison": f"{m_id} vs M0",
                "model_id": m_id,
                "baseline_id": "M0",
                "primary_cluster_axis": primary_cluster_col,
                "primary_cluster_count": p_g,
                "mean_ae_diff_pce": round(p_mean, 4),
                "primary_boot_ci_low": round(p_cil, 4),
                "primary_boot_ci_high": round(p_cih, 4),
                "donor_scaffold_g": axis_results["donor_scaffold"][3],
                "donor_scaffold_ci_low": round(axis_results["donor_scaffold"][1], 4),
                "donor_scaffold_ci_high": round(axis_results["donor_scaffold"][2], 4),
                "acceptor_scaffold_g": axis_results["acceptor_scaffold"][3],
                "acceptor_scaffold_ci_low": round(axis_results["acceptor_scaffold"][1], 4),
                "acceptor_scaffold_ci_high": round(axis_results["acceptor_scaffold"][2], 4),
                "scaffold_pair_g": axis_results["scaffold_pair"][3],
                "scaffold_pair_ci_low": round(axis_results["scaffold_pair"][1], 4),
                "scaffold_pair_ci_high": round(axis_results["scaffold_pair"][2], 4),
                "pair_id_g": axis_results["pair_id"][3],
                "pair_id_ci_low": round(axis_results["pair_id"][1], 4),
                "pair_id_ci_high": round(axis_results["pair_id"][2], 4),
                "doi_norm_g": axis_results["doi_norm"][3],
                "doi_norm_ci_low": round(axis_results["doi_norm"][1], 4),
                "doi_norm_ci_high": round(axis_results["doi_norm"][2], 4),
                "all_5_oneway_margins_exclude_zero": bool(all_exclude_zero),
                "joint_crossed_dependence_tested": False,
            }
        )

    # PCE-range subgroup slices
    subgroup_metrics: List[Dict[str, Any]] = []
    pce_bins = [
        ("< 5.0%", y_test < 5.0),
        ("5.0% - 10.0%", (y_test >= 5.0) & (y_test < 10.0)),
        ("10.0% - 15.0%", (y_test >= 10.0) & (y_test < 15.0)),
        (">= 15.0%", y_test >= 15.0),
    ]
    for pce_label, mask in pce_bins:
        n_slice = int(np.sum(mask))
        if n_slice == 0:
            continue
        for m_id in ["M0", "M1", "M2A", "M3"]:
            y_sl = y_test[mask]
            pred_sl = test_preds[m_id][mask]
            q90 = cal_quantiles[m_id][0.10]
            subgroup_metrics.append(
                {
                    "strategy_id": strategy_id,
                    "slice_type": "PCE Range",
                    "slice_name": pce_label,
                    "sample_size_n": n_slice,
                    "model_id": m_id,
                    "mae_pce": round(float(np.mean(np.abs(y_sl - pred_sl))), 4),
                    "rmse_pce": round(float(np.sqrt(np.mean((y_sl - pred_sl) ** 2))), 4),
                    "bias_pce": round(float(np.mean(pred_sl - y_sl)), 4),
                    "cov_90": round(float(np.mean((y_sl >= pred_sl - q90) & (y_sl <= pred_sl + q90))), 4),
                }
            )

    # Scaffold-level and scaffold-pair-level breakdowns (where n >= MIN_SUBGROUP_N)
    scaffold_pair_subgroups: List[Dict[str, Any]] = []
    for axis_name in ["donor_scaffold", "acceptor_scaffold", "scaffold_pair"]:
        vc = df_test[axis_name].value_counts()
        eligible = vc[vc >= MIN_SUBGROUP_N]
        for grp_val, grp_n in eligible.items():
            mask = (df_test[axis_name] == grp_val).values
            y_sl = y_test[mask]
            for m_id in ["M0", "M1", "M2A", "M3"]:
                pred_sl = test_preds[m_id][mask]
                q90 = cal_quantiles[m_id][0.10]
                scaffold_pair_subgroups.append(
                    {
                        "strategy_id": strategy_id,
                        "group_type": axis_name,
                        "group_id": str(grp_val),
                        "n_records": int(grp_n),
                        "test_share_pct": round(float(grp_n / len(df_test) * 100.0), 2),
                        "pce_mean": round(float(np.mean(y_sl)), 4),
                        "pce_std": round(float(np.std(y_sl)), 4),
                        "model_id": m_id,
                        "mae_pce": round(float(np.mean(np.abs(y_sl - pred_sl))), 4),
                        "bias_pce": round(float(np.mean(pred_sl - y_sl)), 4),
                        "cov_90": round(float(np.mean((y_sl >= pred_sl - q90) & (y_sl <= pred_sl + q90))), 4),
                    }
                )

    # Prediction frame for row-level auditability
    pred_df = pd.DataFrame(
        {
            "strategy_id": strategy_id,
            "row_index": df_test.index.to_numpy(dtype=np.int64),
            "pair_id": df_test["pair_id"].values,
            "donor_canonical": df_test["donor_canonical"].values,
            "acceptor_canonical": df_test["acceptor_canonical"].values,
            "donor_scaffold": df_test["donor_scaffold"].values,
            "acceptor_scaffold": df_test["acceptor_scaffold"].values,
            "scaffold_pair": df_test["scaffold_pair"].values,
            "doi_norm": df_test["doi_norm"].values,
            "pce": y_test,
            "pred_M0": test_preds["M0"],
            "pred_M1": test_preds["M1"],
            "pred_M2A": test_preds["M2A"],
            "pred_M3": test_preds["M3"],
            "q90_M0": cal_quantiles["M0"][0.10],
            "q90_M1": cal_quantiles["M1"][0.10],
            "q90_M2A": cal_quantiles["M2A"][0.10],
            "q90_M3": cal_quantiles["M3"][0.10],
        }
    )

    return {
        "pipeline_medians": {
            "da_median": round(pipe.da_median, 6),
            "thk_median": round(pipe.thk_median, 6),
            "tmp_median": round(pipe.tmp_median, 6),
            "f_donor_median": round(pipe.f_donor_median, 6),
            "thk_log_median": round(pipe.thk_log_median, 6),
            "ann_norm_median": round(pipe.ann_norm_median, 6),
            "bp_median": round(pipe.bp_median, 6),
            "etl_median": round(pipe.etl_median, 6),
            "htl_median": round(pipe.htl_median, 6),
        },
        "model_metrics": model_metrics,
        "interval_metrics": interval_metrics,
        "paired_comparisons": paired_comparisons,
        "subgroup_metrics": subgroup_metrics,
        "scaffold_pair_subgroups": scaffold_pair_subgroups,
        "predictions_df": pred_df,
    }


def run_m3_attribution_decomposition(
    strategy_id: str,
    df_train: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    mol_dict: Dict[Tuple[str, str], np.ndarray],
) -> List[Dict[str, Any]]:
    """Quantify M3 MAE/RMSE/R2 deltas attributable to each distinct Phase 8.4 fix."""
    p82 = Phase82FeaturePipeline()
    p82.mol_dict = dict(mol_dict)
    p82.fit_training(df_train)

    p85 = UnifiedFeaturePipeline()
    p85.mol_dict = dict(mol_dict)
    p85.fit_training(df_train)

    tr_m0 = p85.extract_m0(df_train)
    cl_m0 = p85.extract_m0(df_cal)
    te_m0 = p85.extract_m0(df_test)

    # Variant 0: Historical Phase 8.2 M3
    X_tr_v0 = p82.extract_m3(df_train, tr_m0)
    X_cl_v0 = p82.extract_m3(df_cal, cl_m0)
    X_te_v0 = p82.extract_m3(df_test, te_m0)

    # Variant 1: Phase 8.2 M3 + ONLY regime_thermal_annealed fix
    X_tr_v1 = X_tr_v0.copy()
    X_cl_v1 = X_cl_v0.copy()
    X_te_v1 = X_te_v0.copy()
    X_tr_v1["regime_thermal_annealed"] = df_train["annealing_temp"].apply(is_thermal_annealed_regime).values.astype(np.int64)
    X_cl_v1["regime_thermal_annealed"] = df_cal["annealing_temp"].apply(is_thermal_annealed_regime).values.astype(np.int64)
    X_te_v1["regime_thermal_annealed"] = df_test["annealing_temp"].apply(is_thermal_annealed_regime).values.astype(np.int64)

    # Variant 2: Phase 8.2 M3 + ONLY SOLV_BP/ETL_WF/HTL_WF lookup fixes (reg_ann kept at 0)
    X_tr_v2 = X_tr_v0.copy()
    X_cl_v2 = X_cl_v0.copy()
    X_te_v2 = X_te_v0.copy()
    for frame, df_sub in ((X_tr_v2, df_train), (X_cl_v2, df_cal), (X_te_v2, df_test)):
        frame["solvent_boiling_point_c"] = df_sub["solvent_canonical"].apply(lambda x: _lookup_physical_constant(x, SOLV_BP_MAP)).values.astype(np.float64)
        frame["etl_workfunction_ev"] = df_sub["etl_canonical"].apply(lambda x: _lookup_physical_constant(x, ETL_WF_MAP)).values.astype(np.float64)
        frame["htl_workfunction_ev"] = df_sub["htl_canonical"].apply(lambda x: _lookup_physical_constant(x, HTL_WF_MAP)).values.astype(np.float64)

    # Variant 3: Variant 2 + missing_da_ratio fix (reg_ann still 0)
    X_tr_v3 = X_tr_v2.copy()
    X_cl_v3 = X_cl_v2.copy()
    X_te_v3 = X_te_v2.copy()
    for frame, df_sub in ((X_tr_v3, df_train), (X_cl_v3, df_cal), (X_te_v3, df_test)):
        frame["missing_da_ratio"] = df_sub["d_a_ratio"].apply(lambda v: 0 if pd.notna(parse_f_donor(v)) else 1).values.astype(np.int64)

    # Variant 4: Full Phase 8.5 Unified M3 (both annealing + lookups + missing_da_ratio)
    X_tr_v4 = p85.extract_m3(df_train, tr_m0)
    X_cl_v4 = p85.extract_m3(df_cal, cl_m0)
    X_te_v4 = p85.extract_m3(df_test, te_m0)

    y_tr = df_train["pce"].values.astype(np.float64)
    y_cl = df_cal["pce"].values.astype(np.float64)
    y_te = df_test["pce"].values.astype(np.float64)

    variants = [
        ("v0_phase_8_2_baseline", X_tr_v0, X_cl_v0, X_te_v0),
        ("v1_thermal_annealing_fix_only", X_tr_v1, X_cl_v1, X_te_v1),
        ("v2_lookup_tables_fix_only", X_tr_v2, X_cl_v2, X_te_v2),
        ("v3_lookups_plus_missing_da_fix", X_tr_v3, X_cl_v3, X_te_v3),
        ("v4_phase_8_5_unified_m3", X_tr_v4, X_cl_v4, X_te_v4),
    ]

    rows: List[Dict[str, Any]] = []
    v0_mae = None
    for tag, xtr, xcl, xte in variants:
        model = HistGradientBoostingRegressor(max_iter=100, random_state=MODEL_SEED)
        model.fit(xtr, y_tr)
        pred_cl = model.predict(xcl)
        q90 = compute_conformal_quantile(np.abs(y_cl - pred_cl), 0.10)
        pred_te = model.predict(xte)
        mae = float(mean_absolute_error(y_te, pred_te))
        rmse = float(np.sqrt(mean_squared_error(y_te, pred_te)))
        r2 = float(r2_score(y_te, pred_te))
        bias = float(np.mean(pred_te - y_te))
        cov90 = float(np.mean((y_te >= pred_te - q90) & (y_te <= pred_te + q90)))
        if v0_mae is None:
            v0_mae = mae
        rows.append(
            {
                "strategy_id": strategy_id,
                "variant": tag,
                "mae": round(mae, 4),
                "delta_mae_vs_v0": round(mae - v0_mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "bias": round(bias, 4),
                "cov_90": round(cov90, 4),
                "width_90": round(2.0 * q90, 4),
            }
        )
    return rows


def main() -> None:
    print("=" * 80)
    print("OPV-PredictNet — Phase 8.5: Corrected Structure-Generalization Reproduction")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    os.makedirs(RESULTS_DIR_8_5, exist_ok=True)

    # 1. Dataset & Checksum Verification
    actual_sha = compute_sha256(BENCHMARK_PATH)
    assert actual_sha == EXPECTED_SHA256, f"Dataset checksum mismatch: {actual_sha} != {EXPECTED_SHA256}"
    df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    print(f"Dataset verified: {len(df)} records | SHA-256: {actual_sha}")

    # 2. Frozen Scaffold Definitions & Canonical Discrepancy Audit
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        scaff_cache = json.load(f)

    donor_scaff_diffs: List[Dict[str, str]] = []
    acceptor_scaff_diffs: List[Dict[str, str]] = []
    for s in df["donor_smiles"].unique():
        _, _, sc, _ = validate_and_parse_smiles(s, role="donor")
        if scaff_cache["donor_scaffolds"][s] != sc:
            donor_scaff_diffs.append({"raw_smiles": s, "frozen_cache": scaff_cache["donor_scaffolds"][s], "unified_canonical": sc})
    for s in df["acceptor_smiles"].unique():
        _, _, sc, _ = validate_and_parse_smiles(s, role="acceptor")
        if scaff_cache["acceptor_scaffolds"][s] != sc:
            acceptor_scaff_diffs.append({"raw_smiles": s, "frozen_cache": scaff_cache["acceptor_scaffolds"][s], "unified_canonical": sc})

    # Apply exact frozen Phase 8.2 scaffold grouping so test populations remain strictly frozen
    df["donor_scaffold"] = df["donor_smiles"].map(scaff_cache["donor_scaffolds"])
    df["acceptor_scaffold"] = df["acceptor_smiles"].map(scaff_cache["acceptor_scaffolds"])
    df["scaffold_pair"] = df["donor_scaffold"] + "::" + df["acceptor_scaffold"]
    df["chem_pair"] = df["donor_smiles"] + "::" + df["acceptor_smiles"]
    df["pair_id"] = df["donor_canonical"] + "::" + df["acceptor_canonical"]

    # Pre-extract deterministic 22 RDKit molecular descriptors via UnifiedFeaturePipeline ONLY
    desc_pipe = UnifiedFeaturePipeline()
    t0 = time.time()
    desc_pipe.fit_descriptors(df)
    print(f"UnifiedFeaturePipeline descriptors pre-extracted for {len(desc_pipe.mol_dict)} SMILES pairs in {time.time() - t0:.2f}s")

    # 3. Partition Strategies (Exact Frozen Seeds)
    df_tr_a, df_cl_a, df_te_a = partition_strategy_a(df, seed=SEED_A)
    df_tr_b, df_cl_b, df_te_b = partition_strategy_b(df, seed=SEED_B)
    df_tr_c1, df_cl_c1, df_te_c1 = partition_strategy_c1(df, seed_d=SEED_A, seed_a=SEED_B)
    df_tr_c2, df_cl_c2, df_te_c2 = partition_strategy_c2(df, seed=SEED_C2)

    # Assert exact frozen partition counts from phase_8_2_config.json
    with open(os.path.join(RESULTS_DIR_8_2, "phase_8_2_config.json"), "r", encoding="utf-8") as f:
        p82_cfg = json.load(f)

    assert (len(df_tr_a), len(df_cl_a), len(df_te_a)) == (15750, 5236, 1481)
    assert (len(df_tr_b), len(df_cl_b), len(df_te_b)) == (17658, 1012, 3797)
    assert (len(df_tr_c1), len(df_cl_c1), len(df_te_c1)) == (13324, 260, 284)
    assert (len(df_tr_c2), len(df_cl_c2), len(df_te_c2)) == (16537, 3863, 2067)

    strategies = [
        ("Strategy_A", "Donor Scaffold Holdout", df_tr_a, df_cl_a, df_te_a, "donor_scaffold"),
        ("Strategy_B", "Acceptor Scaffold Holdout", df_tr_b, df_cl_b, df_te_b, "acceptor_scaffold"),
        ("Strategy_C1", "Strict Dual-Scaffold Holdout", df_tr_c1, df_cl_c1, df_te_c1, "scaffold_pair"),
        ("Strategy_C2", "Scaffold-Pair Holdout", df_tr_c2, df_cl_c2, df_te_c2, "scaffold_pair"),
    ]

    all_model_metrics: List[Dict[str, Any]] = []
    all_interval_metrics: List[Dict[str, Any]] = []
    all_paired_comparisons: List[Dict[str, Any]] = []
    all_subgroup_metrics: List[Dict[str, Any]] = []
    all_scaffold_pair_subgroups: List[Dict[str, Any]] = []
    all_pred_dfs: List[pd.DataFrame] = []
    all_attribution_rows: List[Dict[str, Any]] = []
    split_metadata: Dict[str, Any] = {
        "dataset_sha256": actual_sha,
        "total_records": int(len(df)),
        "frozen_scaffold_cache_policy": "Exact Phase 8.2 scaffold_cache.json preserved to freeze identical test populations.",
        "scaffold_canonical_discrepancies": {
            "donor_count": len(donor_scaff_diffs),
            "acceptor_count": len(acceptor_scaff_diffs),
            "donor_details": donor_scaff_diffs,
            "acceptor_details": acceptor_scaff_diffs,
            "impact_note": (
                "Two acyclic molecules had raw vs canonical SMILES inside the [NO_RING]: pseudo-scaffold key. "
                "Re-computing [NO_RING]: keys prior to sorting would perturb list order before rng.shuffle in Strategies B, C1, and C2; "
                "therefore Phase 8.5 freezes scaffold_cache.json to preserve 100% identical train/cal/test partitions."
            ),
        },
        "strategies": {},
    }

    for s_id, s_name, d_tr, d_cl, d_te, p_col in strategies:
        leakage = check_split_leakage(d_tr, d_cl, d_te)
        if s_id == "Strategy_A":
            assert leakage["donor_scaffold"]["tr_cal_overlap"] == 0
            assert leakage["donor_scaffold"]["tr_test_overlap"] == 0
            assert leakage["donor_scaffold"]["cal_test_overlap"] == 0
        elif s_id == "Strategy_B":
            assert leakage["acceptor_scaffold"]["tr_cal_overlap"] == 0
            assert leakage["acceptor_scaffold"]["tr_test_overlap"] == 0
            assert leakage["acceptor_scaffold"]["cal_test_overlap"] == 0
        elif s_id == "Strategy_C1":
            assert leakage["donor_scaffold"]["tr_test_overlap"] == 0
            assert leakage["acceptor_scaffold"]["tr_test_overlap"] == 0
            assert leakage["scaffold_pair"]["tr_test_overlap"] == 0
        elif s_id == "Strategy_C2":
            assert leakage["scaffold_pair"]["tr_cal_overlap"] == 0
            assert leakage["scaffold_pair"]["tr_test_overlap"] == 0
            assert leakage["scaffold_pair"]["cal_test_overlap"] == 0
        assert leakage["chem_pair"]["tr_test_overlap"] == 0

        res = run_strategy(s_id, s_name, d_tr, d_cl, d_te, desc_pipe.mol_dict, p_col)
        all_model_metrics.extend(res["model_metrics"])
        all_interval_metrics.extend(res["interval_metrics"])
        all_paired_comparisons.extend(res["paired_comparisons"])
        all_subgroup_metrics.extend(res["subgroup_metrics"])
        all_scaffold_pair_subgroups.extend(res["scaffold_pair_subgroups"])
        all_pred_dfs.append(res["predictions_df"])

        split_metadata["strategies"][s_id] = {
            "primary_cluster_col": p_col,
            "train_index_sha256": hash_indices(d_tr.index),
            "cal_index_sha256": hash_indices(d_cl.index),
            "test_index_sha256": hash_indices(d_te.index),
            "target_distribution": {
                "train": summarize_target(d_tr["pce"].values),
                "cal": summarize_target(d_cl["pce"].values),
                "test": summarize_target(d_te["pce"].values),
            },
            "leakage_audit": leakage,
            "fitted_training_medians": res["pipeline_medians"],
        }

        attr_rows = run_m3_attribution_decomposition(s_id, d_tr, d_cl, d_te, desc_pipe.mol_dict)
        all_attribution_rows.extend(attr_rows)

    # 4. Save Phase 8.5 Artifacts (Separate Directory Only)
    df_model_metrics = pd.DataFrame(all_model_metrics)
    df_interval_metrics = pd.DataFrame(all_interval_metrics)
    df_paired = pd.DataFrame(all_paired_comparisons)
    df_subgroups = pd.DataFrame(all_subgroup_metrics)
    df_scaff_subgroups = pd.DataFrame(all_scaffold_pair_subgroups)
    df_preds = pd.concat(all_pred_dfs, ignore_index=True)
    df_attr = pd.DataFrame(all_attribution_rows)

    df_model_metrics.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_scaffold_metrics.csv"), index=False)
    df_interval_metrics.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_interval_metrics.csv"), index=False)
    df_paired.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_paired_comparisons.csv"), index=False)
    df_subgroups.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_subgroup_metrics.csv"), index=False)
    df_scaff_subgroups.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_scaffold_pair_subgroups.csv"), index=False)
    df_preds.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_test_predictions.csv"), index=False)
    df_attr.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_m3_attribution_decomposition.csv"), index=False)

    # Historical side-by-side delta CSV (vs read-only Phase 8.2 CSVs)
    p82_metrics = pd.read_csv(os.path.join(RESULTS_DIR_8_2, "phase_8_2_scaffold_metrics.csv"))
    p82_int = pd.read_csv(os.path.join(RESULTS_DIR_8_2, "phase_8_2_interval_metrics.csv"))
    p82_int_90 = p82_int[p82_int["confidence_level"] == "90%"][
        ["strategy_id", "model_id", "empirical_coverage", "median_width"]
    ].rename(columns={"empirical_coverage": "cov_90_p82", "median_width": "width_90_p82"})
    p85_int_90 = df_interval_metrics[df_interval_metrics["confidence_level"] == "90%"][
        ["strategy_id", "model_id", "empirical_coverage", "median_width"]
    ].rename(columns={"empirical_coverage": "cov_90_p85", "median_width": "width_90_p85"})

    comp = pd.merge(
        p82_metrics[["strategy_id", "model_id", "mae", "rmse", "r2", "bias"]],
        df_model_metrics[["strategy_id", "model_id", "mae", "rmse", "r2", "bias"]],
        on=["strategy_id", "model_id"],
        suffixes=("_p82", "_p85"),
    )
    comp = comp.merge(p82_int_90, on=["strategy_id", "model_id"]).merge(p85_int_90, on=["strategy_id", "model_id"])
    comp["delta_mae"] = (comp["mae_p85"] - comp["mae_p82"]).round(4)
    comp["delta_rmse"] = (comp["rmse_p85"] - comp["rmse_p82"]).round(4)
    comp["delta_r2"] = (comp["r2_p85"] - comp["r2_p82"]).round(4)
    comp["delta_bias"] = (comp["bias_p85"] - comp["bias_p82"]).round(4)
    comp["delta_cov_90"] = (comp["cov_90_p85"] - comp["cov_90_p82"]).round(4)
    comp["delta_width_90"] = (comp["width_90_p85"] - comp["width_90_p82"]).round(4)
    comp.to_csv(os.path.join(RESULTS_DIR_8_5, "phase_8_5_historical_comparison.csv"), index=False)

    # Verify M0 exact reproducibility against Phase 8.2
    m0_comp = comp[comp["model_id"] == "M0"]
    assert (m0_comp["delta_mae"].abs() == 0.0).all(), f"M0 MAE mismatch vs Phase 8.2:\n{m0_comp}"
    assert (m0_comp["delta_rmse"].abs() == 0.0).all(), f"M0 RMSE mismatch vs Phase 8.2:\n{m0_comp}"
    assert (m0_comp["delta_r2"].abs() == 0.0).all(), f"M0 R2 mismatch vs Phase 8.2:\n{m0_comp}"

    # Verify Phase 8.2 attribution baseline v0 reproduces Phase 8.2 M3
    v0_rows = df_attr[df_attr["variant"] == "v0_phase_8_2_baseline"].merge(
        p82_metrics[p82_metrics["model_id"] == "M3"][["strategy_id", "mae"]], on="strategy_id", suffixes=("_v0", "_p82")
    )
    assert np.allclose(v0_rows["mae_v0"], v0_rows["mae_p82"], atol=1e-4), f"M3 v0 mismatch:\n{v0_rows}"

    config_data = {
        "phase": "8.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_pipeline": "ml.src.features.processing.UnifiedFeaturePipeline",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "n_jobs": 1,
        },
        "dataset_sha256": actual_sha,
        "seeds": {
            "seed_strategy_a": SEED_A,
            "seed_strategy_b": SEED_B,
            "seed_strategy_c2": SEED_C2,
            "model_seed": MODEL_SEED,
        },
        "feature_counts": {"M0": 22, "M1": 50, "M2A": 37, "M3": 44},
        "models_config": {
            "M0": "RandomForestRegressor(n_estimators=100, random_state=8888, n_jobs=1)",
            "M1": "RandomForestRegressor(n_estimators=100, random_state=8888, n_jobs=1)",
            "M2A": "RandomForestRegressor(n_estimators=100, random_state=8888, n_jobs=1)",
            "M3": "HistGradientBoostingRegressor(max_iter=100, random_state=8888)",
        },
        "partition_counts": p82_cfg["partition_counts"],
    }
    with open(os.path.join(RESULTS_DIR_8_5, "phase_8_5_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    with open(os.path.join(RESULTS_DIR_8_5, "phase_8_5_split_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    print("\n--- Historical Comparison Summary (Phase 8.2 -> Phase 8.5) ---")
    print(comp[["strategy_id", "model_id", "mae_p82", "mae_p85", "delta_mae", "r2_p82", "r2_p85", "cov_90_p85", "width_90_p85"]].to_string(index=False))
    print("\n--- M3 Causal Attribution Decomposition ---")
    print(df_attr[["strategy_id", "variant", "mae", "delta_mae_vs_v0", "r2", "cov_90"]].to_string(index=False))
    print("\nPhase 8.5 execution completed successfully.")


if __name__ == "__main__":
    main()
