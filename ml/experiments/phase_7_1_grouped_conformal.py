"""
OPV-PredictNet — Phase 7.1 Grouped Conformal Calibration Experiment.

Implements the audited and locked 4-stage sequential conformal architecture:
- Target: Marginal coverage for a randomly selected experimental record from an unseen normalized material-name pair.
- Grouping key: df['donor_canonical'] + "::" + df['acceptor_canonical'] (4,730 normalized material-name pairs).
- Partitioning: Pre-registered 60/20/20 grouped split (Train, Cal, Test).
- Stage 1: Fit Phase 5 Baseline Random Forest point predictor on D_train only.
- Stage 2: Compare and select scale estimator on D_train only via 5-fold grouped inner CV.
- Stage 3: Calculate calibration non-conformity scores and conformal quantiles on D_cal only using finite-sample rank k.
- Stage 4: Single final-test evaluation on D_test across pre-registered acceptance gates.

Strict constraints:
- Freeze all models and estimators before accessing D_cal.
- No calibration/test label leakage.
- Raw unclipped intervals evaluated first as primary; clipped [0, inf) evaluated as diagnostic.
- Finite-sample quantile rule: k = ceil((n_cal + 1) * (1 - alpha)).
- If k > n_cal, raise InsufficientCalibrationDataError.
"""

import sys
import os
import json
import time
import platform
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath('.'))
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES

BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
RESULTS_DIR = 'ml/results'
RANDOM_SEED_PARTITION = 42
RANDOM_SEED_CAL_SAMPLE = 42
RANDOM_SEED_TEST_SAMPLE = 43
MODEL_SEED = 42
STABILIZER_EPSILON = 0.10  # PCE percentage points


class InsufficientCalibrationDataError(Exception):
    """Raised when the finite-sample conformal quantile rank k exceeds calibration sample size n_cal."""
    pass


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the exact finite-sample conformal quantile cutoff.

    Formula:
        k = ceil((n_cal + 1) * (1 - alpha))
        cutoff = S_{(k)} (k-th smallest score)

    Raises:
        InsufficientCalibrationDataError: if k > n_cal.
    """
    n = len(scores)
    if n == 0:
        raise InsufficientCalibrationDataError("Calibration score array is empty (n_cal = 0).")
    
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        raise InsufficientCalibrationDataError(
            f"Calibration sample size n_cal={n} is insufficient for confidence level 1-alpha={1.0-alpha:.2f}; "
            f"finite-sample rank k={k} exceeds n_cal={n}."
        )
    
    sorted_scores = np.sort(scores)
    # k is 1-based index; convert to 0-based index (k - 1)
    cutoff = float(sorted_scores[k - 1])
    return cutoff


def split_grouped_dataset(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform deterministic, disjoint grouped 60/20/20 partitioning on material-name pairs.

    Returns:
        (df_train, df_cal, df_test) with zero group overlap.
    """
    unique_groups = np.unique(df[group_col].values)
    total_groups = len(unique_groups)

    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(unique_groups)

    n_train = int(np.round(total_groups * train_frac))
    n_cal = int(np.round(total_groups * cal_frac))
    n_test = total_groups - n_train - n_cal

    train_groups = set(shuffled_groups[:n_train])
    cal_groups = set(shuffled_groups[n_train:n_train + n_cal])
    test_groups = set(shuffled_groups[n_train + n_cal:])

    # Verify zero group leakage
    assert len(train_groups.intersection(cal_groups)) == 0, "Train-Cal group overlap detected!"
    assert len(train_groups.intersection(test_groups)) == 0, "Train-Test group overlap detected!"
    assert len(cal_groups.intersection(test_groups)) == 0, "Cal-Test group overlap detected!"

    df_train = df[df[group_col].isin(train_groups)].copy()
    df_cal = df[df[group_col].isin(cal_groups)].copy()
    df_test = df[df[group_col].isin(test_groups)].copy()

    df_train["partition"] = "train"
    df_cal["partition"] = "cal"
    df_test["partition"] = "test"

    return df_train, df_cal, df_test


def sample_one_record_per_group(
    df_partition: pd.DataFrame,
    group_col: str,
    seed: int
) -> pd.DataFrame:
    """Sample exactly one record uniformly at random from each group.

    Selection is independent of target values, predictions, residuals, or uncertainty scores.
    """
    rng = np.random.RandomState(seed)
    selected_indices = []

    for group_name, group_data in df_partition.groupby(group_col):
        chosen_idx = rng.choice(group_data.index.values)
        selected_indices.append(chosen_idx)

    sampled_df = df_partition.loc[selected_indices].copy()
    assert len(sampled_df) == df_partition[group_col].nunique(), "Mismatch in sampled group count!"
    assert sampled_df[group_col].nunique() == len(sampled_df), "Duplicate groups detected in sampled partition!"
    return sampled_df


class EnsembleDispersionScaleEstimator:
    """Computes uncertainty scale based on tree-to-tree prediction dispersion of a frozen Random Forest.

    Construction is strictly training-derived and does not touch calibration or test labels.
    """
    def __init__(self, fitted_rf: RandomForestRegressor, epsilon: float = STABILIZER_EPSILON):
        self.fitted_rf = fitted_rf
        self.epsilon = float(epsilon)

    def predict_scale(self, X: np.ndarray) -> np.ndarray:
        """Compute standard deviation across individual tree predictions + stabilizer epsilon."""
        # Individual tree predictions: shape (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.fitted_rf.estimators_], dtype=np.float64)
        tree_std = np.std(tree_preds, axis=0, ddof=1)
        scale = tree_std + self.epsilon
        assert np.all(scale > 0), "Scale values must be strictly positive!"
        return scale


class GroupConformalCalibrator:
    """Encapsulates Stage 1-4 conformal prediction for grouped OPV screening."""
    def __init__(self, method: str = "unnormalized", random_state: int = 42):
        assert method in ["unnormalized", "locally_normalized"], f"Unknown method: {method}"
        self.method = method
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.scale_estimator: Optional[EnsembleDispersionScaleEstimator] = None
        self.quantiles: Dict[float, float] = {}
        self.is_frozen: bool = False

    def fit_point_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Stage 1: Fit Phase 5 Baseline Random Forest on training partition records only."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        if self.method == "locally_normalized":
            self.scale_estimator = EnsembleDispersionScaleEstimator(self.model, epsilon=STABILIZER_EPSILON)

    def compute_scale(self, X: np.ndarray) -> np.ndarray:
        """Return unit scale for unnormalized or tree dispersion scale for locally normalized."""
        if self.method == "unnormalized":
            return np.ones(len(X), dtype=np.float64)
        else:
            assert self.scale_estimator is not None, "Scale estimator not initialized!"
            return self.scale_estimator.predict_scale(X)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, alphas: List[float] = [0.20, 0.10, 0.05]):
        """Stage 3: Calculate calibration scores and conformal quantiles using D_cal only."""
        assert self.model is not None, "Model must be fitted before calibration!"
        y_pred_cal = self.model.predict(X_cal)
        scales_cal = self.compute_scale(X_cal)

        # Non-conformity scores
        abs_residuals = np.abs(y_cal - y_pred_cal)
        scores = abs_residuals / scales_cal

        self.quantiles = {}
        for alpha in alphas:
            q = compute_conformal_quantile(scores, alpha)
            self.quantiles[alpha] = q

        self.is_frozen = True
        return scores

    def predict_intervals(
        self,
        X: np.ndarray,
        alpha: float,
        clip_lower_zero: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals for given alpha.

        Returns:
            (y_pred, scale, lower_bound, upper_bound)
        """
        assert self.is_frozen, "Calibrator must be frozen before predicting intervals!"
        assert alpha in self.quantiles, f"Quantile for alpha={alpha} not computed!"

        y_pred = self.model.predict(X)
        scale = self.compute_scale(X)
        q = self.quantiles[alpha]

        lower = y_pred - q * scale
        upper = y_pred + q * scale

        if clip_lower_zero:
            lower = np.maximum(0.0, lower)

        return y_pred, scale, lower, upper


def compute_coverage_and_width(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Compute empirical coverage rate, mean width, and median width."""
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(covered))
    widths = upper - lower
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    return coverage, mean_width, median_width, widths


def compute_winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Compute mean Winkler score (interval score) for predictive efficiency comparison.

    Formula:
        W = (upper - lower) + (2/alpha)*(lower - y) * I(y < lower) + (2/alpha)*(y - upper) * I(y > upper)
    """
    width = upper - lower
    penalty_lower = (2.0 / alpha) * np.maximum(0.0, lower - y_true)
    penalty_upper = (2.0 / alpha) * np.maximum(0.0, y_true - upper)
    winkler = width + penalty_lower + penalty_upper
    return float(np.mean(winkler))


def run_stage_2_inner_scale_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_splits: int = 5
) -> Dict[str, Any]:
    """Stage 2: Evaluate Unnormalized vs. Locally Normalized scale strictly within D_train using 5-fold grouped CV."""
    print("\n--- STAGE 2: Inner Grouped Cross-Validation Scale Selection (D_train only) ---")
    gkf = GroupKFold(n_splits=n_splits)

    unnorm_winkler_list = []
    norm_winkler_list = []
    unnorm_median_width_list = []
    norm_median_width_list = []
    unnorm_cov_list = []
    norm_cov_list = []

    fold = 1
    for inner_train_idx, inner_val_idx in gkf.split(X_train, y_train, groups_train):
        X_tr, y_tr = X_train[inner_train_idx], y_train[inner_train_idx]
        X_va, y_val = X_train[inner_val_idx], y_train[inner_val_idx]
        groups_val = groups_train[inner_val_idx]

        # Fit inner RF
        inner_rf = RandomForestRegressor(
            n_estimators=100, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features=1.0, random_state=42 + fold, n_jobs=-1
        )
        inner_rf.fit(X_tr, y_tr)
        val_pred = inner_rf.predict(X_va)

        # Scale estimates
        unnorm_scale = np.ones(len(X_va))
        scale_est = EnsembleDispersionScaleEstimator(inner_rf, epsilon=STABILIZER_EPSILON)
        norm_scale = scale_est.predict_scale(X_va)

        # Sample 1 record per group in inner validation for calibration simulation
        rng = np.random.RandomState(42 + fold)
        val_df = pd.DataFrame({"group": groups_val, "idx": np.arange(len(X_va))})
        sampled_val_indices = [rng.choice(grp["idx"].values) for _, grp in val_df.groupby("group")]

        # Split inner validation pairs into inner-cal and inner-test (50/50)
        rng.shuffle(sampled_val_indices)
        half = len(sampled_val_indices) // 2
        cal_idx = sampled_val_indices[:half]
        test_idx = sampled_val_indices[half:]

        # Calibrate both methods on inner-cal at alpha=0.10 (90%)
        unnorm_scores_cal = np.abs(y_val[cal_idx] - val_pred[cal_idx])
        norm_scores_cal = np.abs(y_val[cal_idx] - val_pred[cal_idx]) / norm_scale[cal_idx]

        q_unnorm = compute_conformal_quantile(unnorm_scores_cal, alpha=0.10)
        q_norm = compute_conformal_quantile(norm_scores_cal, alpha=0.10)

        # Evaluate on inner-test
        l_unnorm = val_pred[test_idx] - q_unnorm * unnorm_scale[test_idx]
        u_unnorm = val_pred[test_idx] + q_unnorm * unnorm_scale[test_idx]
        cov_u, m_w_u, med_w_u, _ = compute_coverage_and_width(y_val[test_idx], l_unnorm, u_unnorm)
        wink_u = compute_winkler_score(y_val[test_idx], l_unnorm, u_unnorm, alpha=0.10)

        l_norm = val_pred[test_idx] - q_norm * norm_scale[test_idx]
        u_norm = val_pred[test_idx] + q_norm * norm_scale[test_idx]
        cov_n, m_w_n, med_w_n, _ = compute_coverage_and_width(y_val[test_idx], l_norm, u_norm)
        wink_n = compute_winkler_score(y_val[test_idx], l_norm, u_norm, alpha=0.10)

        unnorm_winkler_list.append(wink_u)
        norm_winkler_list.append(wink_n)
        unnorm_median_width_list.append(med_w_u)
        norm_median_width_list.append(med_w_n)
        unnorm_cov_list.append(cov_u)
        norm_cov_list.append(cov_n)
        fold += 1

    summary = {
        "inner_folds": n_splits,
        "unnormalized": {
            "mean_coverage_90": float(np.mean(unnorm_cov_list)),
            "mean_median_width": float(np.mean(unnorm_median_width_list)),
            "mean_winkler_score": float(np.mean(unnorm_winkler_list))
        },
        "locally_normalized": {
            "mean_coverage_90": float(np.mean(norm_cov_list)),
            "mean_median_width": float(np.mean(norm_median_width_list)),
            "mean_winkler_score": float(np.mean(norm_winkler_list))
        },
        "winkler_delta_norm_minus_unnorm": float(np.mean(norm_winkler_list) - np.mean(unnorm_winkler_list)),
        "decision": "Lock and evaluate both Unnormalized (primary benchmark) and Locally Normalized in parallel on D_test."
    }
    print(f"Inner CV Results (alpha=0.10):")
    print(f"  Unnormalized:       Coverage={summary['unnormalized']['mean_coverage_90']*100:.1f}%, Median Width={summary['unnormalized']['mean_median_width']:.2f} PCE pp, Winkler={summary['unnormalized']['mean_winkler_score']:.2f}")
    print(f"  Locally Normalized: Coverage={summary['locally_normalized']['mean_coverage_90']*100:.1f}%, Median Width={summary['locally_normalized']['mean_median_width']:.2f} PCE pp, Winkler={summary['locally_normalized']['mean_winkler_score']:.2f}")
    return summary


def main():
    print("=" * 80)
    print("OPV-PredictNet — Phase 7.1 Grouped Conformal Calibration Experiment")
    print("=" * 80)
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Dataset & Construct Grouping Column
    print(f"\nLoading benchmark dataset from: {BENCHMARK_PATH}")
    df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    total_samples = len(df)
    assert total_samples == 22467, f"Expected 22,467 records, got {total_samples}"

    pair_groups = (df["donor_canonical"] + "::" + df["acceptor_canonical"]).to_numpy()
    df["pair_group"] = pair_groups
    unique_pairs = np.unique(pair_groups)
    assert len(unique_pairs) == 4730, f"Expected 4,730 normalized material-name pairs, got {len(unique_pairs)}"
    print(f"Verified dataset: {total_samples} records, {len(unique_pairs)} normalized material-name pairs.")

    # 2. Extract 22 Features
    print("\nExtracting 22 pure molecular features...")
    unique_donors = df["donor_smiles"].unique()
    unique_acceptors = df["acceptor_smiles"].unique()
    donor_mols = {s: Chem.MolFromSmiles(s) for s in unique_donors}
    acceptor_mols = {s: Chem.MolFromSmiles(s) for s in unique_acceptors}

    smiles_pairs = df[["donor_smiles", "acceptor_smiles"]].drop_duplicates()
    pair_feat_dict = {}
    for _, row in smiles_pairs.iterrows():
        d_mol = donor_mols[row["donor_smiles"]]
        a_mol = acceptor_mols[row["acceptor_smiles"]]
        vec, _ = extract_features_from_mols(d_mol, a_mol)
        pair_feat_dict[(row["donor_smiles"], row["acceptor_smiles"])] = vec

    X = np.array([pair_feat_dict[(r.donor_smiles, r.acceptor_smiles)] for r in df[["donor_smiles", "acceptor_smiles"]].itertuples(index=False)], dtype=np.float64)
    y = df["pce"].to_numpy(dtype=np.float64)

    assert X.shape == (total_samples, 22), f"Feature matrix shape mismatch: {X.shape}"
    assert not np.isnan(X).any(), "NaNs detected in feature matrix!"
    assert not np.isnan(y).any(), "NaNs detected in target array!"
    for i, col in enumerate(FEATURE_NAMES):
        df[col] = X[:, i]

    # 3. Data Partitioning (60/20/20 Pair-Disjoint Split)
    print("\nPartitioning dataset into 60/20/20 pair-disjoint splits...")
    df_train, df_cal, df_test = split_grouped_dataset(
        df, group_col="pair_group", train_frac=0.60, cal_frac=0.20, seed=RANDOM_SEED_PARTITION
    )
    n_train_pairs = df_train["pair_group"].nunique()
    n_cal_pairs = df_cal["pair_group"].nunique()
    n_test_pairs = df_test["pair_group"].nunique()

    print(f"  Training:   {len(df_train)} records across {n_train_pairs} pairs (60.0%)")
    print(f"  Calibration:{len(df_cal)} records across {n_cal_pairs} pairs (20.0%)")
    print(f"  Test:       {len(df_test)} records across {n_test_pairs} pairs (20.0%)")
    assert n_train_pairs == 2838 and n_cal_pairs == 946 and n_test_pairs == 946

    # Sample exactly 1 record per pair for Calibration and Test
    print("\nSampling 1 record per pair for Calibration and Test partitions...")
    sampled_cal = sample_one_record_per_group(df_cal, group_col="pair_group", seed=RANDOM_SEED_CAL_SAMPLE)
    sampled_test = sample_one_record_per_group(df_test, group_col="pair_group", seed=RANDOM_SEED_TEST_SAMPLE)

    assert len(sampled_cal) == 946 and len(sampled_test) == 946
    print(f"  Sampled Calibration: exactly {len(sampled_cal)} records (seed={RANDOM_SEED_CAL_SAMPLE})")
    print(f"  Sampled Test:        exactly {len(sampled_test)} records (seed={RANDOM_SEED_TEST_SAMPLE})")

    # Record cluster breakdown
    partition_meta = {
        "dataset_total_records": total_samples,
        "dataset_total_pairs": len(unique_pairs),
        "random_seed_partition": RANDOM_SEED_PARTITION,
        "random_seed_cal_sample": RANDOM_SEED_CAL_SAMPLE,
        "random_seed_test_sample": RANDOM_SEED_TEST_SAMPLE,
        "train": {
            "records": len(df_train),
            "pairs": n_train_pairs,
            "singletons": int((df_train.groupby("pair_group").size() == 1).sum()),
            "repeated_pairs": int((df_train.groupby("pair_group").size() > 1).sum()),
            "max_records_in_pair": int(df_train.groupby("pair_group").size().max())
        },
        "cal": {
            "records_total": len(df_cal),
            "pairs": n_cal_pairs,
            "singletons": int((df_cal.groupby("pair_group").size() == 1).sum()),
            "repeated_pairs": int((df_cal.groupby("pair_group").size() > 1).sum()),
            "max_records_in_pair": int(df_cal.groupby("pair_group").size().max()),
            "sampled_records": len(sampled_cal)
        },
        "test": {
            "records_total": len(df_test),
            "pairs": n_test_pairs,
            "singletons": int((df_test.groupby("pair_group").size() == 1).sum()),
            "repeated_pairs": int((df_test.groupby("pair_group").size() > 1).sum()),
            "max_records_in_pair": int(df_test.groupby("pair_group").size().max()),
            "sampled_records": len(sampled_test)
        }
    }
    with open(os.path.join(RESULTS_DIR, "phase_7_1_partition_metadata.json"), "w") as f:
        json.dump(partition_meta, f, indent=2)

    # Save selected records for full reproducibility
    selected_records_df = pd.concat([
        sampled_cal[["pair_group", "donor_canonical", "acceptor_canonical", "pce"]].assign(partition="cal"),
        sampled_test[["pair_group", "donor_canonical", "acceptor_canonical", "pce"]].assign(partition="test")
    ])
    selected_records_df.to_csv(os.path.join(RESULTS_DIR, "phase_7_1_selected_records.csv"), index=True)

    # 4. Stage 1: Fit Baseline Point Model on D_train Only
    print("\n--- STAGE 1: Fit Phase 5 Baseline Model on D_train only ---")
    X_train = df_train[FEATURE_NAMES].values
    y_train = df_train["pce"].values
    groups_train = df_train["pair_group"].values

    calibrator_unnorm = GroupConformalCalibrator(method="unnormalized", random_state=MODEL_SEED)
    calibrator_norm = GroupConformalCalibrator(method="locally_normalized", random_state=MODEL_SEED)

    calibrator_unnorm.fit_point_model(X_train, y_train)
    calibrator_norm.fit_point_model(X_train, y_train)

    train_pred = calibrator_unnorm.model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    print(f"Fitted Phase 5 Baseline RF on D_train (N={len(df_train)}): MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}")

    # 5. Stage 2: Inner Grouped CV Scale Selection on D_train Only
    scale_selection_summary = run_stage_2_inner_scale_selection(X_train, y_train, groups_train, n_splits=5)
    with open(os.path.join(RESULTS_DIR, "phase_7_1_inner_cv_scale_selection.json"), "w") as f:
        json.dump(scale_selection_summary, f, indent=2)

    # 6. Stage 3: Calibration on D_cal Only
    print("\n--- STAGE 3: Quantile Calibration on D_cal (N=946 pairs) ---")
    X_cal = sampled_cal[FEATURE_NAMES].values
    y_cal = sampled_cal["pce"].values

    alphas = [0.20, 0.10, 0.05]
    scores_unnorm = calibrator_unnorm.calibrate(X_cal, y_cal, alphas=alphas)
    scores_norm = calibrator_norm.calibrate(X_cal, y_cal, alphas=alphas)

    quantiles_summary = {
        "n_cal": len(sampled_cal),
        "unnormalized_quantiles": {
            "alpha_0.20_nominal_80": {"rank_k": int(np.ceil((946 + 1) * 0.80)), "cutoff_PCE_pp": calibrator_unnorm.quantiles[0.20]},
            "alpha_0.10_nominal_90": {"rank_k": int(np.ceil((946 + 1) * 0.90)), "cutoff_PCE_pp": calibrator_unnorm.quantiles[0.10]},
            "alpha_0.05_nominal_95": {"rank_k": int(np.ceil((946 + 1) * 0.95)), "cutoff_PCE_pp": calibrator_unnorm.quantiles[0.05]}
        },
        "locally_normalized_quantiles": {
            "alpha_0.20_nominal_80": {"rank_k": int(np.ceil((946 + 1) * 0.80)), "cutoff_normalized": calibrator_norm.quantiles[0.20]},
            "alpha_0.10_nominal_90": {"rank_k": int(np.ceil((946 + 1) * 0.90)), "cutoff_normalized": calibrator_norm.quantiles[0.10]},
            "alpha_0.05_nominal_95": {"rank_k": int(np.ceil((946 + 1) * 0.95)), "cutoff_normalized": calibrator_norm.quantiles[0.05]}
        }
    }
    with open(os.path.join(RESULTS_DIR, "phase_7_1_calibration_quantiles.json"), "w") as f:
        json.dump(quantiles_summary, f, indent=2)

    print("Computed Finite-Sample Conformal Quantiles (k = ceil((946+1)*(1-alpha))):")
    print(f"  Unnormalized:       q_80={calibrator_unnorm.quantiles[0.20]:.4f} PCE pp, q_90={calibrator_unnorm.quantiles[0.10]:.4f} PCE pp, q_95={calibrator_unnorm.quantiles[0.05]:.4f} PCE pp")
    print(f"  Locally Normalized: q_80={calibrator_norm.quantiles[0.20]:.4f}, q_90={calibrator_norm.quantiles[0.10]:.4f}, q_95={calibrator_norm.quantiles[0.05]:.4f}")

    # 7. Stage 4: Single Final Test Evaluation on D_test Only
    print("\n--- STAGE 4: Final Test Evaluation on D_test (N=946 pairs) ---")
    X_test = sampled_test[FEATURE_NAMES].values
    y_test = sampled_test["pce"].values

    test_pred = calibrator_unnorm.model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    print(f"Baseline Point Performance on Unseen Test Pairs: MAE={test_mae:.4f} PCE pp, RMSE={test_rmse:.4f} PCE pp, R2={test_r2:.4f}")

    results_table = []
    final_metrics = {
        "test_n": len(y_test),
        "point_metrics": {"mae": float(test_mae), "rmse": float(test_rmse), "r2": float(test_r2)},
        "methods": {}
    }

    test_pred_df = sampled_test[["pair_group", "donor_canonical", "acceptor_canonical", "pce"]].copy()
    test_pred_df["y_pred"] = test_pred

    for method_name, calibrator in [("unnormalized", calibrator_unnorm), ("locally_normalized", calibrator_norm)]:
        method_dict = {}
        for alpha in alphas:
            nominal_level = int((1.0 - alpha) * 100)
            
            # 1. Primary Deliverable: Raw Unconstrained Intervals
            y_p, scale, lower_raw, upper_raw = calibrator.predict_intervals(X_test, alpha, clip_lower_zero=False)
            cov_raw, mean_w_raw, med_w_raw, widths_raw = compute_coverage_and_width(y_test, lower_raw, upper_raw)
            se_raw = np.sqrt(cov_raw * (1.0 - cov_raw) / len(y_test))
            winkler_raw = compute_winkler_score(y_test, lower_raw, upper_raw, alpha)

            # 2. Diagnostic Deliverable: Physically Clipped Intervals [0.0, inf)
            _, _, lower_clip, upper_clip = calibrator.predict_intervals(X_test, alpha, clip_lower_zero=True)
            cov_clip, mean_w_clip, med_w_clip, widths_clip = compute_coverage_and_width(y_test, lower_clip, upper_clip)
            se_clip = np.sqrt(cov_clip * (1.0 - cov_clip) / len(y_test))
            winkler_clip = compute_winkler_score(y_test, lower_clip, upper_clip, alpha)
            clipped_count = int(np.sum(lower_raw < 0.0))

            level_dict = {
                "nominal_coverage": float(1.0 - alpha),
                "raw_intervals": {
                    "empirical_coverage": float(cov_raw),
                    "coverage_se": float(se_raw),
                    "coverage_ci_95": [float(cov_raw - 1.96 * se_raw), float(cov_raw + 1.96 * se_raw)],
                    "mean_width_PCE_pp": float(mean_w_raw),
                    "median_width_PCE_pp": float(med_w_raw),
                    "winkler_score": float(winkler_raw),
                    "width_p10": float(np.percentile(widths_raw, 10)),
                    "width_p25": float(np.percentile(widths_raw, 25)),
                    "width_p75": float(np.percentile(widths_raw, 75)),
                    "width_p90": float(np.percentile(widths_raw, 90))
                },
                "clipped_intervals_diagnostic": {
                    "empirical_coverage": float(cov_clip),
                    "coverage_se": float(se_clip),
                    "mean_width_PCE_pp": float(mean_w_clip),
                    "median_width_PCE_pp": float(med_w_clip),
                    "winkler_score": float(winkler_clip),
                    "records_clipped_at_zero": clipped_count,
                    "fraction_clipped": float(clipped_count / len(y_test))
                }
            }
            method_dict[f"nominal_{nominal_level}"] = level_dict

            # Save interval columns in test_pred_df
            pfx = f"{method_name[:4]}_{nominal_level}"
            test_pred_df[f"{pfx}_lower_raw"] = lower_raw
            test_pred_df[f"{pfx}_upper_raw"] = upper_raw
            test_pred_df[f"{pfx}_lower_clip"] = lower_clip
            test_pred_df[f"{pfx}_upper_clip"] = upper_clip
            test_pred_df[f"{pfx}_width_raw"] = widths_raw

            print(f"[{method_name.upper()} - {nominal_level}% Nominal (k={quantiles_summary[method_name+'_quantiles'][f'alpha_{alpha:.2f}_nominal_{nominal_level}']['rank_k']})]")
            print(f"  Raw Coverage:    {cov_raw*100:.2f}% (SE = {se_raw*100:.2f}%, 95% CI: [{cov_raw*100 - 1.96*se_raw*100:.2f}%, {cov_raw*100 + 1.96*se_raw*100:.2f}%])")
            print(f"  Raw Width:       Mean = {mean_w_raw:.2f} PCE pp, Median = {med_w_raw:.2f} PCE pp, IQR = [{np.percentile(widths_raw, 25):.2f}, {np.percentile(widths_raw, 75):.2f}]")
            print(f"  Clipped Coverage:{cov_clip*100:.2f}%, Clipped Median Width = {med_w_clip:.2f} PCE pp (Clipped count = {clipped_count}/{len(y_test)})")

        final_metrics["methods"][method_name] = method_dict

    with open(os.path.join(RESULTS_DIR, "phase_7_1_final_test_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    test_pred_df.to_csv(os.path.join(RESULTS_DIR, "phase_7_1_test_predictions_and_intervals.csv"), index=True)

    # 8. Sliced Subgroup Audits on D_test
    print("\n--- Sliced Subgroup Analysis on D_test ---")
    subgroup_metrics = {"pce_tiers": {}, "pair_frequency": {}, "device_architecture": {}, "high_pce_slices": {}}

    # A. By PCE Range
    pce_bins = [
        ("< 5.0% PCE", (y_test < 5.0)),
        ("5.0% - 10.0% PCE", (y_test >= 5.0) & (y_test < 10.0)),
        ("10.0% - 15.0% PCE", (y_test >= 10.0) & (y_test < 15.0)),
        (">= 15.0% PCE", (y_test >= 15.0))
    ]
    for label, mask in pce_bins:
        n_sub = int(np.sum(mask))
        if n_sub > 0:
            cov_u_90, mw_u, med_u, _ = compute_coverage_and_width(
                y_test[mask], test_pred_df.loc[mask, "unno_90_lower_raw"].values, test_pred_df.loc[mask, "unno_90_upper_raw"].values
            )
            cov_n_90, mw_n, med_n, _ = compute_coverage_and_width(
                y_test[mask], test_pred_df.loc[mask, "loca_90_lower_raw"].values, test_pred_df.loc[mask, "loca_90_upper_raw"].values
            )
            subgroup_metrics["pce_tiers"][label] = {
                "sample_count": n_sub,
                "fraction_of_test": float(n_sub / len(y_test)),
                "unnormalized_90_coverage": float(cov_u_90),
                "unnormalized_90_median_width": float(med_u),
                "locally_normalized_90_coverage": float(cov_n_90),
                "locally_normalized_90_median_width": float(med_n)
            }
            print(f"  PCE Tier '{label}' (N={n_sub}): Unnorm Cov={cov_u_90*100:.1f}%, Norm Cov={cov_n_90*100:.1f}%")

    # B. By Pair Frequency in Literature (Singletons vs. Repeated Pairs)
    test_pair_counts = df_test.groupby("pair_group").size()
    sampled_test_pair_counts = sampled_test["pair_group"].map(test_pair_counts).values
    freq_bins = [
        ("Singleton Pairs (N_lit = 1)", (sampled_test_pair_counts == 1)),
        ("Repeated Pairs (N_lit > 1)", (sampled_test_pair_counts > 1))
    ]
    for label, mask in freq_bins:
        n_sub = int(np.sum(mask))
        cov_u_90, _, med_u, _ = compute_coverage_and_width(
            y_test[mask], test_pred_df.loc[mask, "unno_90_lower_raw"].values, test_pred_df.loc[mask, "unno_90_upper_raw"].values
        )
        cov_n_90, _, med_n, _ = compute_coverage_and_width(
            y_test[mask], test_pred_df.loc[mask, "loca_90_lower_raw"].values, test_pred_df.loc[mask, "loca_90_upper_raw"].values
        )
        subgroup_metrics["pair_frequency"][label] = {
            "sample_count": n_sub,
            "fraction_of_test": float(n_sub / len(y_test)),
            "unnormalized_90_coverage": float(cov_u_90),
            "unnormalized_90_median_width": float(med_u),
            "locally_normalized_90_coverage": float(cov_n_90),
            "locally_normalized_90_median_width": float(med_n)
        }
        print(f"  Pair Frequency '{label}' (N={n_sub}): Unnorm Cov={cov_u_90*100:.1f}%, Norm Cov={cov_n_90*100:.1f}%")

    # C. By High-PCE Thresholds
    for high_thresh in [12.0, 14.0, 15.0]:
        mask_high = (y_test >= high_thresh)
        n_high = int(np.sum(mask_high))
        if n_high > 0:
            cov_u_90, _, med_u, _ = compute_coverage_and_width(
                y_test[mask_high], test_pred_df.loc[mask_high, "unno_90_lower_raw"].values, test_pred_df.loc[mask_high, "unno_90_upper_raw"].values
            )
            cov_n_90, _, med_n, _ = compute_coverage_and_width(
                y_test[mask_high], test_pred_df.loc[mask_high, "loca_90_lower_raw"].values, test_pred_df.loc[mask_high, "loca_90_upper_raw"].values
            )
            subgroup_metrics["high_pce_slices"][f"PCE >= {high_thresh}%"] = {
                "threshold_PCE": high_thresh,
                "sample_count": n_high,
                "fraction_of_test": float(n_high / len(y_test)),
                "unnormalized_90_coverage": float(cov_u_90),
                "locally_normalized_90_coverage": float(cov_n_90)
            }
            print(f"  High-PCE (>= {high_thresh}%, N={n_high}): Unnorm Cov={cov_u_90*100:.1f}%, Norm Cov={cov_n_90*100:.1f}%")

    # D. By Device Architecture
    # Map device architecture from sampled test records if present
    if "device_architecture" in sampled_test.columns or "device_type" in sampled_test.columns:
        arch_col = "device_architecture" if "device_architecture" in sampled_test.columns else "device_type"
        arch_series = sampled_test[arch_col].fillna("Unknown").astype(str)
        for arch_val in arch_series.value_counts().head(3).index:
            mask_arch = (arch_series == arch_val).values
            n_arch = int(np.sum(mask_arch))
            if n_arch >= 20:
                cov_u_90, _, med_u, _ = compute_coverage_and_width(
                    y_test[mask_arch], test_pred_df.loc[mask_arch, "unno_90_lower_raw"].values, test_pred_df.loc[mask_arch, "unno_90_upper_raw"].values
                )
                cov_n_90, _, med_n, _ = compute_coverage_and_width(
                    y_test[mask_arch], test_pred_df.loc[mask_arch, "loca_90_lower_raw"].values, test_pred_df.loc[mask_arch, "loca_90_upper_raw"].values
                )
                subgroup_metrics["device_architecture"][arch_val] = {
                    "sample_count": n_arch,
                    "unnormalized_90_coverage": float(cov_u_90),
                    "locally_normalized_90_coverage": float(cov_n_90)
                }
                print(f"  Architecture '{arch_val}' (N={n_arch}): Unnorm Cov={cov_u_90*100:.1f}%, Norm Cov={cov_n_90*100:.1f}%")

    with open(os.path.join(RESULTS_DIR, "phase_7_1_subgroup_metrics.json"), "w") as f:
        json.dump(subgroup_metrics, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nPhase 7.1 Experiment completed successfully in {elapsed:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
