"""
OPV-PredictNet — Phase 8.1: Controlled Processing-Aware Model Experiments

Executes the approved Phase 8.0.4 experimental protocol:
- Models: M0 (22 feats), M1 (50 feats), M2 (44 feats), M2A (37 feats), M3 (44 feats)
- Data: opv_devices_strict_molecular_benchmark.csv (22,467 records, 4,730 pairs)
- Partition: Seed 8888 60/20/20 group split (Train: 2838 pairs, Cal: 946 pairs, Test: 946 pairs)
- Alternative B Sampling: Cal (seed 8201), Test (seed 8202) -> 946 records each
- Leakage Controls: All imputers, encoders, and medians fitted strictly on D_train
- Uncertainty: Split-conformal calibration on D_cal (80%, 90%, 95%)
- Single-threaded deterministic execution: n_jobs=1
"""

import sys
import os
import re
import json
import time
import hashlib
import platform
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath('.'))
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES

BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
EXPECTED_SHA256 = 'ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897'
RESULTS_DIR = 'ml/results'

PARTITION_SEED = 8888
CAL_SAMPLE_SEED = 8201
TEST_SAMPLE_SEED = 8202
MODEL_SEED = 8888

# M1 One-Hot columns expected from D_train
M1_OHE_COLUMNS = [
    'solvent_chlorobenzene', 'solvent_chloroform', 'solvent_missing', 'solvent_o-dcb', 'solvent_other',
    'additive_cn', 'additive_dio', 'additive_missing', 'additive_other',
    'architecture_conventional', 'architecture_inverted', 'architecture_missing', 'architecture_other',
    'htl_missing', 'htl_moo3', 'htl_other', 'htl_pedot_pss',
    'etl_ca', 'etl_missing', 'etl_other', 'etl_pdinn', 'etl_pfn_br', 'etl_zno'
]

# Physical lookup mappings
SOLV_BP_MAP = {
    'chloroform': 61.2,
    'chlorobenzene': 131.6,
    'o-dcb': 180.5,
    'toluene': 110.6,
    'o-xylene': 144.4,
    'thf': 66.0
}

ETL_WF_MAP = {
    'zno': 4.30,
    'pdinn': 4.00,
    'ca': 2.87,
    'pfn-br': 4.20
}

HTL_WF_MAP = {
    'pedot': 5.00,
    'moo3': 5.30,
    '2pacz': 5.20
}


def compute_sha256(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute finite-sample split-conformal quantile cutoff:
    k = ceil((n_cal + 1) * (1 - alpha))
    cutoff = sorted_scores[k - 1]
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        raise ValueError(f"k={k} exceeds sample size n={n}")
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def map_solvent_nominal(val):
    if pd.isna(val): return 'missing'
    s = str(val).lower().strip()
    if 'chloroform' in s: return 'chloroform'
    if 'chlorobenzene' in s and 'di' not in s: return 'chlorobenzene'
    if 'dcb' in s or 'dichlorobenzene' in s: return 'o-dcb'
    return 'other'


def map_additive_nominal(val):
    if pd.isna(val): return 'missing'
    s = str(val).lower().strip()
    if 'dio' in s or 'diiodooctane' in s: return 'dio'
    if 'cn' in s or 'chloronaphthalene' in s: return 'cn'
    if s in ['none', 'neat', '0', 'no']: return 'missing'
    return 'other'


def map_arch_nominal(val):
    if pd.isna(val): return 'missing'
    s = str(val).lower().strip()
    if 'conventional' in s or 'regular' in s or 'standard' in s: return 'conventional'
    if 'inverted' in s: return 'inverted'
    return 'other'


def map_htl_nominal(val):
    if pd.isna(val): return 'missing'
    s = str(val).lower().strip()
    if 'pedot' in s: return 'pedot_pss'
    if 'moo3' in s or 'mox' in s or 'molybdenum' in s: return 'moo3'
    return 'other'


def map_etl_nominal(val):
    if pd.isna(val): return 'missing'
    s = str(val).lower().strip()
    if 'zno' in s: return 'zno'
    if 'pdinn' in s: return 'pdinn'
    if s == 'ca' or 'calcium' in s: return 'ca'
    if 'pfn' in s: return 'pfn_br'
    return 'other'


class FeaturePipeline:
    """Fitted strictly on D_train; applies frozen transforms to D_cal and D_test."""
    def __init__(self):
        self.da_median = 0.8333
        self.thk_median = 100.0
        self.tmp_median = 100.0
        self.f_donor_median = 0.4545
        self.thk_log_median = 2.0
        self.ann_norm_median = 0.0
        self.bp_median = 131.6
        self.etl_median = 4.30
        self.htl_median = 5.00
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.mol_dict = {}

    def fit_descriptors(self, df_all: pd.DataFrame):
        unique_donors = df_all['donor_smiles'].unique()
        unique_acceptors = df_all['acceptor_smiles'].unique()
        d_mols = {s: Chem.MolFromSmiles(s) for s in unique_donors}
        a_mols = {s: Chem.MolFromSmiles(s) for s in unique_acceptors}

        pairs = df_all[['donor_smiles', 'acceptor_smiles']].drop_duplicates()
        for _, r in pairs.iterrows():
            d_m = d_mols[r['donor_smiles']]
            a_m = a_mols[r['acceptor_smiles']]
            vec, _ = extract_features_from_mols(d_m, a_m)
            self.mol_dict[(r['donor_smiles'], r['acceptor_smiles'])] = vec

    def fit_training(self, df_train: pd.DataFrame):
        # 1. Continuous parsing for medians
        def parse_da(val):
            if pd.isna(val): return np.nan
            s = str(val).strip()
            m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", s)
            if m:
                try:
                    d, a = float(m.group(1)), float(m.group(2))
                    if a > 0: return d / a
                except: pass
            return np.nan

        def parse_thk(val):
            if pd.isna(val): return np.nan
            s = str(val).replace('nm', '').strip()
            try:
                v = float(s)
                if 15.0 <= v <= 600.0: return v
            except: pass
            return np.nan

        def parse_tmp(val):
            if pd.isna(val): return np.nan
            try:
                v = float(val)
                if 0.0 <= v <= 250.0: return v
            except: pass
            return np.nan

        def parse_f_donor(val):
            if pd.isna(val): return np.nan
            s = str(val).strip()
            m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", s)
            if m:
                try:
                    d, a = float(m.group(1)), float(m.group(2))
                    if d > 0 and a > 0: return d / (d + a)
                except: pass
            return np.nan

        def parse_ann_norm(val):
            if pd.isna(val): return np.nan
            try:
                v = float(val)
                if 40.0 <= v <= 250.0: return (v - 100.0) / 50.0
                if 0.0 <= v <= 35.0: return -1.3
            except: pass
            return np.nan

        da_vals = df_train['d_a_ratio'].apply(parse_da)
        thk_vals = df_train['active_layer_thickness'].apply(parse_thk)
        tmp_vals = df_train['annealing_temp'].apply(parse_tmp)
        f_vals = df_train['d_a_ratio'].apply(parse_f_donor)
        thk_log_vals = thk_vals.apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)
        ann_norm_vals = df_train['annealing_temp'].apply(parse_ann_norm)

        self.da_median = float(da_vals.median())
        self.thk_median = float(thk_vals.median())
        self.tmp_median = float(tmp_vals.median())
        self.f_donor_median = float(f_vals.median())
        self.thk_log_median = float(thk_log_vals.median())
        self.ann_norm_median = float(ann_norm_vals.median())

        # Physical lookups observed medians
        def get_bp(val):
            if pd.isna(val): return np.nan
            s = str(val).lower().strip()
            for k, v in SOLV_BP_MAP.items():
                if k in s: return v
            return np.nan

        def get_etl(val):
            if pd.isna(val): return np.nan
            s = str(val).lower().strip()
            for k, v in ETL_WF_MAP.items():
                if k in s: return v
            return np.nan

        def get_htl(val):
            if pd.isna(val): return np.nan
            s = str(val).lower().strip()
            for k, v in HTL_WF_MAP.items():
                if k in s: return v
            return np.nan

        bp_vals = df_train['solvent_canonical'].apply(get_bp)
        etl_vals = df_train['etl_canonical'].apply(get_etl)
        htl_vals = df_train['htl_canonical'].apply(get_htl)

        self.bp_median = float(bp_vals.median())
        self.etl_median = float(etl_vals.median())
        self.htl_median = float(htl_vals.median())

        # Fit OHE for M1 strictly on D_train
        cats = pd.DataFrame({
            'solvent': df_train['solvent_canonical'].apply(map_solvent_nominal),
            'additive': df_train['additive_canonical'].apply(map_additive_nominal),
            'architecture': df_train['device_type'].apply(map_arch_nominal),
            'htl': df_train['htl_canonical'].apply(map_htl_nominal),
            'etl': df_train['etl_canonical'].apply(map_etl_nominal),
        })
        self.ohe.fit(cats)
        ohe_features = list(self.ohe.get_feature_names_out())
        assert ohe_features == M1_OHE_COLUMNS, f"OHE feature mismatch: {ohe_features}"

    def extract_m0(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        vecs = [self.mol_dict[(r.donor_smiles, r.acceptor_smiles)] for r in df_sub[['donor_smiles', 'acceptor_smiles']].itertuples(index=False)]
        df_m0 = pd.DataFrame(vecs, columns=FEATURE_NAMES, index=df_sub.index)
        assert df_m0.shape == (len(df_sub), 22)
        assert not df_m0.isna().any().any()
        return df_m0

    def extract_m1(self, df_sub: pd.DataFrame, df_m0: pd.DataFrame) -> pd.DataFrame:
        m1_dict = {col: df_m0[col].values for col in FEATURE_NAMES}

        def parse_da(val):
            if pd.isna(val): return self.da_median
            s = str(val).strip()
            m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", s)
            if m:
                try:
                    d, a = float(m.group(1)), float(m.group(2))
                    if a > 0: return d / a
                except: pass
            return self.da_median

        def parse_thk(val):
            if pd.isna(val): return self.thk_median
            s = str(val).replace('nm', '').strip()
            try:
                v = float(s)
                if 15.0 <= v <= 600.0: return v
            except: pass
            return self.thk_median

        def parse_tmp(val):
            if pd.isna(val): return self.tmp_median
            try:
                v = float(val)
                if 0.0 <= v <= 250.0: return v
            except: pass
            return self.tmp_median

        m1_dict['da_ratio_numeric'] = df_sub['d_a_ratio'].apply(parse_da).values
        m1_dict['thickness_nm'] = df_sub['active_layer_thickness'].apply(parse_thk).values
        m1_dict['annealing_temp_c'] = df_sub['annealing_temp'].apply(parse_tmp).values

        # Binary presence
        def has_add(val):
            if pd.isna(val): return 0
            s = str(val).lower().strip()
            if s in ['none', 'neat', '0', 'no']: return 0
            return 1 if len(s) > 0 else 0

        def has_ann(val):
            if pd.isna(val): return 0
            try:
                return 1 if float(val) > 35.0 else 0
            except: return 0

        m1_dict['has_additive'] = df_sub['additive_canonical'].apply(has_add).values
        m1_dict['has_thermal_annealing'] = df_sub['annealing_temp'].apply(has_ann).values

        df_m1_core = pd.DataFrame(m1_dict, index=df_sub.index)

        # OHE categoricals
        cats = pd.DataFrame({
            'solvent': df_sub['solvent_canonical'].apply(map_solvent_nominal),
            'additive': df_sub['additive_canonical'].apply(map_additive_nominal),
            'architecture': df_sub['device_type'].apply(map_arch_nominal),
            'htl': df_sub['htl_canonical'].apply(map_htl_nominal),
            'etl': df_sub['etl_canonical'].apply(map_etl_nominal),
        }, index=df_sub.index)

        ohe_arr = self.ohe.transform(cats)
        df_ohe = pd.DataFrame(ohe_arr, columns=M1_OHE_COLUMNS, index=df_sub.index)

        df_m1 = pd.concat([df_m1_core, df_ohe], axis=1)
        assert df_m1.shape == (len(df_sub), 50), f"M1 shape {df_m1.shape} != ({len(df_sub)}, 50)"
        assert not df_m1.isna().any().any()
        return df_m1

    def extract_m2(self, df_sub: pd.DataFrame, df_m0: pd.DataFrame, is_imputed: bool = True) -> pd.DataFrame:
        m2_dict = {col: df_m0[col].values for col in FEATURE_NAMES}

        # 4 continuous
        def parse_f_donor(val):
            if pd.isna(val): return self.f_donor_median if is_imputed else np.nan
            s = str(val).strip()
            m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", s)
            if m:
                try:
                    d, a = float(m.group(1)), float(m.group(2))
                    if d > 0 and a > 0: return d / (d + a)
                except: pass
            return self.f_donor_median if is_imputed else np.nan

        def parse_thk_log(val):
            if pd.isna(val): return self.thk_log_median if is_imputed else np.nan
            s = str(val).replace('nm', '').strip()
            try:
                v = float(s)
                if 15.0 <= v <= 600.0: return np.log10(v)
            except: pass
            return self.thk_log_median if is_imputed else np.nan

        def parse_ann_norm(val):
            if pd.isna(val): return self.ann_norm_median if is_imputed else np.nan
            try:
                v = float(val)
                if 40.0 <= v <= 250.0: return (v - 100.0) / 50.0
                if 0.0 <= v <= 35.0: return -1.3
            except: pass
            return self.ann_norm_median if is_imputed else np.nan

        def parse_add_conc(val):
            if pd.isna(val): return 0.0 if is_imputed else np.nan
            s = str(val).strip()
            m = re.search(r"([0-9\.]+)", s)
            if m:
                try:
                    v = float(m.group(1))
                    if 0.0 <= v <= 10.0: return v
                except: pass
            return 0.0 if is_imputed else np.nan

        m2_dict['donor_blend_fraction'] = df_sub['d_a_ratio'].apply(parse_f_donor).values
        m2_dict['thickness_log10'] = df_sub['active_layer_thickness'].apply(parse_thk_log).values
        m2_dict['annealing_temp_norm'] = df_sub['annealing_temp'].apply(parse_ann_norm).values
        m2_dict['additive_concentration_pct'] = df_sub['additive_ratio'].apply(parse_add_conc).values

        # 7 missingness indicators
        def is_miss_da(val):
            if pd.isna(val): return 1
            s = str(val).strip()
            m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", s)
            return 0 if m else 1

        def is_miss_thk(val):
            if pd.isna(val): return 1
            s = str(val).replace('nm', '').strip()
            try:
                v = float(s)
                if 15.0 <= v <= 600.0: return 0
            except: pass
            return 1

        def is_miss_ann(val):
            if pd.isna(val): return 1
            try:
                v = float(val)
                if 0.0 <= v <= 250.0: return 0
            except: pass
            return 1

        m2_dict['missing_da_ratio'] = df_sub['d_a_ratio'].apply(is_miss_da).values
        m2_dict['missing_thickness'] = df_sub['active_layer_thickness'].apply(is_miss_thk).values
        m2_dict['missing_annealing'] = df_sub['annealing_temp'].apply(is_miss_ann).values
        m2_dict['missing_solvent'] = df_sub['solvent_canonical'].isna().astype(int).values

        def is_miss_add(val):
            if pd.isna(val): return 1
            s = str(val).lower().strip()
            if s in ['none', 'neat', '0', 'no']: return 0
            return 0 if len(s) > 0 else 1

        m2_dict['missing_additive'] = df_sub['additive_canonical'].apply(is_miss_add).values

        def is_miss_arch(val):
            if pd.isna(val): return 1
            s = str(val).lower().strip()
            return 0 if ('conventional' in s or 'inverted' in s) else 1

        m2_dict['missing_architecture'] = df_sub['device_type'].apply(is_miss_arch).values

        def is_miss_inter(row):
            e_nan = pd.isna(row['etl_canonical'])
            h_nan = pd.isna(row['htl_canonical'])
            return 1 if (e_nan or h_nan) else 0

        m2_dict['missing_interlayers'] = df_sub.apply(is_miss_inter, axis=1).values

        # 7 physical regimes
        def reg_as_cast(val):
            if pd.isna(val): return 0
            try:
                return 1 if float(val) <= 35.0 else 0
            except: return 0

        def reg_ann(val):
            if pd.isna(val): return 0
            try:
                v = float(val)
                return 1 if 40.0 <= v <= 250.0 else 0
            except: return 0

        def reg_add(val):
            if pd.isna(val): return 0
            s = str(val).lower().strip()
            if s in ['none', 'neat', '0', 'no']: return 0
            return 1 if len(s) > 0 else 0

        def reg_chlor(val):
            if pd.isna(val): return 0
            s = str(val).lower().strip()
            return 1 if ('chloroform' in s or 'chlorobenzene' in s or 'dcb' in s) else 0

        def reg_green(val):
            if pd.isna(val): return 0
            s = str(val).lower().strip()
            return 1 if ('toluene' in s or 'xylene' in s or 'thf' in s) else 0

        def reg_inv(val):
            if pd.isna(val): return 0
            return 1 if 'inverted' in str(val).lower() else 0

        def reg_conv(val):
            if pd.isna(val): return 0
            return 1 if ('conventional' in str(val).lower() or 'regular' in str(val).lower()) else 0

        m2_dict['regime_as_cast_room_temp'] = df_sub['annealing_temp'].apply(reg_as_cast).values
        m2_dict['regime_thermal_annealed'] = df_sub['annealing_temp'].apply(reg_ann).values
        m2_dict['regime_solvent_additive_used'] = df_sub['additive_canonical'].apply(reg_add).values
        m2_dict['regime_chlorinated_solvent'] = df_sub['solvent_canonical'].apply(reg_chlor).values
        m2_dict['regime_green_solvent'] = df_sub['solvent_canonical'].apply(reg_green).values
        m2_dict['regime_inverted_stack'] = df_sub['device_type'].apply(reg_inv).values
        m2_dict['regime_conventional_stack'] = df_sub['device_type'].apply(reg_conv).values

        # 4 interaction & interface constants
        donor_mw_log = np.log10(np.maximum(df_m0['donor_mw'].values, 1.0))
        m2_dict['thickness_per_donor_mw'] = m2_dict['thickness_log10'] / donor_mw_log

        def get_bp(val):
            if pd.isna(val): return self.bp_median if is_imputed else np.nan
            s = str(val).lower().strip()
            for k, v in SOLV_BP_MAP.items():
                if k in s: return v
            return self.bp_median if is_imputed else np.nan

        def get_etl(val):
            if pd.isna(val): return self.etl_median if is_imputed else np.nan
            s = str(val).lower().strip()
            for k, v in ETL_WF_MAP.items():
                if k in s: return v
            return self.etl_median if is_imputed else np.nan

        def get_htl(val):
            if pd.isna(val): return self.htl_median if is_imputed else np.nan
            s = str(val).lower().strip()
            for k, v in HTL_WF_MAP.items():
                if k in s: return v
            return self.htl_median if is_imputed else np.nan

        m2_dict['solvent_boiling_point_c'] = df_sub['solvent_canonical'].apply(get_bp).values
        m2_dict['etl_workfunction_ev'] = df_sub['etl_canonical'].apply(get_etl).values
        m2_dict['htl_workfunction_ev'] = df_sub['htl_canonical'].apply(get_htl).values

        df_m2 = pd.DataFrame(m2_dict, index=df_sub.index)
        assert df_m2.shape == (len(df_sub), 44), f"M2 shape {df_m2.shape} != ({len(df_sub)}, 44)"
        if is_imputed:
            assert not df_m2.isna().any().any()
        return df_m2

    def extract_m2a(self, df_m2: pd.DataFrame) -> pd.DataFrame:
        m2a_cols = [c for c in df_m2.columns if not c.startswith('missing_')]
        df_m2a = df_m2[m2a_cols].copy()
        assert df_m2a.shape == (len(df_m2), 37), f"M2A shape {df_m2a.shape} != ({len(df_m2)}, 37)"
        return df_m2a


def compute_subgroup_k(df_sub: pd.DataFrame) -> pd.Series:
    has_ratio = df_sub['d_a_ratio'].notna()
    has_solv = df_sub['solvent_canonical'].notna()
    has_thk = df_sub['active_layer_thickness'].notna()
    has_ann = df_sub['annealing_temp'].notna()
    has_arch = df_sub['device_type'].notna() & df_sub['device_type'].str.lower().isin(['conventional', 'inverted'])
    has_add = df_sub['additive_canonical'].notna()
    has_inter = df_sub['etl_canonical'].notna() & df_sub['htl_canonical'].notna()

    fields_df = pd.DataFrame({
        'ratio': has_ratio, 'solvent': has_solv, 'thk': has_thk,
        'ann': has_ann, 'arch': has_arch, 'add': has_add, 'inter': has_inter
    }, index=df_sub.index)

    return fields_df.sum(axis=1)


def get_tier_label(k: int) -> str:
    if k == 0: return 'Molecular-Only (k=0)'
    if k == 1: return 'Sparse (k=1)'
    if k in [2, 3]: return 'Low-Partial (k=2-3)'
    if k in [4, 5]: return 'High-Partial (k=4-5)'
    return 'Rich (k=6-7)'


def cluster_bootstrap_diff(
    y_true: np.ndarray,
    y_pred_m: np.ndarray,
    y_pred_m0: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 8888
) -> Tuple[float, float, float]:
    """Block bootstrap resampling by material-name pair cluster for paired difference in AE.
    Delta = |y - y_pred_m| - |y - y_pred_m0|. Negative means M is better.
    """
    unique_clusters = np.unique(clusters)
    n_c = len(unique_clusters)
    rng = np.random.RandomState(seed)

    # Pre-index
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    ae_m = np.abs(y_true - y_pred_m)
    ae_m0 = np.abs(y_true - y_pred_m0)
    diff = ae_m - ae_m0
    mean_diff = float(np.mean(diff))

    boot_means = []
    for _ in range(n_boot):
        sampled_c = rng.choice(unique_clusters, size=n_c, replace=True)
        sample_idx = np.concatenate([cluster_indices[c] for c in sampled_c])
        boot_means.append(np.mean(diff[sample_idx]))

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_diff, ci_low, ci_high


def main():
    print("=" * 80)
    print("OPV-PredictNet — Phase 8.1: Controlled Processing-Aware Model Experiments")
    print("=" * 80)
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Verification of Dataset & Environment
    print(f"\n[Step 1] Verifying benchmark dataset integrity...")
    actual_sha256 = compute_sha256(BENCHMARK_PATH)
    assert actual_sha256 == EXPECTED_SHA256, f"Dataset SHA-256 mismatch! Got {actual_sha256}"
    df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    assert len(df) == 22467, f"Expected 22,467 records, got {len(df)}"

    pair_col = df['donor_canonical'] + "::" + df['acceptor_canonical']
    df['pair_id'] = pair_col
    unique_pairs = np.sort(df['pair_id'].unique())
    assert len(unique_pairs) == 4730, f"Expected 4,730 pairs, got {len(unique_pairs)}"
    print(f"Verified dataset: {len(df)} records, {len(unique_pairs)} material-name pairs. Checksum: {actual_sha256}")

    # 2. Partitioning (Seed 8888)
    print(f"\n[Step 2] Executing locked group partition (seed {PARTITION_SEED})...")
    rng = np.random.RandomState(PARTITION_SEED)
    shuffled_pairs = rng.permutation(unique_pairs)

    n_train = int(len(unique_pairs) * 0.6)  # 2838
    n_cal = int(len(unique_pairs) * 0.2)    # 946
    n_test = len(unique_pairs) - n_train - n_cal # 946

    train_pairs = set(shuffled_pairs[:n_train])
    cal_pairs = sorted(list(shuffled_pairs[n_train:n_train + n_cal]))
    test_pairs = sorted(list(shuffled_pairs[n_train + n_cal:]))

    # Verify zero pair overlap
    assert len(train_pairs.intersection(set(cal_pairs))) == 0
    assert len(train_pairs.intersection(set(test_pairs))) == 0
    assert len(set(cal_pairs).intersection(set(test_pairs))) == 0

    df_train = df[df['pair_id'].isin(train_pairs)].copy()
    cal_pool = df[df['pair_id'].isin(set(cal_pairs))].copy()
    test_pool = df[df['pair_id'].isin(set(test_pairs))].copy()

    assert len(df_train) == 14704, f"Expected 14,704 train records, got {len(df_train)}"
    assert len(cal_pool) == 3349, f"Expected 3,349 cal records, got {len(cal_pool)}"
    assert len(test_pool) == 4414, f"Expected 4,414 test records, got {len(test_pool)}"

    # Alternative B single-record draws
    def sample_one_per_pair(pool_df, pairs_list, seed):
        rng_s = np.random.RandomState(seed)
        idxs = []
        for p in pairs_list:
            grp_idxs = pool_df[pool_df['pair_id'] == p].index.values
            idxs.append(rng_s.choice(grp_idxs))
        return pool_df.loc[idxs].copy()

    df_cal = sample_one_per_pair(cal_pool, cal_pairs, seed=CAL_SAMPLE_SEED)
    df_test = sample_one_per_pair(test_pool, test_pairs, seed=TEST_SAMPLE_SEED)

    assert len(df_cal) == 946
    assert len(df_test) == 946
    assert len(set(df_cal.index).intersection(set(df_test.index))) == 0
    print(f"Partitions established: Train={len(df_train)} records, Cal={len(df_cal)} records (1/pair), Test={len(df_test)} records (1/pair).")

    # 3. Feature Extraction
    print(f"\n[Step 3] Extracting features and fitting transformations strictly on D_train...")
    pipe = FeaturePipeline()
    pipe.fit_descriptors(df)
    pipe.fit_training(df_train)

    # Extract M0
    X_train_m0 = pipe.extract_m0(df_train)
    X_cal_m0 = pipe.extract_m0(df_cal)
    X_test_m0 = pipe.extract_m0(df_test)

    # Extract M1
    X_train_m1 = pipe.extract_m1(df_train, X_train_m0)
    X_cal_m1 = pipe.extract_m1(df_cal, X_cal_m0)
    X_test_m1 = pipe.extract_m1(df_test, X_test_m0)

    # Extract M2
    X_train_m2 = pipe.extract_m2(df_train, X_train_m0, is_imputed=True)
    X_cal_m2 = pipe.extract_m2(df_cal, X_cal_m0, is_imputed=True)
    X_test_m2 = pipe.extract_m2(df_test, X_test_m0, is_imputed=True)

    # Extract M2A
    X_train_m2a = pipe.extract_m2a(X_train_m2)
    X_cal_m2a = pipe.extract_m2a(X_cal_m2)
    X_test_m2a = pipe.extract_m2a(X_test_m2)

    # Extract M3 (unimputed NaNs preserved)
    X_train_m3 = pipe.extract_m2(df_train, X_train_m0, is_imputed=False)
    X_cal_m3 = pipe.extract_m2(df_cal, X_cal_m0, is_imputed=False)
    X_test_m3 = pipe.extract_m2(df_test, X_test_m0, is_imputed=False)

    y_train = df_train['pce'].values.astype(np.float64)
    y_cal = df_cal['pce'].values.astype(np.float64)
    y_test = df_test['pce'].values.astype(np.float64)

    # Dimension assertions
    print("Feature matrix dimensions verified:")
    print(f"  M0 : Train {X_train_m0.shape}, Cal {X_cal_m0.shape}, Test {X_test_m0.shape}")
    print(f"  M1 : Train {X_train_m1.shape}, Cal {X_cal_m1.shape}, Test {X_test_m1.shape}")
    print(f"  M2 : Train {X_train_m2.shape}, Cal {X_cal_m2.shape}, Test {X_test_m2.shape}")
    print(f"  M2A: Train {X_train_m2a.shape}, Cal {X_cal_m2a.shape}, Test {X_test_m2a.shape}")
    print(f"  M3 : Train {X_train_m3.shape}, Cal {X_cal_m3.shape}, Test {X_test_m3.shape}")

    # 4. Model Training strictly on D_train with n_jobs=1
    print(f"\n[Step 4] Fitting all 5 candidate models strictly on D_train (n_jobs=1, seed={MODEL_SEED})...")
    models = {
        'M0': RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=MODEL_SEED, n_jobs=1),
        'M1': RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=MODEL_SEED, n_jobs=1),
        'M2': RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=MODEL_SEED, n_jobs=1),
        'M2A': RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=MODEL_SEED, n_jobs=1),
        'M3': HistGradientBoostingRegressor(max_iter=100, max_depth=None, min_samples_leaf=20, random_state=MODEL_SEED)
    }

    train_data = {
        'M0': X_train_m0, 'M1': X_train_m1, 'M2': X_train_m2, 'M2A': X_train_m2a, 'M3': X_train_m3
    }
    cal_data = {
        'M0': X_cal_m0, 'M1': X_cal_m1, 'M2': X_cal_m2, 'M2A': X_cal_m2a, 'M3': X_cal_m3
    }
    test_data = {
        'M0': X_test_m0, 'M1': X_test_m1, 'M2': X_test_m2, 'M2A': X_test_m2a, 'M3': X_test_m3
    }

    fit_times = {}
    for m_id, model in models.items():
        t0 = time.time()
        model.fit(train_data[m_id], y_train)
        fit_times[m_id] = time.time() - t0
        print(f"  Fitted {m_id} in {fit_times[m_id]:.2f}s")

    # 5. Split-Conformal Calibration on D_cal (N = 946)
    print(f"\n[Step 5] Calibrating split-conformal intervals on D_cal (N = {len(y_cal)})...")
    alphas = [0.20, 0.10, 0.05]
    conf_levels = ['80%', '90%', '95%']
    cal_quantiles = {}
    cal_scores = {}

    for m_id, model in models.items():
        y_pred_cal = model.predict(cal_data[m_id])
        scores = np.abs(y_cal - y_pred_cal)
        cal_scores[m_id] = scores
        cal_quantiles[m_id] = {}
        for alpha in alphas:
            q = compute_conformal_quantile(scores, alpha)
            cal_quantiles[m_id][alpha] = q
        print(f"  {m_id} Conformal Quantiles: 80%={cal_quantiles[m_id][0.20]:.4f}, 90%={cal_quantiles[m_id][0.10]:.4f}, 95%={cal_quantiles[m_id][0.05]:.4f}")

    # 6. Final-Test Evaluation on D_test (N = 946)
    print(f"\n[Step 6] Evaluating frozen models on held-out test partition D_test (N = {len(y_test)})...")
    test_preds = {}
    model_metrics = []
    interval_metrics = []

    for m_id, model in models.items():
        y_pred_te = model.predict(test_data[m_id])
        test_preds[m_id] = y_pred_te

        mae = float(mean_absolute_error(y_test, y_pred_te))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
        r2 = float(r2_score(y_test, y_pred_te))
        bias = float(np.mean(y_pred_te - y_test))

        model_metrics.append({
            'model_id': m_id,
            'feature_count': train_data[m_id].shape[1],
            'fit_time_seconds': round(fit_times[m_id], 2),
            'test_mae_pce': round(mae, 4),
            'test_rmse_pce': round(rmse, 4),
            'test_r2': round(r2, 4),
            'test_mean_signed_bias': round(bias, 4)
        })

        for alpha in alphas:
            q = cal_quantiles[m_id][alpha]
            conf_str = f"{int(round((1.0 - alpha) * 100))}%"
            # Raw intervals
            lower_raw = y_pred_te - q
            upper_raw = y_pred_te + q
            cov_raw = float(np.mean((y_test >= lower_raw) & (y_test <= upper_raw)))
            widths_raw = upper_raw - lower_raw
            outside_raw = int(np.sum((y_test < lower_raw) | (y_test > upper_raw)))

            # Clipped intervals [0, inf)
            lower_clip = np.maximum(0.0, lower_raw)
            upper_clip = upper_raw
            cov_clip = float(np.mean((y_test >= lower_clip) & (y_test <= upper_clip)))
            widths_clip = upper_clip - lower_clip
            outside_clip = int(np.sum((y_test < lower_clip) | (y_test > upper_clip)))

            interval_metrics.append({
                'model_id': m_id,
                'confidence_level': conf_str,
                'alpha': alpha,
                'calibration_sample_size': len(y_cal),
                'conformal_quantile_q': round(q, 4),
                # Raw unclipped
                'raw_empirical_coverage': round(cov_raw, 4),
                'raw_median_width_pce': round(float(np.median(widths_raw)), 4),
                'raw_mean_width_pce': round(float(np.mean(widths_raw)), 4),
                'raw_iqr_width_pce': round(float(np.percentile(widths_raw, 75) - np.percentile(widths_raw, 25)), 4),
                'raw_outside_count': outside_raw,
                # Clipped [0, inf)
                'clipped_empirical_coverage': round(cov_clip, 4),
                'clipped_median_width_pce': round(float(np.median(widths_clip)), 4),
                'clipped_mean_width_pce': round(float(np.mean(widths_clip)), 4),
                'clipped_outside_count': outside_clip
            })

    df_model_metrics = pd.DataFrame(model_metrics)
    df_interval_metrics = pd.DataFrame(interval_metrics)

    print("\n--- Point Prediction Final-Test Metrics ---")
    print(df_model_metrics.to_string(index=False))

    print("\n--- Conformal Interval Final-Test Metrics ---")
    print(df_interval_metrics[['model_id', 'confidence_level', 'conformal_quantile_q', 'raw_empirical_coverage', 'raw_median_width_pce', 'clipped_empirical_coverage', 'clipped_median_width_pce']].to_string(index=False))

    # 7. Paired Comparisons against M0 on Identical Test Records
    print(f"\n[Step 7] Computing paired difference comparisons against M0 on identical 946 test records...")
    paired_comparisons = []
    y_pred_m0 = test_preds['M0']
    clusters = df_test['pair_id'].values

    for m_id in ['M1', 'M2', 'M2A', 'M3']:
        y_pred_m = test_preds[m_id]
        ae_m = np.abs(y_test - y_pred_m)
        ae_m0 = np.abs(y_test - y_pred_m0)
        diff_ae = ae_m - ae_m0  # negative = M improves on M0

        mean_diff, ci_low, ci_high = cluster_bootstrap_diff(
            y_test, y_pred_m, y_pred_m0, clusters, n_boot=1000, seed=MODEL_SEED
        )

        t_stat, t_pval = stats.ttest_rel(ae_m, ae_m0)
        w_stat, w_pval = stats.wilcoxon(ae_m, ae_m0)

        paired_comparisons.append({
            'comparison': f"{m_id} vs M0",
            'model_id': m_id,
            'baseline_id': 'M0',
            'mean_ae_diff_pce': round(mean_diff, 4),
            'bootstrap_95ci_low': round(ci_low, 4),
            'bootstrap_95ci_high': round(ci_high, 4),
            'paired_ttest_stat': round(float(t_stat), 4),
            'paired_ttest_pvalue': float(f"{t_pval:.4e}"),
            'wilcoxon_stat': round(float(w_stat), 4),
            'wilcoxon_pvalue': float(f"{w_pval:.4e}"),
            'statistically_significant_05': bool(t_pval < 0.05 and ci_high < 0.0)
        })

    df_paired = pd.DataFrame(paired_comparisons)
    print(df_paired.to_string(index=False))

    # 8. Completeness & Subgroup Analysis
    print(f"\n[Step 8] Evaluating subgroup slices across completeness tiers and field counts...")
    k_test = compute_subgroup_k(df_test)
    df_test['k_reported'] = k_test.values
    df_test['tier'] = df_test['k_reported'].apply(get_tier_label)

    # Also PCE range slices
    def get_pce_range(y: float) -> str:
        if y < 5.0: return '< 5.0%'
        if y < 10.0: return '5.0% - 10.0%'
        if y < 15.0: return '10.0% - 15.0%'
        return '>= 15.0%'

    df_test['pce_range'] = [get_pce_range(y) for y in y_test]

    subgroup_metrics = []

    # Completeness Tiers
    tiers = ['Molecular-Only (k=0)', 'Sparse (k=1)', 'Low-Partial (k=2-3)', 'High-Partial (k=4-5)', 'Rich (k=6-7)']
    for tier in tiers:
        idx = np.where(df_test['tier'] == tier)[0]
        n_sub = len(idx)
        if n_sub == 0: continue

        for m_id in ['M0', 'M1', 'M2', 'M2A', 'M3']:
            sub_y = y_test[idx]
            sub_pred = test_preds[m_id][idx]
            sub_mae = float(mean_absolute_error(sub_y, sub_pred))
            sub_bias = float(np.mean(sub_pred - sub_y))
            q90 = cal_quantiles[m_id][0.10]
            sub_cov = float(np.mean((sub_y >= sub_pred - q90) & (sub_y <= sub_pred + q90)))
            sub_width = 2.0 * q90

            subgroup_metrics.append({
                'slice_type': 'Completeness Tier',
                'subgroup_name': tier,
                'sample_size_n': n_sub,
                'model_id': m_id,
                'mae_pce': round(sub_mae, 4),
                'bias_pce': round(sub_bias, 4),
                'cov_90_raw': round(sub_cov, 4),
                'width_90_pce': round(sub_width, 4),
                'statistical_status': 'Evaluated' if n_sub >= 30 else 'Descriptive (Underpowered)'
            })

    # Individual k counts
    for k_val in range(8):
        idx = np.where(df_test['k_reported'] == k_val)[0]
        n_sub = len(idx)
        if n_sub == 0: continue

        for m_id in ['M0', 'M1', 'M2', 'M2A', 'M3']:
            sub_y = y_test[idx]
            sub_pred = test_preds[m_id][idx]
            sub_mae = float(mean_absolute_error(sub_y, sub_pred))
            sub_bias = float(np.mean(sub_pred - sub_y))
            q90 = cal_quantiles[m_id][0.10]
            sub_cov = float(np.mean((sub_y >= sub_pred - q90) & (sub_y <= sub_pred + q90)))

            subgroup_metrics.append({
                'slice_type': 'Field Count (k)',
                'subgroup_name': f"k = {k_val}",
                'sample_size_n': n_sub,
                'model_id': m_id,
                'mae_pce': round(sub_mae, 4),
                'bias_pce': round(sub_bias, 4),
                'cov_90_raw': round(sub_cov, 4),
                'width_90_pce': round(2.0 * q90, 4),
                'statistical_status': 'Evaluated' if n_sub >= 30 else 'Descriptive (Underpowered)'
            })

    # PCE Range slices
    pce_ranges = ['< 5.0%', '5.0% - 10.0%', '10.0% - 15.0%', '>= 15.0%']
    for prange in pce_ranges:
        idx = np.where(df_test['pce_range'] == prange)[0]
        n_sub = len(idx)
        if n_sub == 0: continue

        for m_id in ['M0', 'M1', 'M2', 'M2A', 'M3']:
            sub_y = y_test[idx]
            sub_pred = test_preds[m_id][idx]
            sub_mae = float(mean_absolute_error(sub_y, sub_pred))
            sub_bias = float(np.mean(sub_pred - sub_y))
            q90 = cal_quantiles[m_id][0.10]
            sub_cov = float(np.mean((sub_y >= sub_pred - q90) & (sub_y <= sub_pred + q90)))

            subgroup_metrics.append({
                'slice_type': 'PCE Range',
                'subgroup_name': prange,
                'sample_size_n': n_sub,
                'model_id': m_id,
                'mae_pce': round(sub_mae, 4),
                'bias_pce': round(sub_bias, 4),
                'cov_90_raw': round(sub_cov, 4),
                'width_90_pce': round(2.0 * q90, 4),
                'statistical_status': 'Evaluated' if n_sub >= 30 else 'Descriptive (Underpowered)'
            })

    df_subgroups = pd.DataFrame(subgroup_metrics)

    # 9. Save all artifacts
    print(f"\n[Step 9] Saving all Phase 8.1 results to {RESULTS_DIR}...")
    df_model_metrics.to_csv(os.path.join(RESULTS_DIR, 'phase_8_1_model_metrics.csv'), index=False)
    df_interval_metrics.to_csv(os.path.join(RESULTS_DIR, 'phase_8_1_interval_metrics.csv'), index=False)
    df_paired.to_csv(os.path.join(RESULTS_DIR, 'phase_8_1_paired_comparisons.csv'), index=False)
    df_subgroups.to_csv(os.path.join(RESULTS_DIR, 'phase_8_1_subgroup_metrics.csv'), index=False)

    # Save test predictions DataFrame
    df_test_preds = df_test[['id', 'donor', 'acceptor', 'donor_canonical', 'acceptor_canonical', 'pair_id', 'pce', 'tier', 'k_reported', 'pce_range']].copy()
    for m_id in ['M0', 'M1', 'M2', 'M2A', 'M3']:
        df_test_preds[f'pred_pce_{m_id}'] = test_preds[m_id]
        df_test_preds[f'residual_{m_id}'] = y_test - test_preds[m_id]
        q90 = cal_quantiles[m_id][0.10]
        df_test_preds[f'covered_90_{m_id}'] = ((y_test >= test_preds[m_id] - q90) & (y_test <= test_preds[m_id] + q90)).astype(int)

    df_test_preds.to_csv(os.path.join(RESULTS_DIR, 'phase_8_1_test_predictions.csv'), index=False)

    # Save config and metadata JSON
    config_dict = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'environment': {
            'python': sys.version.replace('\n', ' '),
            'platform': platform.platform(),
            'sklearn': '1.9.1',
            'rdkit': '2026.09.2',
            'numpy': '2.4.6',
            'scipy': '1.17.1',
            'n_jobs': 1
        },
        'dataset_sha256': actual_sha256,
        'seeds': {
            'partition_seed': PARTITION_SEED,
            'cal_sample_seed': CAL_SAMPLE_SEED,
            'test_sample_seed': TEST_SAMPLE_SEED,
            'model_seed': MODEL_SEED
        },
        'partition_sizes': {
            'train_pairs': len(train_pairs),
            'train_records': len(df_train),
            'cal_pairs': len(cal_pairs),
            'cal_records_evaluated': len(df_cal),
            'test_pairs': len(test_pairs),
            'test_records_evaluated': len(df_test)
        },
        'feature_counts': {
            'M0': 22, 'M1': 50, 'M2': 44, 'M2A': 37, 'M3': 44
        },
        'calibration_quantiles_90': {
            m_id: cal_quantiles[m_id][0.10] for m_id in models
        },
        'total_execution_time_seconds': round(time.time() - start_time, 2)
    }

    with open(os.path.join(RESULTS_DIR, 'phase_8_1_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nPhase 8.1 experiments completed successfully in {time.time() - start_time:.2f} seconds.")
    print("=" * 80)


if __name__ == '__main__':
    main()
