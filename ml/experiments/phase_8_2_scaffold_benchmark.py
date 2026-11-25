"""
OPV-PredictNet — Phase 8.2: Structure-Level Generalization Benchmark

Executes the approved Phase 8.2 experimental protocol:
- Strategies:
  - Strategy A: Donor Scaffold Holdout (Seed 7701)
  - Strategy B: Acceptor Scaffold Holdout (Seed 7702)
  - Strategy C1: Strict Global Dual-Scaffold Holdout (Seeds 7701+7702, N=77, descriptive)
  - Strategy C2: Scaffold-Pair Holdout (Seed 7703)
- Models: M0 (22 feats), M1 (50 feats), M2A (37 feats), M3 (44 feats)
- Uncertainty: Independent split-conformal calibration on disjoint D_cal (80%, 90%, 95%)
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
from rdkit.Chem.Scaffolds import MurckoScaffold

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath('.'))
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES

BENCHMARK_PATH = 'ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv'
EXPECTED_SHA256 = 'ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897'
RESULTS_DIR = 'ml/results'
CACHE_PATH = r'C:\Users\DELL\.gemini\antigravity\brain\f06ca069-f524-44da-9901-d290ece58d0a\scratch\scaffold_cache.json'

SEED_A = 7701
SEED_B = 7702
SEED_C2 = 7703
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
    'dcb': 180.5,
    'dichlorobenzene': 180.5,
    'toluene': 110.6,
    'xylene': 138.5,
    'thf': 66.0
}

ETL_WF_MAP = {
    'zno': 4.30,
    'pfn': 4.10,
    'pfn-br': 4.10,
    'pdinn': 4.15,
    'ca': 2.87,
    'tio2': 4.20
}

HTL_WF_MAP = {
    'pedot:pss': 5.00,
    'pedot': 5.00,
    'moo3': 5.30,
    'vox': 5.40,
    'ni': 5.10
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
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
    if 'moo3' in s or 'molybdenum' in s: return 'moo3'
    return 'other'


def map_etl_nominal(val):
    if pd.isna(val): return 'missing'
    s = str(val).lower().strip()
    if 'zno' in s: return 'zno'
    if 'pfn-br' in s: return 'pfn_br'
    if 'pdinn' in s: return 'pdinn'
    if 'ca' in s or 'calcium' in s: return 'ca'
    return 'other'


class FeaturePipeline:
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

        self.da_median = float(da_vals.median()) if da_vals.notna().any() else 0.8333
        self.thk_median = float(thk_vals.median()) if thk_vals.notna().any() else 100.0
        self.tmp_median = float(tmp_vals.median()) if tmp_vals.notna().any() else 100.0
        self.f_donor_median = float(f_vals.median()) if f_vals.notna().any() else 0.4545
        self.thk_log_median = float(thk_log_vals.median()) if thk_log_vals.notna().any() else 2.0
        self.ann_norm_median = float(ann_norm_vals.median()) if ann_norm_vals.notna().any() else 0.0

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

        self.bp_median = float(bp_vals.median()) if bp_vals.notna().any() else 131.6
        self.etl_median = float(etl_vals.median()) if etl_vals.notna().any() else 4.30
        self.htl_median = float(htl_vals.median()) if htl_vals.notna().any() else 5.00

        # Fit OHE strictly on D_train
        cats = pd.DataFrame({
            'solvent': df_train['solvent_canonical'].apply(map_solvent_nominal),
            'additive': df_train['additive_canonical'].apply(map_additive_nominal),
            'architecture': df_train['device_type'].apply(map_arch_nominal),
            'htl': df_train['htl_canonical'].apply(map_htl_nominal),
            'etl': df_train['etl_canonical'].apply(map_etl_nominal),
        })
        self.ohe.fit(cats)

    def extract_m0(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        if len(df_sub) == 0:
            return pd.DataFrame(columns=FEATURE_NAMES, index=df_sub.index)
        vecs = [self.mol_dict[(r.donor_smiles, r.acceptor_smiles)] for r in df_sub[['donor_smiles', 'acceptor_smiles']].itertuples(index=False)]
        df_m0 = pd.DataFrame(vecs, columns=FEATURE_NAMES, index=df_sub.index)
        assert df_m0.shape == (len(df_sub), 22)
        assert not df_m0.isna().any().any()
        return df_m0

    def extract_m1(self, df_sub: pd.DataFrame, df_m0: pd.DataFrame) -> pd.DataFrame:
        if len(df_sub) == 0:
            core_cols = list(FEATURE_NAMES) + ['da_ratio_numeric', 'thickness_nm', 'annealing_temp_c', 'has_additive', 'has_thermal_annealing']
            return pd.DataFrame(columns=core_cols + M1_OHE_COLUMNS, index=df_sub.index)
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

        cats = pd.DataFrame({
            'solvent': df_sub['solvent_canonical'].apply(map_solvent_nominal),
            'additive': df_sub['additive_canonical'].apply(map_additive_nominal),
            'architecture': df_sub['device_type'].apply(map_arch_nominal),
            'htl': df_sub['htl_canonical'].apply(map_htl_nominal),
            'etl': df_sub['etl_canonical'].apply(map_etl_nominal),
        }, index=df_sub.index)

        ohe_arr = self.ohe.transform(cats)
        ohe_cols = list(self.ohe.get_feature_names_out())
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df_sub.index)

        # Harmonize with standard M1_OHE_COLUMNS (in case rare category missing in D_train)
        for col in M1_OHE_COLUMNS:
            if col not in df_ohe.columns:
                df_ohe[col] = 0.0
        df_ohe = df_ohe[M1_OHE_COLUMNS]

        df_m1 = pd.concat([df_m1_core, df_ohe], axis=1)
        assert df_m1.shape == (len(df_sub), 50), f"M1 shape {df_m1.shape} != ({len(df_sub)}, 50)"
        assert not df_m1.isna().any().any()
        return df_m1

    def extract_m2a(self, df_sub: pd.DataFrame, df_m0: pd.DataFrame) -> pd.DataFrame:
        if len(df_sub) == 0:
            m2a_cols = list(FEATURE_NAMES) + [
                'donor_blend_fraction', 'thickness_log10', 'annealing_temp_norm', 'additive_concentration_pct',
                'regime_as_cast_room_temp', 'regime_thermal_annealed', 'regime_solvent_additive_used',
                'regime_chlorinated_solvent', 'regime_green_solvent', 'regime_inverted_stack', 'regime_conventional_stack',
                'thickness_per_donor_mw', 'solvent_boiling_point_c', 'etl_workfunction_ev', 'htl_workfunction_ev'
            ]
            return pd.DataFrame(columns=m2a_cols, index=df_sub.index)
        m2_dict = {col: df_m0[col].values for col in FEATURE_NAMES}

        def parse_f_donor(val):
            if pd.isna(val): return self.f_donor_median
            s = str(val).strip()
            m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", s)
            if m:
                try:
                    d, a = float(m.group(1)), float(m.group(2))
                    if d > 0 and a > 0: return d / (d + a)
                except: pass
            return self.f_donor_median

        def parse_thk_log(val):
            if pd.isna(val): return self.thk_log_median
            s = str(val).replace('nm', '').strip()
            try:
                v = float(s)
                if 15.0 <= v <= 600.0: return np.log10(v)
            except: pass
            return self.thk_log_median

        def parse_ann_norm(val):
            if pd.isna(val): return self.ann_norm_median
            try:
                v = float(val)
                if 40.0 <= v <= 250.0: return (v - 100.0) / 50.0
                if 0.0 <= v <= 35.0: return -1.3
            except: pass
            return self.ann_norm_median

        def parse_add_conc(val):
            if pd.isna(val): return 0.0
            s = str(val).strip()
            m = re.search(r"([0-9\.]+)", s)
            if m:
                try:
                    v = float(m.group(1))
                    if 0.0 <= v <= 10.0: return v
                except: pass
            return 0.0

        m2_dict['donor_blend_fraction'] = df_sub['d_a_ratio'].apply(parse_f_donor).values
        m2_dict['thickness_log10'] = df_sub['active_layer_thickness'].apply(parse_thk_log).values
        m2_dict['annealing_temp_norm'] = df_sub['annealing_temp'].apply(parse_ann_norm).values
        m2_dict['additive_concentration_pct'] = df_sub['additive_ratio'].apply(parse_add_conc).values

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
            if pd.isna(val): return self.bp_median
            s = str(val).lower().strip()
            for k, v in SOLV_BP_MAP.items():
                if k in s: return v
            return self.bp_median

        def get_etl(val):
            if pd.isna(val): return self.etl_median
            s = str(val).lower().strip()
            for k, v in ETL_WF_MAP.items():
                if k in s: return v
            return self.etl_median

        def get_htl(val):
            if pd.isna(val): return self.htl_median
            s = str(val).lower().strip()
            for k, v in HTL_WF_MAP.items():
                if k in s: return v
            return self.htl_median

        m2_dict['solvent_boiling_point_c'] = df_sub['solvent_canonical'].apply(get_bp).values
        m2_dict['etl_workfunction_ev'] = df_sub['etl_canonical'].apply(get_etl).values
        m2_dict['htl_workfunction_ev'] = df_sub['htl_canonical'].apply(get_htl).values

        df_m2a = pd.DataFrame(m2_dict, index=df_sub.index)
        assert df_m2a.shape == (len(df_sub), 37), f"M2A shape {df_m2a.shape} != ({len(df_sub)}, 37)"
        assert not df_m2a.isna().any().any()
        return df_m2a

    def extract_m3(self, df_sub: pd.DataFrame, df_m0: pd.DataFrame) -> pd.DataFrame:
        if len(df_sub) == 0:
            m3_cols = list(FEATURE_NAMES) + [
                'donor_blend_fraction', 'thickness_log10', 'annealing_temp_norm', 'additive_concentration_pct',
                'missing_da_ratio', 'missing_thickness', 'missing_annealing', 'missing_solvent',
                'missing_additive', 'missing_architecture', 'missing_interlayers',
                'regime_as_cast_room_temp', 'regime_thermal_annealed', 'regime_solvent_additive_used',
                'regime_chlorinated_solvent', 'regime_green_solvent', 'regime_inverted_stack', 'regime_conventional_stack',
                'thickness_per_donor_mw', 'solvent_boiling_point_c', 'etl_workfunction_ev', 'htl_workfunction_ev'
            ]
            return pd.DataFrame(columns=m3_cols, index=df_sub.index)
        # M3 has 44 columns, preserving raw NaNs for HistGradientBoosting
        m3_dict = {col: df_m0[col].values for col in FEATURE_NAMES}

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

        def parse_thk_log(val):
            if pd.isna(val): return np.nan
            s = str(val).replace('nm', '').strip()
            try:
                v = float(s)
                if 15.0 <= v <= 600.0: return np.log10(v)
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

        def parse_add_conc(val):
            if pd.isna(val): return np.nan
            s = str(val).strip()
            m = re.search(r"([0-9\.]+)", s)
            if m:
                try:
                    v = float(m.group(1))
                    if 0.0 <= v <= 10.0: return v
                except: pass
            return np.nan

        m3_dict['donor_blend_fraction'] = df_sub['d_a_ratio'].apply(parse_f_donor).values
        m3_dict['thickness_log10'] = df_sub['active_layer_thickness'].apply(parse_thk_log).values
        m3_dict['annealing_temp_norm'] = df_sub['annealing_temp'].apply(parse_ann_norm).values
        m3_dict['additive_concentration_pct'] = df_sub['additive_ratio'].apply(parse_add_conc).values

        # 7 missingness flags
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

        m3_dict['missing_da_ratio'] = df_sub['d_a_ratio'].apply(is_miss_da).values
        m3_dict['missing_thickness'] = df_sub['active_layer_thickness'].apply(is_miss_thk).values
        m3_dict['missing_annealing'] = df_sub['annealing_temp'].apply(is_miss_ann).values
        m3_dict['missing_solvent'] = df_sub['solvent_canonical'].isna().astype(int).values

        def is_miss_add(val):
            if pd.isna(val): return 1
            s = str(val).lower().strip()
            if s in ['none', 'neat', '0', 'no']: return 0
            return 0 if len(s) > 0 else 1

        m3_dict['missing_additive'] = df_sub['additive_canonical'].apply(is_miss_add).values

        def is_miss_arch(val):
            if pd.isna(val): return 1
            s = str(val).lower().strip()
            return 0 if ('conventional' in s or 'inverted' in s) else 1

        m3_dict['missing_architecture'] = df_sub['device_type'].apply(is_miss_arch).values

        def is_miss_inter(row):
            e_nan = pd.isna(row['etl_canonical'])
            h_nan = pd.isna(row['htl_canonical'])
            return 1 if (e_nan or h_nan) else 0

        m3_dict['missing_interlayers'] = df_sub.apply(is_miss_inter, axis=1).values

        # 7 physical regimes
        def reg_as_cast(val):
            if pd.isna(val): return 0
            try: return 1 if float(val) <= 35.0 else 0
            except: return 0

        def reg_ann(val):
            if pd.isna(val): return 0
            try: return 1 if 40.0 <= v <= 250.0 else 0
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

        m3_dict['regime_as_cast_room_temp'] = df_sub['annealing_temp'].apply(reg_as_cast).values
        m3_dict['regime_thermal_annealed'] = df_sub['annealing_temp'].apply(reg_ann).values
        m3_dict['regime_solvent_additive_used'] = df_sub['additive_canonical'].apply(reg_add).values
        m3_dict['regime_chlorinated_solvent'] = df_sub['solvent_canonical'].apply(reg_chlor).values
        m3_dict['regime_green_solvent'] = df_sub['solvent_canonical'].apply(reg_green).values
        m3_dict['regime_inverted_stack'] = df_sub['device_type'].apply(reg_inv).values
        m3_dict['regime_conventional_stack'] = df_sub['device_type'].apply(reg_conv).values

        donor_mw_log = np.log10(np.maximum(df_m0['donor_mw'].values, 1.0))
        m3_dict['thickness_per_donor_mw'] = m3_dict['thickness_log10'] / donor_mw_log

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

        m3_dict['solvent_boiling_point_c'] = df_sub['solvent_canonical'].apply(get_bp).values
        m3_dict['etl_workfunction_ev'] = df_sub['etl_canonical'].apply(get_etl).values
        m3_dict['htl_workfunction_ev'] = df_sub['htl_canonical'].apply(get_htl).values

        df_m3 = pd.DataFrame(m3_dict, index=df_sub.index)
        assert df_m3.shape == (len(df_sub), 44), f"M3 shape {df_m3.shape} != ({len(df_sub)}, 44)"
        return df_m3


def cluster_bootstrap_diff(
    y_true: np.ndarray,
    y_pred_m: np.ndarray,
    y_pred_m0: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 8888
) -> Tuple[float, float, float]:
    unique_clusters = np.unique(clusters)
    n_c = len(unique_clusters)
    rng = np.random.RandomState(seed)

    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    diff = np.abs(y_true - y_pred_m) - np.abs(y_true - y_pred_m0)
    mean_diff = float(np.mean(diff))

    boot_means = []
    for _ in range(n_boot):
        sampled_c = rng.choice(unique_clusters, size=n_c, replace=True)
        sample_idx = np.concatenate([cluster_indices[c] for c in sampled_c])
        boot_means.append(np.mean(diff[sample_idx]))

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_diff, ci_low, ci_high


def run_strategy_benchmark(
    strategy_id: str,
    strategy_name: str,
    df_train: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    pipe: FeaturePipeline,
    cluster_col: str,
    alphas: List[float] = [0.20, 0.10, 0.05]
) -> Dict[str, Any]:
    print(f"\n" + "="*80)
    print(f"Executing {strategy_id}: {strategy_name}")
    print(f"  Train records: {len(df_train)}, Cal records: {len(df_cal)}, Test records: {len(df_test)}")

    pipe.fit_training(df_train)

    X_train_m0 = pipe.extract_m0(df_train)
    X_cal_m0 = pipe.extract_m0(df_cal)
    X_test_m0 = pipe.extract_m0(df_test)

    X_train_m1 = pipe.extract_m1(df_train, X_train_m0)
    X_cal_m1 = pipe.extract_m1(df_cal, X_cal_m0)
    X_test_m1 = pipe.extract_m1(df_test, X_test_m0)

    X_train_m2a = pipe.extract_m2a(df_train, X_train_m0)
    X_cal_m2a = pipe.extract_m2a(df_cal, X_cal_m0)
    X_test_m2a = pipe.extract_m2a(df_test, X_test_m0)

    X_train_m3 = pipe.extract_m3(df_train, X_train_m0)
    X_cal_m3 = pipe.extract_m3(df_cal, X_cal_m0)
    X_test_m3 = pipe.extract_m3(df_test, X_test_m0)

    y_train = df_train['pce'].values.astype(np.float64)
    y_cal = df_cal['pce'].values.astype(np.float64)
    y_test = df_test['pce'].values.astype(np.float64)

    models = {
        'M0': RandomForestRegressor(n_estimators=100, random_state=MODEL_SEED, n_jobs=1),
        'M1': RandomForestRegressor(n_estimators=100, random_state=MODEL_SEED, n_jobs=1),
        'M2A': RandomForestRegressor(n_estimators=100, random_state=MODEL_SEED, n_jobs=1),
        'M3': HistGradientBoostingRegressor(max_iter=100, random_state=MODEL_SEED)
    }

    train_data = {'M0': X_train_m0, 'M1': X_train_m1, 'M2A': X_train_m2a, 'M3': X_train_m3}
    cal_data = {'M0': X_cal_m0, 'M1': X_cal_m1, 'M2A': X_cal_m2a, 'M3': X_cal_m3}
    test_data = {'M0': X_test_m0, 'M1': X_test_m1, 'M2A': X_test_m2a, 'M3': X_test_m3}

    fit_times = {}
    test_preds = {}
    model_metrics = []
    interval_metrics = []
    cal_quantiles = {}

    for m_id, model in models.items():
        t0 = time.time()
        model.fit(train_data[m_id], y_train)
        fit_times[m_id] = time.time() - t0

        # Conformal calibration on D_cal
        if len(df_cal) > 0:
            y_pred_cal = model.predict(cal_data[m_id])
            scores_cal = np.abs(y_cal - y_pred_cal)
            cal_quantiles[m_id] = {alpha: compute_conformal_quantile(scores_cal, alpha) for alpha in alphas}
        else:
            cal_quantiles[m_id] = {alpha: np.nan for alpha in alphas}

        # Predict test
        y_pred_te = model.predict(test_data[m_id])
        test_preds[m_id] = y_pred_te

        mae = float(mean_absolute_error(y_test, y_pred_te))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_te)))
        r2 = float(r2_score(y_test, y_pred_te))
        bias = float(np.mean(y_pred_te - y_test))

        model_metrics.append({
            'strategy_id': strategy_id,
            'strategy': strategy_id,
            'strategy_name': strategy_name,
            'model_id': m_id,
            'feature_count': train_data[m_id].shape[1],
            'fit_time_seconds': round(fit_times[m_id], 2),
            'n_train': len(df_train),
            'n_cal': len(df_cal),
            'n_test': len(y_test),
            'test_n': len(y_test),
            'mae': round(mae, 4),
            'test_mae_pce': round(mae, 4),
            'rmse': round(rmse, 4),
            'test_rmse_pce': round(rmse, 4),
            'r2': round(r2, 4),
            'test_r2': round(r2, 4),
            'bias': round(bias, 4),
            'test_mean_signed_bias': round(bias, 4),
            'pce_mean': round(float(np.mean(y_test)), 4),
            'pce_std': round(float(np.std(y_test)), 4)
        })

        if len(df_cal) > 0:
            for alpha in alphas:
                q = cal_quantiles[m_id][alpha]
                conf_str = f"{int(round((1.0 - alpha) * 100))}%"
                cov = float(np.mean((y_test >= y_pred_te - q) & (y_test <= y_pred_te + q)))
                raw_width = 2.0 * q
                interval_metrics.append({
                    'strategy_id': strategy_id,
                    'strategy': strategy_id,
                    'strategy_name': strategy_name,
                    'model_id': m_id,
                    'confidence_level': conf_str,
                    'alpha': alpha,
                    'conformal_cutoff_q': round(q, 4),
                    'coverage': round(cov, 4),
                    'empirical_coverage': round(cov, 4),
                    'mean_width': round(raw_width, 4),
                    'median_width': round(raw_width, 4),
                    'interval_width_pce': round(raw_width, 4)
                })
        else:
            for alpha in alphas:
                conf_str = f"{int(round((1.0 - alpha) * 100))}%"
                interval_metrics.append({
                    'strategy_id': strategy_id,
                    'strategy': strategy_id,
                    'strategy_name': strategy_name,
                    'model_id': m_id,
                    'confidence_level': conf_str,
                    'alpha': alpha,
                    'conformal_cutoff_q': np.nan,
                    'coverage': np.nan,
                    'empirical_coverage': np.nan,
                    'mean_width': np.nan,
                    'median_width': np.nan,
                    'interval_width_pce': np.nan
                })

    # Paired comparisons against M0
    paired_comparisons = []
    y_pred_m0 = test_preds['M0']
    clusters = df_test[cluster_col].values

    for m_id in ['M1', 'M2A', 'M3']:
        y_pred_m = test_preds[m_id]
        ae_m = np.abs(y_test - y_pred_m)
        ae_m0 = np.abs(y_test - y_pred_m0)

        mean_diff, ci_l, ci_h = cluster_bootstrap_diff(
            y_test, y_pred_m, y_pred_m0, clusters, n_boot=1000, seed=MODEL_SEED
        )
        t_stat, t_pval = stats.ttest_rel(ae_m, ae_m0)
        w_stat, w_pval = stats.wilcoxon(ae_m, ae_m0)

        paired_comparisons.append({
            'strategy_id': strategy_id,
            'strategy': strategy_id,
            'strategy_name': strategy_name,
            'comparison': f"{m_id} vs M0",
            'model_id': m_id,
            'baseline_id': 'M0',
            'mean_ae_diff_pce': round(mean_diff, 4),
            'bootstrap_95ci_low': round(ci_l, 4),
            'bootstrap_95ci_high': round(ci_h, 4),
            'paired_ttest_stat': round(float(t_stat), 4),
            'paired_ttest_pvalue': float(f"{t_pval:.4e}"),
            'wilcoxon_stat': round(float(w_stat), 4),
            'wilcoxon_pvalue': float(f"{w_pval:.4e}")
        })

    # Slices by PCE Range
    subgroup_metrics = []
    pce_bins = [
        ('< 5.0%', y_test < 5.0),
        ('5.0% - 10.0%', (y_test >= 5.0) & (y_test < 10.0)),
        ('10.0% - 15.0%', (y_test >= 10.0) & (y_test < 15.0)),
        ('>= 15.0%', y_test >= 15.0)
    ]

    for pce_label, mask in pce_bins:
        n_slice = int(np.sum(mask))
        if n_slice == 0: continue
        for m_id in ['M0', 'M1', 'M2A', 'M3']:
            y_sl = y_test[mask]
            pred_sl = test_preds[m_id][mask]
            mae_sl = float(np.mean(np.abs(y_sl - pred_sl)))
            bias_sl = float(np.mean(pred_sl - y_sl))
            cov_90 = np.nan
            if len(df_cal) > 0:
                q90 = cal_quantiles[m_id][0.10]
                cov_90 = float(np.mean((y_sl >= pred_sl - q90) & (y_sl <= pred_sl + q90)))

            subgroup_metrics.append({
                'strategy_id': strategy_id,
                'strategy': strategy_id,
                'slice_type': 'PCE Range',
                'slice_name': pce_label,
                'sample_size_n': n_slice,
                'model_id': m_id,
                'mae_pce': round(mae_sl, 4),
                'bias_pce': round(bias_sl, 4),
                'cov_90': round(cov_90, 4) if not np.isnan(cov_90) else np.nan
            })

    return {
        'model_metrics': model_metrics,
        'interval_metrics': interval_metrics,
        'paired_comparisons': paired_comparisons,
        'subgroup_metrics': subgroup_metrics,
        'test_preds': test_preds,
        'cal_quantiles': cal_quantiles
    }


def partition_strategy_a(df: pd.DataFrame, seed: int = 7701) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_d_scaff = df['donor_scaffold'].value_counts()
    top_d_scaff = unique_d_scaff.index[0]
    all_d_scaff = sorted(list(df['donor_scaffold'].unique()))
    all_d_scaff.remove(top_d_scaff)

    rng_a = np.random.RandomState(seed)
    rng_a.shuffle(all_d_scaff)

    n_tot_d = len(all_d_scaff) + 1
    n_tr_d = int(n_tot_d * 0.60)
    n_cal_d = int(n_tot_d * 0.20)

    train_d_scaff = set([top_d_scaff] + all_d_scaff[:n_tr_d - 1])
    cal_d_scaff = set(all_d_scaff[n_tr_d - 1 : n_tr_d - 1 + n_cal_d])
    test_d_scaff = set(all_d_scaff[n_tr_d - 1 + n_cal_d :])

    df_train = df[df['donor_scaffold'].isin(train_d_scaff)].copy()
    df_cal = df[df['donor_scaffold'].isin(cal_d_scaff)].copy()
    df_test = df[df['donor_scaffold'].isin(test_d_scaff)].copy()
    return df_train, df_cal, df_test


def partition_strategy_b(df: pd.DataFrame, seed: int = 7702) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_a_scaff = df['acceptor_scaffold'].value_counts()
    top_a_scaff = unique_a_scaff.index[0]
    all_a_scaff = sorted(list(df['acceptor_scaffold'].unique()))
    all_a_scaff.remove(top_a_scaff)

    rng_b = np.random.RandomState(seed)
    rng_b.shuffle(all_a_scaff)

    n_tot_a = len(all_a_scaff) + 1
    n_tr_a = int(n_tot_a * 0.60)
    n_cal_a = int(n_tot_a * 0.20)

    train_a_scaff = set([top_a_scaff] + all_a_scaff[:n_tr_a - 1])
    cal_a_scaff = set(all_a_scaff[n_tr_a - 1 : n_tr_a - 1 + n_cal_a])
    test_a_scaff = set(all_a_scaff[n_tr_a - 1 + n_cal_a :])

    df_train = df[df['acceptor_scaffold'].isin(train_a_scaff)].copy()
    df_cal = df[df['acceptor_scaffold'].isin(cal_a_scaff)].copy()
    df_test = df[df['acceptor_scaffold'].isin(test_a_scaff)].copy()
    return df_train, df_cal, df_test


def partition_strategy_c1(df: pd.DataFrame, seed_d: int = 7701, seed_a: int = 7702) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_d_scaff = df['donor_scaffold'].value_counts()
    top_d_scaff = unique_d_scaff.index[0]
    all_d_scaff = sorted(list(df['donor_scaffold'].unique()))
    all_d_scaff.remove(top_d_scaff)
    rng_d = np.random.RandomState(seed_d)
    rng_d.shuffle(all_d_scaff)
    n_tot_d = len(all_d_scaff) + 1
    n_tr_d = int(n_tot_d * 0.60)
    n_cal_d = int(n_tot_d * 0.20)
    train_d = set([top_d_scaff] + all_d_scaff[:n_tr_d - 1])
    cal_d = set(all_d_scaff[n_tr_d - 1 : n_tr_d - 1 + n_cal_d])
    test_d = set(all_d_scaff[n_tr_d - 1 + n_cal_d :])

    unique_a_scaff = df['acceptor_scaffold'].value_counts()
    top_a_scaff = unique_a_scaff.index[0]
    all_a_scaff = sorted(list(df['acceptor_scaffold'].unique()))
    all_a_scaff.remove(top_a_scaff)
    rng_a = np.random.RandomState(seed_a)
    rng_a.shuffle(all_a_scaff)
    n_tot_a = len(all_a_scaff) + 1
    n_tr_a = int(n_tot_a * 0.60)
    n_cal_a = int(n_tot_a * 0.20)
    train_a = set([top_a_scaff] + all_a_scaff[:n_tr_a - 1])
    cal_a = set(all_a_scaff[n_tr_a - 1 : n_tr_a - 1 + n_cal_a])
    test_a = set(all_a_scaff[n_tr_a - 1 + n_cal_a :])

    df_train = df[df['donor_scaffold'].isin(train_d) & df['acceptor_scaffold'].isin(train_a)].copy()
    df_cal = df[df['donor_scaffold'].isin(cal_d) & df['acceptor_scaffold'].isin(cal_a)].copy()
    df_test = df[df['donor_scaffold'].isin(test_d) & df['acceptor_scaffold'].isin(test_a)].copy()
    return df_train, df_cal, df_test


def partition_strategy_c2(df: pd.DataFrame, seed: int = 7703) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_sp = np.sort(df['scaffold_pair'].unique())
    rng = np.random.RandomState(seed)
    shuffled_sp = rng.permutation(unique_sp)

    n_tot_sp = len(shuffled_sp)
    n_tr_sp = int(n_tot_sp * 0.60)
    n_cal_sp = int(n_tot_sp * 0.20)

    train_sp = set(shuffled_sp[:n_tr_sp])
    cal_sp = set(shuffled_sp[n_tr_sp : n_tr_sp + n_cal_sp])
    test_sp = set(shuffled_sp[n_tr_sp + n_cal_sp :])

    df_train = df[df['scaffold_pair'].isin(train_sp)].copy()
    df_cal = df[df['scaffold_pair'].isin(cal_sp)].copy()
    df_test = df[df['scaffold_pair'].isin(test_sp)].copy()
    return df_train, df_cal, df_test


def main():
    print("=" * 80)
    print("OPV-PredictNet — Phase 8.2: Structure-Level Generalization Benchmark")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Dataset Verification
    print("\n[Step 1] Verifying dataset integrity...")
    actual_sha = compute_sha256(BENCHMARK_PATH)
    assert actual_sha == EXPECTED_SHA256, f"Checksum mismatch: {actual_sha}"
    df = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    print(f"Dataset verified: {len(df)} records. Checksum: {actual_sha}")

    # 2. Scaffold Cache & Identification
    print("\n[Step 2] Loading Bemis-Murcko scaffolds...")
    with open(CACHE_PATH, 'r') as f:
        scaff_cache = json.load(f)

    df['donor_scaffold'] = df['donor_smiles'].map(scaff_cache['donor_scaffolds'])
    df['acceptor_scaffold'] = df['acceptor_smiles'].map(scaff_cache['acceptor_scaffolds'])
    df['scaffold_pair'] = df['donor_scaffold'] + "::" + df['acceptor_scaffold']
    df['pair_id'] = df['donor_canonical'] + "::" + df['acceptor_canonical']

    print(f"Unique donor scaffolds: {df['donor_scaffold'].nunique()}")
    print(f"Unique acceptor scaffolds: {df['acceptor_scaffold'].nunique()}")
    print(f"Unique scaffold pairs: {df['scaffold_pair'].nunique()}")

    pipe = FeaturePipeline()
    print("\n[Step 3] Pre-extracting molecular descriptors...")
    t0 = time.time()
    pipe.fit_descriptors(df)
    print(f"Molecular descriptors pre-extracted for {len(pipe.mol_dict)} molecular pairs in {time.time()-t0:.2f}s")

    all_model_metrics = []
    all_interval_metrics = []
    all_paired_comparisons = []
    all_subgroup_metrics = []
    saved_test_predictions = {}

    # -------------------------------------------------------------------------
    # STRATEGY A: Donor Scaffold Holdout (Seed 7701)
    # -------------------------------------------------------------------------
    print("\n" + "#"*80)
    print("# STRATEGY A: Donor Scaffold Holdout (Seed 7701)")
    print("#"*80)

    df_train_a, df_cal_a, df_test_a = partition_strategy_a(df, seed=SEED_A)
    assert len(set(df_train_a['donor_scaffold']).intersection(set(df_cal_a['donor_scaffold']))) == 0
    assert len(set(df_train_a['donor_scaffold']).intersection(set(df_test_a['donor_scaffold']))) == 0
    assert len(set(df_cal_a['donor_scaffold']).intersection(set(df_test_a['donor_scaffold']))) == 0

    res_a = run_strategy_benchmark(
        strategy_id='Strategy_A',
        strategy_name='Donor Scaffold Holdout',
        df_train=df_train_a,
        df_cal=df_cal_a,
        df_test=df_test_a,
        pipe=pipe,
        cluster_col='donor_scaffold'
    )
    all_model_metrics.extend(res_a['model_metrics'])
    all_interval_metrics.extend(res_a['interval_metrics'])
    all_paired_comparisons.extend(res_a['paired_comparisons'])
    all_subgroup_metrics.extend(res_a['subgroup_metrics'])

    # -------------------------------------------------------------------------
    # STRATEGY B: Acceptor Scaffold Holdout (Seed 7702)
    # -------------------------------------------------------------------------
    print("\n" + "#"*80)
    print("# STRATEGY B: Acceptor Scaffold Holdout (Seed 7702)")
    print("#"*80)

    df_train_b, df_cal_b, df_test_b = partition_strategy_b(df, seed=SEED_B)
    assert len(set(df_train_b['acceptor_scaffold']).intersection(set(df_cal_b['acceptor_scaffold']))) == 0
    assert len(set(df_train_b['acceptor_scaffold']).intersection(set(df_test_b['acceptor_scaffold']))) == 0
    assert len(set(df_cal_b['acceptor_scaffold']).intersection(set(df_test_b['acceptor_scaffold']))) == 0

    res_b = run_strategy_benchmark(
        strategy_id='Strategy_B',
        strategy_name='Acceptor Scaffold Holdout',
        df_train=df_train_b,
        df_cal=df_cal_b,
        df_test=df_test_b,
        pipe=pipe,
        cluster_col='acceptor_scaffold'
    )
    all_model_metrics.extend(res_b['model_metrics'])
    all_interval_metrics.extend(res_b['interval_metrics'])
    all_paired_comparisons.extend(res_b['paired_comparisons'])
    all_subgroup_metrics.extend(res_b['subgroup_metrics'])

    # -------------------------------------------------------------------------
    # STRATEGY C1: Strict Global Dual-Scaffold Holdout
    # -------------------------------------------------------------------------
    print("\n" + "#"*80)
    print("# STRATEGY C1: Strict Global Dual-Scaffold Holdout (Seed 7701/7702)")
    print("#"*80)

    df_train_c1, df_cal_c1, df_test_c1 = partition_strategy_c1(df, seed_d=SEED_A, seed_a=SEED_B)
    assert len(set(df_train_c1['donor_scaffold']).intersection(set(df_cal_c1['donor_scaffold']))) == 0
    assert len(set(df_train_c1['donor_scaffold']).intersection(set(df_test_c1['donor_scaffold']))) == 0
    assert len(set(df_cal_c1['donor_scaffold']).intersection(set(df_test_c1['donor_scaffold']))) == 0

    assert len(set(df_train_c1['acceptor_scaffold']).intersection(set(df_cal_c1['acceptor_scaffold']))) == 0
    assert len(set(df_train_c1['acceptor_scaffold']).intersection(set(df_test_c1['acceptor_scaffold']))) == 0
    assert len(set(df_cal_c1['acceptor_scaffold']).intersection(set(df_test_c1['acceptor_scaffold']))) == 0

    res_c1 = run_strategy_benchmark(
        strategy_id='Strategy_C1',
        strategy_name='Strict Dual-Scaffold Holdout',
        df_train=df_train_c1,
        df_cal=df_cal_c1,
        df_test=df_test_c1,
        pipe=pipe,
        cluster_col='scaffold_pair'
    )
    all_model_metrics.extend(res_c1['model_metrics'])
    all_interval_metrics.extend(res_c1['interval_metrics'])
    all_paired_comparisons.extend(res_c1['paired_comparisons'])
    all_subgroup_metrics.extend(res_c1['subgroup_metrics'])

    # -------------------------------------------------------------------------
    # STRATEGY C2: Scaffold-Pair Holdout (Seed 7703)
    # -------------------------------------------------------------------------
    print("\n" + "#"*80)
    print("# STRATEGY C2: Scaffold-Pair Holdout (Seed 7703)")
    print("#"*80)

    df_train_c2, df_cal_c2, df_test_c2 = partition_strategy_c2(df, seed=SEED_C2)
    assert len(set(df_train_c2['scaffold_pair']).intersection(set(df_cal_c2['scaffold_pair']))) == 0
    assert len(set(df_train_c2['scaffold_pair']).intersection(set(df_test_c2['scaffold_pair']))) == 0
    assert len(set(df_cal_c2['scaffold_pair']).intersection(set(df_test_c2['scaffold_pair']))) == 0

    res_c2 = run_strategy_benchmark(
        strategy_id='Strategy_C2',
        strategy_name='Scaffold-Pair Holdout',
        df_train=df_train_c2,
        df_cal=df_cal_c2,
        df_test=df_test_c2,
        pipe=pipe,
        cluster_col='scaffold_pair'
    )
    all_model_metrics.extend(res_c2['model_metrics'])
    all_interval_metrics.extend(res_c2['interval_metrics'])
    all_paired_comparisons.extend(res_c2['paired_comparisons'])
    all_subgroup_metrics.extend(res_c2['subgroup_metrics'])

    # -------------------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("[Step 4] Persisting Benchmark Artifacts...")

    df_model_metrics = pd.DataFrame(all_model_metrics)
    df_interval_metrics = pd.DataFrame(all_interval_metrics)
    df_paired_comp = pd.DataFrame(all_paired_comparisons)
    df_subgroup_metrics = pd.DataFrame(all_subgroup_metrics)

    metrics_csv = os.path.join(RESULTS_DIR, 'phase_8_2_scaffold_metrics.csv')
    interval_csv = os.path.join(RESULTS_DIR, 'phase_8_2_interval_metrics.csv')
    paired_csv = os.path.join(RESULTS_DIR, 'phase_8_2_paired_comparisons.csv')
    subgroup_csv = os.path.join(RESULTS_DIR, 'phase_8_2_subgroup_metrics.csv')

    df_model_metrics.to_csv(metrics_csv, index=False)
    df_interval_metrics.to_csv(interval_csv, index=False)
    df_paired_comp.to_csv(paired_csv, index=False)
    df_subgroup_metrics.to_csv(subgroup_csv, index=False)

    print(f"Saved: {metrics_csv}")
    print(f"Saved: {interval_csv}")
    print(f"Saved: {paired_csv}")
    print(f"Saved: {subgroup_csv}")

    # Config JSON
    config_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'sklearn': '1.9.1',
            'rdkit': '2026.09.2',
            'n_jobs': 1
        },
        'dataset_sha256': actual_sha,
        'seeds': {
            'seed_strategy_a': SEED_A,
            'seed_strategy_b': SEED_B,
            'seed_strategy_c2': SEED_C2,
            'model_seed': MODEL_SEED
        },
        'feature_counts': {
            'M0': 22,
            'M1': 50,
            'M2A': 37,
            'M3': 44
        },
        'partition_counts': {
            'Strategy_A': {'train_records': len(df_train_a), 'cal_records': len(df_cal_a), 'test_records': len(df_test_a), 'test_donor_scaffolds': df_test_a['donor_scaffold'].nunique()},
            'Strategy_B': {'train_records': len(df_train_b), 'cal_records': len(df_cal_b), 'test_records': len(df_test_b), 'test_acceptor_scaffolds': df_test_b['acceptor_scaffold'].nunique()},
            'Strategy_C1': {'train_records': len(df_train_c1), 'cal_records': len(df_cal_c1), 'test_records': len(df_test_c1), 'test_scaffold_pairs': df_test_c1['scaffold_pair'].nunique(), 'status': 'strict_dual_scaffold_holdout'},
            'Strategy_C2': {'train_records': len(df_train_c2), 'cal_records': len(df_cal_c2), 'test_records': len(df_test_c2), 'test_scaffold_pairs': df_test_c2['scaffold_pair'].nunique()}
        }
    }

    config_json = os.path.join(RESULTS_DIR, 'phase_8_2_config.json')
    with open(config_json, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved: {config_json}")

    print("\nMaster Model Comparison across Scaffold Strategies:")
    print(df_model_metrics[['strategy_id', 'model_id', 'n_train', 'n_cal', 'n_test', 'mae', 'rmse', 'r2', 'bias']].to_string(index=False))

    print("\nPhase 8.2 Scaffold Generalization Benchmark execution finished successfully.")


if __name__ == '__main__':
    main()
