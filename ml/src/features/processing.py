"""
Authoritative Unified Feature Pipeline for OPV-PredictNet (Phase 8.4).

Consolidates deterministic feature extraction for M0 (22), M1 (50), M2 (44),
M2A (37), and M3 (44) under locked Phase 8.0.2 / 8.0.3 / 8.1 specifications,
eliminating Phase 8.2 script drift and fixing the Phase 8.2 extract_m3
thermal annealing NameError.
"""

from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import OneHotEncoder

from ml.src.features.descriptors import FEATURE_NAMES, extract_features_from_mols


# =============================================================================
# Typed Validation Exceptions
# =============================================================================

class FeaturePipelineError(ValueError):
    """Base error for deterministic feature validation and extraction failures."""


class MissingMolecularIdentityError(FeaturePipelineError):
    """Raised when donor or acceptor SMILES identity is missing or empty."""


class InvalidMolecularInputError(FeaturePipelineError):
    """Raised when SMILES or BigSMILES fails RDKit parsing or sanitization."""


class UnsupportedPolymerOrMixtureError(FeaturePipelineError):
    """Raised when unsupported wildcard (*) or mixture (.) structures are rejected."""


class PhysicalInputValidationError(FeaturePipelineError):
    """Raised when processing parameters violate physical boundaries or invalid types."""


# =============================================================================
# Authoritative Feature Schemas & Ordered Column Lists
# =============================================================================

M0_FEATURE_NAMES: Tuple[str, ...] = tuple(FEATURE_NAMES)

M1_CORE_PROCESSING_NAMES: Tuple[str, ...] = (
    "da_ratio_numeric",
    "thickness_nm",
    "annealing_temp_c",
    "has_additive",
    "has_thermal_annealing",
)

M1_OHE_COLUMNS: Tuple[str, ...] = (
    "solvent_chlorobenzene", "solvent_chloroform", "solvent_missing", "solvent_o-dcb", "solvent_other",
    "additive_cn", "additive_dio", "additive_missing", "additive_other",
    "architecture_conventional", "architecture_inverted", "architecture_missing", "architecture_other",
    "htl_missing", "htl_moo3", "htl_other", "htl_pedot_pss",
    "etl_ca", "etl_missing", "etl_other", "etl_pdinn", "etl_pfn_br", "etl_zno",
)

M1_FEATURE_NAMES: Tuple[str, ...] = M0_FEATURE_NAMES + M1_CORE_PROCESSING_NAMES + M1_OHE_COLUMNS

CONTINUOUS_PROCESSING_NAMES: Tuple[str, ...] = (
    "donor_blend_fraction",
    "thickness_log10",
    "annealing_temp_norm",
    "additive_concentration_pct",
)

MISSINGNESS_INDICATOR_NAMES: Tuple[str, ...] = (
    "missing_da_ratio",
    "missing_thickness",
    "missing_annealing",
    "missing_solvent",
    "missing_additive",
    "missing_architecture",
    "missing_interlayers",
)

PHYSICAL_REGIME_NAMES: Tuple[str, ...] = (
    "regime_as_cast_room_temp",
    "regime_thermal_annealed",
    "regime_solvent_additive_used",
    "regime_chlorinated_solvent",
    "regime_green_solvent",
    "regime_inverted_stack",
    "regime_conventional_stack",
)

INTERFACE_CONSTANT_NAMES: Tuple[str, ...] = (
    "thickness_per_donor_mw",
    "solvent_boiling_point_c",
    "etl_workfunction_ev",
    "htl_workfunction_ev",
)

M2_FEATURE_NAMES: Tuple[str, ...] = (
    M0_FEATURE_NAMES
    + CONTINUOUS_PROCESSING_NAMES
    + MISSINGNESS_INDICATOR_NAMES
    + PHYSICAL_REGIME_NAMES
    + INTERFACE_CONSTANT_NAMES
)

M2A_FEATURE_NAMES: Tuple[str, ...] = (
    M0_FEATURE_NAMES
    + CONTINUOUS_PROCESSING_NAMES
    + PHYSICAL_REGIME_NAMES
    + INTERFACE_CONSTANT_NAMES
)

M3_FEATURE_NAMES: Tuple[str, ...] = M2_FEATURE_NAMES

M3_NAN_ALLOWED_COLUMNS: Tuple[str, ...] = CONTINUOUS_PROCESSING_NAMES + INTERFACE_CONSTANT_NAMES

MODEL_SCHEMAS: Dict[str, Tuple[str, ...]] = {
    "M0": M0_FEATURE_NAMES,
    "M1": M1_FEATURE_NAMES,
    "M2": M2_FEATURE_NAMES,
    "M2A": M2A_FEATURE_NAMES,
    "M3": M3_FEATURE_NAMES,
}

# Locked Phase 8.0.3 / Phase 8.1 Physical Lookup Tables
SOLV_BP_MAP: Dict[str, float] = {
    "chloroform": 61.2,
    "chlorobenzene": 131.6,
    "o-dcb": 180.5,
    "toluene": 110.6,
    "o-xylene": 144.4,
    "thf": 66.0,
}

ETL_WF_MAP: Dict[str, float] = {
    "zno": 4.30,
    "pdinn": 4.00,
    "ca": 2.87,
    "pfn-br": 4.20,
}

HTL_WF_MAP: Dict[str, float] = {
    "pedot": 5.00,
    "moo3": 5.30,
    "2pacz": 5.20,
}


# =============================================================================
# Molecular Validation & Scaffold Extraction
# =============================================================================

def validate_and_parse_smiles(
    smiles: Any,
    role: str = "molecule",
    reject_wildcards_or_mixtures: bool = False,
) -> Tuple[Chem.Mol, str, str, List[str]]:
    """Validate SMILES, canonicalize, extract Bemis-Murcko scaffold, and flag edge cases.

    Args:
        smiles: Raw SMILES input for donor or acceptor.
        role: Human-readable entity role ('donor' or 'acceptor') for error messages.
        reject_wildcards_or_mixtures: If True, raises UnsupportedPolymerOrMixtureError
            when dummy atoms (*) or disconnected fragments (.) are present.

    Returns:
        Tuple of (mol, canonical_smiles, scaffold_id, warnings).
    """
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        raise MissingMolecularIdentityError(f"Missing {role} SMILES identity.")
    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidMolecularInputError(f"Empty or non-string {role} SMILES input.")

    raw = smiles.strip()
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        raise InvalidMolecularInputError(f"Invalid or unparseable {role} SMILES: '{raw}'.")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    warnings: List[str] = []

    has_wildcard = any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
    has_mixture = len(Chem.GetMolFrags(mol)) > 1

    if has_wildcard:
        warnings.append(f"{role.upper()}_POLYMER_WILDCARD_PRESENT")
    if has_mixture:
        warnings.append(f"{role.upper()}_MULTI_COMPONENT_MIXTURE_PRESENT")

    if reject_wildcards_or_mixtures and (has_wildcard or has_mixture):
        raise UnsupportedPolymerOrMixtureError(
            f"Unsupported polymer wildcard (*) or disconnected mixture (.) in {role} SMILES: '{canonical_smiles}'."
        )

    murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if not murcko:
        scaffold_id = f"[NO_RING]:{canonical_smiles}"
        warnings.append(f"{role.upper()}_NON_RING_PSEUDO_SCAFFOLD")
    else:
        scaffold_id = murcko

    return mol, canonical_smiles, scaffold_id, warnings


# =============================================================================
# Nominal Mappers & Physical Lookup Helpers (Canonical Phase 8.1 Definitions)
# =============================================================================

def map_solvent_nominal(val: Any) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).lower().strip()
    if "chloroform" in s:
        return "chloroform"
    if "chlorobenzene" in s and "di" not in s:
        return "chlorobenzene"
    if "dcb" in s or "dichlorobenzene" in s:
        return "o-dcb"
    return "other"


def map_additive_nominal(val: Any) -> str:
    """Locked M1 additive mapper (note: explicit neat/'none' maps to 'missing' per Phase 8.1 schema)."""
    if pd.isna(val):
        return "missing"
    s = str(val).lower().strip()
    if "dio" in s or "diiodooctane" in s:
        return "dio"
    if "cn" in s or "chloronaphthalene" in s:
        return "cn"
    if s in {"none", "neat", "0", "no"}:
        return "missing"
    return "other"


def map_additive_nominal_v2_spec(val: Any) -> str:
    """Proposed v2 additive mapper separating explicit absence ('absent_explicit') from unreported ('missing').

    NOT used in the locked 50-feature M1 schema; requires a new model version and evaluation protocol.
    """
    if pd.isna(val):
        return "missing"
    s = str(val).lower().strip()
    if not s:
        raise PhysicalInputValidationError("Empty string is not a valid categorical additive input.")
    if "dio" in s or "diiodooctane" in s:
        return "dio"
    if "cn" in s or "chloronaphthalene" in s:
        return "cn"
    if s in {"none", "neat", "0", "no", "additive-free"}:
        return "absent_explicit"
    return "other"


def map_arch_nominal(val: Any) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).lower().strip()
    if "conventional" in s or "regular" in s or "standard" in s:
        return "conventional"
    if "inverted" in s:
        return "inverted"
    return "other"


def map_htl_nominal(val: Any) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).lower().strip()
    if "pedot" in s:
        return "pedot_pss"
    if "moo3" in s or "mox" in s or "molybdenum" in s:
        return "moo3"
    return "other"


def map_etl_nominal(val: Any) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).lower().strip()
    if "zno" in s:
        return "zno"
    if "pdinn" in s:
        return "pdinn"
    if s == "ca" or "calcium" in s:
        return "ca"
    if "pfn" in s:
        return "pfn_br"
    return "other"


def _lookup_physical_constant(val: Any, mapping: Dict[str, float]) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    for key, constant in mapping.items():
        if key in s:
            return constant
    return np.nan


# =============================================================================
# Numeric Parsers & Validation Helpers (Shared across M1, M2, M2A, M3)
# =============================================================================

def parse_da_ratio(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", str(val).strip())
    if m:
        try:
            d, a = float(m.group(1)), float(m.group(2))
            if a > 0.0:
                return d / a
        except ValueError:
            pass
    return np.nan


def parse_f_donor(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", str(val).strip())
    if m:
        try:
            d, a = float(m.group(1)), float(m.group(2))
            if d > 0.0 and a > 0.0:
                return d / (d + a)
        except ValueError:
            pass
    return np.nan


def parse_thickness_nm(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).replace("nm", "").strip()
    try:
        v = float(s)
        if 15.0 <= v <= 600.0:
            return v
    except ValueError:
        pass
    return np.nan


def parse_thickness_log10(val: Any) -> float:
    v = parse_thickness_nm(val)
    return float(np.log10(v)) if pd.notna(v) and v > 0.0 else np.nan


def parse_annealing_c(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    try:
        v = float(val)
        if 0.0 <= v <= 250.0:
            return v
    except (ValueError, TypeError):
        pass
    return np.nan


def parse_annealing_norm(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    try:
        v = float(val)
        if 40.0 <= v <= 250.0:
            return (v - 100.0) / 50.0
        if 0.0 <= v <= 35.0:
            return -1.3
    except (ValueError, TypeError):
        pass
    return np.nan


def parse_additive_conc(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    m = re.search(r"([0-9\.]+)", str(val).strip())
    if m:
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 10.0:
                return v
        except ValueError:
            pass
    return np.nan


def is_thermal_annealed_regime(val: Any) -> int:
    """Authoritative thermal annealing regime check (fixes Phase 8.2 extract_m3 NameError)."""
    if pd.isna(val):
        return 0
    try:
        v = float(val)
        return 1 if 40.0 <= v <= 250.0 else 0
    except (ValueError, TypeError):
        return 0


def is_as_cast_regime(val: Any) -> int:
    if pd.isna(val):
        return 0
    try:
        v = float(val)
        return 1 if 0.0 <= v <= 35.0 else 0
    except (ValueError, TypeError):
        return 0


def validate_processing_record(
    record: Dict[str, Any],
    strict_categorical_types: bool = True,
) -> List[str]:
    """Validate single-record processing inputs against physical impossibility and type bounds.

    Never silently clips or coerces impossible values; raises PhysicalInputValidationError on
    Tier 1/2 violations and returns explicit Tier 3 domain-range warnings when valid but unusual.
    """
    warnings: List[str] = []

    # Thickness check
    thk = record.get("active_layer_thickness")
    if thk is not None and not (isinstance(thk, float) and np.isnan(thk)):
        s_thk = str(thk).replace("nm", "").strip()
        try:
            v_thk = float(s_thk)
        except ValueError as exc:
            raise PhysicalInputValidationError(f"Unparseable active_layer_thickness: '{thk}'.") from exc
        if not np.isfinite(v_thk) or v_thk <= 0.0 or v_thk > 2000.0:
            raise PhysicalInputValidationError(
                f"Physically impossible active_layer_thickness ({v_thk} nm); expected (0, 2000] nm."
            )
        if v_thk < 15.0 or v_thk > 600.0:
            warnings.append("THICKNESS_OUT_OF_TRAINING_RANGE_WARNING")

    # Annealing temperature check
    tmp = record.get("annealing_temp")
    if tmp is not None and not (isinstance(tmp, float) and np.isnan(tmp)):
        try:
            v_tmp = float(tmp)
        except (ValueError, TypeError) as exc:
            raise PhysicalInputValidationError(f"Unparseable annealing_temp: '{tmp}'.") from exc
        if not np.isfinite(v_tmp) or v_tmp < -273.15 or v_tmp > 350.0:
            raise PhysicalInputValidationError(
                f"Physically impossible annealing_temp ({v_tmp} C); expected [-273.15, 350.0] C."
            )
        if v_tmp < 0.0 or v_tmp > 250.0 or (35.0 < v_tmp < 40.0):
            warnings.append("ANNEALING_TEMP_OUT_OF_TRAINING_RANGE_WARNING")

    # D:A ratio check
    da = record.get("d_a_ratio")
    if da is not None and not (isinstance(da, float) and np.isnan(da)):
        m = re.match(r"^([0-9\.]+)\s*[:/]\s*([0-9\.]+)$", str(da).strip())
        if not m:
            raise PhysicalInputValidationError(f"Malformed d_a_ratio '{da}'; expected 'D:A' ratio.")
        d_val, a_val = float(m.group(1)), float(m.group(2))
        if d_val <= 0.0 or a_val <= 0.0:
            raise PhysicalInputValidationError(f"D:A ratio components must be > 0; got '{da}'.")

    # Additive concentration check
    add_r = record.get("additive_ratio")
    if add_r is not None and not (isinstance(add_r, float) and np.isnan(add_r)):
        m = re.search(r"([+-]?[0-9\.]+)", str(add_r).strip())
        if not m:
            raise PhysicalInputValidationError(f"Unparseable additive_ratio: '{add_r}'.")
        v_add = float(m.group(1))
        if v_add < 0.0 or v_add > 100.0:
            raise PhysicalInputValidationError(f"Physically impossible additive_ratio ({v_add}%); expected [0, 100].")
        if v_add > 10.0:
            warnings.append("ADDITIVE_CONCENTRATION_OUT_OF_TRAINING_RANGE_WARNING")

    # Categorical type checks
    if strict_categorical_types:
        cat_fields = ("solvent_canonical", "additive_canonical", "device_type", "htl_canonical", "etl_canonical")
        for field in cat_fields:
            val = record.get(field)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if not isinstance(val, str) or not val.strip() or val.strip().lower() == "nan":
                raise PhysicalInputValidationError(
                    f"Invalid categorical input for '{field}': {val!r}; expected non-empty string or null."
                )

    return warnings


# =============================================================================
# Unified Authoritative Feature Pipeline
# =============================================================================

class UnifiedFeaturePipeline:
    """Authoritative shared feature extractor for M0 (22), M1 (50), M2 (44), M2A (37), M3 (44).

    Fitted strictly on D_train or hydrated from frozen pipeline_state dictionaries.
    Zero calibration or test records may influence fitted medians or encoders.
    """

    def __init__(self) -> None:
        self.da_median: float = 0.8333333333333334
        self.thk_median: float = 100.0
        self.tmp_median: float = 100.0
        self.f_donor_median: float = 0.45454545454545453
        self.thk_log_median: float = 2.0
        self.ann_norm_median: float = 0.0
        self.bp_median: float = 131.6
        self.etl_median: float = 4.30
        self.htl_median: float = 5.00
        self.ohe: OneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.mol_dict: Dict[Tuple[str, str], np.ndarray] = {}
        self._is_fitted: bool = False

    def fit_descriptors(self, df_all: pd.DataFrame, reject_wildcards_or_mixtures: bool = False) -> None:
        """Deterministically extract and cache 22 RDKit molecular descriptors per SMILES pair."""
        unique_donors = df_all["donor_smiles"].unique()
        unique_acceptors = df_all["acceptor_smiles"].unique()

        d_mols = {}
        for s in unique_donors:
            mol, _, _, _ = validate_and_parse_smiles(
                s, role="donor", reject_wildcards_or_mixtures=reject_wildcards_or_mixtures
            )
            d_mols[s] = mol

        a_mols = {}
        for s in unique_acceptors:
            mol, _, _, _ = validate_and_parse_smiles(
                s, role="acceptor", reject_wildcards_or_mixtures=reject_wildcards_or_mixtures
            )
            a_mols[s] = mol

        pairs = df_all[["donor_smiles", "acceptor_smiles"]].drop_duplicates()
        for _, row in pairs.iterrows():
            vec, _ = extract_features_from_mols(d_mols[row["donor_smiles"]], a_mols[row["acceptor_smiles"]])
            self.mol_dict[(row["donor_smiles"], row["acceptor_smiles"])] = vec

    def fit_training(self, df_train: pd.DataFrame) -> None:
        """Fit continuous imputation medians and M1 OneHotEncoder strictly on D_train."""
        da_vals = df_train["d_a_ratio"].apply(parse_da_ratio)
        thk_vals = df_train["active_layer_thickness"].apply(parse_thickness_nm)
        tmp_vals = df_train["annealing_temp"].apply(parse_annealing_c)
        f_vals = df_train["d_a_ratio"].apply(parse_f_donor)
        thk_log_vals = thk_vals.apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)
        ann_norm_vals = df_train["annealing_temp"].apply(parse_annealing_norm)

        self.da_median = float(da_vals.median()) if da_vals.notna().any() else 0.8333333333333334
        self.thk_median = float(thk_vals.median()) if thk_vals.notna().any() else 100.0
        self.tmp_median = float(tmp_vals.median()) if tmp_vals.notna().any() else 100.0
        self.f_donor_median = float(f_vals.median()) if f_vals.notna().any() else 0.45454545454545453
        self.thk_log_median = float(thk_log_vals.median()) if thk_log_vals.notna().any() else 2.0
        self.ann_norm_median = float(ann_norm_vals.median()) if ann_norm_vals.notna().any() else 0.0

        bp_vals = df_train["solvent_canonical"].apply(lambda x: _lookup_physical_constant(x, SOLV_BP_MAP))
        etl_vals = df_train["etl_canonical"].apply(lambda x: _lookup_physical_constant(x, ETL_WF_MAP))
        htl_vals = df_train["htl_canonical"].apply(lambda x: _lookup_physical_constant(x, HTL_WF_MAP))

        self.bp_median = float(bp_vals.median()) if bp_vals.notna().any() else 131.6
        self.etl_median = float(etl_vals.median()) if etl_vals.notna().any() else 4.30
        self.htl_median = float(htl_vals.median()) if htl_vals.notna().any() else 5.00

        cats = self._build_nominal_df(df_train)
        self.ohe.fit(cats)
        self._is_fitted = True

    @staticmethod
    def _build_nominal_df(df_sub: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "solvent": df_sub["solvent_canonical"].apply(map_solvent_nominal),
                "additive": df_sub["additive_canonical"].apply(map_additive_nominal),
                "architecture": df_sub["device_type"].apply(map_arch_nominal),
                "htl": df_sub["htl_canonical"].apply(map_htl_nominal),
                "etl": df_sub["etl_canonical"].apply(map_etl_nominal),
            },
            index=df_sub.index,
        )

    def _ensure_m0(self, df_sub: pd.DataFrame, df_m0: Optional[pd.DataFrame]) -> pd.DataFrame:
        return df_m0 if df_m0 is not None else self.extract_m0(df_sub)

    def extract_m0(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        if len(df_sub) == 0:
            return pd.DataFrame(columns=list(M0_FEATURE_NAMES), index=df_sub.index, dtype=np.float64)

        vecs: List[np.ndarray] = []
        for row in df_sub[["donor_smiles", "acceptor_smiles"]].itertuples(index=False):
            key = (row.donor_smiles, row.acceptor_smiles)
            if key in self.mol_dict:
                vecs.append(self.mol_dict[key])
            else:
                d_mol, _, _, _ = validate_and_parse_smiles(row.donor_smiles, role="donor")
                a_mol, _, _, _ = validate_and_parse_smiles(row.acceptor_smiles, role="acceptor")
                vec, _ = extract_features_from_mols(d_mol, a_mol)
                self.mol_dict[key] = vec
                vecs.append(vec)

        df_m0 = pd.DataFrame(vecs, columns=list(M0_FEATURE_NAMES), index=df_sub.index, dtype=np.float64)
        self._assert_contract("M0", df_m0, allow_nans=False)
        return df_m0

    def extract_m1(self, df_sub: pd.DataFrame, df_m0: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._is_fitted:
            raise FeaturePipelineError("Pipeline must be fitted on D_train before calling extract_m1.")
        if len(df_sub) == 0:
            return pd.DataFrame(columns=list(M1_FEATURE_NAMES), index=df_sub.index, dtype=np.float64)

        m0_frame = self._ensure_m0(df_sub, df_m0)
        m1_dict: Dict[str, np.ndarray] = {col: m0_frame[col].values for col in M0_FEATURE_NAMES}

        m1_dict["da_ratio_numeric"] = (
            df_sub["d_a_ratio"].apply(parse_da_ratio).fillna(self.da_median).values.astype(np.float64)
        )
        m1_dict["thickness_nm"] = (
            df_sub["active_layer_thickness"].apply(parse_thickness_nm).fillna(self.thk_median).values.astype(np.float64)
        )
        m1_dict["annealing_temp_c"] = (
            df_sub["annealing_temp"].apply(parse_annealing_c).fillna(self.tmp_median).values.astype(np.float64)
        )

        def has_add(val: Any) -> int:
            if pd.isna(val):
                return 0
            s = str(val).lower().strip()
            if s in {"none", "neat", "0", "no"}:
                return 0
            return 1 if len(s) > 0 else 0

        def has_ann(val: Any) -> int:
            if pd.isna(val):
                return 0
            try:
                return 1 if float(val) > 35.0 else 0
            except (ValueError, TypeError):
                return 0

        m1_dict["has_additive"] = df_sub["additive_canonical"].apply(has_add).values.astype(np.int64)
        m1_dict["has_thermal_annealing"] = df_sub["annealing_temp"].apply(has_ann).values.astype(np.int64)

        df_core = pd.DataFrame(m1_dict, index=df_sub.index)
        cats = self._build_nominal_df(df_sub)
        ohe_arr = self.ohe.transform(cats)
        ohe_cols = list(self.ohe.get_feature_names_out())
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df_sub.index, dtype=np.float64)

        for col in M1_OHE_COLUMNS:
            if col not in df_ohe.columns:
                df_ohe[col] = 0.0
        df_ohe = df_ohe[list(M1_OHE_COLUMNS)]

        df_m1 = pd.concat([df_core, df_ohe], axis=1)[list(M1_FEATURE_NAMES)]
        self._assert_contract("M1", df_m1, allow_nans=False)
        return df_m1

    def _build_processing_frame(
        self,
        model_id: str,
        df_sub: pd.DataFrame,
        df_m0: Optional[pd.DataFrame],
        impute_missing: bool,
        include_missing_flags: bool,
    ) -> pd.DataFrame:
        target_cols = list(MODEL_SCHEMAS[model_id])
        if len(df_sub) == 0:
            return pd.DataFrame(columns=target_cols, index=df_sub.index, dtype=np.float64)

        m0_frame = self._ensure_m0(df_sub, df_m0)
        data: Dict[str, np.ndarray] = {col: m0_frame[col].values for col in M0_FEATURE_NAMES}

        # 4 continuous normalized features
        f_donor = df_sub["d_a_ratio"].apply(parse_f_donor)
        thk_log = df_sub["active_layer_thickness"].apply(parse_thickness_log10)
        ann_norm = df_sub["annealing_temp"].apply(parse_annealing_norm)
        add_conc = df_sub["additive_ratio"].apply(parse_additive_conc)

        if impute_missing:
            f_donor = f_donor.fillna(self.f_donor_median)
            thk_log = thk_log.fillna(self.thk_log_median)
            ann_norm = ann_norm.fillna(self.ann_norm_median)
            add_conc = add_conc.fillna(0.0)

        data["donor_blend_fraction"] = f_donor.values.astype(np.float64)
        data["thickness_log10"] = thk_log.values.astype(np.float64)
        data["annealing_temp_norm"] = ann_norm.values.astype(np.float64)
        data["additive_concentration_pct"] = add_conc.values.astype(np.float64)

        # 7 missingness flags (for M2 and M3)
        if include_missing_flags:
            data["missing_da_ratio"] = df_sub["d_a_ratio"].apply(
                lambda v: 0 if pd.notna(parse_f_donor(v)) else 1
            ).values.astype(np.int64)
            data["missing_thickness"] = df_sub["active_layer_thickness"].apply(
                lambda v: 0 if pd.notna(parse_thickness_nm(v)) else 1
            ).values.astype(np.int64)
            data["missing_annealing"] = df_sub["annealing_temp"].apply(
                lambda v: 0 if pd.notna(parse_annealing_c(v)) else 1
            ).values.astype(np.int64)
            data["missing_solvent"] = df_sub["solvent_canonical"].isna().astype(np.int64).values

            def is_miss_add(val: Any) -> int:
                if pd.isna(val):
                    return 1
                s = str(val).lower().strip()
                if s in {"none", "neat", "0", "no"}:
                    return 0
                return 0 if len(s) > 0 else 1

            def is_miss_arch(val: Any) -> int:
                if pd.isna(val):
                    return 1
                s = str(val).lower().strip()
                return 0 if ("conventional" in s or "inverted" in s) else 1

            data["missing_additive"] = df_sub["additive_canonical"].apply(is_miss_add).values.astype(np.int64)
            data["missing_architecture"] = df_sub["device_type"].apply(is_miss_arch).values.astype(np.int64)
            data["missing_interlayers"] = df_sub.apply(
                lambda r: 1 if (pd.isna(r["etl_canonical"]) or pd.isna(r["htl_canonical"])) else 0, axis=1
            ).values.astype(np.int64)

        # 7 physical regimes (strictly non-imputed binary {0,1}; missing stays 0)
        def reg_add(val: Any) -> int:
            if pd.isna(val):
                return 0
            s = str(val).lower().strip()
            if s in {"none", "neat", "0", "no"}:
                return 0
            return 1 if len(s) > 0 else 0

        data["regime_as_cast_room_temp"] = df_sub["annealing_temp"].apply(is_as_cast_regime).values.astype(np.int64)
        data["regime_thermal_annealed"] = df_sub["annealing_temp"].apply(is_thermal_annealed_regime).values.astype(np.int64)
        data["regime_solvent_additive_used"] = df_sub["additive_canonical"].apply(reg_add).values.astype(np.int64)
        data["regime_chlorinated_solvent"] = df_sub["solvent_canonical"].apply(
            lambda v: 0 if pd.isna(v) else (1 if any(k in str(v).lower().strip() for k in ("chloroform", "chlorobenzene", "dcb")) else 0)
        ).values.astype(np.int64)
        data["regime_green_solvent"] = df_sub["solvent_canonical"].apply(
            lambda v: 0 if pd.isna(v) else (1 if any(k in str(v).lower().strip() for k in ("toluene", "xylene", "thf")) else 0)
        ).values.astype(np.int64)
        data["regime_inverted_stack"] = df_sub["device_type"].apply(
            lambda v: 0 if pd.isna(v) else (1 if "inverted" in str(v).lower() else 0)
        ).values.astype(np.int64)
        data["regime_conventional_stack"] = df_sub["device_type"].apply(
            lambda v: 0 if pd.isna(v) else (1 if ("conventional" in str(v).lower() or "regular" in str(v).lower()) else 0)
        ).values.astype(np.int64)

        # 4 interaction & interface constants
        donor_mw_log = np.log10(np.maximum(m0_frame["donor_mw"].values, 1.0))
        data["thickness_per_donor_mw"] = (data["thickness_log10"] / donor_mw_log).astype(np.float64)

        bp_series = df_sub["solvent_canonical"].apply(lambda x: _lookup_physical_constant(x, SOLV_BP_MAP))
        etl_series = df_sub["etl_canonical"].apply(lambda x: _lookup_physical_constant(x, ETL_WF_MAP))
        htl_series = df_sub["htl_canonical"].apply(lambda x: _lookup_physical_constant(x, HTL_WF_MAP))

        if impute_missing:
            bp_series = bp_series.fillna(self.bp_median)
            etl_series = etl_series.fillna(self.etl_median)
            htl_series = htl_series.fillna(self.htl_median)

        data["solvent_boiling_point_c"] = bp_series.values.astype(np.float64)
        data["etl_workfunction_ev"] = etl_series.values.astype(np.float64)
        data["htl_workfunction_ev"] = htl_series.values.astype(np.float64)

        df_out = pd.DataFrame(data, index=df_sub.index)[target_cols]
        self._assert_contract(model_id, df_out, allow_nans=(not impute_missing))
        return df_out

    def extract_m2(self, df_sub: pd.DataFrame, df_m0: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self._build_processing_frame("M2", df_sub, df_m0, impute_missing=True, include_missing_flags=True)

    def extract_m2a(self, df_sub: pd.DataFrame, df_m0: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self._build_processing_frame("M2A", df_sub, df_m0, impute_missing=True, include_missing_flags=False)

    def extract_m3(self, df_sub: pd.DataFrame, df_m0: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self._build_processing_frame("M3", df_sub, df_m0, impute_missing=False, include_missing_flags=True)

    @staticmethod
    def _assert_contract(model_id: str, df_feat: pd.DataFrame, allow_nans: bool) -> None:
        expected_cols = list(MODEL_SCHEMAS[model_id])
        assert list(df_feat.columns) == expected_cols, f"{model_id} column mismatch."
        assert df_feat.shape[1] == len(expected_cols), f"{model_id} width {df_feat.shape[1]} != {len(expected_cols)}."
        assert not np.isinf(df_feat.to_numpy(dtype=np.float64)).any(), f"{model_id} contains +/-inf."

        if not allow_nans:
            assert not df_feat.isna().any().any(), f"{model_id} must not contain NaN values."
        else:
            allowed_nan_set = set(M3_NAN_ALLOWED_COLUMNS)
            for col in df_feat.columns:
                if col not in allowed_nan_set:
                    assert not df_feat[col].isna().any(), f"{model_id} column '{col}' contains prohibited NaN."

        for col in df_feat.columns:
            if col.startswith(("missing_", "regime_", "has_")):
                vals = set(df_feat[col].unique())
                assert vals.issubset({0, 1}), f"{model_id} binary column '{col}' has invalid values: {vals}."

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize fitted training state (medians and OHE categories) for reproducible inference."""
        if not self._is_fitted:
            raise FeaturePipelineError("Cannot export pipeline state before fitting on D_train.")
        return {
            "da_median": self.da_median,
            "thk_median": self.thk_median,
            "tmp_median": self.tmp_median,
            "f_donor_median": self.f_donor_median,
            "thk_log_median": self.thk_log_median,
            "ann_norm_median": self.ann_norm_median,
            "bp_median": self.bp_median,
            "etl_median": self.etl_median,
            "htl_median": self.htl_median,
            "ohe_categories": [list(cats) for cats in self.ohe.categories_],
            "ohe_feature_names": list(self.ohe.get_feature_names_out()),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "UnifiedFeaturePipeline":
        """Restore frozen training state without accessing raw training CSV data."""
        pipe = cls()
        pipe.da_median = float(state["da_median"])
        pipe.thk_median = float(state["thk_median"])
        pipe.tmp_median = float(state["tmp_median"])
        pipe.f_donor_median = float(state["f_donor_median"])
        pipe.thk_log_median = float(state["thk_log_median"])
        pipe.ann_norm_median = float(state["ann_norm_median"])
        pipe.bp_median = float(state["bp_median"])
        pipe.etl_median = float(state["etl_median"])
        pipe.htl_median = float(state["htl_median"])

        # Reconstruct fitted OneHotEncoder deterministically from frozen category lists
        dummy_rows = pd.DataFrame(
            {
                col: [cats[i % len(cats)] for i in range(max(len(c) for c in state["ohe_categories"]))]
                for col, cats in zip(["solvent", "additive", "architecture", "htl", "etl"], state["ohe_categories"])
            }
        )
        pipe.ohe = OneHotEncoder(
            categories=[np.array(c, dtype=object) for c in state["ohe_categories"]],
            sparse_output=False,
            handle_unknown="ignore",
        )
        pipe.ohe.fit(dummy_rows)
        pipe._is_fitted = True
        return pipe
