# OPV-PredictNet — Phase 8.4: Authoritative Feature Pipeline Specification

**Date:** 2026-11-25  
**Implementation Module:** [processing.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/features/processing.py) & [descriptors.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/features/descriptors.py)  
**Scope:** Single authoritative, deterministic feature extraction specification shared across future training, evaluation, and inference for **M0 (22), M1 (50), M2A (37), and M3 (44)** (plus archived **M2 [44]**).

---

## 1. Architecture & Shared Design

Prior to Phase 8.4, processing feature engineering was duplicated across `ml/experiments/phase_8_1_processing_aware_models.py` and `ml/experiments/phase_8_2_scaffold_benchmark.py`, introducing lookup drift and a silent `NameError` in Phase 8.2's `extract_m3` thermal-annealing regime helper.

`UnifiedFeaturePipeline` in [ml/src/features/processing.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/features/processing.py) consolidates all feature logic into one module with:
1. **Locked Schema Constants:** `M0_FEATURE_NAMES` (22), `M1_FEATURE_NAMES` (50), `M2_FEATURE_NAMES` (44), `M2A_FEATURE_NAMES` (37), `M3_FEATURE_NAMES` (44), and `M3_NAN_ALLOWED_COLUMNS` (8).
2. **Single Shared Processing Builder:** `_build_processing_frame()` parameterizes `impute_missing` and `include_missing_flags` across M2, M2A, and M3—eliminating copy-paste divergence.
3. **Training/Inference State Serialization:** `to_state_dict()` and `from_state_dict()` freeze exact $D_{\text{train}}$ imputation medians and `OneHotEncoder` categories for bit-identical inference without touching raw training CSVs.
4. **Explicit Input Validation Without Silent Coercion:**
   - `validate_and_parse_smiles()` rejects missing, empty, or unparseable SMILES/BigSMILES (`MissingMolecularIdentityError`, `InvalidMolecularInputError`), assigns deterministic `[NO_RING]:<canonical_smiles>` for ringless molecules, and detects wildcard `*` atoms or disconnected `.` salts/mixtures (`UnsupportedPolymerOrMixtureError` when strict rejection is enabled).
   - `validate_processing_record()` rejects physically impossible values (`thickness <= 0` or `> 2000`, `annealing_temp < -273.15` or `> 350.0`, malformed/non-positive D:A ratios, `additive_ratio < 0` or `> 100`, or invalid non-string/empty categoricals) rather than clipping or inventing values, while emitting non-fatal range warnings for unusual valid values outside training bounds (`thickness` outside `[15, 600]`, `annealing_temp` outside `[0, 250]`, `additive_ratio > 10`).

---

## 2. Authoritative Model Schemas & Missing-Value Semantics

| Model | Column Count | Column Composition (Exact Order) | Data Types | Missing-Value Semantics |
| :--- | :---: | :--- | :--- | :--- |
| **`M0`** | **22** | `M0_FEATURE_NAMES` (22 RDKit molecular descriptors) | `float64` | Strictly finite; zero `NaN`/`Inf` permitted. |
| **`M1`** | **50** | `22` M0 + `5` numeric/binary (`da_ratio_numeric`, `thickness_nm`, `annealing_temp_c`, `has_additive`, `has_thermal_annealing`) + `23` `M1_OHE_COLUMNS` | `float64` / `int64` | Strictly finite. Continuous missing $\to$ frozen $D_{\text{train}}$ medians. Null nominals $\to$ `*_missing = 1.0`. Unrecognized strings $\to$ `*_other = 1.0`. |
| **`M2A`** | **37** | `22` M0 + `4` continuous normalized + `7` physical regimes + `4` interface/interaction constants | `float64` / `int64` | Strictly finite. Unobserved continuous/interface $\to$ frozen $D_{\text{train}}$ medians (`additive_concentration_pct` $\to$ `0.0`); missing regimes remain `0`. |
| **`M3`** | **44** | `22` M0 + `4` continuous normalized + `7` `missing_*` indicators + `7` physical regimes + `4` interface/interaction constants | `float64` / `int64` | **Controlled `NaN`:** Unobserved columns in `M3_NAN_ALLOWED_COLUMNS` (4 continuous + 4 interface/constants) retain `IEEE 754 NaN` for native HGBT split routing. All 22 molecular, 7 `missing_*`, and 7 `regime_*` columns strictly non-NaN (`{0, 1}` for binary). |

---

## 3. Reconciliation of Phase 8.1 vs. Phase 8.2 Extraction Differences

To preserve full consistency with the locked Phase 8.0.3 / Phase 8.1 specification (`PHASE_8_0_3_FINAL_CORRECTIONS.md`) and eliminate Phase 8.2 defects:

| Component | Phase 8.1 (Locked Canonical) | Phase 8.2 (Drifted / Defective) | Unified Authoritative Resolution (`processing.py`) |
| :--- | :--- | :--- | :--- |
| **`extract_m3` `reg_ann`** | Properly converts `v = float(val); return 1 if 40.0 <= v <= 250.0 else 0` | Omitted `v = float(val)` $\to$ `NameError` swallowed $\to$ always `0` | Unified `is_thermal_annealed_regime(val)` converts `v = float(val)` cleanly for M2, M2A, and M3. |
| **`SOLV_BP_MAP`** | `'o-dcb': 180.5`, `'o-xylene': 144.4` (`chloroform`, `cb`, `toluene`, `thf`) | Changed to `'dcb'`, `'dichlorobenzene'`, `'xylene': 138.5` (34 OPV-DB diffs) | Adopts locked Phase 8.0.3 / Phase 8.1 values (`'o-dcb': 180.5`, `'o-xylene': 144.4`). |
| **`ETL_WF_MAP`** | `'zno': 4.30`, `'pdinn': 4.00`, `'ca': 2.87`, `'pfn-br': 4.20` | Changed `'pdinn': 4.15`, `'pfn-br': 4.10`, added `'pfn'`, `'tio2'` (118 diffs) | Adopts locked Phase 8.0.3 / Phase 8.1 values (`'pdinn': 4.00`, `'pfn-br': 4.20`). |
| **`HTL_WF_MAP`** | `'pedot': 5.00`, `'moo3': 5.30`, `'2pacz': 5.20` | Dropped `'2pacz'`, added `'vox': 5.40`, `'ni': 5.10` (80 diffs) | Adopts locked Phase 8.0.3 / Phase 8.1 values (`'2pacz': 5.20` preserved). |
| **`map_etl_nominal`** | `s == 'ca' or 'calcium' in s`; `'pfn' in s` | `'ca' in s` (falsely matched 46 strings e.g. `ZrAcac`, `PDIN:DCA` as Ca!) | Adopts Phase 8.1 exact token check (`s == 'ca' or 'calcium' in s`; `'pfn' in s`). |
| **`map_htl_nominal`** | `'pedot'`, `'moo3' or 'mox' or 'molybdenum'` | `'pedot'`, `'moo3' or 'molybdenum'` (0 OPV-DB diffs) | Adopts Phase 8.1 definition (`'moo3' or 'mox' or 'molybdenum'`). |

---

## 4. Audit of Additive Semantics & Proposed v2 Representation

### 4.1 Documented Limitation in Locked M1 Schema (50 Features)
In the locked Phase 8.1 `map_additive_nominal()` function:
- Missing/unrecorded values (`None`, `NaN`) map to `'missing'` $\to$ `additive_missing = 1.0`.
- Explicit additive-free values (`'none'`, `'neat'`, `'0'`, `'no'`) **also map to `'missing'`** $\to$ `additive_missing = 1.0`.
- *Consequence:* The 50-feature M1 representation **conflates verified neat blends with unreported additive metadata** (whereas `missing_additive` in M2/M3 properly marks explicit `'none'/'neat'/'0'/'no'` as `missing_additive = 0` and `regime_solvent_additive_used = 0`).
- *Preservation Rule:* Per Phase 8.4 governance, the 50-feature M1 schema is **not silently altered**, because doing so would invalidate comparison against Phase 8.1's locked M1 results.

### 4.2 Proposed Corrected v2 Specification (Future Separate Protocol Only)
`map_additive_nominal_v2_spec()` is specified in [processing.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/features/processing.py) as an isolated reference helper for future versioned models (e.g., `M1-v2`, 51 features):
- `pd.isna(val)` $\to$ `'missing'` (metadata unrecorded)
- `s in {'none', 'neat', '0', 'no', 'additive-free'}` $\to$ `'absent_explicit'` (verified neat blend; dedicated OHE column `additive_absent_explicit`)
- `'dio'`/`'diiodooctane'` $\to$ `'dio'`, `'cn'`/`'chloronaphthalene'` $\to$ `'cn'`, other non-empty strings $\to$ `'other'`.
- Empty strings `""` raise `PhysicalInputValidationError`.

Any model adopting `map_additive_nominal_v2_spec` **must receive a distinct model version tag and undergo a separate pre-registered validation protocol**; historical Phase 8.1/8.2 M1 metrics do not validate a v2 representation.
