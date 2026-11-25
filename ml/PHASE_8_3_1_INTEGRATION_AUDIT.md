# OPV-PredictNet — Phase 8.3.1: Integration Contract Verification & Technical Audit

**Date:** 2026-11-25  
**Scope:** Focused technical and scientific verification of the five Phase 8.3 documents against actual model artifacts, preprocessing code, `ml/src/models/predictor.py`, and dataset SMILES edge cases prior to any backend or frontend integration.  
**Strict Boundaries:** Zero training, zero hyperparameter tuning, zero production/frontend edits, zero binary artifact creation, zero Git commits.

---

## 1. Executive Audit Scorecard

| Audit Item | Initial Pre-Correction Verdict | Final Post-Correction Status | Primary Evidence Inspected |
| :--- | :---: | :---: | :--- |
| **1. Strategy A Statistical Language** | **FAIL** *(Clarification Needed)* | **PASS** *(Corrected in Phase 8.3 Docs)* | `ml/PHASE_8_3_FINAL_MODEL_SELECTION.md`, `ml/PHASE_8_3_FINAL_READINESS_REPORT.md`, `ml/PHASE_8_2_1_STATISTICAL_CORRECTIONS.md` |
| **2. M1 Categorical Handling** | **FAIL** *(Ambiguous Wording & Undocumented Conflation)* | **PASS** *(Contract & Limitations Corrected)* | `ml/experiments/phase_8_1_processing_aware_models.py` (L95–137), `ml/experiments/phase_8_2_scaffold_benchmark.py` (L102–144) |
| **3. Scaffold OOD Handling** | **FAIL** *(Missing Polymer, Salt, Collision & Missing-ID Rules)* | **PASS** *(Contract & Governance Hardened)* | `ml/PHASE_8_3_INTEGRATION_CONTRACT.md`, `ml/src/features/descriptors.py`, `opv_devices_strict_molecular_benchmark.csv` |
| **4. Actual Artifact Compatibility** | **NOT YET IMPLEMENTED** *(No Artifacts on Disk + `predictor.py` & M3 Bug Found)* | **NOT YET IMPLEMENTED** *(Explicitly Flagged as Future Task)* | `ml/models/saved_models/` (only `README.md`), `ml/src/models/predictor.py` (L14–98), `phase_8_1` & `phase_8_2` scripts |
| **5. Uncertainty Output Contract** | **FAIL** *(Conformal Scope & 4-Way Output Separation Needed)* | **PASS** *(Schema & Non-Guarantees Updated)* | `ml/PHASE_8_3_INTEGRATION_CONTRACT.md`, `ml/results/phase_8_1_interval_metrics.csv`, `ml/results/phase_8_2_interval_metrics.csv` |

---

## 2. Detailed Evidence-Backed Audit Findings

### 2.1 Item 1: Strategy A Statistical Language Verification
- **Evidence Inspected:** `PHASE_8_3_FINAL_MODEL_SELECTION.md` (Section 1 Table 1 & Section 2.2 Regime 1 & Regime 2) and `PHASE_8_3_FINAL_READINESS_REPORT.md` (Section 1.3).
- **Deficiencies Found Prior to Correction:**
  1. `PHASE_8_3_FINAL_MODEL_SELECTION.md` Section 2.2 (Regime 1) wrote *"across all 5 one-way cluster-bootstrap margins in Phase 8.1 and C2"*, whereas Phase 8.1 (`PHASE_8_1_1_STATISTICAL_REVIEW.md`) evaluated **4** one-way margins (`pair_id`, `donor_canonical`, `acceptor_canonical`, `doi_norm`); only Phase 8.2 evaluated 5 margins (adding `scaffold_pair`).
  2. Strategy A descriptions needed explicit three-tier separation distinguishing:
     - **(a) Observed held-out sample performance:** Unweighted device population ($N=1,481$ records across 216 test donor scaffolds; M3 `MAE=2.0025`, M1 `MAE=2.0916`, M0 `MAE=2.2302`).
     - **(b) Separate one-way cluster-bootstrap sensitivity analyses:** Margin-by-margin exploratory bounds (`donor_scaffold`, `acceptor_scaffold`, `scaffold_pair`, `pair_id`, `doi_norm`) that happen to lie below zero for M1 and M3.
     - **(c) Unattained joint crossed-dependence confirmation:** Genuine statistical confirmation would require simultaneous crossed-cluster variance estimation over donor scaffold $\times$ acceptor scaffold $\times$ DOI, equal-weight per-scaffold evaluation, and elimination of the 41.9% (`215/513`) DOI overlap between Strategy A train and test sets.
- **Resolution:** Updated `PHASE_8_3_FINAL_MODEL_SELECTION.md` and `PHASE_8_3_FINAL_READINESS_REPORT.md` while preserving the finding that **M1 and M3 are exploratory research candidates for unseen donor scaffolds paired with known acceptors**.

---

### 2.2 Item 2: M1 Categorical Handling Verification
- **Evidence Inspected:** `phase_8_1_processing_aware_models.py` (lines 45–51, 95–137, 262–271) and `phase_8_2_scaffold_benchmark.py` (lines 48–54, 102–144, 265–273).
- **Exact Actual Code Behavior (`map_*_nominal` $\to$ `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`):**
  1. **Known Category:** Non-null string matching substring rules (`'chloroform'`, `'chlorobenzene'`, `'dcb'`/`'dichlorobenzene'`; `'dio'`, `'cn'`; `'conventional'`/`'regular'`/`'standard'`, `'inverted'`; `'pedot'`, `'moo3'`; `'zno'`, `'pdinn'`, `'ca'`, `'pfn'`) produces its nominal label and sets exactly one OHE bit to `1.0`.
  2. **Missing Category (`None`, `NaN`, `pd.NA`):** `pd.isna(val)` returns `'missing'` $\to$ sets `*_missing = 1.0`.
     - *Documented Pipeline Limitation (`map_additive_nominal`):* Explicitly additive-free strings (`'none'`, `'neat'`, `'0'`, `'no'`) **also return `'missing'`**, setting `additive_missing = 1.0`. The pipeline **cannot distinguish additive-absent ("neat") devices from unreported additive metadata**.
  3. **Unknown Category (e.g., `'acetone'`, `'2PACz'`, `'WS2'`):** Intercepted by `map_*_nominal` *before* `OneHotEncoder` and mapped to the string `'other'`. Because `'other'` is present in full OPV-DB $D_{\text{train}}$ for all five categorical variables, **`OneHotEncoder` encodes every unrecognized string as `*_other = 1.0`—NOT as all zeros**.
     - *Documented Pipeline Limitation:* The M1 pipeline **cannot distinguish rare known categories observed in training (e.g., `'toluene'`) from brand-new unknown strings (e.g., `'foo_solvent'`)**. An all-zero OHE slice occurs *only* if `OneHotEncoder(handle_unknown='ignore')` receives a post-mapper token absent from its training fit (which cannot happen on full OPV-DB $D_{\text{train}}$ where all 23 columns exist).
  4. **Invalid Category Input (empty string `""`, whitespace `"  "`, literal `"NaN"`, or non-string objects `123`, `True`):** Because `pd.isna("")` is `False`, `str(val).lower().strip()` silently coerces empty strings, `"NaN"` text, and arbitrary numbers into **`'other'`** (or `'missing'` if `'0'` in additive). **The existing research scripts do NOT reject invalid categorical inputs.**
  5. **Script Drift Discovered:** `map_htl_nominal` checks `'mox'` in Phase 8.1 vs `'molybdenum'` in Phase 8.2; `map_etl_nominal` checks `s == 'ca'` in Phase 8.1 vs `'ca' in s` in Phase 8.2 (which causes any string containing `'ca'`—e.g. `'polycarbonate'` or `'mica'`—to match `'ca'`).
- **Resolution:** Removed `"other or all zeros"` ambiguity across Phase 8.3 docs; documented exact Known / Missing / Unknown / Invalid behavior and required pre-mapper validation in the integration contract.

---

### 2.3 Item 3: Scaffold OOD Handling Verification
- **Evidence Inspected:** `opv_devices_strict_molecular_benchmark.csv` SMILES inspection, `ml/src/features/descriptors.py`, `PHASE_8_3_INTEGRATION_CONTRACT.md` Section 3.1 & 4.1.
- **Empirical Dataset Edge Cases Found:**
  - `3` unique acceptor SMILES in OPV-DB contain attachment wildcards `*` (`*/C(C#N)=C1...`). In RDKit, `Descriptors.MolWt` assigns atomic weight `0.0` to `*`.
  - `8` unique donor SMILES and `26` unique acceptor SMILES contain disconnected components `.` (salts like `N#C[S-].[Cu+]`, co-solvents `.CCl.CCl`, or multi-component blends). RDKit descriptors sum across all disconnected fragments.
  - `2` donor SMILES (`8` rows) and `3` acceptor SMILES (`4` rows) yield empty Murcko scaffolds `""`.
  - `79` donor and `68` acceptor canonical names map to multiple distinct SMILES.
- **Resolution:** Hardened deterministic contract rules covering all six edge cases (`empty Murcko`, `invalid SMILES`, `unsupported polymer/mixture`, `material-name collisions`, `absent from scaffold_registry.json`, `missing donor/acceptor identity`) so unknown or ambiguous identity can never silently resolve as in-domain.

---

### 2.4 Item 4: Actual Artifact Compatibility Verification (`NOT YET IMPLEMENTED`)
- **Evidence Inspected:**
  - `ml/models/saved_models/`: Contains **only `README.md` (589 bytes)**. Zero serialized `.joblib`, `.meta.json`, `.pipeline_state.json`, or `scaffold_registry.json` files exist for M0, M1, M2A, or M3.
  - `ml/src/models/predictor.py`:
    - `validate_scientific_artifact` hardcodes `model.n_features_in_ != len(FEATURE_NAMES)` (22) at line 33 and `list(metadata["features"]) != FEATURE_NAMES` at line 93 (note also key mismatch: `predictor.py` checks `"features"` whereas `test_model_safety.py` line 113 passes `"feature_names"`).
    - **Incompatible** as-is with M1 (`50`), M2A (`37`), and M3 (`44`), and lacks `.pipeline_state.json`, `scaffold_registry.json`, and M3 `NaN`-preserving extraction.
  - Experiment Estimators Inspected (`phase_8_1` L704–710 & `phase_8_2` L747–752):
    - `M0 (22)`, `M1 (50)`, `M2A (37)`: `RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=8888, n_jobs=1)`.
    - `M3 (44)`: `HistGradientBoostingRegressor(max_iter=100, max_depth=None, min_samples_leaf=20, random_state=8888)`. **Important compatibility note:** `HistGradientBoostingRegressor` does not accept `n_jobs=1`; deterministic single-threaded execution requires `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1`.
  - **Code Defect Found in `phase_8_2_scaffold_benchmark.py` (`extract_m3`):**
    - Lines 614–617 define `reg_ann(val)` as `try: return 1 if 40.0 <= v <= 250.0 else 0` without assigning `v = float(val)`, causing a `NameError` caught by `except: return 0`. Thus `regime_thermal_annealed` in Phase 8.2 M3 evaluated to `0` on all rows (whereas `phase_8_1_processing_aware_models.py` line 460 properly assigns `v = float(val)`).
    - Lookup dictionaries (`SOLV_BP_MAP`, `ETL_WF_MAP`, `HTL_WF_MAP`) also drift slightly between Phase 8.1 and Phase 8.2 scripts.
- **Verdict:** **Artifact packaging, unified `ml/src/features/` extractor refactoring (from the Phase 8.1 bug-free `reg_ann` baseline), and `OPVModelRegistry` multi-schema support are `NOT YET IMPLEMENTED`.** Saved artifacts cannot currently reproduce the integration contract because no saved artifacts exist yet.

---

### 2.5 Item 5: Uncertainty Output Audit
- **Evidence Inspected:** `phase_8_1_interval_metrics.csv`, `phase_8_2_interval_metrics.csv`, `PHASE_8_3_INTEGRATION_CONTRACT.md` Section 4.2.
- **Exact Interval Provenance & Coverage Reality:**
  - All candidate conformal quantiles in the integration contract derive from **Phase 8.1 $D_{\text{cal}}$** ($N_{\text{cal}}=946$ held-out material-name pairs, partition seed `8888`, cal draw seed `8201`, order statistic $k=\lceil 947 \times 0.90 \rceil = 853$):
    - **M0:** $q_{0.10}=4.4836$ (width `8.9671 pp`, Phase 8.1 test cov `90.06%`)
    - **M1:** $q_{0.10}=4.3244$ (width `8.6488 pp`, Phase 8.1 test cov `90.17%`)
    - **M2A:** $q_{0.10}=4.3300$ (width `8.6600 pp`, Phase 8.1 test cov `90.38%`)
    - **M3:** $q_{0.10}=4.2141$ (width `8.4283 pp`, Phase 8.1 test cov `88.69%`)
- **Deficiencies Fixed:** Refactored the API JSON contract into **four strictly separated top-level fields** (`point_prediction`, `interval_estimate`, `ood_status`, `scientific_support_status`) with explicit boolean flags proving no nominal 90% coverage guarantee applies to unseen scaffolds, high-PCE ($\ge 12\%/\ge 15\%$), arbitrary individual records, or distribution-shifted inputs.
