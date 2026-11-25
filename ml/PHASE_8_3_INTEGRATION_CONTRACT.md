# OPV-PredictNet — Phase 8.3: ML Integration Architecture & Runtime Contract (Phase 8.3.1 Verified)

**Date:** 2026-11-25  
**Scope:** Formal software, data, schema, uncertainty, and safety integration specification for candidate models **M0, M1, M2A, and M3** without modifying production code.  
**Implementation Status:** **`NOT YET IMPLEMENTED`** — As of Phase 8.3.1, `ml/models/saved_models/` contains only `README.md` (zero serialized `.joblib`/`.meta.json`/`.pipeline_state.json`/`scaffold_registry.json` files exist on disk), and [ml/src/models/predictor.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/models/predictor.py) currently hardcodes `len(FEATURE_NAMES) == 22`. Packaging artifacts and refactoring `OPVModelRegistry` are explicitly deferred to an authorized future implementation phase.  
**Integration Prerequisite:** Models must be exposed strictly as **experimental research estimators** under the constraints of [PHASE_8_3_MODEL_GOVERNANCE.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_3_MODEL_GOVERNANCE.md).

---

## 1. Artifact Format, Packaging, & Versioning (`NOT YET IMPLEMENTED`)

### 1.1 Directory Layout & Bundle Structure
Future authorized integration must package each model family inside `ml/models/saved_models/<model_id>_v8.3.0/` (verified inside `OPVModelRegistry.models_dir` with path-traversal protection). **No production or research bundles exist on disk yet.** Each future bundle must consist of **four mandatory files**:

| File | Format | Responsibility | Current Status |
| :--- | :--- | :--- | :---: |
| `<model_id>.joblib` | `joblib` | Serialized scikit-learn estimator (`RandomForestRegressor` for M0/M1/M2A; `HistGradientBoostingRegressor` for M3). | **NOT YET IMPLEMENTED** |
| `<model_id>.meta.json` | UTF-8 JSON | Provenance, SHA-256 checksums, exact `feature_names` list, conformal quantiles, evaluated scope, and runtime limitations. | **NOT YET IMPLEMENTED** |
| `<model_id>.pipeline_state.json` | UTF-8 JSON | Frozen $D_{\text{train}}$ medians, `OneHotEncoder` categories (for M1), and canonical unified lookup tables (`SOLV_BP_MAP`, `ETL_WF_MAP`, `HTL_WF_MAP`). | **NOT YET IMPLEMENTED** |
| `scaffold_registry.json` | UTF-8 JSON | Frozen training `donor_scaffolds`, `acceptor_scaffolds`, `scaffold_pairs`, `donor_canonical_smiles`, `acceptor_canonical_smiles`, ambiguous-name map, and training partition SHA-256. | **NOT YET IMPLEMENTED** |

### 1.2 Semantic Versioning, Determinism, & Code Unification Requirements
- **Model Version Tags:** `opv-m0-rf-v8.3.0-research`, `opv-m1-rf-v8.3.0-research`, `opv-m2a-rf-v8.3.0-research`, `opv-m3-hgbt-v8.3.0-research`.
- **Single-Threaded Execution Pinning:**
  - `M0`, `M1`, `M2A` (`RandomForestRegressor`, `n_estimators=100`, `random_state=8888`): enforce `n_jobs = 1` at load time (`model.set_params(n_jobs=1)`).
  - `M3` (`HistGradientBoostingRegressor`, `max_iter=100`, `min_samples_leaf=20`, `random_state=8888`): does **not** expose an `n_jobs` constructor parameter in scikit-learn; deterministic single-thread inference requires pinning OpenMP/BLAS threads (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` / `threadpoolctl`) at runtime.
- **Prerequisite Code Unification (`NOT YET IMPLEMENTED`):** Before packaging artifacts, the duplicated feature extractors in `phase_8_1_processing_aware_models.py` and `phase_8_2_scaffold_benchmark.py` must be unified into a canonical module under `ml/src/features/` using Phase 8.1's bug-free `reg_ann` (`v = float(val)` was omitted in Phase 8.2's `extract_m3` `reg_ann`, causing `regime_thermal_annealed` to evaluate to `0` in Phase 8.2 M3), reconciling `SOLV_BP_MAP`/`ETL_WF_MAP`/`HTL_WF_MAP` drift, and enforcing exact `'ca'` token matching in `map_etl_nominal`.
- **Environment & Checksum Verification:** Record `python_version`, `sklearn_version`, `rdkit_version`, `numpy_version`, and SHA-256 hashes of `<model_id>.joblib`, `<model_id>.pipeline_state.json`, and `scaffold_registry.json`. Deserialization fails if any SHA-256 verification fails.

### 1.3 Provenance & Scope Certification Schema (`.meta.json`)
Currently, [ml/src/models/predictor.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/models/predictor.py) (`validate_scientific_artifact`) hardcodes `model.n_features_in_ != len(FEATURE_NAMES)` (22) and `list(metadata["features"]) != FEATURE_NAMES`, and lacks `.pipeline_state.json` / `scaffold_registry.json` loaders. Future integration must refactor `validate_scientific_artifact` to validate against `metadata["feature_count"]` (`22`, `50`, `37`, or `44`), align the `"features"` vs `"feature_names"` metadata key, and enforce explicit scope flags:

```json
{
  "model_id": "M3",
  "version_tag": "opv-m3-hgbt-v8.3.0-research",
  "estimator_class": "HistGradientBoostingRegressor",
  "n_jobs": 1,
  "feature_count": 44,
  "feature_names": ["donor_mw", "..."],
  "artifact_sha256": "<sha256>",
  "pipeline_state_sha256": "<sha256>",
  "training_provenance": "OPV-DB v1.1.0 experimental literature dataset (Phase 8.1/8.2 research protocols); retrospective benchmark only",
  "dataset_source": "opv_devices_strict_molecular_benchmark.csv",
  "dataset_sha256": "ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897",
  "dataset_version": "OPV-DB v1.1.0 (N=22,467)",
  "is_scientifically_validated": true,
  "research_estimate_only": true,
  "validated_for_production": false,
  "evaluated_scopes": {
    "familiar_scaffold_pairs": "SUPPORTED_RESEARCH_ESTIMATE",
    "unseen_donor_known_acceptor": "SUPPORTED_EXPLORATORY_WITH_OOD_WARNING",
    "unseen_acceptor_known_donor": "INCONCLUSIVE_HIGH_UNCERTAINTY_WARNING",
    "dual_unseen_scaffolds": "UNSUPPORTED_REJECT_OR_BLOCK",
    "high_pce_discovery_ge_15": "UNSUPPORTED_SEVERE_UNDERPREDICTION"
  },
  "conformal_quantiles_pair_holdout": {
    "alpha_0.20_nominal_80": {"q": 3.1843, "n_cal": 946, "observed_cov": 0.8044},
    "alpha_0.10_nominal_90": {"q": 4.2141, "n_cal": 946, "observed_cov": 0.8869},
    "alpha_0.05_nominal_95": {"q": 5.6556, "n_cal": 946, "observed_cov": 0.9609}
  }
}
```
*Note:* Synthetic/fixture keywords (`"synthetic"`, `"test fixture"`, `"dummy"`, `"mock"`, `"random"`, `"placeholder"`) remain strictly rejected by `validate_scientific_artifact`.

---

## 2. Exact Feature Schemas (Locked Column Order & Types)

Every inference call must construct a `float64`/`int64` numeric array whose columns match the exact names and order below (matching [PHASE_8_0_2_FEATURE_MATRIX_CONTRACT.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_0_2_FEATURE_MATRIX_CONTRACT.md) and [phase_8_2_scaffold_benchmark.py](file:///c:/Users/DELL/OPV-PredictNet/ml/experiments/phase_8_2_scaffold_benchmark.py)):

### 2.1 Base 22 Molecular Descriptors (`M0`, Columns 1–22 of All Models)
Extracted via `ml.src.features.descriptors.extract_features_from_mols(donor_mol, acceptor_mol)` (`Float64`, zero `NaN`s, zero `Inf`s):
1. `donor_mw`, 2. `donor_logp`, 3. `donor_tpsa`, 4. `donor_hbd`, 5. `donor_hba`, 6. `donor_rotatable_bonds`, 7. `donor_aromatic_rings`, 8. `donor_fraction_csp3`, 9. `donor_heavy_atoms`, 10. `acceptor_mw`, 11. `acceptor_logp`, 12. `acceptor_tpsa`, 13. `acceptor_hbd`, 14. `acceptor_hba`, 15. `acceptor_rotatable_bonds`, 16. `acceptor_aromatic_rings`, 17. `acceptor_fraction_csp3`, 18. `acceptor_heavy_atoms`, 19. `mw_diff`, 20. `logp_diff`, 21. `tpsa_total`, 22. `aromatic_rings_total`.

### 2.2 Model `M1` Schema (`50` Columns, Strictly Finite)
- **1–22:** `M0` columns.
- **23–27:** `da_ratio_numeric` (`Float64`), `thickness_nm` (`Float64`), `annealing_temp_c` (`Float64`), `has_additive` (`{0,1}`), `has_thermal_annealing` (`{0,1}`).
- **28–50 (One-Hot Encoded via training-fitted `OneHotEncoder(handle_unknown='ignore')`):**
  `solvent_chlorobenzene`, `solvent_chloroform`, `solvent_missing`, `solvent_o-dcb`, `solvent_other`, `additive_cn`, `additive_dio`, `additive_missing`, `additive_other`, `architecture_conventional`, `architecture_inverted`, `architecture_missing`, `architecture_other`, `htl_missing`, `htl_moo3`, `htl_other`, `htl_pedot_pss`, `etl_ca`, `etl_missing`, `etl_other`, `etl_pdinn`, `etl_pfn_br`, `etl_zno`.

### 2.3 Model `M2A` Schema (`37` Columns, Strictly Finite)
- **1–22:** `M0` columns.
- **23–26 (Continuous Normalized):** `donor_blend_fraction`, `thickness_log10`, `annealing_temp_norm`, `additive_concentration_pct`.
- **27–33 (Physical Regimes, `{0,1}`):** `regime_as_cast_room_temp`, `regime_thermal_annealed`, `regime_solvent_additive_used`, `regime_chlorinated_solvent`, `regime_green_solvent`, `regime_inverted_stack`, `regime_conventional_stack`.
- **34–37 (Interface & Physical Constants):** `thickness_per_donor_mw`, `solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`.

### 2.4 Model `M3` Schema (`44` Columns, Controlled IEEE 754 `NaN`s)
- **1–22:** `M0` columns (strictly finite).
- **23–26 (Continuous Normalized, `NaN` permitted if unobserved):** `donor_blend_fraction`, `thickness_log10`, `annealing_temp_norm`, `additive_concentration_pct`.
- **27–33 (Missingness Indicators, strictly `{0,1}`, zero `NaN`s):** `missing_da_ratio`, `missing_thickness`, `missing_annealing`, `missing_solvent`, `missing_additive`, `missing_architecture`, `missing_interlayers`.
- **34–40 (Physical Regimes, strictly `{0,1}`, zero `NaN`s):** `regime_as_cast_room_temp`, `regime_thermal_annealed`, `regime_solvent_additive_used`, `regime_chlorinated_solvent`, `regime_green_solvent`, `regime_inverted_stack`, `regime_conventional_stack`.
- **41–44 (Interface & Physical Constants, `NaN` permitted if unobserved):** `thickness_per_donor_mw`, `solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`.

---

## 3. Input Validation, Missingness, & Unknown Categorical Handling

### 3.1 Multi-Tier Input Validation Rules
All inputs are validated prior to feature vector construction. Invalid or unphysical inputs are **rejected explicitly—never silently coerced**:

| Tier | Validation Check | Condition / Rule | Action |
| :--- | :--- | :--- | :--- |
| **Tier 1: Identity & Syntax** | Missing Donor/Acceptor | `donor_smiles` or `acceptor_smiles` is `None`, omitted, empty `""`, or whitespace | **HARD REJECT** (`MissingMolecularIdentityError` / `InvalidMolecularInputError`, HTTP 422) |
| **Tier 1: Name Collision** | Material-Name Query | Query provides material name without canonical SMILES, OR name maps to $>1$ canonical structure in reference map (79 donor / 68 acceptor collisions) | **HARD REJECT** (`AmbiguousMaterialIdentityError`, HTTP 422) if SMILES omitted; if SMILES provided and conflicts with reference name mapping, emit `MATERIAL_NAME_STRUCTURE_CONFLICT_WARNING` and key strictly on canonical SMILES |
| **Tier 1: SMILES Parse** | Invalid SMILES / BigSMILES | `Chem.MolFromSmiles(s) is None` or RDKit sanitization fails (including unparseable BigSMILES `{[]...[]}`) | **HARD REJECT** (`InvalidMolecularInputError`, HTTP 422) |
| **Tier 1: Polymer / Mixture** | Wildcards `*` or `.` Salts/Blends | SMILES contains dummy/attachment wildcard `*` (atomic number 0) or disconnected fragments `.` | If absent from frozen `train_canonical_smiles`: **HARD REJECT** (`UnsupportedPolymerOrMixtureError`, HTTP 422). If matched in training or diagnostic override: emit `UNSUPPORTED_POLYMER_OR_MIXTURE_WARNING` and set `scientific_support_status = UNSUPPORTED_STRUCTURE_APPROXIMATION` |
| **Tier 1: Non-Ring Check** | Empty Murcko Scaffold | `MurckoScaffoldSmiles(mol) == ""` | Assign `[NO_RING]:<canonical_smiles>`; execute full `scaffold_registry.json` OOD check (never bypass); emit `NON_RING_PSEUDO_SCAFFOLD_WARNING` |
| **Tier 1: Categorical Type** | Invalid M1 Category Input | Non-string, empty `""`, whitespace, or `"NaN"` text in categorical fields | **HARD REJECT** in API layer (`PhysicalInputValidationError`, HTTP 422) prior to `map_*_nominal` (do not allow `str()` coercion into `'other'`) |
| **Tier 2: Physical Impossibility** | Unphysical Processing | `thickness_nm <= 0` or `> 2000`; `annealing_temp_c < -273.15` or `> 350.0`; `d_a_ratio` with donor `<= 0` or acceptor `<= 0`; `additive_ratio < 0` or `> 100` | **HARD REJECT** (`PhysicalInputValidationError`, HTTP 422) |
| **Tier 3: Training Domain Bounds** | Valid Outside Training Fit Range | `thickness_nm` outside `[15.0, 600.0]`; `annealing_temp_c` outside `[0.0, 250.0]` (`(35, 40)` transition gap); `additive_concentration_pct > 10.0` | **PERMIT WITH `OUT_OF_TRAINING_RANGE_WARNING`**; treat continuous value as unobserved/missing for M1/M2A/M3 imputation/NaN contract rather than silently clipping |
| **Tier 4: Output Sanity** | Thermodynamic Ceiling | Predicted `pce < 0.0` or `pce > 34.0` via `validate_prediction_sanity` | **HARD REJECT** (`UnphysicalPredictionError`) |

### 3.2 Missing-Value Behavior Across Models
- **Zero-Processing Dispatch (`k = 0`):** If all 7 processing groups (`d_a_ratio`, `solvent`, `active_layer_thickness`, `annealing_temp`, `device_type`, `additive`, `etl`/`htl`) are omitted, route exclusively to **`M0`** (unless caller explicitly requests M2A/M3 counterfactual with declared missing fields, which emits `NO_PROCESSING_INPUTS_WARNING`).
- **M1 Missing Behavior:** Missing continuous scalars (`da_ratio_numeric`, `thickness_nm`, `annealing_temp_c`) are filled using frozen $D_{\text{train}}$ medians (`pipeline_state.json`). Null nominals (`None`, `NaN`) route to `*_missing` one-hot columns (`solvent_missing`, `additive_missing`, `architecture_missing`, `htl_missing`, `etl_missing`).
- **M2A Missing Behavior:** Missing continuous scalars and physical constants (`solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`) are filled with frozen $D_{\text{train}}$ medians; `additive_concentration_pct` defaults to `0.0`. All associated `regime_*` binary columns remain `0`.
- **M3 Missing Behavior:** Unobserved continuous and physical constant columns remain `np.nan` (`IEEE 754 NaN`) so `HistGradientBoostingRegressor` applies learned training missing-value split directions. Corresponding `missing_*` flags are set to `1`, and associated `regime_*` flags remain `0`.

### 3.3 Exact M1 Categorical Handling Contract (Observed Pipeline Behavior & Limitations)
In the actual `FeaturePipeline`, categorical inputs pass sequentially through `map_*_nominal(val)` and then `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` fitted on $D_{\text{train}}$:

| Input State | Actual `FeaturePipeline` Execution | Exact OHE Output Vector | Known Pipeline Limitation & API Control |
| :--- | :--- | :--- | :--- |
| **1. Known Category** | Matches substring keyword rule (e.g. `'chloroform'`, `'dio'`, `'inverted'`, `'pedot'`, `'zno'`) | Single `1.0` in matching bucket (e.g. `solvent_chloroform=1.0`), `0.0` elsewhere | Substring rules must be canonicalized (avoid `'ca' in s` false positives). |
| **2. Missing Category (`None`/`NaN`)** | `pd.isna(val)` returns `'missing'` | Single `1.0` in `*_missing` (e.g. `solvent_missing=1.0`) | **Additive Conflation:** `map_additive_nominal` maps `'none'`, `'neat'`, `'0'`, `'no'` to `'missing'` (`additive_missing=1.0`), so M1 cannot separate additive-free ("neat") devices from unreported additive metadata. |
| **3. Unknown Category (e.g. `'acetone'`, `'WS2'`)** | Intercepted by `map_*_nominal` *before* OHE and returned as `'other'` | Single `1.0` in `*_other` (e.g. `solvent_other=1.0`), **NOT all zeros** when `'other'` was fitted in $D_{\text{train}}$ | **Indistinguishable from Rare Knowns:** M1 cannot distinguish a rare training category (e.g. `'toluene'` $\to$ `'other'`) from a novel unseen string (`'acetone'` $\to$ `'other'`). An all-zero OHE slice occurs only if a post-mapper token was absent from training fit. API must emit `<field>_UNRECOGNIZED_MAPPED_TO_OTHER_WARNING`. |
| **4. Invalid Input (`""`, `"  "`, `"NaN"` text, `123`, `True`)** | Existing script coerces `str(val).lower().strip()` $\to$ `'other'` (or `'missing'` for `'0'` in additive) | Silently encodes as `*_other = 1.0` in research scripts | **API Must Reject:** Future API integration must strictly validate type/non-empty string before calling `map_*_nominal` (rejecting invalid inputs under Tier 1 rather than silently collapsing to `'other'`). |

Physical Constant Mapping (`M2A` & `M3`): If a non-null `solvent`, `etl`, or `htl` string is absent from `SOLV_BP_MAP`, `ETL_WF_MAP`, or `HTL_WF_MAP`, M2A assigns the frozen $D_{\text{train}}$ median while M3 preserves `np.nan` and emits `UNMAPPED_PHYSICAL_CONSTANT_WARNING`.

---

## 4. Scaffold Out-of-Distribution (OOD) Detection & Four-Block Output Contract

### 4.1 Runtime Scaffold OOD Classification
At inference time, compute RDKit canonical SMILES and Bemis–Murcko scaffolds (assigning `[NO_RING]:<canonical_smiles>` if empty) for both donor and acceptor, and query `scaffold_registry.json`. **Unknown or ambiguous identity must never silently resolve as in-domain.**

| Scaffold OOD Class | Lookup Condition | API Action, Support Status, & Interval Policy |
| :--- | :--- | :--- |
| **`FAMILIAR_SCAFFOLD_PAIR`** | `(donor_scaff, acceptor_scaff) in train_scaffold_pairs` (and neither is `[NO_RING]`/`*`/`.` approximation) | Allow (`SUPPORTED_RESEARCH_ESTIMATE`); `interval_estimate.available = true` (marginal Phase 8.1 guarantee only unless `PCE >= 12%`). |
| **`UNSEEN_PAIR_KNOWN_SCAFFOLDS`** | `donor_scaff in train_d` AND `acceptor_scaff in train_a` (pair unseen) | Allow (`EXPLORATORY_PAIR_COMBINATION`); emit `UNSEEN_SCAFFOLD_PAIR_WARNING` + set `interval_estimate.exchangeability_valid = false` (C2 80% undercoverage `75.7–78.5%`). |
| **`UNSEEN_DONOR_KNOWN_ACCEPTOR`** | `donor_scaff not in train_d` AND `acceptor_scaff in train_a` | Allow for M1/M3 (`EXPLORATORY_UNSEEN_DONOR_ONLY`); warn M2A inconclusive; emit `UNSEEN_DONOR_SCAFFOLD_OOD_WARNING`; set `interval_estimate.exchangeability_valid = false`. |
| **`UNSEEN_ACCEPTOR_KNOWN_DONOR`** | `donor_scaff in train_d` AND `acceptor_scaff not in train_a` | Set `INCONCLUSIVE_EXPLORATORY_ONLY`; emit `CRITICAL_UNSEEN_ACCEPTOR_OOD_WARNING`; **suppress calibrated interval (`interval_estimate.available = false`)**. |
| **`DUAL_UNSEEN_SCAFFOLDS`** | `donor_scaff not in train_d` AND `acceptor_scaff not in train_a` | **HARD REJECT by default** (`OutOfDistributionScaffoldError`, HTTP 422). If `allow_unsupported_ood_diagnostic=true`, return `UNSUPPORTED_EXTRAPOLATION` with `interval_estimate.available = false`. |
| **`INVALID_OR_AMBIGUOUS_IDENTITY`** | Missing D/A, unparseable SMILES, ambiguous name-only query, or unseen `*`/`.` polymer/mixture | **HARD REJECT** (HTTP 422). Never compute or silently pass as in-domain. |

### 4.2 Uncertainty Provenance & Four-Block Response Schema

Conformal 90% intervals originate strictly from **Phase 8.1 $D_{\text{cal}}$** ($N_{\text{cal}}=946$ held-out material-name pairs, partition seed `8888`, Alternative B cal draw seed `8201`, finite-sample rank $k=\lceil 947 \times 0.90 \rceil = 853$; source: `ml/results/phase_8_1_interval_metrics.csv`): `M0` $q_{0.10}=4.4836$ (width `8.9671`, obs `90.06%`), `M1` $q_{0.10}=4.3244$ (width `8.6488`, obs `90.17%`), `M2A` $q_{0.10}=4.3300$ (width `8.6600`, obs `90.38%`), `M3` $q_{0.10}=4.2141$ (width `8.4283`, obs `88.69%`).

The API response **must strictly separate four top-level objects**: `point_prediction`, `interval_estimate`, `ood_status`, and `scientific_support_status`:

```json
{
  "status": "success_research_estimate",
  "model": {
    "model_id": "M3",
    "version_tag": "opv-m3-hgbt-v8.3.0-research",
    "feature_count": 44,
    "dataset_version": "OPV-DB v1.1.0",
    "dataset_sha256": "ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897"
  },
  "point_prediction": {
    "pce_pct": 11.42,
    "processing_mode": "processing_conditional_partial",
    "processing_completeness_k": 4,
    "provided_fields": ["d_a_ratio", "solvent", "annealing_temp", "device_type"],
    "missing_fields": ["active_layer_thickness", "additive", "etl_htl"],
    "active_regimes": ["thermal_annealed", "chlorinated_solvent", "inverted_stack"]
  },
  "interval_estimate": {
    "available": true,
    "nominal_level": 0.90,
    "lower_pce_pct": 7.21,
    "upper_pce_pct": 15.63,
    "half_width_q": 4.2141,
    "full_width_pce_pp": 8.4283,
    "calibration_source": "phase_8_1_interval_metrics.csv (Phase 8.1 Held-Out Material Pairs, N_cal=946, rank k=853)",
    "exchangeability_valid": false,
    "coverage_guaranteed_for_individual_record": false,
    "coverage_guaranteed_for_unseen_scaffolds": false,
    "coverage_guaranteed_for_high_pce": false,
    "coverage_guaranteed_under_distribution_shift": false,
    "interval_limitation_notice": "Split-conformal coverage applies marginally only across exchangeable held-out material-name pairs; NOT guaranteed for individual devices, unseen scaffolds, high-PCE (>=12%/15%) cells, or distribution-shifted inputs."
  },
  "ood_status": {
    "scaffold_ood_class": "UNSEEN_PAIR_KNOWN_SCAFFOLDS",
    "donor_in_train_scaffolds": true,
    "acceptor_in_train_scaffolds": true,
    "pair_in_train_scaffolds": false,
    "donor_canonical_smiles": "...",
    "acceptor_canonical_smiles": "...",
    "donor_scaffold": "...",
    "acceptor_scaffold": "...",
    "structure_edge_cases": []
  },
  "scientific_support_status": {
    "support_tier": "EXPLORATORY_PAIR_COMBINATION",
    "research_estimate_only": true,
    "validated_for_production": false,
    "authorized_use": "Retrospective research comparison only; not prospectively validated.",
    "prohibited_claims": ["autonomous_discovery", "high_pce_screening", "zero_shot_dual_scaffold_extrapolation"]
  },
  "warnings": [
    "RESEARCH_ESTIMATE_ONLY: Retrospective OPV-DB model without prospective experimental validation.",
    "UNSEEN_SCAFFOLD_PAIR_WARNING: Conformal 80%/90% coverage may under-cover novel scaffold pairings (C2 80% cov 75.7%-78.5%)."
  ]
}
```
*Rule on Interval Suppression:* Whenever `scaffold_ood_class` is `UNSEEN_ACCEPTOR_KNOWN_DONOR` or `DUAL_UNSEEN_SCAFFOLDS` (in diagnostic override mode), OR when `structure_edge_cases` indicates unsupported polymer/mixture approximations, `interval_estimate.available` must be **`false`** (`lower_pce_pct = null`, `upper_pce_pct = null`) and `NON_EXCHANGEABLE_INTERVAL_INVALIDATED_WARNING` must be emitted. Clipped intervals must never be presented as unclipped conformal bounds.

---

## 5. Error Handling & Audit Logging

- **Typed Exceptions (No Silent Fallbacks):**
  - `MissingMolecularIdentityError` (missing donor or acceptor identity)
  - `AmbiguousMaterialIdentityError` (material-name query without SMILES or ambiguous name collision)
  - `InvalidMolecularInputError` (unparseable SMILES / BigSMILES syntax)
  - `UnsupportedPolymerOrMixtureError` (unseen wildcard `*` polymer or `.` salt/mixture outside training structures)
  - `PhysicalInputValidationError` (impossible thickness, temperature, D:A ratio, or invalid categorical type/empty string)
  - `OutOfDistributionScaffoldError` (dual-unseen scaffolds when diagnostic override is disabled)
  - `UnphysicalPredictionError` (prediction outside `[0.0%, 34.0%]`)
  - `ArtifactIntegrityError` (checksum mismatch, `n_jobs != 1` / OpenMP unpinned, schema mismatch)
- **Structured Audit Logging:** Every inference request logs a deterministic request ID, timestamp, `model_id`, `version_tag`, `ood_status.scaffold_ood_class`, `scientific_support_status.support_tier`, `processing_completeness_k`, `point_prediction.pce_pct`, `interval_estimate.available`, triggered `warning_codes`, and `status` (`success_research_estimate` vs `rejected_<error_type>`).
