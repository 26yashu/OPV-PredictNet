# OPV-PredictNet — Phase 8.3.1: Final Contract Corrections & Implementation Backlog

**Date:** 2026-11-25  
**Scope:** Exact specification of all corrections applied to Phase 8.3 documentation and remaining pre-integration engineering prerequisites identified during Phase 8.3.1 audit.

---

## 1. Corrections Applied to Phase 8.3 Documents

### 1.1 Strategy A Statistical Language (`PHASE_8_3_FINAL_MODEL_SELECTION.md`, `PHASE_8_3_FINAL_READINESS_REPORT.md`)
- Corrected Phase 8.1 one-way cluster-bootstrap count from `5` to **`4`** (`pair_id`, `donor_canonical`, `acceptor_canonical`, `doi_norm`), reserving `5` margins specifically for Phase 8.2 (`donor_scaffold`, `acceptor_scaffold`, `scaffold_pair`, `pair_id`, `doi_norm`).
- Explicitly separated Strategy A (unseen donor scaffolds) into three distinct evidentiary levels:
  1. **Observed held-out sample metrics** ($N=1,481$ unweighted device rows across 216 donor scaffolds).
  2. **Separate one-way cluster-bootstrap sensitivity analyses** (exploratory margin-by-margin bounds).
  3. **Unattained joint crossed-dependence confirmation** (would require simultaneous donor $\times$ acceptor $\times$ DOI variance estimation, balanced per-scaffold evaluation, and removal of 41.9% DOI overlap).
- Preserved `M1` and `M3` strictly as **exploratory candidates** for unseen donor scaffolds paired with known acceptors.

### 1.2 M1 Categorical Pipeline Exact Semantics (`PHASE_8_3_FINAL_MODEL_SELECTION.md`, `PHASE_8_3_INTEGRATION_CONTRACT.md`)
- Removed ambiguous phrasing (`"'other' bucket or out-of-vocabulary"` / `"all zeros"`).
- Documented exact four-case behavior of `map_*_nominal` + `OneHotEncoder(handle_unknown='ignore')`:
  1. **Known category:** Maps via keyword rule $\to$ single active `1.0` OHE column.
  2. **Missing category (`None`/`NaN`):** Maps to `'missing'` $\to$ `*_missing = 1.0`. Explicitly documented that `map_additive_nominal` also routes `'none'`, `'neat'`, `'0'`, `'no'` to `'missing'` (cannot separate additive-free from unreported).
  3. **Unknown category (non-null unrecognized string):** Intercepted by `map_*_nominal` and mapped to `'other'` $\to$ activates `*_other = 1.0` (never all zeros when fitted on full OPV-DB $D_{\text{train}}$). Explicitly documented that the pipeline **cannot distinguish rare known training categories from brand-new unknown strings**.
  4. **Invalid category input (`""`, whitespace, `"NaN"` text, numeric/boolean):** Currently coerced by `str(val).lower().strip()` into `'other'` (or `'missing'` for `'0'` in additive); future integration must reject non-string/empty inputs prior to `map_*_nominal` unless explicitly null.

### 1.3 Hardened Scaffold & Identity Edge-Case Contract (`PHASE_8_3_INTEGRATION_CONTRACT.md`, `PHASE_8_3_MODEL_GOVERNANCE.md`)
Added explicit deterministic rules for all six edge cases:
1. **Empty Bemis–Murcko scaffold:** `[NO_RING]:<canonical_smiles>` + full `scaffold_registry.json` OOD lookup + `NON_RING_PSEUDO_SCAFFOLD_WARNING`; hard reject if both D & A unseen.
2. **Invalid SMILES (`None`, `""`, whitespace, unparseable/sanitization failure):** Hard reject (`InvalidMolecularInputError`).
3. **Unsupported polymer/mixture SMILES (BigSMILES syntax, wildcard `*`, disconnected `.` salts/blends):** Hard reject unparseable BigSMILES; hard reject unseen `*`/`.` structures (`UnsupportedPolymerOrMixtureError`) or emit `UNSUPPORTED_POLYMER_OR_MIXTURE_WARNING` + `UNSUPPORTED_STRUCTURE_APPROXIMATION` if matched or overridden.
4. **Material-name collisions:** Hard reject name-only queries without explicit SMILES (`AmbiguousMaterialIdentityError`); emit `MATERIAL_NAME_STRUCTURE_CONFLICT_WARNING` if supplied name conflicts with SMILES; key strictly on canonical SMILES/scaffold.
5. **Absent from scaffold registry:** Strictly classified as unseen (`UNSEEN_DONOR_KNOWN_ACCEPTOR`, `UNSEEN_ACCEPTOR_KNOWN_DONOR`, or `DUAL_UNSEEN_SCAFFOLDS`), never silently in-domain.
6. **Missing donor or acceptor identity:** Hard reject (`MissingMolecularIdentityError`).

### 1.4 Artifact Compatibility & Unimplemented Prerequisites (`PHASE_8_3_INTEGRATION_CONTRACT.md`, `PHASE_8_3_FINAL_READINESS_REPORT.md`)
- Marked serialized artifact generation (`.joblib`, `.meta.json`, `.pipeline_state.json`, `scaffold_registry.json`), `OPVModelRegistry` refactoring (`predictor.py` currently hardcodes `22` features), and `ml/src/features/` unification as **`NOT YET IMPLEMENTED`**.
- Documented `HistGradientBoostingRegressor` OpenMP thread-pinning requirement (`OMP_NUM_THREADS=1`) since it lacks `n_jobs=1`.
- Documented the `phase_8_2_scaffold_benchmark.py` `extract_m3` `reg_ann` bug (`v` undefined $\to$ `regime_thermal_annealed=0`) and `phase_8_1` vs `phase_8_2` lookup/mapper drift so future production code strictly unifies from the bug-free Phase 8.1 `reg_ann` logic.

### 1.5 Four-Block Uncertainty Output Schema (`PHASE_8_3_INTEGRATION_CONTRACT.md`, `PHASE_8_3_MODEL_GOVERNANCE.md`)
- Separated API response schema into four explicit top-level objects: `point_prediction`, `interval_estimate`, `ood_status`, and `scientific_support_status`.
- Added explicit boolean non-guarantee fields (`coverage_guaranteed_for_individual_record: false`, `coverage_guaranteed_for_unseen_scaffolds: false`, `coverage_guaranteed_for_high_pce: false`, `coverage_guaranteed_under_distribution_shift: false`).

---

## 2. Mandatory Future Implementation Tasks (Before Backend/Frontend Integration)

1. **Task A (`NOT YET IMPLEMENTED`):** Refactor `FeaturePipeline` (using Phase 8.1 bug-free `reg_ann`, unified `SOLV_BP_MAP`/`ETL_WF_MAP`/`HTL_WF_MAP`, and strict `map_etl_nominal` exact `'ca'` token matching) into `ml/src/features/processing.py`.
2. **Task B (`NOT YET IMPLEMENTED`):** Update `ml/src/models/predictor.py` (`validate_scientific_artifact` & `OPVModelRegistry`) to support `feature_count in {22, 37, 44, 50}`, verify SHA-256 checksums, enforce `n_jobs=1` / `OMP_NUM_THREADS=1`, and load `pipeline_state.json` + `scaffold_registry.json`.
3. **Task C (`NOT YET IMPLEMENTED`):** Train and serialize locked Phase 8.1 / full-contract research bundles (`M0`, `M1`, `M2A`, `M3`) into `ml/models/saved_models/` only under an authorized Phase 9 integration plan.
