# OPV-PredictNet — Phase 9 Implementation Report

**Status:** Complete (Model Packaging & FastAPI Integration; stopped prior to Frontend integration)  
**Authoritative References:**
- Phase 8.3 Governance Contract (`ml/PHASE_8_3_MODEL_GOVERNANCE.md`)
- Phase 8.4 Unified Feature Pipeline (`ml/src/features/processing.py`)
- Phase 8.5 Corrected Structure-Generalization Benchmark (`ml/PHASE_8_5_CORRECTED_BENCHMARK_REPORT.md`)

---

## 1. Model Packaging Summary

**Packaging Script:** `ml/scripts/package_models_phase9.py`  
**Artifact Directory:** `ml/models/saved_models/phase9_v1/`

### 1.1 Dataset & Pipeline State
- **Dataset:** `ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv`
- **SHA-256 Verified:** `ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897`
- **Training Records:** 22,467 records (`pce` range: 0.00%–21.83%, mean: 9.89%)
- **Feature Pipeline:** `UnifiedFeaturePipeline` fitted on the full 22,467-record dataset and serialized to `phase9_pipeline_state.json`.
- **Scaffold Registry:** 1,073 unique donor Bemis–Murcko scaffolds and 812 unique acceptor Bemis–Murcko scaffolds serialized to `phase9_scaffold_registry.json`.

### 1.2 Packaged Models & Approved Configurations

| Model | Algorithm | Features | Artifact | Strategy C2 MAE / RMSE / R² | Reload Max Delta |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **M0** | `RandomForestRegressor(n_estimators=100, random_state=8888, n_jobs=1)` | 22 | `phase9_M0_model.joblib` | 2.3163 / 3.0573 / 0.6172 | `0.00e+00` |
| **M1** | `RandomForestRegressor(n_estimators=100, random_state=8888, n_jobs=1)` | 50 | `phase9_M1_model.joblib` | 2.1278 / 2.8724 / 0.6621 | `0.00e+00` |
| **M2A** | `RandomForestRegressor(n_estimators=100, random_state=8888, n_jobs=1)` | 37 | `phase9_M2A_model.joblib` | 2.1611 / 2.8878 / 0.6584 | `0.00e+00` |
| **M3** | `HistGradientBoostingRegressor(max_iter=100, random_state=8888)` | 44 | `phase9_M3_model.joblib` | 2.1638 / 2.8640 / 0.6640 | `0.00e+00` |

*Note:* **M2** remains archived and is excluded from packaging per Phase 8.3 governance.

### 1.3 Conformal Calibration Parameters (Phase 8.5 Strategy C2, $N_{\text{cal}} = 3,863$)

| Model | $q(80\%)$ | $q(90\%)$ | $q(95\%)$ |
| :--- | :---: | :---: | :---: |
| **M0** | 3.3730 | 4.6503 | 6.2258 |
| **M1** | 3.2106 | 4.2726 | 5.5214 |
| **M2A** | 3.2626 | 4.3230 | 5.5246 |
| **M3** | 3.1379 | 4.1624 | 5.2954 |

Global metadata is stored in `ml/models/saved_models/phase9_v1/phase9_model_metadata.json`.

---

## 2. FastAPI Prediction Service & Endpoints

**Service Module:** `backend/app/services/phase9_prediction.py` (`Phase9PredictionService`, `phase9_service`)  
**Schemas:** `backend/app/schemas/prediction.py`  
**Router:** `backend/app/api/v1/predictions.py`

### 2.1 Endpoints
- `GET /api/predictions/models` — Lists packaged Phase 9 models (`M0`, `M1`, `M2A`, `M3`), conformal quantiles, scaffold counts, and governance flags (`research_estimate_only=True`, `validated_for_production=False`).
- `POST /api/predictions/pce` — Authenticated (`Depends(get_current_user)`) PCE prediction endpoint returning `Phase9PredictionResponse` with strictly separated sections:
  - `point_prediction`: `predicted_pce`, `unit` (`"%"`)
  - `interval_estimate`: `prediction_interval_low`, `prediction_interval_high`, `confidence_level`, `conformal_quantile_q`, `interval_width`, `calibration_strategy`
  - `ood_status`: `ood_donor_scaffold`, `ood_acceptor_scaffold`, `ood_dual_scaffold`, `donor_scaffold_known`, `acceptor_scaffold_known`, `donor_scaffold_id`, `acceptor_scaffold_id`, `processing_ood_flags`, `severity`
  - `scientific_support`: `research_estimate_only=True`, `validated_for_production=False`, `support_level`, `governance_role`, `benchmark_metrics`, `limitations`
  - `warnings`: explicit list of molecular, processing, and OOD warnings

### 2.2 Input Validation & Safety Guards
- **Molecular Validation:** Uses `validate_and_parse_smiles(..., reject_wildcards_or_mixtures=True)` to reject invalid SMILES, empty strings, polymer wildcards (`*`), BigSMILES, and multi-component mixtures (`.`) with HTTP `422`.
- **Processing Validation:** Uses `validate_processing_record(..., strict_categorical_types=True)` to reject unphysical thickness/temperature/ratio values with HTTP `422` and surface Tier 3 range warnings.
- **Physical Output Sanity:** Uses `validate_prediction_sanity(pce=...)` to reject unphysical predictions (`< 0.0%` or `> 34.0%`) without silent clipping.

### 2.3 Out-of-Distribution (OOD) Policy
- **Unseen Donor Scaffold (`ood_donor_scaffold=True`):** Returns HTTP `200` with `OOD_DONOR_SCAFFOLD` warning (`severity="warning"`), noting Strategy A exploratory status for `M1` and `M3`.
- **Unseen Acceptor Scaffold (`ood_acceptor_scaffold=True`):** Returns HTTP `200` with `OOD_ACCEPTOR_SCAFFOLD` warning (`severity="warning"`), noting Strategy B high-PCE 90% empirical coverage collapse (~45%).
- **Dual-Scaffold OOD (`ood_dual_scaffold=True`):** Rejected with HTTP `422` (`DUAL_SCAFFOLD_OOD_REJECTED`) by default (`allow_dual_scaffold_ood=False`), citing Strategy C1 $R^2 \approx 0.08$. Explicit opt-in returns HTTP `200` with `severity="critical"` and `CRITICAL_DUAL_SCAFFOLD_OOD` warning.
- **Missing Processing Flags:** Explicitly surfaces `etl_missing`, `htl_missing`, and `missing_architecture`.

### 2.4 Prediction History Integration
- Persists every successful `POST /api/predictions/pce` response to the existing SQLite `predictions` table (`Prediction`) and `model_versions` table (`ModelVersion`) with user isolation (`user_id = current_user.id`).
- Stores the full conformal interval, OOD status, warnings, and hardcoded governance flags (`research_estimate_only=True`, `validated_for_production=False`) in `Prediction.uncertainty_json`.
- Retrievable via existing `GET /api/predictions` and `GET /api/predictions/{prediction_id}` endpoints.

---

## 3. Governance Compliance Checklist

- [x] `research_estimate_only=true` and `validated_for_production=false` hardcoded in all responses and persisted records
- [x] Dual-scaffold extrapolation rejected by default (`422`)
- [x] No test data used for model tuning
- [x] Only `UnifiedFeaturePipeline` used for feature extraction
- [x] Existing backend authentication (`get_current_user`) and user isolation preserved
- [x] No frontend modifications
- [x] Historical Phase 8.2 / 8.5 result files untouched
- [x] No Git commits made

---

## 4. Known Limitations (Phase 8.3 Registry)
1. **Research Estimate Only:** Not validated for production OPV device engineering or autonomous synthesis.
2. **Acceptor OOD High-PCE Collapse:** Under Strategy B, 90% conformal coverage drops to ~45% in top-performing PCE regimes.
3. **Dual-Scaffold Extrapolation:** Strategy C1 yields $R^2 \approx 0.08$ with wide intervals (16–19 pp at 90% nominal).
4. **Strategy C2 Calibration:** Empirical 90% coverage on Strategy C2 is ~88.0%–89.0% (mild undercoverage).
5. **Polymer / Mixture Exclusion:** Wildcard (`*`), BigSMILES, and multi-component (`.`) inputs are explicitly rejected.
