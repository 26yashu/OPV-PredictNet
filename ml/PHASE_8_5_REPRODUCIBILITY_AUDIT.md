# OPV-PredictNet — Phase 8.5: Reproducibility & Protocol Compliance Audit

**Date:** 2026-11-25  
**Scope:** Independent verification of frozen protocol fidelity, split isolation, pipeline authority, determinism, and repository boundary compliance for Phase 8.5.  
**Verification Script:** [test_phase_8_5_reproduction.py](file:///c:/Users/DELL/OPV-PredictNet/tests/ml/test_phase_8_5_reproduction.py)

---

## 1. Frozen Protocol Fidelity & Discrepancy Documentation

Per Phase 8.5 Section 1 (*"If any original split or configuration cannot be reproduced exactly, document the discrepancy before execution"*):

| Protocol Element | Phase 8.2 Original Reference | Phase 8.5 Status | Verification Detail |
| :--- | :--- | :---: | :--- |
| **Dataset & SHA-256** | `opv_devices_strict_molecular_benchmark.csv` (`ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897`) | **EXACT MATCH** | `22,467` records; SHA-256 verified at runtime. |
| **Partition Seeds** | `SEED_A=7701`, `SEED_B=7702`, `SEED_C2=7703`, `MODEL_SEED=8888` | **EXACT MATCH** | Recorded in [phase_8_5_config.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_config.json). |
| **Scaffold Grouping Cache** | `scaffold_cache.json` (`donor_scaffolds`, `acceptor_scaffolds`) | **FROZEN WITH AUDIT NOTE** | **Discrepancy Documented:** `2` acyclic peptide SMILES (`1` donor, `1` acceptor) stored raw SMILES after `[NO_RING]:` in `scaffold_cache.json` whereas `validate_and_parse_smiles()` places canonical SMILES after `[NO_RING]:`. Re-sorting strings prior to `rng.shuffle` would alter partition indices in Strategies B, C1, and C2. To keep test populations 100% identical to Phase 8.2, `scaffold_cache.json` groupings were frozen verbatim (exact discrepancy logged in [phase_8_5_split_metadata.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_split_metadata.json)). |
| **Partition Counts** | A: `15750/5236/1481`, B: `17658/1012/3797`, C1: `13324/260/284`, C2: `16537/3863/2067` | **EXACT MATCH** | Asserted against `phase_8_2_config.json`; index SHA-256 digests archived. |
| **Model Configs & Threads** | `RF(100, seed=8888, n_jobs=1)` (`M0/M1/M2A`), `HGBR(max_iter=100, seed=8888)` (`M3`) | **EXACT MATCH** | Identical estimators, seeds, and `n_jobs=1`. |
| **Feature Extraction** | Phase 8.2 used drifted `FeaturePipeline` | **REPLACED BY UNIFIED** | 100% of `M0/M1/M2A/M3` extraction routed through `ml.src.features.processing.UnifiedFeaturePipeline`. `M2` kept archived. |
| **Conformal Calibration** | Disjoint `D_cal` quantile $k=\lceil(n+1)(1-\alpha)\rceil$, $\alpha\in\{0.20,0.10,0.05\}$ | **EXACT MATCH** | Unclipped $[ \hat{y}-q_\alpha, \hat{y}+q_\alpha ]$. |

---

## 2. Zero-Leakage & Isolation Verification

*(Source: [phase_8_5_split_metadata.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_split_metadata.json))*

| Strategy | Held-Out Structural Overlap (`Tr&Cl / Tr&Ts / Cl&Ts`) | `chem_pair` Overlap | Pipeline Fitting Isolation |
| :--- | :---: | :---: | :--- |
| **Strategy A** | `donor_scaffold`: `0 / 0 / 0` (`donor_smiles`: `0 / 0 / 0`) | `0 / 0 / 0` | Fresh `UnifiedFeaturePipeline` per strategy; `fit_training(df_train)` called strictly on `D_train`. |
| **Strategy B** | `acceptor_scaffold`: `0 / 0 / 0` (`acceptor_smiles`: `0 / 0 / 0`) | `0 / 0 / 0` | Fresh `UnifiedFeaturePipeline`; `D_cal` / `D_test` medians untouched. |
| **Strategy C1** | `donor_scaffold`: `0 / 0 / 0`; `acceptor_scaffold`: `0 / 0 / 0` | `0 / 0 / 0` | Disjoint dual-scaffold holdout (`8,599` cross-edges dropped identically to Phase 8.2). |
| **Strategy C2** | `scaffold_pair`: `0 / 0 / 0` (`donor_scaffold` `123` Tr&Ts, `acceptor_scaffold` `106` Tr&Ts — pair holdout only) | `0 / 0 / 0` | Disjoint `scaffold_pair` holdout; exact known-scaffold overlap retained per Phase 8.2.1. |

**Cross-Verification Checks Passing:**
- **M0 Bit-for-Bit Reproduction:** `delta_mae = 0.0000`, `delta_rmse = 0.0000`, `delta_r2 = 0.0000` across all 4 strategies vs. Phase 8.2.
- **Attribution Baseline Check:** `v0_phase_8_2_baseline` reproduces frozen `phase_8_2_scaffold_metrics.csv` `M3` MAE across all 4 strategies within `< 1e-4`.

---

## 3. Environment & Determinism Record

- **Python / OS:** `3.11.9` / `Windows-10-10.0.26200-SP0` (`n_jobs=1`)
- **ML / Chemistry Libraries:** `scikit-learn 1.9.1`, `RDKit 2026.09.2`
- **Artifacts Generated Under `ml/results/phase_8_5/`:**
  - `phase_8_5_config.json`
  - `phase_8_5_split_metadata.json`
  - `phase_8_5_scaffold_metrics.csv`
  - `phase_8_5_interval_metrics.csv`
  - `phase_8_5_paired_comparisons.csv`
  - `phase_8_5_subgroup_metrics.csv`
  - `phase_8_5_scaffold_pair_subgroups.csv`
  - `phase_8_5_test_predictions.csv` (`7,629` rows = `1,481 + 3,797 + 284 + 2,067`)
  - `phase_8_5_historical_comparison.csv`
  - `phase_8_5_m3_attribution_decomposition.csv`

---

## 4. Acceptance & Boundary Compliance Checklist

- [x] Frozen split definitions and test populations reproduced identically.
- [x] Zero train/calibration/test structural leakage on held-out margins (`chem_pair` overlap `0` across all strategies).
- [x] All active models (`M0`, `M1`, `M2A`, `M3`) use `UnifiedFeaturePipeline` (`M2` archived).
- [x] Original Phase 5–8.2 CSV/JSON result files preserved unmodified.
- [x] All historical Phase 8.2 vs. Phase 8.5 differences causally decomposed (`M1` nominal ETL mapping, `M2A` physical lookup tables + shifted medians, `M3` lookups + `missing_da_ratio`; `regime_thermal_annealed` verified `0.0000` delta).
- [x] Statistical limitations retained (record-level tests invalid; separate one-way cluster bootstraps only; no joint crossed-dependence test claimed; no prospective validation claimed).
- [x] Zero hyperparameter tuning, zero backend/frontend modifications, zero `.joblib` production artifacts, zero Git commits.
