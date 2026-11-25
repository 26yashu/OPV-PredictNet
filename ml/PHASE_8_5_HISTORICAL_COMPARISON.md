# OPV-PredictNet — Phase 8.5: Historical Comparison & Feature Attribution Analysis

**Date:** 2026-11-25  
**Comparison Scope:** Frozen Historical Phase 8.2 ([phase_8_2_scaffold_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_scaffold_metrics.csv)) vs. Corrected Phase 8.5 ([phase_8_5_scaffold_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_scaffold_metrics.csv)).  
**Side-by-Side Artifacts:** [phase_8_5_historical_comparison.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_historical_comparison.csv) & [phase_8_5_m3_attribution_decomposition.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_m3_attribution_decomposition.csv).

---

## 1. Executive Summary of Historical Differences

Re-running Phase 8.2 with the authoritative Phase 8.4 `UnifiedFeaturePipeline` ([processing.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/features/processing.py)) yields **critical mechanistic findings that overturn prior assumptions**:

1. **Exact `M0` Parity (`0.0000` Delta):** All 4 strategies reproduce Phase 8.2 `M0` MAE, RMSE, $R^2$, bias, 90% coverage, and 90% width identically (`delta = 0.0000`), confirming zero partition or descriptor drift.
2. **`regime_thermal_annealed` Fix Contributed `0.0000` to `M3` Metric Changes:** Although Phase 8.2 `extract_m3` suffered a silent `NameError` that zeroed `regime_thermal_annealed` across 100% of rows (`5,436` rows affected vs Phase 8.5), controlled counterfactual decomposition proves **`0.0000` difference across all 4 strategies** between historical M3 (`v0`) and thermal-annealing-fixed M3 (`v1`), with max absolute prediction difference `0.0` on Strategy A.
   - *Mathematical Reason:* Across all `22,467` records, `(regime_thermal_annealed == 1)` is logically equivalent to `(annealing_temp_norm >= -1.2)` (where `regime == 0` has `max(annealing_temp_norm) == -1.3` or `NaN`). `HistGradientBoostingRegressor` already isolates thermally annealed regimes via a single native split on `annealing_temp_norm`, making the binary `regime_thermal_annealed` column completely redundant informationally in `M3`.
3. **100% of `M3` Deltas Are Driven by Physical Lookups & `missing_da_ratio` Edge Cases:**
   - **Solvent & Interlayer Lookup Alignment (`SOLV_BP_MAP`, `ETL_WF_MAP`, `HTL_WF_MAP`):** Phase 8.2 drifted lookup dictionaries (adding `'tio2': 4.20`, `'vox': 5.40`, `'ni': 5.10`, `'xylene': 138.5`, shifting `'pdinn'` `4.00 -> 4.15`, `'pfn-br'` `4.20 -> 4.10`, and dropping `'2pacz': 5.20`), altering `723` solvent, `3,896` ETL, and `691` HTL entries in non-imputed `M3` (and `15,860` ETL rows in `M2A` via shifted `D_train` `etl_median`). Restoring locked Phase 8.0.3 / 8.1 tables accounts for `+0.0052` (A), `+0.0413` (B), `+0.0638` (C1), and `+0.0138` (C2) MAE change in `M3`.
   - **`missing_da_ratio` Edge-Case Correction (`29` rows):** Phase 8.2 marked `29` degenerate zero-component D:A ratios (`'1:0'`, `'0:1'`, `'100:0'`, etc.) as `missing_da_ratio = 0` despite `donor_blend_fraction = NaN`. Fixing this via `parse_f_donor` parity drove the remaining `+0.0562 pp` shift in small-sample Strategy C1 (`N=284`) and minor shifts in A/B/C2.
4. **Nominal Category Mapping Improved `M1` Across Every Strategy:** Fixing `map_etl_nominal` (`'ca' in s` collision on 46 non-calcium strings, `'pfn' in s`, `'PDINN/PFN-Br'` precedence) and `map_htl_nominal` (`'mox'`) **reduced `M1` MAE in all four strategies** (`-0.0037`, `-0.0148`, `-0.0185`, and `-0.0115 pp`), elevating `M1` to the lowest-MAE model on Strategy C2 (`2.1278%`).

---

## 2. Side-by-Side Comparison Table (Phase 8.2 vs. Phase 8.5)

*(Source: [phase_8_5_historical_comparison.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_historical_comparison.csv))*

| Strategy | Model | Phase 8.2 MAE | Phase 8.5 MAE | $\Delta\text{MAE}$ | Phase 8.2 $R^2$ | Phase 8.5 $R^2$ | $\Delta R^2$ | 90% Cov (8.2 $\to$ 8.5) | 90% Width (8.2 $\to$ 8.5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** | **M0** | 2.2302 | 2.2302 | **0.0000** | 0.5839 | 0.5839 | **0.0000** | 97.57% $\to$ 97.57% | 13.84 $\to$ 13.84 |
| `(N=1,481)` | **M1** | 2.0916 | 2.0879 | **-0.0037** | 0.6318 | 0.6321 | +0.0003 | 93.79% $\to$ 94.06% | 10.25 $\to$ 10.30 |
| | **M2A** | 2.1470 | 2.1494 | +0.0024 | 0.6175 | 0.6167 | -0.0008 | 94.46% $\to$ 94.94% | 10.48 $\to$ 10.75 |
| | **M3** | 2.0025 | 2.0150 | +0.0125 | 0.6428 | 0.6431 | +0.0003 | 92.57% $\to$ 92.37% | 9.26 $\to$ 9.43 |
| **Strategy B** | **M0** | 2.2243 | 2.2243 | **0.0000** | 0.4396 | 0.4396 | **0.0000** | 94.05% $\to$ 94.05% | 10.84 $\to$ 10.84 |
| `(N=3,797)` | **M1** | 2.1820 | 2.1672 | **-0.0148** | 0.4415 | 0.4472 | +0.0057 | 95.02% $\to$ 94.76% | 11.15 $\to$ 11.11 |
| | **M2A** | 2.1158 | 2.0853 | **-0.0305** | 0.4795 | 0.4876 | +0.0081 | 96.29% $\to$ 96.08% | 11.83 $\to$ 11.67 |
| | **M3** | 1.9463 | 1.9870 | +0.0407 | 0.5375 | 0.5212 | -0.0163 | 94.63% $\to$ 93.86% | 10.38 $\to$ 9.95 |
| **Strategy C1** | **M0** | 2.5740 | 2.5740 | **0.0000** | 0.0786 | 0.0786 | **0.0000** | 99.65% $\to$ 99.65% | 16.52 $\to$ 16.52 |
| `(N=284)` | **M1** | 2.3999 | 2.3814 | **-0.0185** | 0.1672 | 0.1791 | +0.0119 | 98.94% $\to$ 99.65% | 16.47 $\to$ 17.04 |
| | **M2A** | 2.3314 | 2.3660 | +0.0346 | 0.2440 | 0.2093 | -0.0347 | 99.65% $\to$ 99.65% | 18.62 $\to$ 17.69 |
| | **M3** | 2.3013 | 2.4213 | +0.1200 | 0.2385 | 0.1435 | -0.0950 | 98.59% $\to$ 98.59% | 15.69 $\to$ 16.58 |
| **Strategy C2** | **M0** | 2.3163 | 2.3163 | **0.0000** | 0.6172 | 0.6172 | **0.0000** | 88.63% $\to$ 88.63% | 9.30 $\to$ 9.30 |
| `(N=2,067)` | **M1** | 2.1393 | 2.1278 | **-0.0115** | 0.6578 | 0.6621 | +0.0043 | 88.05% $\to$ 88.05% | 8.57 $\to$ 8.55 |
| | **M2A** | 2.1436 | 2.1611 | +0.0175 | 0.6582 | 0.6584 | +0.0002 | 89.89% $\to$ 88.97% | 8.77 $\to$ 8.65 |
| | **M3** | 2.1536 | 2.1638 | +0.0102 | 0.6667 | 0.6640 | -0.0027 | 88.44% $\to$ 88.20% | 8.28 $\to$ 8.32 |

---

## 3. Causal Attribution Decomposition by Mechanism

### 3.1 `M3` Multi-Stage Attribution Decomposition

*(Source: [phase_8_5_m3_attribution_decomposition.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_m3_attribution_decomposition.csv))*

| Strategy | `v0` Phase 8.2 Baseline MAE | `v1` Thermal Annealing Fix Only ($\Delta$ vs `v0`) | `v2` Lookup Tables Fix Only ($\Delta$ vs `v0`) | `v3` Lookups + `missing_da_ratio` ($\Delta$ vs `v2`) | `v4` Full Phase 8.5 Unified M3 ($\Delta$ vs `v3`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** | 2.0025 | 2.0025 (**0.0000**) | 2.0077 (**+0.0052**) | 2.0150 (**+0.0073**) | 2.0150 (**0.0000**) |
| **Strategy B** | 1.9463 | 1.9463 (**0.0000**) | 1.9876 (**+0.0413**) | 1.9870 (**-0.0006**) | 1.9870 (**0.0000**) |
| **Strategy C1** | 2.3013 | 2.3013 (**0.0000**) | 2.3651 (**+0.0638**) | 2.4213 (**+0.0562**) | 2.4213 (**0.0000**) |
| **Strategy C2** | 2.1536 | 2.1536 (**0.0000**) | 2.1674 (**+0.0138**) | 2.1638 (**-0.0037**) | 2.1638 (**0.0000**) |

**Interpretation:**
- **Thermal Annealing Fix (`v1 - v0` and `v4 - v3`):** Literally **0.0000 pp** in every strategy because `HistGradientBoostingRegressor` learns the exact annealing split from `annealing_temp_norm >= -1.2`.
- **Solvent & Interlayer Lookup Tables (`v2 - v0`):** Phase 8.2's unapproved lookup expansion (`tio2`, `vox`, `ni`, `xylene`) happened to supply numeric work-function proxies for several unseen-acceptor and small-C1 subsets that reduced test MAE marginally in Phase 8.2 (`0.0052–0.0638 pp`), whereas the locked Phase 8.0.3 / 8.1 tables restrict lookups to the audited curated materials (`SOLV_BP_MAP`, `ETL_WF_MAP`, `HTL_WF_MAP`) and retain `NaN` for unverified interlayers.
- **`missing_da_ratio` Consistency Fix (`v3 - v2`):** Reclassifying the 29 zero-component D:A strings (`'1:0'`, `'0:1'`, etc.) from `0` to `1` alters `M3` training splits slightly, producing `+0.0562 pp` variance on the small `N=284` C1 test sample and minor shifts elsewhere.

### 3.2 `M1` & `M2A` Mechanism Attribution

- **`M1` (`etl_ca`, `etl_pfn_br`, `etl_pdinn`, `etl_other`):** Column-diff verification across all `22,467` records shows zero continuous or non-ETL differences in M1. Every improvement in M1 (`-0.0037` to `-0.0185 pp`) comes exclusively from fixing `map_etl_nominal` (stopping `'ca' in s` false collisions on `ZrAcac`/`PDIN:DCA`, capturing `'pfn'`, and resolving `'PDINN/PFN-Br'` precedence) and `map_htl_nominal` (`'mox'`).
- **`M2A` (`solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`):** Column-diff verification confirms only these 3 physical lookup columns differ in M2A (`723`, `15,860`, and `691` records respectively across the full dataset; `15,860` ETL records differ because `D_train` `etl_median` shifted from `4.15` to `4.20 eV` in Strategies B/C2 and to `4.30 eV` in Strategy C1). Swapping individual columns in Strategy B showed `etl_workfunction_ev` alone (`2.0855%`), `solvent_boiling_point_c` (`2.0939%`), and `htl_workfunction_ev` (`2.1096%`) together explain the `-0.0305 pp` gain over Phase 8.2's `2.1158%`.

---

## 4. Impact on Governance & Model Roles (Phase 8.3/8.3.1 Continuity)

1. **Historical Freeze Preserved:** Original `ml/results/phase_8_2_*.csv` and `phase_8_2_config.json` remain untouched.
2. **Governance Conclusions Confirmed & Strengthened:**
   - **No Model Is Ready for Prospective Quantitative PCE Prediction Under Dual-Scaffold Shift:** Strategy C1 remains collapsed (`R² = 0.079–0.209`) with descriptive/inconclusive model differences.
   - **Strategy B Remains Statistically Inconclusive:** `M3 vs M0` acceptor-scaffold bootstrap CI `[-0.3238, +0.0094]` still crosses zero due to `PC61BM` (`61.18%`) dominance.
   - **Exploratory Candidate Status Updated:** `M1` (fixed nominal ETL mapping) now shows stronger error stability across all 4 strategies (`-0.0037` to `-0.0185 pp` vs Phase 8.2, lowest MAE on C2 at `2.1278%`) while **`M3`** retains lowest MAE on Strategy A (`2.0150%`) and Strategy B (`1.9870%`). Both remain exploratory processing-aware candidates subject to strict out-of-domain and conformal distribution-shift warnings.
