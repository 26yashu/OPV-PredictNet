# OPV-PredictNet — Phase 8.5: Corrected Structure-Generalization Benchmark Report

**Date:** 2026-11-25  
**Protocol Status:** Controlled reproduction of Phase 8.2 using ONLY the authoritative Phase 8.4 `UnifiedFeaturePipeline` ([processing.py](file:///c:/Users/DELL/OPV-PredictNet/ml/src/features/processing.py)).  
**Execution Script:** [phase_8_5_corrected_benchmark.py](file:///c:/Users/DELL/OPV-PredictNet/ml/experiments/phase_8_5_corrected_benchmark.py)  
**Result Artifacts Directory:** `ml/results/phase_8_5/`  
**Governance & Restrictions:** Zero hyperparameter tuning, zero model architecture changes, zero split redesign, zero modification of historical Phase 5–8.2 CSV/JSON files, zero production backend/frontend edits, zero Git commits.

---

## 1. Executive Summary

Phase 8.5 re-executed all four structure-generalization strategies (**Strategy A, B, C1, C2**) under frozen Phase 8.2 partitions, random seeds (`7701`, `7702`, `7703`, `8888`), model architectures (`n_jobs=1`), and conformal calibration rules while replacing Phase 8.2's ad-hoc extractor with `UnifiedFeaturePipeline` across **M0 (22)**, **M1 (50)**, **M2A (37)**, and **M3 (44)** (`M2` remains archived).

Key findings:
1. **Exact Baseline Reproducibility (`M0`):** `M0` reproduces historical Phase 8.2 metrics to `0.0000` across all 4 strategies, proving identical partitions, descriptor extraction, and random forest execution.
2. **Corrected Relative Performance Across Strategies:**
   - **Strategy A (`N=1,481`, Unseen Donor Scaffold):** `M3` achieves the lowest MAE (`2.0150%`) and `R² = 0.6431` (macro donor-scaffold MAE `1.9913%`), followed by `M1` (`MAE = 2.0879%`, `R² = 0.6321`). Both `M1 vs M0` (`-0.1424 pp`) and `M3 vs M0` (`-0.2152 pp`) exclude zero across all five separate one-way cluster-bootstrap sensitivity margins (exploratory directional signal; no joint crossed-dependence inference).
   - **Strategy B (`N=3,797`, Unseen Acceptor Scaffold):** `M3` (`1.9870%`) and `M2A` (`2.0853%`) reduce MAE relative to `M0` (`2.2243%`), **but all models remain statistically inconclusive under acceptor-scaffold clustering** because `PC61BM` represents `2,323 / 3,797 (61.18%)` of test records (`M3 vs M0` acceptor-scaffold 95% CI `[-0.3238, +0.0094]`).
   - **Strategy C1 (`N=284`, Strict Dual-Scaffold Holdout):** All models degrade severely (`R² = 0.0786` for `M0`, `0.1791` for `M1`, `0.2093` for `M2A`, `0.1435` for `M3`), and all five separate one-way bootstrap margins cross zero (`M3 vs M0` primary 95% CI `[-0.5417, +0.2102]`). All C1 comparisons are purely descriptive.
   - **Strategy C2 (`N=2,067`, Scaffold-Pair Holdout):** Fixing the nominal ETL mapping improved `M1` to the lowest MAE (`2.1278%`, macro scaffold-pair MAE `2.1729%`), ahead of `M2A` (`2.1611%`) and `M3` (`2.1638%`, `R² = 0.6640`). Recall from Phase 8.2.1 that C2 tests novel scaffold combinations rather than unseen individual scaffolds (`73.25%` share training donor scaffolds; `84.86%` share training acceptor scaffolds).
3. **Conformal Exchangeability Failures Persist:** As diagnosed in Phase 8.2.1, unnormalized split-conformal coverage fluctuates from severe overcoverage in Strategy C1 (`98.59%–99.65%` at 90% nominal; width `16.52–17.69 pp`) and Strategy A (`92.37%–97.57%`) to systematic undercoverage in Strategy C2 (`75.71%–77.94%` at 80% nominal; `88.05%–88.97%` at 90%) and collapses on High-PCE (`>= 15%`) unseen acceptors in Strategy B (`45.07%–59.15%`).

---

## 2. Target Distributions & Frozen Partition Summary

*(Source: [phase_8_5_split_metadata.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_split_metadata.json) and [phase_8_5_config.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_config.json))*

Dataset verified: `ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv` (`22,467` records; SHA-256 `ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897`).

| Strategy (Split Basis) | Partition | $N$ | PCE Mean ± Std (%) | Median [IQR] (%) | Range [Min, Max] (%) | High-PCE $\ge 15\%$ $n$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** | Train | 15,750 | 10.69 ± 5.41 | 10.64 [9.28] | [0.004, 21.83] | 4,792 (30.43%) |
| *(Donor Scaffold)* | Cal | 5,236 | 8.42 ± 6.57 | 5.82 [12.50] | [0.000, 21.26] | 1,365 (26.07%) |
| | Test | 1,481 | **6.60 ± 4.43** | **5.66 [5.92]** | [0.020, 19.20] | 115 (7.77%) |
| **Strategy B** | Train | 17,658 | 11.04 ± 5.60 | 11.15 [9.83] | [0.004, 21.83] | 6,057 (34.30%) |
| *(Acceptor Scaffold)* | Cal | 1,012 | 8.12 ± 5.19 | 7.00 [7.06] | [0.050, 21.26] | 144 (14.23%) |
| | Test | 3,797 | **5.03 ± 3.83** | **3.72 [5.13]** | [0.000, 21.04] | 71 (1.87%) |
| **Strategy C1** | Train | 13,324 | 11.37 ± 5.29 | 11.72 [8.88] | [0.004, 21.83] | 4,662 (34.99%) |
| *(Dual Scaffold)* | Cal | 260 | 7.79 ± 6.53 | 5.22 [9.95] | [0.070, 21.26] | 55 (21.15%) |
| *(8,599 cross-edges dropped)* | Test | 284 | **4.84 ± 3.29** | **4.10 [3.91]** | [0.040, 16.70] | 6 (2.11%) |
| **Strategy C2** | Train | 16,537 | 10.62 ± 6.02 | 10.62 [11.35] | [0.000, 21.63] | 5,881 (35.56%) |
| *(Scaffold Pair)* | Cal | 3,863 | 8.17 ± 4.23 | 8.40 [6.19] | [0.005, 21.83] | 186 (4.81%) |
| | Test | 2,067 | **7.32 ± 4.94** | **6.42 [7.08]** | [0.013, 21.26] | 205 (9.92%) |

---

## 3. Master Point-Prediction Performance

*(Source: [phase_8_5_scaffold_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_scaffold_metrics.csv))*

| Strategy ($N_{\text{test}}$) | Model ($p$) | MAE (% PCE) | RMSE (% PCE) | $R^2$ | Signed Bias (% PCE) | Macro Donor-Scaff MAE | Macro Acceptor-Scaff MAE | Macro Pair MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** | **M0** (22) | 2.2302 | 2.8574 | 0.5839 | -0.2930 | 2.2257 | 2.3800 | 2.3023 |
| `(N=1,481)` | **M1** (50) | 2.0879 | 2.6868 | 0.6321 | -0.0712 | 2.0183 | 2.2793 | 2.1042 |
| | **M2A** (37) | 2.1494 | 2.7427 | 0.6167 | -0.0355 | 2.0692 | 2.3855 | 2.1681 |
| | **M3** (44) | **2.0150** | **2.6465** | **0.6431** | -0.0508 | **1.9913** | **2.1324** | **2.0623** |
| **Strategy B** | **M0** (22) | 2.2243 | 2.8694 | 0.4396 | +0.7019 | 2.3204 | 2.6466 | 2.4867 |
| `(N=3,797)` | **M1** (50) | 2.1672 | 2.8496 | 0.4472 | +0.7791 | 2.1718 | 2.5275 | 2.3420 |
| | **M2A** (37) | 2.0853 | 2.7437 | 0.4876 | +0.6812 | 2.1610 | 2.5206 | 2.3501 |
| | **M3** (44) | **1.9870** | **2.6522** | **0.5212** | +0.4957 | **1.9731** | **2.4054** | **2.1859** |
| **Strategy C1** | **M0** (22) | 2.5740 | 3.1599 | 0.0786 | +0.2447 | 2.7280 | 2.6351 | 2.7406 |
| `(N=284)` | **M1** (50) | 2.3814 | 2.9827 | 0.1791 | +0.5875 | 2.4952 | 2.4361 | 2.5172 |
| | **M2A** (37) | **2.3660** | **2.9274** | **0.2093** | +0.5656 | 2.5406 | **2.3356** | 2.5348 |
| | **M3** (44) | 2.4213 | 3.0466 | 0.1435 | +0.3846 | **2.1982** | 2.6016 | **2.2469** |
| **Strategy C2** | **M0** (22) | 2.3163 | 3.0573 | 0.6172 | -0.0898 | 2.2027 | 2.5400 | 2.3509 |
| `(N=2,067)` | **M1** (50) | **2.1278** | 2.8724 | 0.6621 | -0.0732 | 2.0077 | **2.4059** | **2.1729** |
| | **M2A** (37) | 2.1611 | 2.8878 | 0.6584 | -0.0172 | **1.9941** | 2.4428 | 2.1831 |
| | **M3** (44) | 2.1638 | **2.8640** | **0.6640** | +0.0034 | 2.0195 | 2.4366 | 2.2072 |

**Cluster Imbalance Effect:** Notice how macro-averaged `acceptor_scaffold` MAE in Strategy B (`2.4054–2.6466%`) is `+0.42 pp` higher than record-weighted MAE (`1.9870–2.2243%`), confirming that `PC61BM` (`n=2,323`, `61.18%` of Strategy B test rows) heavily depresses record-weighted error.

---

## 4. Conformal Uncertainty Calibration Results (80%, 90%, 95%)

*(Source: [phase_8_5_interval_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_interval_metrics.csv))*

Unnormalized split conformal calibration applies a single threshold $q_\alpha$ per strategy and model, so `mean_width == median_width == 2 * q_alpha` across every test row.

| Strategy ($N_{\text{cal}}/N_{\text{test}}$) | Model | 80% Cov (Width, $q_{0.20}$) | 90% Cov (Width, $q_{0.10}$) | 95% Cov (Width, $q_{0.05}$) | High-PCE $\ge 15\%$ 90% Cov ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** | M0 | 94.87% (11.26, 5.6315) | 97.57% (13.84, 6.9215) | 98.51% (15.42, 7.7115) | 96.52% (115) |
| `(5,236 / 1,481)` | M1 | 87.44% (7.98, 3.9899) | 94.06% (10.30, 5.1524) | 97.23% (12.36, 6.1778) | 88.70% (115) |
| | M2A | 88.66% (8.41, 4.2026) | 94.94% (10.75, 5.3746) | 97.03% (12.59, 6.2967) | 91.30% (115) |
| | M3 | 86.29% (7.47, 3.7344) | 92.37% (9.43, 4.7129) | 95.68% (11.65, 5.8255) | 87.83% (115) |
| **Strategy B** | M0 | 84.20% (7.50, 3.7479) | 94.05% (10.84, 5.4202) | 97.79% (13.86, 6.9278) | **45.07%** (71) |
| `(1,012 / 3,797)` | M1 | 88.12% (8.04, 4.0208) | 94.76% (11.11, 5.5526) | 97.81% (15.15, 7.5743) | **50.70%** (71) |
| | M2A | 88.25% (7.95, 3.9756) | 96.08% (11.67, 5.8366) | 98.21% (15.22, 7.6095) | **54.93%** (71) |
| | M3 | 87.75% (7.49, 3.7469) | 93.86% (9.95, 4.9732) | 96.37% (12.09, 6.0455) | **59.15%** (71) |
| **Strategy C1** | M0 | **95.42%** (12.33, 6.1662) | **99.65%** (16.52, 8.2587) | **99.65%** (18.24, 9.1187) | 83.33% (6) |
| `(260 / 284)` | M1 | **96.48%** (11.53, 5.7674) | **99.65%** (17.04, 8.5186) | **99.65%** (19.06, 9.5295) | 83.33% (6) |
| | M2A | **96.13%** (11.83, 5.9128) | **99.65%** (17.69, 8.8465) | **99.65%** (19.59, 9.7955) | 83.33% (6) |
| | M3 | **93.31%** (11.80, 5.9003) | **98.59%** (16.58, 8.2879) | **98.94%** (17.89, 8.9437) | 83.33% (6) |
| **Strategy C2** | M0 | **75.71%** (6.75, 3.3730) | **88.63%** (9.30, 4.6503) | 95.31% (12.45, 6.2258) | **73.66%** (205) |
| `(3,863 / 2,067)` | M1 | **77.75%** (6.42, 3.2106) | **88.05%** (8.55, 4.2726) | **94.44%** (11.04, 5.5214) | **75.12%** (205) |
| | M2A | **77.79%** (6.53, 3.2626) | **88.97%** (8.65, 4.3230) | **94.63%** (11.05, 5.5246) | **77.56%** (205) |
| | M3 | **77.94%** (6.28, 3.1379) | **88.20%** (8.32, 4.1624) | **93.76%** (10.59, 5.2954) | **78.05%** (205) |

---

## 5. Multi-Margin Separate One-Way Cluster-Bootstrap Sensitivity Analysis

*(Source: [phase_8_5_paired_comparisons.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_paired_comparisons.csv))*

Per Phase 8.2.1 governance, record-level i.i.d. tests (`ttest_rel`, `wilcoxon`) are invalid under repeated donors, acceptors, scaffolds, and papers and are omitted from inferential claims. Because no joint crossed-dependence inferential test is implemented (`joint_crossed_dependence_tested = False`), all intervals below are **separate one-way cluster-bootstrap sensitivity analyses** ($B=1,000$, `seed=8888`) on $\Delta \text{AE} = |y - \hat{y}_{M}| - |y - \hat{y}_{M0}|$:

| Strategy ($N$) | Comparison | Mean $\Delta\text{AE}$ | `donor_scaffold` 95% CI ($G$) | `acceptor_scaffold` 95% CI ($G$) | `scaffold_pair` 95% CI ($G$) | `pair_id` 95% CI ($G$) | `doi_norm` 95% CI ($G$) | Exploratory Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Strategy A** | **M1 vs M0** | -0.1424 | [-0.2860, -0.0198] (216) | [-0.2210, -0.0285] (71) | [-0.2791, -0.0197] (384) | [-0.2714, -0.0248] (623) | [-0.2513, -0.0421] (513) | Consistent directional signal (all 5 < 0) |
| `(1,481)` | **M2A vs M0** | -0.0808 | [-0.2531, +0.0591] (216) | [-0.1651, +0.0291] (71) | [-0.2373, +0.0524] (384) | [-0.2236, +0.0497] (623) | [-0.2031, +0.0187] (513) | Not significant (5/5 cross 0) |
| | **M3 vs M0** | -0.2152 | [-0.3638, -0.0711] (216) | [-0.3661, -0.0967] (71) | [-0.3507, -0.0853] (384) | [-0.3570, -0.0726] (623) | [-0.3427, -0.1001] (513) | Consistent directional signal (all 5 < 0) |
| **Strategy B** | **M1 vs M0** | -0.0570 | [-0.1635, -0.0193] (323) | [-0.1179, +0.0334] (163) | [-0.1499, -0.0122] (603) | [-0.1262, +0.0259] (887) | [-0.1186, +0.0112] (1163) | Inconclusive (`acceptor_scaffold` crosses 0) |
| `(3,797)` | **M2A vs M0** | -0.1390 | [-0.1915, -0.0765] (323) | [-0.1849, +0.0284] (163) | [-0.1864, -0.0711] (603) | [-0.2070, -0.0485] (887) | [-0.1959, -0.0729] (1163) | Inconclusive (`acceptor_scaffold` crosses 0) |
| | **M3 vs M0** | -0.2373 | [-0.3399, -0.1489] (323) | [-0.3238, +0.0094] (163) | [-0.3231, -0.1344] (603) | [-0.3364, -0.1086] (887) | [-0.3193, -0.1541] (1163) | Inconclusive (`acceptor_scaffold` crosses 0) |
| **Strategy C1** | **M1 vs M0** | -0.1925 | [-0.4226, +0.0347] (73) | [-0.3703, +0.1707] (17) | [-0.4197, +0.0271] (92) | [-0.4039, +0.0171] (114) | [-0.4267, +0.0270] (116) | Not significant (5/5 cross 0) |
| `(284)` | **M2A vs M0** | -0.2079 | [-0.4997, +0.0374] (73) | [-0.6998, +0.1089] (17) | [-0.4632, +0.0160] (92) | [-0.4293, -0.0161] (114) | [-0.4365, -0.0057] (116) | Inconclusive (structural axes cross 0) |
| | **M3 vs M0** | -0.1527 | [-0.6365, +0.2391] (73) | [-0.5167, +0.8762] (17) | [-0.5417, +0.2102] (92) | [-0.4855, +0.1606] (114) | [-0.4806, +0.1895] (116) | Not significant (5/5 cross 0) |
| **Strategy C2** | **M1 vs M0** | -0.1885 | [-0.2725, -0.1024] (310) | [-0.3184, -0.0715] (231) | [-0.2806, -0.0926] (565) | [-0.2693, -0.1113] (921) | [-0.2644, -0.1186] (864) | Consistent directional signal (all 5 < 0) |
| `(2,067)` | **M2A vs M0** | -0.1552 | [-0.2424, -0.0706] (310) | [-0.2665, -0.0380] (231) | [-0.2485, -0.0631] (565) | [-0.2225, -0.0876] (921) | [-0.2232, -0.0879] (864) | Consistent directional signal (all 5 < 0) |
| | **M3 vs M0** | -0.1525 | [-0.2413, -0.0451] (310) | [-0.2636, -0.0294] (231) | [-0.2627, -0.0410] (565) | [-0.2429, -0.0538] (921) | [-0.2461, -0.0766] (864) | Consistent directional signal (all 5 < 0) |

---

## 6. Subgroup Performance: PCE Bins and Scaffold / Pair Slices

### 6.1 PCE-Range Subgroup Breakdown

*(Source: [phase_8_5_subgroup_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_subgroup_metrics.csv))*

| Strategy | PCE Slice ($n$) | M0 MAE (Bias, 90% Cov) | M1 MAE (Bias, 90% Cov) | M2A MAE (Bias, 90% Cov) | M3 MAE (Bias, 90% Cov) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Strategy A** | `< 5.0%` (628) | 1.7756 (+1.33, 98.1%) | 1.7475 (+1.29, 95.2%) | 1.8157 (+1.33, 96.2%) | 1.6409 (+1.24, 95.5%) |
| | `5.0% - 10.0%` (543) | 2.4625 (-0.98, 98.2%) | 2.2181 (-0.70, 95.0%) | 2.3250 (-0.70, 95.0%) | 2.2534 (-0.64, 90.8%) |
| | `10.0% - 15.0%` (195) | 2.2668 (-1.69, 94.9%) | 2.3657 (-1.07, 90.8%) | 2.4508 (-1.08, 92.8%) | 2.2795 (-1.19, 89.2%) |
| | `>= 15.0%` (**115**) | **3.5546** (-3.55, 96.5%) | **2.8605** (-2.83, 88.7%) | **2.6318** (-2.59, 91.3%) | **2.4840** (-2.38, 87.8%) |
| **Strategy B** | `< 5.0%` (2,325) | 2.1211 (+2.01, 95.3%) | 2.1223 (+2.01, 95.8%) | 1.9897 (+1.86, 97.3%) | 1.8024 (+1.59, 95.6%) |
| | `5.0% - 10.0%` (974) | 1.9966 (-0.52, 96.7%) | 1.8389 (-0.31, 97.5%) | 1.8553 (-0.33, 98.3%) | 1.9758 (-0.51, 96.0%) |
| | `10.0% - 15.0%` (427) | 2.6232 (-2.48, 89.5%) | 2.5366 (-2.35, 90.2%) | 2.5047 (-2.33, 91.3%) | 2.4784 (-2.25, 85.3%) |
| | `>= 15.0%` (**71**) | **6.3272** (-6.33, **45.1%**) | **5.9218** (-5.92, **50.7%**) | **5.8499** (-5.85, **54.9%**) | **5.2309** (-5.21, **59.2%**) |
| **Strategy C1** | `< 5.0%` (174) | 2.3622 (+2.02, 100.0%) | 2.3953 (+2.18, 100.0%) | 2.3967 (+2.13, 100.0%) | 2.3337 (+1.91, 98.3%) |
| | `5.0% - 10.0%` (85) | 2.4614 (-2.03, 100.0%) | 2.0033 (-1.48, 100.0%) | 1.9732 (-1.45, 100.0%) | 2.2015 (-1.51, 100.0%) |
| | `10.0% - 15.0%` (19) | 4.4600 (-4.32, 100.0%) | 3.2369 (-3.13, 100.0%) | 3.0730 (-3.04, 100.0%) | 3.2366 (-3.24, 100.0%) |
| | `>= 15.0%` (**6**) | **4.3364** (-4.34, 83.3%) | **4.6263** (-4.63, 83.3%) | **4.8024** (-4.80, 83.3%) | **5.4911** (-5.49, 83.3%) |
| **Strategy C2** | `< 5.0%` (791) | 2.2691 (+1.98, 89.0%) | 2.0810 (+1.76, 88.2%) | 2.1429 (+1.85, 88.2%) | 2.1881 (+1.86, 88.2%) |
| | `5.0% - 10.0%` (698) | 1.9219 (-0.48, 93.4%) | 1.7692 (-0.38, 94.1%) | 1.8078 (-0.29, 94.3%) | 1.7756 (-0.33, 93.3%) |
| | `10.0% - 15.0%` (373) | 2.6025 (-2.02, 87.1%) | 2.4341 (-1.88, 83.4%) | 2.4316 (-1.90, 86.9%) | 2.4951 (-1.86, 84.2%) |
| | `>= 15.0%` (**205**) | **3.3203** (-3.22, **73.7%**) | **2.9719** (-2.85, **75.1%**) | **2.9420** (-2.85, **77.6%**) | **2.7887** (-2.64, **78.0%**) |

### 6.2 Scaffold-Level & Scaffold-Pair Subgroup Highlights ($n \ge 20$)

Full scaffold/pair subgroup tables (`524` rows covering all eligible subgroups with $n \ge 20$: `8/15/12` groups in A, `12/14/17` in B, `2/2/2` in C1, and `16/15/16` in C2 across acceptor/donor/pair levels) are archived in [phase_8_5_scaffold_pair_subgroups.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_5/phase_8_5_scaffold_pair_subgroups.csv):
- **Strategy A Largest Held-Out Donor Scaffolds:** Largest cluster (`n=173`, `PCE mean=4.00%`): M3 MAE `1.3474%` vs M0 `1.7137%`; second largest (`n=104`, `PCE mean=10.77%`): M3 MAE `2.4569%` vs M0 `2.8085%`; third largest (`n=88`, `PCE mean=11.38%`): M0 `1.8464%` outperforms M3 `1.9433%` and M1 `2.0861%`.
- **Strategy B `PC61BM` Dominance vs Non-Fullerene Acceptors:** `PC61BM` (`n=2,323`, `61.18%` share, `PCE mean=3.30%`): M3 MAE `1.5955%` vs M0 `1.9206%` (`cov_90=98.79%`). Conversely, on the #3 and #4 NFA acceptor scaffolds (`n=151`, `PCE mean=9.43%` and `n=132`, `PCE mean=6.63%`), **M3 underperforms M0** (`2.6199%` vs `2.4071%`, and `2.5676%` vs `2.2611%`). On high-efficiency benzothiadiazole NFA scaffold (`n=31`, `PCE mean=7.36%`), `cov_90` collapses to `32.26%–41.94%` across all models.
- **Strategy C1 Dual-Scaffold Hubs:** `202 / 284 (71.13%)` of test rows belong to the `PC61BM` acceptor scaffold (`M3 MAE=1.9355%`, `M0=2.2541%`), whereas the secondary ITIC-family acceptor (`n=22`, `7.75%`) rises to `M3 MAE=5.6537%` (`bias=+5.18%`, `cov_90=86.36%`).
- **Strategy C2 Largest Novel Pairs ($n \ge 20$):** Across the 16 eligible scaffold pairs (`n=20–52`), `M1` improves or matches `M0` on mid-range pairs (e.g., `n=49`, `PCE mean=7.06%`: M1 `1.8647%` vs M0 `2.5978%`), while high-PCE mixture pair (`n=35`, `PCE mean=19.07%`) drops to `M1 MAE=2.7328%` vs `M0 1.9904%` and `M3 1.8812%`.
