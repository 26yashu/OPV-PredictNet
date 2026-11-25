# OPV-PredictNet — Phase 8.2.1: Independent Statistical and Scientific Audit

**Date:** 2026-11-25  
**Scope:** Independent verification of Phase 8.2 benchmark results, Phase 8.1 metric reconciliation, split integrity, conformal interval validity, and scientific interpretations prior to Phase 8.3 authorization.  
**Constraint Compliance:** Original Phase 8.1 and Phase 8.2 CSV/JSON result artifacts preserved untouched. Zero retraining, zero recalibration, zero production code changes, zero Git commits.

---

## 1. Executive Audit Verdict

An independent forensic audit of `ml/experiments/phase_8_2_scaffold_benchmark.py`, the generated Phase 8.2 result CSVs, and `ml/PHASE_8_2_STRUCTURE_GENERALIZATION_REPORT.md` identified **four critical errors and methodological distortions** that require correction before Phase 8.3:

1. **Critical Metric Hallucination in Phase 8.2 Report (Phase 8.1 Baseline Misquoted):**
   `PHASE_8_2_STRUCTURE_GENERALIZATION_REPORT.md` reported Phase 8.1 test MAE as `1.28%–1.34%` (M0 `1.3418%`, M3 `1.2829%`), $R^2$ as `0.8354–0.8492`, and M3 90% interval width as `3.52%`. **None of these numbers exist in any Phase 8.1 result artifact.** Actual verified Phase 8.1 test metrics ($N=946$) are **M0 MAE = 2.1727% ($R^2=0.6110$, width 8.9671%)** and **M3 MAE = 2.0540% ($R^2=0.6502$, width 8.4283%)**. Because of this error, the magnitude of the "generalization gap" claimed in Section 5 of the Phase 8.2 report was mathematically invalid.
2. **Invalid Inferential Interpretation of Paired Comparisons:**
   In Strategy B (`M3 vs M0`), the report simultaneously printed a cluster bootstrap 95% CI crossing zero (`[-0.3781, +0.0386]`) and an i.i.d. paired t-test $p = 4.29 \times 10^{-47}$. Because a single acceptor scaffold (`PC61BM`) constitutes **2,323 of 3,797 (61.18%)** test records, the record-level t-test suffers extreme pseudo-replication and is invalid. Likewise, Strategy C1's `scaffold_pair` bootstrap CIs exclude zero (`[-0.5552, -0.0231]` for M3), yet bootstrapping by `donor_scaffold` (`[-0.6274, +0.0031]`) or `acceptor_scaffold` (`[-0.5846, +0.4291]`) crosses zero.
3. **Conformal Exchangeability Violations Driving Extreme Overcoverage (C1) and Undercoverage (C2 & High-PCE):**
   In Strategy C1, nominal 90% intervals exhibit **98.59%–99.65%** coverage with **15.69–18.62 pp** widths because $D_{\text{cal}}$ ($N=260$, $\text{std}=6.53\%$, $21.2\%$ high-PCE) is dramatically more variable and heavier-tailed than $D_{\text{test}}$ ($N=284$, $\text{std}=3.29\%$, $2.1\%$ high-PCE). Conversely, Strategy C2 exhibits undercoverage across all nominal levels (75.7%–78.5% for 80%; 88.0%–89.9% for 90%). Exchangeability does **not** hold across scaffold partitions.
4. **Protocol-to-Execution Divergences & Overstated C2 Isolation:**
   While Strategy A, B, and C1 cleanly enforce scaffold disjointness on canonical structures, Strategy C2 holds out *scaffold pairs* only: **73.25% of C2 test records share their donor scaffold with training, 84.86% share their acceptor scaffold, and 58.35% have BOTH individual scaffolds present in training.** Additionally, actual C1 counts (`Train=13,324, Cal=260, Test=284`, 8,599 cross-edges dropped) differ from the pre-execution protocol (`N=77`, no `D_cal`), and Strategy A omitted the planned Alternative B per-scaffold evaluation.

---

## 2. Phase 8.1 vs. Phase 8.2 Metric Reconciliation

### 2.1 Audit of Misquoted Phase 8.1 Metrics

Direct inspection of [phase_8_1_model_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_1_model_metrics.csv), [phase_8_1_interval_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_1_interval_metrics.csv), [phase_8_1_config.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_1_config.json), and a full-directory search across `ml/*.md` confirmed:

| Metric | Incorrectly Claimed in Phase 8.2 Report | Actual Phase 8.1 CSV Value ([phase_8_1_model_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_1_model_metrics.csv)) | Discrepancy |
| :--- | :---: | :---: | :---: |
| **M0 MAE (% PCE)** | 1.3418 | **2.1727** | -0.8309 pp fabricated |
| **M0 $R^2$** | 0.8354 | **0.6110** | +0.2244 fabricated |
| **M0 90% Width (% PCE)** | — | **8.9671** ($q=4.4836$) | — |
| **M1 MAE (% PCE)** | ("1.28–1.34" range) | **2.0321** | -0.69 to -0.75 pp fabricated |
| **M2A MAE (% PCE)** | ("1.28–1.34" range) | **2.0689** | -0.73 to -0.79 pp fabricated |
| **M3 MAE (% PCE)** | 1.2829 | **2.0540** | -0.7711 pp fabricated |
| **M3 $R^2$** | 0.8492 | **0.6502** | +0.1990 fabricated |
| **M3 90% Width (% PCE)** | 3.52% | **8.4283** ($q=4.2141$) | -4.91 pp fabricated |

**Origin & Cause:** The figures `1.28%–1.34%`, `0.8354`, `0.8492`, and `3.52%` do not match Phase 5 (`MAE = 2.0628, R^2 = 0.7099`), Phase 7.1, or Phase 8.1. They were erroneously fabricated in prose during authoring of `PHASE_8_2_STRUCTURE_GENERALIZATION_REPORT.md` without verifying the Phase 8.1 CSV artifacts.

### 2.2 Reconciled Phase 8.1 vs. Phase 8.2 Comparison & Variance Confounding

When true Phase 8.1 values are compared against Phase 8.2, the generalization pattern changes substantially:

| Metric | Phase 8.1 ($N=946$, 1 rec/pair) | Strategy A ($N=1,481$, Donor) | Strategy B ($N=3,797$, Acceptor) | Strategy C1 ($N=284$, Dual) | Strategy C2 ($N=2,067$, Pair) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test PCE Mean (Std)** | 7.48% (4.65%) | 6.60% (4.43%) | 5.03% (3.83%) | 4.84% (3.29%) | 7.32% (4.94%) |
| **Test PCE Variance** | **21.66** | **19.62** | **14.69** | **10.84** | **24.42** |
| **M0 MAE / RMSE / $R^2$** | 2.17 / 2.90 / 0.611 | 2.23 / 2.86 / 0.584 | 2.22 / 2.87 / 0.440 | 2.57 / 3.16 / 0.079 | 2.32 / 3.06 / 0.617 |
| **M1 MAE / RMSE / $R^2$** | 2.03 / 2.78 / 0.642 | 2.09 / 2.69 / 0.632 | 2.18 / 2.86 / 0.442 | 2.40 / 3.00 / 0.167 | 2.14 / 2.89 / 0.658 |
| **M2A MAE / RMSE / $R^2$** | 2.07 / 2.82 / 0.633 | 2.15 / 2.74 / 0.618 | 2.12 / 2.77 / 0.480 | 2.33 / 2.86 / 0.244 | 2.14 / 2.89 / 0.658 |
| **M3 MAE / RMSE / $R^2$** | 2.05 / 2.75 / 0.650 | 2.00 / 2.65 / 0.643 | 1.95 / 2.61 / 0.538 | 2.30 / 2.87 / 0.239 | 2.15 / 2.85 / 0.667 |
| **M3 90% Width (pp)** | 8.43 | 9.26 | 10.38 | 15.69 | 8.28 |

**Why Naive Direct Comparison Is Methodologically Flawed:**
1. **Sampling Unit Mismatch:** Phase 8.1 used **Alternative B sampling** (1 randomly sampled record per held-out material pair, $N=946$). Phase 8.2 evaluated **all unweighted device rows** in each test scaffold partition ($N=1,481$ to $3,797$), heavily weighting over-represented hubs (e.g., PC61BM = 61.18% of Strategy B test rows).
2. **MAE Actually Decreases Slightly for M3 in A & B:** Relative to true Phase 8.1 (`2.0540`), M3 MAE is **lower** in Strategy A (`2.0025`) and Strategy B (`1.9463`), and RMSE is **lower** in both (`2.6478`, `2.6066` vs `2.7525`). However, this reflects different test sample composition (mean PCE 5.03% in B vs 7.48% in P8.1) rather than superior scaffold generalization.
3. **Target Variance Compression Explains Much of the $R^2$ Drop:** Because $R^2 = 1 - \text{MSE}/\text{Var}(y)$, halving test target variance from $21.66$ (Phase 8.1) to $10.84$ (Strategy C1) mechanically drives $R^2$ sharply downward even though M3's RMSE only increased by $+4.4\%$ ($2.7525 \to 2.8727$). Conversely, M0's RMSE increased from $2.9026 \to 3.1599$ ($+8.9\%$) on this restricted variance, pushing M0 $R^2$ down to $0.0786$.
4. **Confirmed Scaffold Leakage in Phase 8.1:** Independent verification confirms that in Phase 8.1's 946 test rows, **790/946 (83.5%)** shared a donor scaffold with training, **812/946 (85.8%)** shared an acceptor scaffold, **659/946 (69.7%)** shared *both* individual scaffolds, and **445/946 (47.0%)** shared the *exact* scaffold pair. Thus, Phase 8.1 truly did not test novel scaffolds—even though its actual MAE/R² metrics were misreported in Phase 8.2.

---

## 3. Protocol-to-Execution Discrepancy Audit

Direct code inspection of [phase_8_2_scaffold_benchmark.py](file:///c:/Users/DELL/OPV-PredictNet/ml/experiments/phase_8_2_scaffold_benchmark.py) against [PHASE_8_2_STRUCTURE_GENERALIZATION_PROTOCOL.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_STRUCTURE_GENERALIZATION_PROTOCOL.md) identified three execution divergences:

1. **Strategy C1 Partition Construction:** The locked protocol specified $N_{\text{test}}=77$ with no calibration set (`D_cal` impossible). The actual script implemented a 3-way bipartite intersection (`donor in train_d & acceptor in train_a`, `cal_d & cal_a`, `test_d & test_a`), yielding **$N_{\text{train}}=13,324$, $N_{\text{cal}}=260$, $N_{\text{test}}=284$**, while dropping **8,599 cross-edges (38.27% of the dataset)**. While chemically leak-free, this 3-way subsampling deviates from the locked protocol specification.
2. **Omitted Alternative B Sampling in Strategy A:** Protocol Section 3.1 required reporting Alternative B (1 record per test scaffold) alongside the unweighted population. Only the unweighted device population was computed.
3. **Strategy C2 Record Counts:** Protocol and Mapping Audit documents stated $N_{\text{test}}=3,604$ for Strategy C2; actual execution produced **$N_{\text{test}}=2,067$** (`Train=16,537, Cal=3,863`).

---

## 4. Machine-Verifiable Scaffold & Structure Split Integrity

Independent verification across all 22,467 records produced the following exact intersection counts across `Train (Tr)`, `Cal (Cl)`, and `Test (Ts)`:

| Identifier Level | Strategy A (`15,750 / 5,236 / 1,481`) Tr&Cl / Tr&Ts / Cl&Ts | Strategy B (`17,658 / 1,012 / 3,797`) Tr&Cl / Tr&Ts / Cl&Ts | Strategy C1 (`13,324 / 260 / 284`) Tr&Cl / Tr&Ts / Cl&Ts | Strategy C2 (`16,537 / 3,863 / 2,067`) Tr&Cl / Tr&Ts / Cl&Ts |
| :--- | :---: | :---: | :---: | :---: |
| **`donor_scaffold`** | **0 / 0 / 0** (643/214/216) | 63 / 196 / 43 (938/71/323) | **0 / 0 / 0** (562/17/73) | 128 / 123 / 73 (719/313/310) |
| **`acceptor_scaffold`** | 136 / 53 / 45 (658/279/71) | **0 / 0 / 0** (487/162/163) | **0 / 0 / 0** (387/42/17) | 91 / 106 / 53 (567/220/231) |
| **`scaffold_pair`** | **0 / 0 / 0** (1781/654/384) | **0 / 0 / 0** (1943/273/603) | **0 / 0 / 0** (1213/58/92) | **0 / 0 / 0** (1691/563/565) |
| **`donor_smiles`** | **0 / 0 / 0** (1155/432/410) | 99 / 242 / 62 (1748/124/470) | **0 / 0 / 0** (1013/45/97) | 178 / 168 / 93 (1348/493/526) |
| **`acceptor_smiles`** | 186 / 83 / 64 (1180/442/136) | **0 / 0 / 0** (903/297/280) | **0 / 0 / 0** (718/67/28) | 141 / 124 / 74 (1017/381/361) |
| **`chem_pair`** | **0 / 0 / 0** (3042/1155/635) | **0 / 0 / 0** (3477/459/896) | **0 / 0 / 0** (2142/103/115) | **0 / 0 / 0** (2985/931/916) |
| **`donor_canonical`** | 17 / 6 / 3 | 89 / 233 / 56 | 6 / 3 / 0 | 167 / 161 / 86 |
| **`acceptor_canonical`** | 185 / 78 / 59 | 4 / 8 / 4 | 0 / 0 / 0 | 146 / 130 / 75 |
| **`pair_id` (names)** | 21 / 8 / 2 | 0 / 3 / 1 | 0 / 0 / 0 | 18 / 14 / 4 |
| **`doi_norm` (papers)** | 509 / 215 / 110 | 193 / 435 / 67 | 12 / 21 / 0 | 678 / 485 / 200 |

### Key Integrity Audit Observations:
1. **Canonical Molecular Structures Do Not Leak on Held-Out Axes:** Because each canonical SMILES maps deterministically to exactly one scaffold (or pseudo-scaffold `[NO_RING]:<smiles>` for the 5 ringless molecules / 12 records), `donor_smiles` has **0** overlap in A and C1, `acceptor_smiles` has **0** overlap in B and C1, and `chem_pair` has **0** overlap across all strategies.
2. **Material-Name Homonyms Explained:** Small non-zero overlaps in `donor_canonical` (6 in A, 3 in C1) and `acceptor_canonical` (8 in B) occur because of **raw dataset homonyms** (the same canonical name string assigned to two distinct SMILES with different scaffolds). Structure-level isolation remained intact (`0` SMILES overlap).
3. **Literature DOI Leakage Across Splits:** In Strategy A, **215 / 513 (41.9%)** test DOIs also appear in training; in Strategy B, **435 / 1,163 (37.4%)**; in C1, **21 / 116 (18.1%)**. Thus, same-paper laboratory effects are not fully isolated.
4. **Explicit Classification of Strategy C2:** Strategy C2 holds out **unseen scaffold combinations (`scaffold_pair`) only**. At the record level ($N_{\text{test}}=2,067$):
   - **73.25% (1,514/2,067)** use a `donor_scaffold` present in training (`62.94%` exact `donor_smiles` in train).
   - **84.86% (1,754/2,067)** use an `acceptor_scaffold` present in training (`77.65%` exact `acceptor_smiles` in train).
   - **58.35% (1,206/2,067)** have **BOTH** individual scaffolds present in training.
   - **0.24% (5/2,067)** have neither scaffold in training.
   Strategy C2 must **never** be described as testing unseen individual scaffolds.

---

## 5. Conformal Interval & Distribution Shift Audit

### 5.1 Complete Conformal Metrics Across All 48 Evaluations
*(Source: [phase_8_2_interval_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_interval_metrics.csv) and [phase_8_2_subgroup_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_subgroup_metrics.csv))*

Note: Because unnormalized split-conformal calibration produces a single constant half-width $q_\alpha$ per strategy/model, `mean_width == median_width == 2 * q_alpha` across every record.

| Strategy ($N_{\text{cal}} / N_{\text{test}}$) | Model | 80% Cov (Width, $q_{0.20}$) | 90% Cov (Width, $q_{0.10}$) | 95% Cov (Width, $q_{0.05}$) | High-PCE ($\ge 15\%$) 90% Cov ($N_{\text{sub}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** | M0 | 94.87% (11.26, 5.6315) | 97.57% (13.84, 6.9215) | 98.51% (15.42, 7.7115) | 96.52% (115) |
| ($5,236 / 1,481$) | M1 | 87.24% (7.89, 3.9460) | 93.79% (10.25, 5.1258) | 97.30% (12.30, 6.1520) | 88.70% (115) |
| | M2A | 88.32% (8.33, 4.1665) | 94.46% (10.48, 5.2421) | 96.83% (12.14, 6.0723) | 88.70% (115) |
| | M3 | 85.01% (7.23, 3.6139) | 92.57% (9.26, 4.6297) | 95.81% (11.81, 5.9040) | 86.96% (115) |
| **Strategy B** | M0 | 84.20% (7.50, 3.7479) | 94.05% (10.84, 5.4202) | 97.79% (13.86, 6.9278) | **45.07%** (71) |
| ($1,012 / 3,797$) | M1 | 87.65% (7.98, 3.9897) | 95.02% (11.15, 5.5764) | 98.10% (15.36, 7.6792) | **50.70%** (71) |
| | M2A | 87.38% (7.74, 3.8701) | 96.29% (11.83, 5.9142) | 98.42% (15.57, 7.7875) | **53.52%** (71) |
| | M3 | 89.36% (7.75, 3.8746) | 94.63% (10.38, 5.1911) | 96.58% (12.18, 6.0901) | **59.15%** (71) |
| **Strategy C1** | M0 | **95.42%** (12.33, 6.1662) | **99.65%** (16.52, 8.2587) | **99.65%** (18.24, 9.1187) | 83.33% (6) |
| ($260 / 284$) | M1 | **96.83%** (11.84, 5.9208) | **98.94%** (16.47, 8.2346) | **99.65%** (18.26, 9.1318) | 83.33% (6) |
| | M2A | **96.83%** (11.90, 5.9509) | **99.65%** (18.62, 9.3105) | **100.00%** (21.01, 10.5057)| 83.33% (6) |
| | M3 | **95.77%** (11.59, 5.7944) | **98.59%** (15.69, 7.8444) | **99.30%** (17.45, 8.7265) | 83.33% (6) |
| **Strategy C2** | M0 | **75.71%** (6.75, 3.3730) | **88.63%** (9.30, 4.6503) | 95.31% (12.45, 6.2258) | **73.66%** (205) |
| ($3,863 / 2,067$) | M1 | **77.65%** (6.41, 3.2034) | **88.05%** (8.57, 4.2830) | **94.73%** (11.11, 5.5572) | **75.12%** (205) |
| | M2A | **78.47%** (6.49, 3.2428) | **89.89%** (8.77, 4.3832) | **94.63%** (11.37, 5.6828) | **82.93%** (205) |
| | M3 | **78.13%** (6.28, 3.1380) | **88.44%** (8.28, 4.1387) | **93.90%** (10.70, 5.3510) | **75.61%** (205) |

### 5.2 Diagnosis of C1 98.6%–99.7% Overcoverage and Exchangeability Failure

Split-conformal calibration guarantees finite-sample coverage **only when calibration and test records are exchangeable**. Across scaffold splits, **exchangeability is violated**:
1. **Severe Calibration-vs-Test Target Shift in C1:**
   - $D_{\text{cal}}$ ($N=260$, only **17 donor scaffolds** and **42 acceptor scaffolds**) has $\text{mean PCE} = 7.79\%$, $\text{std} = 6.53\%$, and **55 high-PCE cells (21.15%)**. Its 90th-percentile absolute residual is **7.82–9.24 pp**, setting $q_{0.10} = 7.84\text{–}9.31\text{ pp}$ (interval widths **15.69–18.62 pp**).
   - $D_{\text{test}}$ ($N=284$, 73 donor and only **17 acceptor scaffolds**) has $\text{mean PCE} = 4.84\%$, $\text{std} = 3.29\%$, and only **6 high-PCE cells (2.11%)**. Its 90th-percentile absolute residual is only **4.35–5.15 pp**.
   - Applying wide cutoffs derived from a high-variance calibration slice to a narrow, low-PCE test slice yields **excessively conservative overcoverage (98.59%–99.65% at 90% nominal)** and uninformative 16–21 pp intervals.
2. **Symmetric Reverse Shift in Strategy C2 (Undercoverage):**
   - In C2, $D_{\text{cal}}$ has lower variance ($\text{std}=4.23\%$, $4.8\%$ high-PCE) than $D_{\text{test}}$ ($\text{std}=4.94\%$, $9.9\%$ high-PCE), causing nominal 80% coverage to drop to **75.71%–78.47%** and nominal 90% coverage to miss target (**88.05%–89.89%**).
3. **Conclusion:** Nominal conformal guarantees do **not** survive arbitrary scaffold distribution shift. Observed coverage fluctuates arbitrarily from **75.7%** (C2 80%) to **99.7%** (C1 90%) and collapses to **45.1%–59.2%** on high-PCE unseen acceptors (Strategy B).

---

## 6. Audit of Scientific Interpretations

| Claim in `PHASE_8_2_STRUCTURE_GENERALIZATION_REPORT.md` | Audit Finding | Required Correction |
| :--- | :--- | :--- |
| *"Processing metadata acts as an indispensable physical regularizer..."* | **Unsupported mechanistic claim.** While adding processing features reduces sample MAE across A, B, C1, and C2, this experiment did not isolate physical regularization from dataset shortcut effects (e.g., publication era, lab conventions, or solvent/temperature co-occurrence with device families). | Classify strictly as an **unverified explanatory hypothesis** consistent with observed error reductions, not a demonstrated physical mechanism. |
| *"M3 achieved the lowest MAE across A, B, and C1... statistically significant improvement"* | **Partially false / invalid inference.** In Strategy B (`M3 vs M0`), cluster bootstrap 95% CI `[-0.3781, +0.0386]` crosses zero; in C1, `donor_scaffold` (`[-0.6274, +0.0031]`) and `acceptor_scaffold` (`[-0.5846, +0.4291]`) CIs cross zero; and Strategy C2 M3 MAE (`2.1536`) is worse than M1 (`2.1393`) and M2A (`2.1436`). | Downgrade Strategy B and C1 comparisons to **descriptive / non-confirmatory**; state explicitly that no model is universally superior. |
| *"Valid finite-sample coverage... uncertainty estimation correctly flags novel scaffolds"* | **Overstated.** High coverage in A, B, and C1 is an artifact of target variance mismatch ($D_{\text{cal}} \text{ std} > D_{\text{test}} \text{ std}$). Under C2, coverage drops below nominal (75.7%–78.5% at 80%), and under Strategy B high-PCE, coverage collapses to 45.1%–59.2%. | Explicitly warn that conformal guarantees fail under scaffold shift and cannot be relied upon as calibrated uncertainty for out-of-distribution scaffolds. |
| *"Catastrophic drop vs Phase 8.1 (1.28–1.34% -> 2.30–2.57%)"* | **Factually incorrect baseline.** True Phase 8.1 MAE is 2.03–2.17%. | Replace with exact Phase 8.1 figures and explain variance/sampling confounding. |
