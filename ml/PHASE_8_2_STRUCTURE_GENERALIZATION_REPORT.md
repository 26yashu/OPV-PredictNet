# OPV-PredictNet — Phase 8.2: Structure-Level Generalization Benchmark Report (Audit-Corrected)

**Document Status:** Corrected per Phase 8.2.1 Independent Audit ([PHASE_8_2_1_INDEPENDENT_AUDIT.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_1_INDEPENDENT_AUDIT.md))  
**Date:** 2026-11-25  
**Execution Environment:** Python 3.11.9, Scikit-Learn 1.9.1, RDKit 2026.09.2, Windows 10, Single-Threaded Inference (`n_jobs=1`)  
**Dataset Checksum:** `ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897` (22,467 records)  
**Configuration & Metadata:** [phase_8_2_config.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_config.json)  

---

## 1. Executive Summary & Objective

In Phase 8.1, OPV-PredictNet evaluated held-out **material-name pairs** ($N_{\text{test}}=946$, 1 sampled device per pair), achieving test MAE of **2.0321%–2.1727% PCE**, $R^2$ of **0.6110–0.6502**, and 90% conformal interval widths of **8.43–8.97 pp**. However, independent structure mapping confirmed that **69.7% (659/946)** of Phase 8.1 test records shared both individual donor and acceptor Bemis–Murcko scaffolds with training data, and **47.0% (445/946)** shared the exact scaffold pair. Consequently, Phase 8.1 did not evaluate generalization to genuinely novel chemical scaffolds.

Phase 8.2 establishes a structure-level benchmark evaluating molecular baseline (**M0**) and processing-aware models (**M1, M2A, M3**) across four scaffold-partitioned splits.

### Key Audit-Corrected Findings:

1. **Collapse of Explained Variance Under Strict Dual-Scaffold Holdout (Strategy C1):**
   When both donor and acceptor scaffolds are completely absent from training ($N_{\text{test}} = 284$, with $8,599$ cross-edges dropped to maintain 3-way isolation), $R^2$ collapses across every model:
   - **M0 (Molecular Baseline):** $R^2 = \mathbf{0.0786}$, MAE = **2.5740%**, RMSE = **3.1599%** PCE.
   - **M1 (One-Hot Processing):** $R^2 = \mathbf{0.1672}$, MAE = **2.3999%**, RMSE = **3.0042%** PCE.
   - **M2A (Processing Ablation):** $R^2 = \mathbf{0.2440}$, MAE = **2.3314%**, RMSE = **2.8623%** PCE.
   - **M3 (Native NaN GBDT):** $R^2 = \mathbf{0.2385}$, MAE = **2.3013%**, RMSE = **2.8727%** PCE.
   Note: Reduced target variance in C1 test ($\text{std}=3.29\%$, $\text{Var}=10.84$ vs. Phase 8.1 $\text{std}=4.65\%$, $\text{Var}=21.66$) amplifies the $R^2$ drop alongside higher prediction error.

2. **Asymmetric Single-Axis Scaffold Generalization (Strategies A vs. B):**
   - **Unseen Donor Scaffolds (Strategy A, $N_{\text{test}} = 1,481$):** Models achieve $R^2 = 0.5839\text{–}0.6428$ and MAE $2.0025\%\text{–}2.2302\%$. M1 and M3 show consistent directional MAE improvement over M0 across all five one-way cluster-bootstrap margins.
   - **Unseen Acceptor Scaffolds (Strategy B, $N_{\text{test}} = 3,797$):** Models exhibit lower $R^2$ ($0.4396\text{–}0.5375$) and positive signed bias ($+0.43$ to $+0.80\text{ pp}$). Critically, one cluster (`PC61BM`) constitutes **61.18% ($2,323/3,797$)** of Strategy B test rows; under acceptor-scaffold cluster bootstrapping, **no processing model achieves statistical significance over M0** (M3 vs. M0 95% CI `[-0.3781, +0.0386]`).

3. **Processing Features Reduce Sample MAE, Mechanism Unproven:**
   Across all four splits, processing-aware models (M1, M2A, M3) achieved lower sample MAE than M0 (`-0.04` to `-0.28 pp`), though gains are statistically inconclusive under scaffold clustering in Strategy B and Strategy C1. One plausible explanation is that fabrication metadata constrains the physically plausible efficiency range when 2D structural extrapolation degrades, though dataset shortcut correlations (laboratory reporting practices or historical era confounding) were not ruled out.

4. **Conformal Uncertainty Does Not Survive Scaffold Distribution Shift:**
   Because scaffold splits violate calibration–test exchangeability, empirical coverage varies erratically: Strategy C1 yields **excessively conservative overcoverage (98.59%–99.65% at 90% nominal, widths 15.69–18.62 pp)** due to severe calibration-vs-test variance mismatch ($D_{\text{cal}} \text{ std}=6.53\%$ vs. $D_{\text{test}} \text{ std}=3.29\%$), whereas Strategy C2 exhibits **undercoverage across all models (75.71%–78.47% at 80% nominal; 88.05%–89.89% at 90% nominal)**.

5. **Severe High-PCE Underprediction ($\text{PCE} \ge 15.0\%$):**
   On unseen high-efficiency acceptor scaffolds (Strategy B, $N=71$), models exhibit systematic negative bias ($-5.25\%$ to $-6.33\%$), and 90% conformal coverage collapses to **45.07%–59.15%**.

---

## 2. Structure Representation & Bipartite Graph Audit

### 2.1 Chemical Entity Cardinality & Bemis–Murcko Scaffolds

Across all 22,467 benchmark records:
- **Raw Material Names:** 2,192 donors, 1,616 acceptors.
- **Canonical Material Names:** 1,920 donors, 1,435 acceptors (4,730 unique material-name pairs).
- **RDKit Canonical SMILES:** 1,984 donors, 1,462 acceptors (4,807 unique canonical pairs; 4,832 raw SMILES pairs).
- **Bemis–Murcko Scaffolds:** 1,073 unique donor scaffolds, 812 unique acceptor scaffolds (2,819 unique scaffold pairs).

### 2.2 Acyclic Molecules & Non-Ring Handling (Zero Silent Drops in Splits A, B, C2)

Exactly 2 donor SMILES (8 records: CuSCN 7, PR1 1) and 3 acceptor SMILES (4 records: GT-s 2, TiO2 1, ITN 1) lack ring systems. Deterministic pseudo-scaffolds `"[NO_RING]:" + smiles` ensure all 22,467 records are retained in Strategies A, B, and C2 without silent deletion.

### 2.3 Bipartite Graph Connectivity & Protocol-to-Execution Reconciliation

The donor–acceptor scaffold bipartite graph forms a giant connected component containing **22,455 records (99.95%)**. In pre-execution planning ([PHASE_8_2_STRUCTURE_GENERALIZATION_PROTOCOL.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_STRUCTURE_GENERALIZATION_PROTOCOL.md)), Strategy C1 was estimated as $N=77$ without a calibration partition. In actual execution ([phase_8_2_scaffold_benchmark.py](file:///c:/Users/DELL/OPV-PredictNet/ml/experiments/phase_8_2_scaffold_benchmark.py)), enforcing 3-way bipartite isolation (`donor in part_d & acceptor in part_a`) yielded **$N_{\text{train}}=13,324$, $N_{\text{cal}}=260$, $N_{\text{test}}=284$**, while dropping **8,599 cross-edges (38.27%)**. Additionally, Strategy A evaluated the unweighted full device population rather than Alternative B (1 record per scaffold), and Strategy C2 yielded $N_{\text{test}}=2,067$.

---

## 3. Benchmark Partitioning Strategies & Verified Isolation

| Strategy ID | Description | Split Basis | Seed | $N_{\text{train}}$ | $N_{\text{cal}}$ | $N_{\text{test}}$ | Verified Structural Isolation & Single-Side Overlap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Strategy A** | Donor Scaffold Holdout | Donor Scaffolds | 7701 | 15,750 | 5,236 | 1,481 | **0** donor scaffold & **0** `donor_smiles` overlap across Tr/Cl/Ts (216 test donor scaffolds). 74.6% of unique test acceptors overlap train. |
| **Strategy B** | Acceptor Scaffold Holdout | Acceptor Scaffolds | 7702 | 17,658 | 1,012 | 3,797 | **0** acceptor scaffold & **0** `acceptor_smiles` overlap across Tr/Cl/Ts (163 test acceptor scaffolds). 60.7% of unique test donors overlap train. |
| **Strategy C1** | Strict Dual-Scaffold Holdout | Joint Donor & Acceptor | 7701/7702 | 13,324 | 260 | 284 | **0** donor AND **0** acceptor scaffold/`smiles` overlap across Tr/Cl/Ts (73 test donors, 17 test acceptors; 8,599 cross-edges dropped). |
| **Strategy C2** | Scaffold-Pair Holdout | Scaffold Pairs | 7703 | 16,537 | 3,863 | 2,067 | **0** `scaffold_pair` & **0** `chem_pair` overlap. **Does NOT hold out individual scaffolds:** 73.25% of test records have a seen donor scaffold, 84.86% have a seen acceptor scaffold, 58.35% have BOTH seen individually, and only 0.24% (5/2,067) have neither. |

*Note on Bibliographic Overlap:* Across all four strategies, 18.1%–41.9% of test DOIs (`doi_norm`) also appear in training, so same-publication experimental effects are not isolated.

---

## 4. Master Model Performance Comparison

*(Source: [phase_8_2_scaffold_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_scaffold_metrics.csv))*

| Strategy | Model | Features | $N_{\text{train}}$ | $N_{\text{cal}}$ | $N_{\text{test}}$ | MAE (% PCE) | RMSE (% PCE) | $R^2$ | Mean Bias (% PCE) | Test PCE Mean (Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** (Donor Holdout) | M0 | 22 | 15,750 | 5,236 | 1,481 | 2.2302 | 2.8574 | 0.5839 | -0.2930 | 6.60% (4.43%) |
| | M1 | 50 | 15,750 | 5,236 | 1,481 | 2.0916 | 2.6880 | 0.6318 | -0.0716 | 6.60% (4.43%) |
| | M2A | 37 | 15,750 | 5,236 | 1,481 | 2.1470 | 2.7399 | 0.6175 | -0.0040 | 6.60% (4.43%) |
| | M3 | 44 | 15,750 | 5,236 | 1,481 | 2.0025 | 2.6478 | 0.6428 | -0.0285 | 6.60% (4.43%) |
| **Strategy B** (Acceptor Holdout) | M0 | 22 | 17,658 | 1,012 | 3,797 | 2.2243 | 2.8694 | 0.4396 | +0.7019 | 5.03% (3.83%) |
| | M1 | 50 | 17,658 | 1,012 | 3,797 | 2.1820 | 2.8645 | 0.4415 | +0.7972 | 5.03% (3.83%) |
| | M2A | 37 | 17,658 | 1,012 | 3,797 | 2.1158 | 2.7652 | 0.4795 | +0.7445 | 5.03% (3.83%) |
| | M3 | 44 | 17,658 | 1,012 | 3,797 | 1.9463 | 2.6066 | 0.5375 | +0.4255 | 5.03% (3.83%) |
| **Strategy C1** (Strict Dual Holdout) | M0 | 22 | 13,324 | 260 | 284 | 2.5740 | 3.1599 | 0.0786 | +0.2447 | 4.84% (3.29%) |
| | M1 | 50 | 13,324 | 260 | 284 | 2.3999 | 3.0042 | 0.1672 | +0.5443 | 4.84% (3.29%) |
| | M2A | 37 | 13,324 | 260 | 284 | 2.3314 | 2.8623 | 0.2440 | +0.5326 | 4.84% (3.29%) |
| | M3 | 44 | 13,324 | 260 | 284 | 2.3013 | 2.8727 | 0.2385 | +0.3810 | 4.84% (3.29%) |
| **Strategy C2** (Scaffold-Pair Holdout)| M0 | 22 | 16,537 | 3,863 | 2,067 | 2.3163 | 3.0573 | 0.6172 | -0.0898 | 7.32% (4.94%) |
| | M1 | 50 | 16,537 | 3,863 | 2,067 | 2.1393 | 2.8906 | 0.6578 | -0.0345 | 7.32% (4.94%) |
| | M2A | 37 | 16,537 | 3,863 | 2,067 | 2.1436 | 2.8887 | 0.6582 | -0.0071 | 7.32% (4.94%) |
| | M3 | 44 | 16,537 | 3,863 | 2,067 | 2.1536 | 2.8526 | 0.6667 | +0.0154 | 7.32% (4.94%) |

---

## 5. Reconciled Comparison: Phase 8.1 Material Pairs vs. Phase 8.2 Chemical Scaffolds

Direct comparison with actual Phase 8.1 artifacts ([phase_8_1_model_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_1_model_metrics.csv), [phase_8_1_interval_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_1_interval_metrics.csv)):

| Metric | Phase 8.1 ($N=946$, 1 rec/pair) | Strategy A ($N=1,481$, Donor) | Strategy B ($N=3,797$, Acceptor) | Strategy C1 ($N=284$, Dual) | Strategy C2 ($N=2,067$, Pair) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test PCE Variance** | 21.66 ($\text{std}=4.65\%$) | 19.62 ($\text{std}=4.43\%$) | 14.69 ($\text{std}=3.83\%$) | 10.84 ($\text{std}=3.29\%$) | 24.42 ($\text{std}=4.94\%$) |
| **M0 MAE (% PCE)** | 2.1727 | 2.2302 (+2.6%) | 2.2243 (+2.4%) | 2.5740 (+18.5%) | 2.3163 (+6.6%) |
| **M0 $R^2$** | 0.6110 | 0.5839 | 0.4396 | **0.0786** | 0.6172 |
| **M1 MAE (% PCE)** | **2.0321** | 2.0916 (+2.9%) | 2.1820 (+7.4%) | 2.3999 (+18.1%) | **2.1393** (+5.3%) |
| **M3 MAE (% PCE)** | 2.0540 | **2.0025** (-2.5%) | **1.9463** (-5.2%) | **2.3013** (+12.0%) | 2.1536 (+4.8%) |
| **M3 RMSE (% PCE)** | 2.7525 | 2.6478 (-3.8%) | 2.6066 (-5.3%) | 2.8727 (+4.4%) | 2.8526 (+3.6%) |
| **M3 $R^2$** | 0.6502 | 0.6428 | 0.5375 | **0.2385** | 0.6667 |
| **M3 90% Width (pp)** | 8.4283 | 9.2594 (+9.9%) | 10.3822 (+23.2%) | 15.6888 (+86.1%) | 8.2775 (-1.8%) |

### Interpretation Caveats:
1. **Non-Comparable Sampling Designs:** Phase 8.1 evaluated 1 randomly sampled device per held-out pair ($N=946$), whereas Phase 8.2 evaluated all unweighted records per test scaffold ($N=284\text{–}3,797$). Consequently, M3 MAE appears slightly *lower* in Strategy A (`2.0025`) and Strategy B (`1.9463`) than Phase 8.1 (`2.0540`), reflecting lower-PCE sample composition rather than improved generalization.
2. **Variance Compression vs. True Error Increase in C1:** Halving target variance from `21.66` (Phase 8.1) to `10.84` (C1) drives part of the $R^2$ collapse (`0.6502 -> 0.2385` for M3), alongside genuine MAE deterioration (`2.0540 -> 2.3013`, $+0.247\text{ pp}$). For M0, RMSE rises from `2.9026 -> 3.1599` and $R^2$ falls to `0.0786`, leaving almost no useful predictive signal on dual unseen scaffolds.

---

## 6. Conformal Prediction & Distribution Shift Audit

*(Source: [phase_8_2_interval_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_interval_metrics.csv))*

| Strategy ($N_{\text{cal}} / N_{\text{test}}$) | Model | 80% Cov (Width, $q_{0.20}$) | 90% Cov (Width, $q_{0.10}$) | 95% Cov (Width, $q_{0.05}$) | High-PCE ($\ge 15\%$) 90% Cov |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** ($5,236 / 1,481$) | M0 | 94.87% (11.26, 5.6315) | 97.57% (13.84, 6.9215) | 98.51% (15.42, 7.7115) | 96.52% ($N=115$) |
| | M1 | 87.24% (7.89, 3.9460) | 93.79% (10.25, 5.1258) | 97.30% (12.30, 6.1520) | 88.70% |
| | M2A | 88.32% (8.33, 4.1665) | 94.46% (10.48, 5.2421) | 96.83% (12.14, 6.0723) | 88.70% |
| | M3 | 85.01% (7.23, 3.6139) | 92.57% (9.26, 4.6297) | 95.81% (11.81, 5.9040) | 86.96% |
| **Strategy B** ($1,012 / 3,797$) | M0 | 84.20% (7.50, 3.7479) | 94.05% (10.84, 5.4202) | 97.79% (13.86, 6.9278) | **45.07%** ($N=71$) |
| | M1 | 87.65% (7.98, 3.9897) | 95.02% (11.15, 5.5764) | 98.10% (15.36, 7.6792) | **50.70%** |
| | M2A | 87.38% (7.74, 3.8701) | 96.29% (11.83, 5.9142) | 98.42% (15.57, 7.7875) | **53.52%** |
| | M3 | 89.36% (7.75, 3.8746) | 94.63% (10.38, 5.1911) | 96.58% (12.18, 6.0901) | **59.15%** |
| **Strategy C1** ($260 / 284$) | M0 | **95.42%** (12.33, 6.1662) | **99.65%** (16.52, 8.2587) | **99.65%** (18.24, 9.1187) | 83.33% ($N=6$) |
| | M1 | **96.83%** (11.84, 5.9208) | **98.94%** (16.47, 8.2346) | **99.65%** (18.26, 9.1318) | 83.33% |
| | M2A | **96.83%** (11.90, 5.9509) | **99.65%** (18.62, 9.3105) | **100.00%** (21.01, 10.5057)| 83.33% |
| | M3 | **95.77%** (11.59, 5.7944) | **98.59%** (15.69, 7.8444) | **99.30%** (17.45, 8.7265) | 83.33% |
| **Strategy C2** ($3,863 / 2,067$) | M0 | **75.71%** (6.75, 3.3730) | **88.63%** (9.30, 4.6503) | 95.31% (12.45, 6.2258) | **73.66%** ($N=205$) |
| | M1 | **77.65%** (6.41, 3.2034) | **88.05%** (8.57, 4.2830) | **94.73%** (11.11, 5.5572) | **75.12%** |
| | M2A | **78.47%** (6.49, 3.2428) | **89.89%** (8.77, 4.3832) | **94.63%** (11.37, 5.6828) | **82.93%** |
| | M3 | **78.13%** (6.28, 3.1380) | **88.44%** (8.28, 4.1387) | **93.90%** (10.70, 5.3510) | **75.61%** |

*(Mean width equals median width across all entries because constant unnormalized conformal quantiles $2 q_\alpha$ are applied.)*

### Conformal Audit Findings:
- **Exchangeability Failure Across Scaffold Splits:** Nominal conformal guarantees do **not** survive arbitrary scaffold distribution shift. In C1, $D_{\text{cal}}$ ($\text{std}=6.53\%$, $21.2\%$ high-PCE) is much heavier-tailed than $D_{\text{test}}$ ($\text{std}=3.29\%$, $2.1\%$ high-PCE), producing excessive 98.6%–99.7% empirical coverage with uninformative 15.7–18.6 pp intervals. Conversely, in C2 ($D_{\text{cal}} \text{ std}=4.23\%$ vs. $D_{\text{test}} \text{ std}=4.94\%$), coverage drops below nominal across all models at 80% (75.7%–78.5%) and 90% (88.1%–89.9%).

---

## 7. Paired Statistical Significance & Cluster Dependence Audit

*(Full forensic review in [PHASE_8_2_1_STATISTICAL_CORRECTIONS.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_1_STATISTICAL_CORRECTIONS.md); data in [phase_8_2_paired_comparisons.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_paired_comparisons.csv))*

| Strategy | Comparison | Mean $\Delta$AE (pp) | Primary `cluster_col` 95% CI | Record-Level t-test $p$ *(Invalid under clustering)* | Audit-Corrected Inferential Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Strategy A** | M1 vs M0 | -0.1386 | [-0.2849, -0.0152] ($G=216$) | $8.81 \times 10^{-6}$ | **Exploratory Directional Signal** (all 5 one-way CIs < 0) |
| | M2A vs M0 | -0.0833 | [-0.2387, +0.0456] ($G=216$) | $8.24 \times 10^{-3}$ | **Not Significant** (all 5 one-way CIs cross 0) |
| | M3 vs M0 | -0.2278 | [-0.3915, -0.0853] ($G=216$) | $1.87 \times 10^{-9}$ | **Exploratory Directional Signal** (all 5 one-way CIs < 0) |
| **Strategy B** | M1 vs M0 | -0.0423 | [-0.1191, +0.0443] ($G=163$) | $5.02 \times 10^{-3}$ | **Not Significant / Inconclusive** |
| | M2A vs M0 | -0.1085 | [-0.1471, +0.0168] ($G=163$) | $2.61 \times 10^{-14}$ | **Inconclusive / Descriptive** (`acceptor_scaffold` CI crosses 0) |
| | M3 vs M0 | -0.2780 | [-0.3781, +0.0386] ($G=163$) | $4.29 \times 10^{-47}$ | **Inconclusive / Descriptive** (61.2% of rows in `PC61BM`; CI crosses 0) |
| **Strategy C1** | M1 vs M0 | -0.1741 | [-0.4059, +0.0485] ($G=92$) | $8.82 \times 10^{-3}$ | **Not Significant** (all 5 one-way CIs cross 0) |
| | M2A vs M0 | -0.2426 | [-0.4912, -0.0180] ($G=92$) | $5.30 \times 10^{-5}$ | **Downgraded to Descriptive** (`donor_scaffold` & `acceptor_scaffold` CIs cross 0) |
| | M3 vs M0 | -0.2727 | [-0.5552, -0.0231] ($G=92$) | $2.77 \times 10^{-4}$ | **Downgraded to Descriptive** (`donor_scaffold` & `acceptor_scaffold` CIs cross 0) |
| **Strategy C2** | M1 vs M0 | -0.1769 | [-0.2684, -0.0858] ($G=565$) | $4.62 \times 10^{-15}$ | **Exploratory Directional Signal** (all 5 one-way CIs < 0) |
| | M2A vs M0 | -0.1727 | [-0.2643, -0.0858] ($G=565$) | $3.49 \times 10^{-13}$ | **Exploratory Directional Signal** (all 5 one-way CIs < 0) |
| | M3 vs M0 | -0.1626 | [-0.2818, -0.0397] ($G=565$) | $6.10 \times 10^{-8}$ | **Exploratory Directional Signal** (all 5 one-way CIs < 0) |

---

## 8. High-PCE Subgroup Failure ($\text{PCE} \ge 15.0\%$)

*(Source: [phase_8_2_subgroup_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_subgroup_metrics.csv))*

| Strategy | Model | Subgroup $N$ | MAE (% PCE) | Mean Signed Bias (% PCE) | 90% Conformal Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A** (Donor Holdout) | M0 / M1 / M2A / M3 | 115 | 3.55 / 2.83 / 2.73 / 2.35 | -3.55 / -2.79 / -2.67 / -2.24 | 96.5% / 88.7% / 88.7% / 87.0% |
| **Strategy B** (Acceptor Holdout) | M0 / M1 / M2A / M3 | 71 | 6.33 / 5.88 / 5.97 / 5.26 | **-6.33 / -5.88 / -5.95 / -5.25** | **45.1% / 50.7% / 53.5% / 59.2%** |
| **Strategy C1** (Strict Dual) | M0 / M1 / M2A / M3 | 6 | 4.34 / 4.91 / 4.66 / 5.34 | -4.34 / -4.91 / -4.66 / -5.34 | 83.3% / 83.3% / 83.3% / 83.3% |
| **Strategy C2** (Scaffold-Pair) | M0 / M1 / M2A / M3 | 205 | 3.32 / 2.96 / 2.76 / 2.82 | -3.22 / -2.84 / -2.69 / -2.67 | 73.7% / 75.1% / 82.9% / 75.6% |

Across every benchmark split, tree-based ensembles severely underpredict champion efficiency devices by `2.2 to 6.3 pp`, and conformal prediction intervals fail nominal coverage on unseen high-PCE acceptors.

---

## 9. Regime-Specific Model Classification (No Universal Winner)

*(Full readiness matrix in [PHASE_8_2_1_FINAL_MODEL_READINESS.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_1_FINAL_MODEL_READINESS.md))*

1. **Known-Scaffold Combinatorial Research (Phase 8.1 & Strategy C2):** M1, M2A, and M3 are supported as exploratory research tools (M1 for lowest sample MAE on standard lab vocabularies; M2A for continuous parameter sweeps; M3 for lowest RMSE/$R^2$ and native missingness). M0 serves as baseline comparator.
2. **Unseen Donor Scaffolds (Strategy A):** M1 and M3 are supported for exploratory research when paired with characterized acceptors (intervals $\approx 9.3\text{–}10.3\text{ pp}$); M2A improvement over M0 is not statistically distinguishable.
3. **Unseen Acceptor Scaffolds (Strategy B):** All models are **weak / inconclusive**; no processing model demonstrates statistically robust superiority across independent unseen acceptor scaffolds.
4. **Strict Dual-Scaffold Extrapolation (Strategy C1):** **ALL MODELS (M0, M1, M2A, M3) ARE UNSUPPORTED & PROHIBITED** ($R^2 = 0.08\text{–}0.24$, 90% widths $15.7\text{–}18.6\text{ pp}$).
5. **High-PCE Discovery ($\text{PCE} \ge 15.0\%$):** **ALL MODELS ARE UNSUPPORTED & PROHIBITED** for autonomous champion selection due to severe negative bias and coverage collapse.

---

## 10. Verification & Artifact Index

- **Phase 8.2.1 Audit Deliverables:**
  - [PHASE_8_2_1_INDEPENDENT_AUDIT.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_1_INDEPENDENT_AUDIT.md)
  - [PHASE_8_2_1_STATISTICAL_CORRECTIONS.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_1_STATISTICAL_CORRECTIONS.md)
  - [PHASE_8_2_1_FINAL_MODEL_READINESS.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_1_FINAL_MODEL_READINESS.md)
- **Protocol, Code, Tests, and Preserved CSVs:**
  - [PHASE_8_2_STRUCTURE_MAPPING_AUDIT.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_STRUCTURE_MAPPING_AUDIT.md)
  - [PHASE_8_2_STRUCTURE_GENERALIZATION_PROTOCOL.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_2_STRUCTURE_GENERALIZATION_PROTOCOL.md)
  - [phase_8_2_scaffold_benchmark.py](file:///c:/Users/DELL/OPV-PredictNet/ml/experiments/phase_8_2_scaffold_benchmark.py)
  - [test_phase_8_2_scaffold_benchmark.py](file:///c:/Users/DELL/OPV-PredictNet/tests/ml/test_phase_8_2_scaffold_benchmark.py)
  - [phase_8_2_scaffold_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_scaffold_metrics.csv)
  - [phase_8_2_interval_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_interval_metrics.csv)
  - [phase_8_2_paired_comparisons.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_paired_comparisons.csv)
  - [phase_8_2_subgroup_metrics.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_subgroup_metrics.csv)
  - [phase_8_2_config.json](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_config.json)
