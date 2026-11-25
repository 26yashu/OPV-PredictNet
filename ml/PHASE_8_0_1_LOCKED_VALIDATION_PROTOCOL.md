# OPV-PredictNet — Phase 8.0.1: Locked Validation and Experimental Protocol

**Date**: 2026-11-24  
**Scope**: Experimental protocol, partition locking, non-leakage contracts, model candidate definitions, and uncertainty evaluation for Phase 8.1.  
**Deliverable**: Pre-Registered, Contamination-Free Protocol for Processing-Aware OPV Modeling.

---

## 1. Executive Summary & Anti-Contamination Mandate

This protocol establishes the experimental and validation framework for Phase 8.1. To prevent data contamination, statistical bias, and retrospective leakage:
1. **Uncontaminated Partitions**: The Phase 7.1 final test partition is strictly retired and will **not** be reused for Phase 8 evaluation. An independent, locked 60 / 20 / 20 pair-group partition is established with random seed `8888`.
2. **Concurrent M0 Baseline**: The molecular-only baseline (Model M0) will be refitted on $D_{\text{train}}$ and recalibrated on $D_{\text{cal}}$ under the new Phase 8 partition. All comparative claims for processing-aware models M1, M2, and M3 will be evaluated against this concurrent Phase 8 M0 baseline on identical test instances.
3. **Mandatory Shortcut-Learning Ablation**: An explicit ablation candidate (M2A, 37 features) will be evaluated alongside M2 (44 features) to test whether missingness flags provide genuine physical signal or merely capture text-mining extraction artifacts.
4. **Single-Threaded Determinism**: Single-threaded execution (`n_jobs=1`) is enforced during prediction and evaluation to guarantee 100% bit-for-bit reproducible coverage counts and metrics.

---

## 2. Partition Locking and Group Disjointness

The benchmark dataset ($N = 22,467$, 4,730 normalized material-name pairs) is partitioned deterministically:

```
+---------------------------------------------------------------------------------------------------------+
| Phase 8 Locked Partition Specifications (Seed = 8888)                                                   |
+--------------------------+---------------------+-------------------+------------------------------------+
| Partition                | Pair Groups Count   | Raw Record Count  | Evaluation Sampling Rule           |
+--------------------------+---------------------+-------------------+------------------------------------+
| **Training (D_train)**   | 2,838 pairs (60.0%) | 14,704 records    | All records used for fitting       |
| **Calibration (D_cal)**  | 946 pairs (20.0%)   | 3,349 records     | Exactly 1 record/pair (Seed 8201)  |
| **Final Test (D_test)**  | 946 pairs (20.0%)   | 4,414 records     | Exactly 1 record/pair (Seed 8202)  |
+--------------------------+---------------------+-------------------+------------------------------------+
```

### 2.1 Formal Proof of Partition Disjointness
$$\text{Pairs}(D_{\text{train}}) \cap \text{Pairs}(D_{\text{cal}}) = \emptyset$$
$$\text{Pairs}(D_{\text{train}}) \cap \text{Pairs}(D_{\text{test}}) = \emptyset$$
$$\text{Pairs}(D_{\text{cal}}) \cap \text{Pairs}(D_{\text{test}}) = \emptyset$$
Zero material pairs appear in more than one partition.

### 2.2 Fresh Test Set Independence Verification
The Phase 8 test partition shares exactly 184 pairs (19.5%) with the historical Phase 7.1 test partition, consistent with an independent random draw from 4,730 total pairs ($946 \times \frac{946}{4730} \approx 189$). This confirms that Phase 8 testing is statistically independent and uncontaminated by Phase 7.1 inspection.

### 2.3 Generalization Scope and Sensitivity Analyses
- **Primary Operational Claim**: Prediction-interval coverage for one randomly selected experimental record from an unseen normalized material-name pair under known chemical spaces.
- **Sensitivity Tier 1**: Canonical molecular-structure grouping (4,788 pairs).
- **Sensitivity Tier 2**: Bemis-Murcko core scaffold disjoint splitting.
- *Boundary Disclaimer*: Material-name pair evaluation does **not** establish generalization to novel uncharacterized chemical scaffolds.

---

## 3. Candidate Models and Ablation Architecture

All models are trained strictly on $D_{\text{train}}$ (14,704 records across 2,838 pairs) and calibrated on $D_{\text{cal}}$ (946 pairs, Alternative B):

```
+---------------------------------------------------------------------------------------------------------+
| Candidate Models and Feature Spaces                                                                     |
+-----+---------------------------+-----------+-----------------------+-----------------------------------+
| ID  | Model Description         | Features  | Feature Set           | Missingness & Split Handling      |
+-----+---------------------------+-----------+-----------------------+-----------------------------------+
| M0  | **Molecular Baseline RF** | 22 feats  | Config A (Pure Mol)   | Median imputation on D_train      |
| M1  | **Naive Processing RF**   | 32 feats  | Config B (Mol + Direct| Standard median/mode imputation   |
|     |                           |           | Processing)           | without missingness flags         |
| M2  | **Principled Proc. RF**   | 44 feats  | Config C (Mol + Norm +| Explicit binary missingness flags |
|     |                           |           | Regimes + Constants)  | and domain-standard defaults      |
| M2A | **Ablation Model RF**     | 37 feats  | Config C minus the 7  | Tests whether missingness flags   |
|     | (No Missingness Flags)    |           | missingness flags     | exploit extraction shortcuts      |
| M3  | **Native-NaN GBDT**       | 44 feats  | Config C              | LightGBM / HistGradientBoosting;  |
|     | (Gradient Boosted Trees)  |           |                       | native missing split routing      |
+-----+---------------------------+-----------+-----------------------+-----------------------------------+
```

---

## 4. Anti-Leakage Contracts and Inner CV Rules

```
+---------------------------------------------------------------------------------------------------------+
| Information Isolation Protocol                                                                          |
+------------------------------------+--------------------------------------------------------------------+
| Boundary                           | Operational Enforcement Rule                                       |
+------------------------------------+--------------------------------------------------------------------+
| **Feature Transformation Fitting** | Imputation medians, standard scalers, and frequency maps are       |
|                                    | fitted strictly on D_train records.                                |
| **Hyperparameter Selection**       | Selected strictly via 5-fold GroupKFold CV grouped by material     |
|                                    | pair within D_train. D_cal and D_test are never touched.           |
| **Quantile Calibration**           | Conformal cutoffs q are computed on D_cal residuals after point     |
|                                    | models and scale estimators are frozen.                            |
| **Final Test Blindness**           | Predictions and interval boundaries are constructed from X_test    |
|                                    | before test labels y_test are accessed for metric evaluation.      |
| **Single-Threaded Enforcement**    | n_jobs=1 is enforced during prediction to prevent multi-threaded   |
|                                    | floating-point LSB boundary shifts.                                |
+------------------------------------+--------------------------------------------------------------------+
```

---

## 5. Comparative Evaluation Criteria and Acceptance Gates

### 5.1 Primary Performance Targets (Head-to-Head vs. Concurrent M0)
1. **Accuracy Superiority**:
   $$\text{MAE}(M_k) < \text{MAE}(M_0) \quad \text{and} \quad \text{RMSE}(M_k) < \text{RMSE}(M_0) \quad \text{on } D_{\text{test}}$$
2. **Uncertainty Interval Tightness**:
   $$\text{Median Width}(M_k) < \text{Median Width}(M_0) \quad \text{at 90% confidence}$$
   - *Pre-Registered Target Gate*: **Median Width $\le 5.80\text{ PCE pp}$** (retained as a pre-specified target gate, not a guaranteed outcome).
   - *Historical Context*: Phase 7.1 historical width was $8.84\text{ PCE pp}$; the direct comparison must be against Phase 8 M0.

### 5.2 Marginal Conformal Acceptance Gates (Evaluated on $N_{\text{test}} = 946$)
- **80% Marginal Coverage Band**: $[77.0\%, 83.0\%]$.
- **90% Marginal Coverage Band**: $[88.0\%, 92.5\%]$.
- **95% Marginal Coverage Band**: $[93.5\%, 96.5\%]$.
- *Scope Rule*: Marginal coverage guarantees marginal validity over unseen pairs on average. It does **not** mathematically guarantee conditional coverage within specific efficiency or architecture slices.

### 5.3 Diagnostic Subgroup Audits (Descriptive Reporting)
1. **Efficiency Tiers**:
   - Low PCE: $< 5.0\%$ ($N = 347$)
   - Moderate PCE: $5.0\% - 10.0\%$ ($N = 320$)
   - High PCE: $10.0\% - 15.0\%$ ($N = 216$)
   - Champion PCE: $\ge 15.0\%$ ($N = 63$)
2. **Device Architecture**:
   - `conventional` ($N = 400$)
   - `inverted` ($N = 241$)
   - `Unknown / Other` ($N = 305$)
3. **Processing Metadata Completeness Tiers**:
   - **Sparse Tier (0–1 fields reported)** ($N = 127$ test pairs)
   - **Partial Tier (2–4 fields reported)** ($N = 405$ test pairs)
   - **Rich Tier (5–7 fields reported)** ($N = 414$ test pairs)
4. **Champion Warning Trigger ($\hat{y} \ge 14.0\%$)**:
   - Slices where predicted PCE $\ge 14.0\%$ will be audited descriptively for sample count, empirical coverage, mean signed bias, and median interval width.
   - Independent-binomial significance tests on conditional slices are avoided to prevent ungrounded claims.

---

## 6. Execution Deliverables Authorized for Phase 8.1

Upon formal user authorization, Phase 8.1 is authorized to execute:
```bash
backend\.venv\Scripts\python.exe ml/experiments/phase_8_1_processing_aware_models.py
```
generating exclusively:
1. `ml/results/phase_8_1_partition_metadata.json`
2. `ml/results/phase_8_1_selected_records.csv`
3. `ml/results/phase_8_1_point_metrics.json`
4. `ml/results/phase_8_1_calibration_quantiles.json`
5. `ml/results/phase_8_1_test_predictions_and_intervals.csv`
6. `ml/results/phase_8_1_final_test_metrics.json`
7. `ml/results/phase_8_1_subgroup_metrics.json`
8. `ml/PHASE_8_1_PROCESSING_IMPROVEMENT_REPORT.md`

All existing Phase 5 and Phase 7 artifacts remain untouched.

---

**Protocol locked. Execution halted awaiting explicit Phase 8.1 authorization.**
