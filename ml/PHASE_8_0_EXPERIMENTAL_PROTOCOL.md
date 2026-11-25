# OPV-PredictNet — Phase 8.0: Experimental and Validation Protocol

**Date**: 2026-11-24  
**Scope**: Experimental methodology, grouping definitions, leakage isolation rules, model candidate architectures, comparative evaluation criteria, and uncertainty benchmarking for Phase 8.  
**Deliverable**: Pre-Registered Research Protocol for Processing-Aware OPV Modeling.

---

## 1. Objective and Scientific Hypothesis

### 1.1 Objective
To establish whether augmenting intrinsic molecular descriptors with device fabrication and processing parameters (blend ratio, solvent, additives, annealing temperature, active-layer thickness, and electrode interlayers) significantly improves:
1. **Point-Prediction Accuracy**: Reducing test MAE below $2.22\text{ PCE pp}$ and RMSE below $2.92\text{ PCE pp}$, with positive $R^2 > 0.586$.
2. **Prediction Interval Tightness**: Shrinking 90% conformal prediction interval median width below the $8.84\text{ PCE pp}$ unnormalized baseline toward the target gate of $\le 5.80\text{ PCE pp}$.
3. **Champion-Cell Calibration**: Eliminating conditional undercoverage on high-efficiency devices ($\hat{y} \ge 14.0\%$) and mitigating selection bias.

### 1.2 Scientific Hypothesis
In literature OPV devices, variation among devices sharing identical donor-acceptor molecular names stems from processing choices that dictate nanoscale phase separation and morphological charge percolation. Incorporating processing parameters and their explicit missingness indicators will enable tree algorithms to distinguish low-efficiency unoptimized formulations from high-efficiency champion recipes, explaining within-pair variance that purely molecular descriptors cannot resolve.

---

## 2. Partitioning, Grouping, and Generalization Scope

To maintain rigorous comparability with Phase 5 and Phase 7.1, the evaluation framework enforces exact structural grouping:

```
+---------------------------------------------------------------------------------------------------------+
| Evaluation Grouping Hierarchy and Generalization Boundaries                                             |
+--------------------------+-----------------------+------------------------------------------------------+
| Evaluation Tier          | Grouping Key          | Scientific Generalization Claim                      |
+--------------------------+-----------------------+------------------------------------------------------+
| **Primary Benchmark**    | Normalized Material-  | Generalization to unseen experimental device records |
| (Mandatory Comparison)   | Name Pair (4,730)     | of **unseen material-name pairs** under known        |
|                          | `donor::acceptor`     | literature chemistry.                                |
+--------------------------+-----------------------+------------------------------------------------------+
| **Sensitivity Tier 1**   | Canonical Molecular-  | Sensitivity to molecular structural equivalence.     |
| (Secondary Audit)        | Structure (4,788)     | Evaluates generalization across unique RDKit SMILES. |
+--------------------------+-----------------------+------------------------------------------------------+
| **Sensitivity Tier 2**   | Bemis-Murcko Scaffold | Generalization across core chemical scaffolds.       |
| (Exploratory Audit)      | Clustering            | Strict out-of-distribution scaffold evaluation.       |
+--------------------------+-----------------------+------------------------------------------------------+
```

### 2.1 Boundary Disclaimer on Chemical Scaffolds
Generalization across material-name pairs does **not** constitute proof of generalization across novel molecular scaffolds. A prominent donor (e.g., PM6) or acceptor (e.g., PC71BM, Y6) may appear in training pairs while paired with a new partner in testing. Formal claims must be strictly bounded to *"device-record prediction for unseen material-name combinations."*

### 2.2 Locked Benchmark Partitions (60 / 20 / 20 Split)
The benchmark dataset ($N = 22,467$, 4,730 pairs) is partitioned via locked random seed `seed=8888` (as specified in `ml/PHASE_8_0_1_LOCKED_VALIDATION_PROTOCOL.md`):
- **Training Partition ($D_{\text{train}}$)**: Exactly 2,838 material-name pairs (60.0% of groups, 14,704 records). Used exclusively for model training, feature selection, and inner cross-validation.
- **Calibration Partition ($D_{\text{cal}}$)**: Exactly 946 material-name pairs (20.0% of groups, 3,349 records; sampled 1 record per pair via Alternative B, `cal_seed=8201`). Used exclusively for non-conformity score quantile calculation.
- **Test Partition ($D_{\text{test}}$)**: Exactly 946 material-name pairs (20.0% of groups, 4,414 records; sampled 1 record per pair via Alternative B, `test_seed=8202`). Used strictly for final frozen evaluation.

---

## 3. Candidate Model Architectures and Baselines

Five distinct model candidates will be trained strictly on $D_{\text{train}}$:

```
+---------------------------------------------------------------------------------------------------------+
| Candidate Model Architectures                                                                           |
+-----+-----------------------------+---------------------+-------------------+---------------------------+
| ID  | Model Architecture          | Feature Matrix      | Missingness Treat.| Description               |
+-----+-----------------------------+---------------------+-------------------+---------------------------+
| M0  | **Concurrent Baseline RF**  | Config A (22 feats) | Intrinsic median  | Concurrently trained on   |
|     | (RandomForestRegressor)     | Pure Molecular      | on D_train        | D_train, locked baseline. |
+-----+-----------------------------+---------------------+-------------------+---------------------------+
| M1  | **Naive Processing RF**     | Config B (32 feats) | Standard median / | Conventional integration; |
|     | (RandomForestRegressor)     | Molecular + Process | mode imputation   | no missingness indicators.|
+-----+-----------------------------+---------------------+-------------------+---------------------------+
| M2  | **Principled Processing RF**| Config C (44 feats) | Explicit binary   | Domain-aware regimes,     |
|     | (RandomForestRegressor)     | Molecular + Regimes | flags + defaults  | flags, and interaction WF.|
+-----+-----------------------------+---------------------+-------------------+---------------------------+
| M2A | **Ablation Processing RF**  | Config C-Ablation   | Defaults without  | Diagnostic model: tests   |
|     | (RandomForestRegressor)     | (37 feats, no flags)| missingness flags | shortcut learning on flags|
+-----+-----------------------------+---------------------+-------------------+---------------------------+
| M3  | **Native-NaN Gradient GBDT**| Config C (44 feats) | Native split-path | LightGBM / HistGradient;  |
|     | (HistGradientBoostingReg.)  | Molecular + Regimes | routing for NaNs  | non-linear split on NaNs. |
+-----+-----------------------------+---------------------+-------------------+---------------------------+
```

### 3.1 Inner Cross-Validation and Hyperparameter Locking
- **Tuning Boundary**: Hyperparameters will be selected via 5-fold `GroupKFold` cross-validation grouped by material-name pair **strictly within $D_{\text{train}}$**.
- **Point Model Tuning Grid**:
  - `max_depth`: `[None, 12, 16, 20]`
  - `min_samples_leaf`: `[1, 2, 4]`
  - `learning_rate` (for GBDT): `[0.03, 0.05, 0.10]`
- **Freezing Contract**: All hyperparameters and feature transformers will be locked before accessing $D_{\text{cal}}$ or $D_{\text{test}}$.

---

## 4. Comparative Evaluation Protocol

Candidate models will be evaluated head-to-head on the identical held-out test set ($N_{\text{test}} = 946$ unseen material pairs, Alternative B):

### 4.1 Primary Quantitative Metrics
1. **Mean Absolute Error (MAE)**: $\text{MAE} = \frac{1}{N} \sum |y_i - \hat{y}_i|$ (Target: $< 2.22\text{ PCE pp}$).
2. **Root Mean Squared Error (RMSE)**: $\text{RMSE} = \sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$ (Target: $< 2.92\text{ PCE pp}$).
3. **Coefficient of Determination ($R^2$)**: $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ (Target: $> 0.586$).
4. **Mean Signed Bias (MSB)**: $\text{Bias} = \frac{1}{N} \sum (\hat{y}_i - y_i)$ (Evaluating global systematic drift).

### 4.2 Sliced Subgroup Comparisons
To prevent global metrics from masking localized degradation, models must be compared across three pre-registered subgroup slices:
1. **Efficiency Slices**:
   - Low PCE: $< 5.0\%$ ($N = 347$)
   - Moderate PCE: $5.0\% - 10.0\%$ ($N = 320$)
   - High PCE: $10.0\% - 15.0\%$ ($N = 216$)
   - Champion PCE: $\ge 15.0\%$ ($N = 63$)
2. **Device Architecture Slices**:
   - `conventional` ($N = 400$)
   - `inverted` ($N = 241$)
   - `Unknown / Other` ($N = 305$)
3. **Processing Metadata Completeness Slices**:
   - **Sparse Tier (0–1 fields reported)** ($N = 127$ test pairs)
   - **Partial Tier (2–4 fields reported)** ($N = 405$ test pairs)
   - **Rich Tier (5–7 fields reported)** ($N = 414$ test pairs)

---

## 5. Uncertainty Quantification Research Protocol

To determine whether processing-aware modeling resolves the uncertainty interval limitations identified in Phase 7.1, split conformal prediction will be applied to all candidate models.

### 5.1 Uncertainty Metrics and Acceptance Gates
1. **Marginal Coverage Evaluation**:
   - 80% Nominal Gate: Target band $[77.0\%, 83.0\%]$.
   - 90% Nominal Gate: Target band $[88.0\%, 92.5\%]$.
   - 95% Nominal Gate: Target band $[93.5\%, 96.5\%]$.
2. **Interval Tightness Gate (90% Confidence)**:
   - Primary Gate: **Median width $\le 5.80\text{ PCE pp}$**.
   - Benchmark Comparison: Reduction compared to the Phase 7.1 unnormalized width ($8.84\text{ PCE pp}$).
3. **Champion-Cell Calibration Gate**:
   - For predicted high-efficiency cells ($\hat{y} \ge 14.0\%$), empirical 90% interval coverage must exceed **85.0%** (mitigating the $80.3\%$ undercoverage failure in Phase 7.1).
4. **Stability across Completeness Tiers**:
   - Conformal intervals must maintain nominal coverage ($\ge 88.0\%$ at 90%) across all three completeness tiers (Sparse, Partial, Rich), proving that missing processing metadata does not destabilize uncertainty estimates.

---

## 6. Execution Safeguards and Non-Leakage Contract

```
+---------------------------------------------------------------------------------------------------------+
| Protocol Safeguards Checklist                                                                           |
+------------------------------------+--------------------------------------------------------------------+
| Safeguard Principle                | Verification Protocol                                              |
+------------------------------------+--------------------------------------------------------------------+
| **Isolation of Test Data**         | Features X_test and targets y_test must never be passed to inner   |
|                                    | cross-validation, imputation fitting, or hyperparameter selection. |
| **Independence of Calibration**    | Calibration residuals must be computed on D_cal after point model  |
|                                    | parameters and scale estimators are frozen.                        |
| **Discrete Finite-Sample Ranks**   | Cutoffs must use exact rank k = ceil((946 + 1) * (1 - alpha))      |
|                                    | without continuous percentile interpolation.                       |
| **Reproducibility Contract**       | Single-threaded execution (n_jobs=1) must be used for final frozen |
|                                    | metric generation to ensure bit-for-bit determinism.               |
| **Artifact Preservation**          | All Phase 8 experiment scripts must output to ml/results/phase_8_* |
|                                    | and never overwrite Phase 5 or Phase 7 artifacts.                  |
+------------------------------------+--------------------------------------------------------------------+
```

---

**Protocol documented and locked. Ready for experimental execution upon authorization.**
