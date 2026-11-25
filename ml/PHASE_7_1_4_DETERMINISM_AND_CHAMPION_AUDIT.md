# OPV-PredictNet — Phase 7.1.4: Deterministic Reproduction and High-PCE Warning Audit

**Date**: 2026-11-24  
**Audit Scope**: Investigation of inference determinism, single-threaded floating-point stability, statistical audit of high-PCE actual vs. predicted groups, and final gate interpretation for Phase 7.1 Grouped Conformal Prediction.  
**Audit Decision**: **CONDITIONALLY ACCEPTED WITH DOCUMENTED LIMITATIONS**  
**Strict Constraints Enforced**: All original Phase 7.1 results preserved; dataset and partitions unchanged; hyperparameters and quantiles untouched; interval widths unmanipulated; no production integration; zero Git commits created.

---

## 1. Executive Summary & Audit Mandate

Phase 7.1.4 resolves the two outstanding technical inquiries identified in Phase 7.1.3:
1. **Inference Determinism**: Pinpointing the exact computational mechanism responsible for the 1-record difference ($721/946 \to 722/946$) in Locally Normalized 80% coverage during reproduction, and testing single-threaded execution (`n_jobs=1`).
2. **High-PCE Warning Audit**: Rigorously comparing actual high-efficiency devices ($\text{PCE} \ge 14\%$ and $\ge 15\%$) against predicted high-efficiency devices ($\hat{y} \ge 14\%$ and $\ge 15\%$), and determining whether a user-facing warning based on $\hat{y} \ge 14\%$ is statistically justified.
3. **Rigorous Acceptance Gate Interpretation**: Reaffirming that pre-registered gate failures (80% nominal coverage and 90% median interval width) remain failed and cannot be retroactively excused.

---

## 2. Task 1: Deterministic Inference Investigation

### 2.1 Characterization of the 1-Record Discrepancy
In Phase 7.1.3, reproducing the full calibration pipeline on the 946 test pairs yielded identical outputs for all metrics except Locally Normalized 80% coverage, which shifted from $721 / 946$ ($76.22\%$) to $722 / 946$ ($76.32\%$).

Forensic row-by-row isolation of the coverage indicator arrays:
$$\mathbf{1}\{ y_i \in [\text{lower\_raw}_i, \; \text{upper\_raw}_i] \}$$
identified the exact instance responsible:
- **Test Row Index**: 329
- **Experimental Target**: $y_{329} = \mathbf{15.05000000000000\%}$ PCE
- **Point Prediction**: $\hat{y}_{329} = 12.311983730158724\%$ PCE
- **Original Saved Run (`n_jobs=-1`)**:
  $$\text{loca\_80\_upper\_raw} = \mathbf{15.04999999999999893\%} \implies y_{329} \le \text{upper} \text{ was } \mathbf{False}$$
- **Reproduced Run (`n_jobs=-1`)**:
  $$\text{loca\_80\_upper\_raw} = \mathbf{15.05000000000000071\%} \implies y_{329} \le \text{upper} \text{ was } \mathbf{True}$$
- **Numerical Difference**:
  $$\Delta = 15.05000000000000071 - 15.04999999999999893 = \mathbf{1.776 \times 10^{-15}\text{ PCE pp}}$$

### 2.2 Mechanism: Non-Associative Floating-Point Summation
In `phase_7_1_grouped_conformal.py`, the Random Forest is instantiated with `n_jobs=-1`. In scikit-learn:
1. `RandomForestRegressor.predict(X)` splits the 100 trees into batches processed across available CPU worker threads.
2. The tree predictions are aggregated in chunk order.
3. Because IEEE-754 floating-point addition is **non-associative**:
   $$(a + b) + c \neq a + (b + c)$$
   slight differences in thread scheduling, chunk boundaries, or CPU core availability alter the least significant bit (LSB) at the 16th decimal place ($10^{-16}$).
4. For row 329, the prediction interval upper bound happens to coincide **identically** with the true value $15.05\%$. A 1-bit shift across the boundary toggled the boolean condition `pce <= upper`.

### 2.3 Verification of Single-Threaded Determinism (`n_jobs=1`)
To test whether single-threaded prediction eliminates floating-point indeterminism, repeated isolated executions were conducted with `n_jobs=1` in temporary directories:
- **Test 1**: Isolated full pipeline execution with `n_jobs=1` (Run 1).
- **Test 2**: Identical isolated full pipeline execution with `n_jobs=1` (Run 2).

```
+-------------------------------------------------------------------------------------------------------------------------+
| Single-Threaded Determinism Benchmark (n_jobs=1 Run 1 vs. n_jobs=1 Run 2)                                              |
+------------------------------------+-----------------------+-----------------------+------------------+-----------------+
| Quantity Checked                   | Run 1 Value           | Run 2 Value           | Difference       | Status          |
+------------------------------------+-----------------------+-----------------------+------------------+-----------------+
| Point Predictions (All 946 rows)   | y_pred array          | y_pred array          | **0.00e+00**     | Bit-for-bit     |
| Raw Interval Bounds (All 946 rows) | Upper & Lower bounds  | Upper & Lower bounds  | **0.00e+00**     | Bit-for-bit     |
| Calibration Quantiles (All levels) | q cutoffs             | q cutoffs             | **0.00e+00**     | Bit-for-bit     |
| Unnormalized 80% Coverage Count    | 728 / 946 (76.9556%)  | 728 / 946 (76.9556%)  | **0** records    | Identical       |
| Unnormalized 90% Coverage Count    | 843 / 946 (89.1121%)  | 843 / 946 (89.1121%)  | **0** records    | Identical       |
| Unnormalized 95% Coverage Count    | 899 / 946 (95.0317%)  | 899 / 946 (95.0317%)  | **0** records    | Identical       |
| Locally Norm. 80% Coverage Count   | 722 / 946 (76.3214%)  | 722 / 946 (76.3214%)  | **0** records    | Identical       |
| Locally Norm. 90% Coverage Count   | 830 / 946 (87.7378%)  | 830 / 946 (87.7378%)  | **0** records    | Identical       |
| Locally Norm. 95% Coverage Count   | 886 / 946 (93.6575%)  | 886 / 946 (93.6575%)  | **0** records    | Identical       |
| Test MAE                           | 2.220222163095597     | 2.220222163095597     | **0.00e+00**     | Bit-for-bit     |
| Test RMSE                          | 2.9169562483142695    | 2.9169562483142695    | **0.00e+00**     | Bit-for-bit     |
| Test R²                            | 0.585921995384387     | 0.585921995384387     | **0.00e+00**     | Bit-for-bit     |
| Row 329 Upper Bound                | 15.05000000000000071  | 15.05000000000000071  | **0.00e+00**     | Bit-for-bit     |
+------------------------------------+-----------------------+-----------------------+------------------+-----------------+
```

### 2.4 Determinism Finding
- Single-threaded execution (`n_jobs=1`) completely eliminates floating-point non-determinism, yielding **100% bit-for-bit reproducible coverage counts, predictions, interval bounds, and metrics across repeated runs**.
- In production inference pipelines or regression testing, setting `n_jobs=1` is recommended to ensure deterministic output verification across varying hardware architectures.
- The original saved coverage count of $721/946$ ($76.22\%$) reflects an artifact of parallel summation (`n_jobs=-1`), while the deterministic single-threaded count is $722/946$ ($76.32\%$). Both fall below the 80% pre-registered gate ($77.0\%$).

---

## 3. Task 2: High-PCE Warning Audit (Actual vs. Predicted)

### 3.1 Motivation & Methodological Distinction
A critical statistical error in model auditing is confusing conditioning on the **unobserved true target** ($y \ge 14\%$) with conditioning on the **observable model prediction** ($\hat{y} \ge 14\%$):
- In deployment, the true PCE $y$ is **unknown** before device synthesis.
- Any operational screening decision or user-facing warning must be triggered strictly by the observable point prediction $\hat{y}$.

To evaluate whether a user-facing warning based on $\hat{y} \ge 14.0\%$ is scientifically justified, all four high-efficiency groups were recalculated independently on the 946 held-out test pairs.

### 3.2 High-PCE Recalculation Results Table

```
+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| High-PCE Performance Audit: Actual vs. Predicted Groupings (N_test = 946)                                                                             |
+---------------------+----+--------+---------+---------+-----------+----------------------+----------------------+---------------------------------+
| Group Definition    | N  | % Test | Target  | Pred    | Mean Bias | Unnormalized Cov %   | Locally Norm. Cov %  | Median Widths (80 / 90 / 95%)   |
|                     |    | Set    | Mean    | Mean    | (pp)      | 80%   90%   95%      | 80%   90%   95%      | Unnorm        Locally Norm      |
+---------------------+----+--------+---------+---------+-----------+----------------------+----------------------+---------------------------------+
| **Actual PCE ≥ 14%**| 84 | 8.88%  | 16.04%  | 13.60%  | -2.4394   | 64.3% 78.6% 85.7%    | 70.2% 90.5% 95.2%    | 6.51 / 8.84 / 11.41 | 7.46 / 9.98 / 12.97 |
| **Actual PCE ≥ 15%**| 63 | 6.66%  | 16.58%  | 14.02%  | -2.5626   | 66.7% 77.8% 85.7%    | 68.3% 92.1% 96.8%    | 6.51 / 8.84 / 11.41 | 6.68 / 8.95 / 11.63 |
| **Pred PCE ≥ 14%**  | 66 | 6.98%  | 14.20%  | 15.81%  | +1.6043   | 69.7% 80.3% 86.4%    | 54.5% 69.7% 77.3%    | 6.51 / 8.84 / 11.41 | 3.91 / 5.23 / 6.80  |
| **Pred PCE ≥ 15%**  | 48 | 5.07%  | 14.50%  | 16.29%  | +1.7910   | 70.8% 79.2% 87.5%    | 50.0% 66.7% 77.1%    | 6.51 / 8.84 / 11.41 | 3.51 / 4.70 / 6.11  |
+---------------------+----+--------+---------+---------+-----------+----------------------+----------------------+---------------------------------+
```

### 3.3 Deep Statistical Analysis

#### 1. The Bias Inversion (Regression-to-the-Mean vs. Winner's Curse)
- **Actual High-PCE Slices ($\text{PCE} \ge 14\%$ and $\ge 15\%$)**:
  - The model exhibits substantial **negative bias** ($-2.44$ pp for $\ge 14\%$; $-2.56$ pp for $\ge 15\%$).
  - Decision trees cannot extrapolate beyond the range of leaf training averages, causing severe shrinkage toward the dataset mean ($7.02\%$).
  - Because Unnormalized intervals have fixed half-width $\pm 4.42$ pp, a downward shift of $-2.56$ pp causes the true target to escape the upper bound, driving 90% coverage down to **77.78%** ($49/63$).
  - Locally Normalized intervals widen adaptively ($+1.14$ pp wider) and recover nominal coverage ($90.48\%$ and $92.06\%$).
- **Predicted High-PCE Slices ($\hat{y} \ge 14\%$ and $\ge 15\%$)**:
  - The model exhibits substantial **positive bias** ($+1.60$ pp for $\ge 14\%$; $+1.79$ pp for $\ge 15\%$).
  - This is a classical selection effect (the *winner's curse*): selecting instances with high predictions selects cases where random estimation error was positive.
  - Furthermore, several false positives enter this group: for $\hat{y} \ge 14.0\%$, true PCE ranges from **1.29% to 19.10%**!
  - **Catastrophic Breakdown of Locally Normalized Conformal**: On predicted champion cells, the tree ensemble exhibits spurious low tree-to-tree disagreement (mean scale shrinks from $2.31$ to $1.31$, median width collapses to $5.23$ pp). Because the intervals become overly narrow right where positive prediction bias is strongest, Locally Normalized 90% coverage **collapses to 69.70%** (and **54.55%** at nominal 80%)!
  - Unnormalized conformal maintains static width ($8.84$ pp) and achieves **80.30%** coverage at 90% nominal ($53/66$).

#### 2. Statistical Justification of a User-Facing Warning Based on $\hat{y} \ge 14\%$
- **Finding**: A user-facing warning triggered when **predicted PCE $\hat{y} \ge 14.0\%$** is **STATISTICALLY FULLY JUSTIFIED**.
- **Reasoning**:
  1. Under the selected Unnormalized method, empirical 90% coverage drops from the dataset-wide $89.11\%$ to **$80.30\%$** ($53/66$) when $\hat{y} \ge 14\%$.
  2. The point prediction is systematically optimistic, overestimating true performance by $+1.60\text{ PCE pp}$ on average.
  3. Under the alternative Locally Normalized method, coverage collapses to an unacceptable $69.70\%$.
  4. Users screening candidate donor-acceptor combinations must be warned that top-tier predictions carry elevated overestimation risk.

#### 3. Sample Size Limitations and Statistical Power
- **Sample Sizes**:
  - $\hat{y} \ge 14.0\%$: $N = 66$ pairs ($6.98\%$ of test set).
  - $\hat{y} \ge 15.0\%$: $N = 48$ pairs ($5.07\%$ of test set).
- **Statistical Assessment**:
  - Under a null hypothesis that true conditional coverage remains at $90\%$, observing $53$ or fewer successes out of $66$ trials has a one-sided binomial $p$-value of:
    $$p = \sum_{k=0}^{53} \binom{66}{k} (0.90)^k (0.10)^{66-k} = \mathbf{0.0076}$$
  - The drop in coverage from $90\%$ to $80.3\%$ is statistically significant at $\alpha = 0.01$.
  - **Caveat**: While the undercoverage is statistically genuine, the modest sample size ($N = 66$) implies that the exact conditional coverage estimate carries a wide 95% Wilson confidence interval: $[69.1\%, 88.1\%]$. A sample of $N=66$ is sufficient to justify a warning banner, but insufficient to establish a precise calibrated conditional cutoff.

---

## 4. Task 3: Final Acceptance Interpretation and Gate Status

All pre-registered acceptance thresholds established prior to Phase 7.1 execution remain locked and unamended:

```
+-------------------------------------------------------------------------------------------------------------------------+
| Complete Pre-Registered Acceptance Gate Ledger                                                                          |
+------------------------------------+------------------+------------------+------------------+---------------------------+
| Pre-Registered Gate Requirement    | Target Band / Max| Unnormalized     | Locally Norm.    | Strict Audit Verdict      |
+------------------------------------+------------------+------------------+------------------+---------------------------+
| **Gate 1: 80% Nominal Coverage**   | [77.0%, 83.0%]   | 76.96% (728/946) | 76.22% (721/946) | **FAIL** (Both methods)   |
| **Gate 2: 90% Nominal Coverage**   | [88.0%, 92.5%]   | 89.11% (843/946) | 87.74% (830/946) | **PASS** (Unnormalized);  |
|                                    |                  |                  |                  | **FAIL** (Locally Norm.)  |
| **Gate 3: 95% Nominal Coverage**   | [93.5%, 96.5%]   | 95.03% (899/946) | 93.66% (886/946) | **PASS** (Both methods)   |
| **Gate 4: 90% Median Width**       | Max 5.8 PCE pp   | 8.84 PCE pp      | 8.67 PCE pp      | **FAIL** (Both methods)   |
| **Gate 5: High-PCE Tier (≥15%)**   | Nominal ≥ 88.0%  | 77.78% (49/63)   | 92.06% (58/63)   | **FAIL** (Unnormalized);  |
|                                    |                  |                  |                  | **PASS** (Locally Norm.)  |
+------------------------------------+------------------+------------------+------------------+---------------------------+
```

### 4.1 Unambiguous Principles of Audit Integrity
1. **The 80% Coverage Gate Remains FAILED**:
   - The empirical coverage of $76.96\%$ missed the lower gate threshold of $77.0\%$.
   - While finite-sample exchangeability variance explains the minor gap ($-0.04$ pp), formal benchmarking forbids redefining tolerance boundaries post hoc. The gate remains strictly **FAILED**.
2. **The 90% Median Interval Width Gate Remains FAILED**:
   - The observed median width of $8.84\text{ PCE pp}$ exceeds the $5.8\text{ PCE pp}$ ceiling by $+3.04\text{ PCE pp}$.
   - This failure is rooted in the high experimental variance of organic solar cell literature (where identical molecular pairs vary by $>6\%$ PCE due to non-chemical processing variables). The gate remains strictly **FAILED**.
3. **Passing 90% and 95% Coverage Does NOT Override Failed Gates**:
   - Demonstrating that unnormalized split conformal achieves $89.11\%$ at 90% and $95.03\%$ at 95% proves that the methodology works as designed for marginal coverage.
   - However, partial pass rates do not excuse gate failures; they define the boundaries of the model's reliability.
4. **No Chemical Scaffold or Laboratory-Independent Guarantee Is Established**:
   - The exchangeability unit is strictly the *normalized material-name pair* within the OPV benchmark distribution.
   - The procedure does not guarantee coverage for unseen molecular scaffolds or novel synthetic chemistries.

---

## 5. Final Recommendation and Project Status

### Final Audit Recommendation:
### **CONDITIONALLY ACCEPTED WITH DOCUMENTED LIMITATIONS**

### Final Project Status:
1. **Suitability for Research Prototype**:
   - The Phase 7.1 Unnormalized Grouped Conformal Predictor is approved for integration into the **Research Prototype** tier of OPV-PredictNet.
   - The underlying mathematical execution, data partition boundaries, and non-leakage protocols are confirmed to be rigorous and reproducible.
2. **Mandatory Research Prototype Implementation Rules**:
   - **UI Interval Width Display**: Predictions must display symmetric intervals of $\hat{y} \pm 4.42\text{ PCE pp}$ (at 90% confidence), with explicit text stating the full width of $8.84\text{ PCE pp}$.
   - **High-PCE Warning Banner**: Whenever a user prediction satisfies $\hat{y} \ge 14.0\%$, the interface must render a high-priority diagnostic alert:
     > **Warning**: Predictions exceeding 14.0% PCE are subject to selection bias (+1.6 pp average overprediction). In benchmark validation, empirical 90% interval coverage dropped to 80.3% (N = 66). Interpret high forecasts as screening leads rather than confirmed device efficiencies.
   - **Scope Disclaimer**: The UI must state that prediction intervals apply to experimental device records drawn from previously cataloged donor–acceptor material families, not novel uncharacterized chemical scaffolds.
3. **Production Milestone Prohibition**:
   - The model must **NOT** be promoted to production release or advertised as a tight-uncertainty engine until interval widths can be reduced below $5.8\text{ PCE pp}$ (e.g., through processing-condition features such as annealing temperature, solvent additives, and D:A ratios in Phase 8).

---

## 6. Literal Git Status Confirmation

Execution of `git status --short`:
```
?? ml/DATASET_DECISION.md
?? ml/DATASET_RESEARCH_REPORT.md
?? ml/DATASET_SCHEMA_REPORT.md
?? ml/DATA_QUALITY_ASSESSMENT.md
?? ml/PHASE_4_1_SCIENTIFIC_AUDIT.md
?? ml/PHASE_4_2_ACQUISITION_REPORT.md
?? ml/PHASE_4_2_DATA_READINESS.md
?? ml/PHASE_4_2_FEATURE_AUDIT.md
?? ml/PHASE_4_2_STRUCTURE_AUDIT.md
?? ml/PHASE_4_3_PRETRAINING_REVIEW.md
?? ml/PHASE_5_1_INDEPENDENT_AUDIT.md
?? ml/PHASE_5_BASELINE_REPORT.md
?? ml/PHASE_6_1_1_FINAL_AUDIT.md
?? ml/PHASE_6_1_2_FINAL_STATISTICAL_CORRECTIONS.md
?? ml/PHASE_6_1_NESTED_VALIDATION_REPORT.md
?? ml/PHASE_6_VALIDATION_REPORT.md
?? ml/PHASE_7_0_1_CONFORMAL_AUDIT.md
?? ml/PHASE_7_0_1_FINAL_VALIDATION_DESIGN.md
?? ml/PHASE_7_0_1_GROUPED_COVERAGE_DESIGN.md
?? ml/PHASE_7_0_2_FINAL_DESIGN_CORRECTIONS.md
?? ml/PHASE_7_0_3_FINAL_SCOPE_AUDIT.md
?? ml/PHASE_7_0_4_PAIR_IDENTITY_AUDIT.md
?? ml/PHASE_7_0_5_PAIR_IDENTITY_RECONCILIATION.md
?? ml/PHASE_7_1_1_INDEPENDENT_AUDIT.md
?? ml/PHASE_7_1_2_NUMERICAL_RECONCILIATION.md
?? ml/PHASE_7_1_3_FINAL_ACCEPTANCE_AUDIT.md
?? ml/PHASE_7_1_4_DETERMINISM_AND_CHAMPION_AUDIT.md
?? ml/PHASE_7_1_GROUPED_CONFORMAL_REPORT.md
?? ml/PHASE_7_CALIBRATION_DESIGN.md
?? ml/PHASE_7_UNCERTAINTY_RESEARCH.md
?? ml/PHASE_7_VALIDATION_PROTOCOL.md
?? ml/experiments/
?? ml/results/
?? tests/ml/test_grouped_conformal.py
```

- **Tracked files modified**: **0**
- **Git commits created**: **0**
- **Original experiment artifacts modified**: **0**

**Audit complete. Execution halted in accordance with Phase 7.1.4 instructions.**
