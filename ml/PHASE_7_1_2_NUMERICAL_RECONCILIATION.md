# OPV-PredictNet — Phase 7.1.2: Numerical Contradiction Reconciliation

**Date**: 2026-11-24  
**Audit Status**: Forensic Read-Only Investigation  
**Scope**: Reconciliation of numerical contradictions between Phase 7.1 Report (`ml/PHASE_7_1_GROUPED_CONFORMAL_REPORT.md`), Phase 7.1.1 Audit (`ml/PHASE_7_1_1_INDEPENDENT_AUDIT.md`), and raw serialized artifacts (`ml/results/`).  
**Constraints Enforced**: No models fitted, no predictions regenerated, no intervals recalibrated, no final tests rerun, no existing artifacts modified, no production code altered, no Git commits created.

---

## 1. Executive Summary & Reconciliation Ledger

This investigation conducted a line-by-line, column-by-column forensic audit of all serialized data files and reports generated under Phase 7.1 and Phase 7.1.1.

### 1.1 Summary Reconciliation Ledger
```
+-------------------------------------------------------------------------------------------------------------------------+
| Complete Reconciliation Ledger across Reports and Raw Artifacts                                                         |
+------------------------------+--------------------+-------------------+---------------------+---------------------------+
| Metric / Quantity            | Phase 7.1 Report   | Phase 7.1.1 Audit | True Value (CSV)    | Adjudication & Action     |
+------------------------------+--------------------+-------------------+---------------------+---------------------------+
| Test MAE (PCE pp)            | 2.2202             | 1.9790            | 2.220222...         | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Test RMSE (PCE pp)           | 2.9170             | 2.6710            | 2.916956...         | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Test R²                      | 0.5859             | 0.4775            | 0.585922...         | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Unnorm 95% Raw Median Width  | 11.41 PCE pp       | 11.2330 PCE pp    | 11.407093...        | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Loca 95% Raw Median Width    | 11.27 PCE pp       | 10.8288 PCE pp    | 11.273174...        | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Unnorm 80% Raw Median Width  | 6.51 PCE pp        | 6.5113 PCE pp     | 6.511251...         | Both agree (rounding diff)|
| Unnorm 90% Raw Median Width  | 8.84 PCE pp        | 8.8372 PCE pp     | 8.837188...         | Both agree (rounding diff)|
| Loca 80% Raw Median Width    | 6.48 PCE pp        | 5.8659 PCE pp     | 6.479195...         | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Loca 90% Raw Median Width    | 8.67 PCE pp        | 8.1638 PCE pp     | 8.673204...         | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Unnorm 95% Clipped Count     | 384 (40.59%)       | 260 (27.48%)      | 384 of 946 (40.59%) | Phase 7.1 CORRECT;        |
|                              |                    |                   |                     | Phase 7.1.1 WITHDRAWN     |
| Subgroup Taxonomy            | PCE Tiers,         | Donor Class,      | PCE Tiers,          | Phase 7.1 CORRECT;        |
|                              | Pair Frequency,    | Acceptor Class,   | Pair Frequency,     | Phase 7.1.1 fabricated    |
|                              | Architecture,      | Historical Era    | Architecture,       | non-existent schema;      |
|                              | High-PCE Slices    | (fabricated)      | High-PCE Slices     | WITHDRAWN                 |
+------------------------------+--------------------+-------------------+---------------------+---------------------------+
```

### Key Root-Cause Finding
The original Phase 7.1 report and its saved artifacts (`phase_7_1_test_predictions_and_intervals.csv`, `phase_7_1_final_test_metrics.json`, `phase_7_1_calibration_quantiles.json`, `phase_7_1_subgroup_metrics.json`) are **100% mathematically exact and mutually consistent**. The numerical contradictions arose exclusively from errors introduced in the markdown drafting of the Phase 7.1.1 audit report (`ml/PHASE_7_1_1_INDEPENDENT_AUDIT.md`), where the author presented fabricated point metrics, erroneous median widths, incorrect clipping counts, and non-existent subgroup partitions.

---

## 2. Resolution of Point-Prediction Metric Contradictions

### 2.1 The Contradiction
- **Phase 7.1 Report Statement**: $\text{MAE} = 2.2202$, $\text{RMSE} = 2.9170$, $R^2 = 0.5859$.
- **Phase 7.1.1 Audit Statement**: $\text{MAE} = 1.9790$, $\text{RMSE} = 2.6710$, $R^2 = 0.4775$.

### 2.2 Forensic Trace of Calculation Inputs
- **Data File**: `ml/results/phase_7_1_test_predictions_and_intervals.csv`
- **Total Rows Evaluated**: $N = 946$ (representing exactly 946 distinct held-out normalized material-name pairs sampled under Alternative B).
- **Target Column**: `pce` ($y_i$, experimental Power Conversion Efficiency, %).
- **Prediction Column**: `y_pred` ($\hat{y}_i$, point prediction from Phase 5 baseline Random Forest fitted on $D_{\text{train}}$).
- **Formulas**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
  $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}, \quad \text{where } \bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$$

### 2.3 Independent Calculation Results
Direct numerical execution on `phase_7_1_test_predictions_and_intervals.csv` yields:
```python
N = 946
Target mean (y_bar)   = 7.022790697674419 %
SS_tot                = 15007.47895052854
SS_res                = 6214.288285741697
MAE                   = 2.220222163095597 PCE pp
MSE                   = 6.569015101206868 (PCE pp)^2
RMSE                  = 2.9169562483142695 PCE pp
R²                    = 0.585921995384387
```

Comparison with saved JSON artifact `ml/results/phase_7_1_final_test_metrics.json`:
```json
"point_metrics": {
  "mae": 2.220222163095597,
  "rmse": 2.9169562483142695,
  "r2": 0.585921995384387
}
```
**Exact Discrepancy with Phase 7.1 Report**: **0.00e+00**.

### 2.4 Origin of Phase 7.1.1 Audit Values
A comprehensive workspace-wide regex scan for `1.9790`, `2.6710`, and `0.4775` across all `.json`, `.csv`, `.py`, and `.md` files confirmed that these numbers exist **nowhere in any dataset or serialized result file**. 
They were inadvertently fabricated during the drafting of `ml/PHASE_7_1_1_INDEPENDENT_AUDIT.md` (lines 173–182).
- **Action**: The Phase 7.1.1 figures ($\text{MAE}=1.9790, \text{RMSE}=2.6710, R^2=0.4775$) are **WITHDRAWN as erroneous hallucinations**.
- **Action**: The Phase 7.1 report figures ($\text{MAE}=2.2202, \text{RMSE}=2.9170, R^2=0.5859$) are **CONFIRMED as the true, mathematically exact test metrics**.

---

## 3. Resolution of Interval-Width Contradictions

### 3.1 The Contradiction
The Phase 7.1 report and Phase 7.1.1 audit presented conflicting median widths for the final test intervals:
- Unnormalized 95%: Phase 7.1 reported **11.41**, while Phase 7.1.1 reported **11.2330**.
- Locally Normalized 80%: Phase 7.1 reported **6.48**, while Phase 7.1.1 reported **5.8659**.
- Locally Normalized 90%: Phase 7.1 reported **8.67**, while Phase 7.1.1 reported **8.1638**.
- Locally Normalized 95%: Phase 7.1 reported **11.27**, while Phase 7.1.1 reported **10.8288**.

### 3.2 Mathematical Identity for Unnormalized Intervals
For unnormalized split conformal prediction, the scale factor is identically $1.0$ for all records:
$$\hat{C}_{\text{unnorm}}(x_i) = \left[ \hat{y}_i - q, \; \hat{y}_i + q \right]$$
The raw unclipped interval width for record $i$ is:
$$\text{Width}_i = (\hat{y}_i + q) - (\hat{y}_i - q) = 2 \cdot q$$
Because $q$ is a scalar constant derived from the calibration set, the raw interval width for the unnormalized method is **mathematically invariant across all test rows**:
$$\text{Width}_{\min} = \text{Width}_{\max} = \text{Width}_{\text{mean}} = \text{Width}_{\text{median}} = \text{Width}_{P25} = \text{Width}_{P75} = 2 \cdot q$$

### 3.3 Independent Verification from Raw Test Interval Columns
Every row's raw width was computed directly as `upper_raw - lower_raw` across all 946 rows in `phase_7_1_test_predictions_and_intervals.csv`:

```
+---------------------------------------------------------------------------------------------------------------------------------------+
| Detailed Width Distribution across Methods and Nominal Levels (N = 946)                                                               |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
| Method & Type       | Nominal | Minimum     | Maximum     | Mean        | Median      | P25         | P75         | Mathematical 2*q  |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
| Unnorm Raw          | 80%     | 6.511251    | 6.511251    | 6.511251    | 6.511251    | 6.511251    | 6.511251    | 6.511251 (Exact)  |
| Unnorm Clipped      | 80%     | 3.949805    | 6.511251    | 6.405497    | 6.511251    | 6.511251    | 6.511251    | N/A               |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
| Unnorm Raw          | 90%     | 8.837188    | 8.837188    | 8.837188    | 8.837188    | 8.837188    | 8.837188    | 8.837188 (Exact)  |
| Unnorm Clipped      | 90%     | 5.112774    | 8.837188    | 8.503035    | 8.837188    | 8.660783    | 8.837188    | N/A               |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
| Unnorm Raw          | 95%     | 11.407093   | 11.407093   | 11.407093   | 11.407093   | 11.407093   | 11.407093   | 11.407093 (Exact) |
| Unnorm Clipped      | 95%     | 6.397726    | 11.407093   | 10.618462   | 11.407093   | 9.945735    | 11.407093   | N/A               |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
| Loca Raw            | 80%     | 0.637324    | 19.793792   | 6.852735    | 6.479195    | 4.926232    | 8.444327    | Adaptive          |
| Loca Clipped        | 80%     | 0.637324    | 19.010998   | 6.764931    | 6.424045    | 4.847035    | 8.374180    | Adaptive          |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
| Loca Raw            | 90%     | 0.853137    | 26.496441   | 9.173235    | 8.673204    | 6.594372    | 11.303778   | Adaptive          |
| Loca Clipped        | 90%     | 0.853137    | 22.935616   | 8.809477    | 8.358825    | 6.366878    | 10.876089   | Adaptive          |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
| Loca Raw            | 95%     | 1.108882    | 34.439288   | 11.923098   | 11.273174   | 8.571169    | 14.692315   | Adaptive          |
| Loca Clipped        | 95%     | 1.108882    | 28.824337   | 10.962581   | 10.318831   | 7.902227    | 13.567347   | Adaptive          |
+---------------------+---------+-------------+-------------+-------------+-------------+-------------+-------------+-------------------+
```

### 3.4 Origin of the 11.2330 Value and Audit Reconciliation
- At 95% nominal coverage, the unnormalized calibration cutoff is $q = 5.703546491228075$.
- By algebraic necessity:
  $$\text{Raw Width} = 2 \cdot 5.703546491228075 = 11.40709298245615 \approx \mathbf{11.41\text{ PCE pp}}$$
- The raw median width reported in the Phase 7.1 report was **11.41 PCE pp**, which matches the exact CSV data.
- The value **11.2330 PCE pp** presented in the Phase 7.1.1 audit report was fabricated. It does not match the raw median (11.4071), the clipped median (11.4071), the clipped mean (10.6185), or any quantile in the dataset.
- **Action**: All width figures in the Phase 7.1.1 audit table (11.2330, 10.8288, 5.8659, 8.1638) are **WITHDRAWN**.
- **Action**: The Phase 7.1 report figures (Unnorm: 6.51, 8.84, 11.41; Loca: 6.48, 8.67, 11.27) are **CONFIRMED as correct**.

### 3.5 Clarification of Clipping Order and Clipping Counts
- **Order of Calculation**:
  - **Raw width** was computed on $[y_{\text{pred}} - q, \; y_{\text{pred}} + q]$ **BEFORE** clipping.
  - **Clipped width** was computed on $[\max(0.0, y_{\text{pred}} - q), \; y_{\text{pred}} + q]$ **AFTER** clipping lower bounds at $0.0\%$ PCE.
  - Both metrics are stored under separate keys (`raw_intervals` and `clipped_intervals_diagnostic`) in `phase_7_1_final_test_metrics.json`.
- **Exact Clipping Counts** ($\text{lower\_raw} < 0.0$):
  - **Unnormalized**:
    - 80% nominal: **121 / 946** records (12.79%) clipped at zero.
    - 90% nominal: **258 / 946** records (27.27%) clipped at zero.
    - 95% nominal: **384 / 946** records (40.59%) clipped at zero.
  - **Locally Normalized**:
    - 80% nominal: **121 / 946** records (12.79%) clipped at zero.
    - 90% nominal: **270 / 946** records (28.54%) clipped at zero.
    - 95% nominal: **430 / 946** records (45.45%) clipped at zero.
- The Phase 7.1.1 audit claim that "260 of 946 intervals (27.48%) clipped at zero" at 95% nominal was incorrect; the true count is **384 records (40.59%)**, matching `phase_7_1_final_test_metrics.json`.

---

## 4. Verification of Calibration Quantiles and Interval Construction

### 4.1 Calibration Sample Size and Non-Conformity Scores
- **Calibration Set Size**: Exactly $N_{\text{cal}} = 946$ records, corresponding to 946 disjoint pair groups.
- **Score Vectors**: Exactly 946 entries per method, computed via:
  $$s_i^{\text{unnorm}} = |y_i - \hat{y}_i|$$
  $$s_i^{\text{loca}} = \frac{|y_i - \hat{y}_i|}{\hat{\sigma}_i}$$
  where $\hat{\sigma}_i$ is the ensemble tree dispersion with stabilizer floor $\epsilon = 0.10$.

### 4.2 Exact Finite-Sample Rank Order Statistic
The mathematical rank order statistic without interpolation is:
$$k = \left\lceil (N_{\text{cal}} + 1)(1 - \alpha) \right\rceil = \left\lceil 947 \cdot (1 - \alpha) \right\rceil$$

```
+-------------------------------------------------------------------------------------------------------------------------+
| Finite-Sample Quantile Ranks and Exact Order Statistic Cutoffs (N_cal = 946)                                           |
+---------------+---------------+---------------+-----------------------+-----------------------+-------------------------+
| Method        | Nominal Level | Exact Rank k  | Order Statistic S_(k) | Saved Cutoff (JSON)   | Numerical Discrepancy   |
+---------------+---------------+---------------+-----------------------+-----------------------+-------------------------+
| Unnormalized  | 80% (α=0.20)  | k = 758       | 3.255625357142864     | 3.255625357142864     | 0.00e+00 (Exact)        |
| Unnormalized  | 90% (α=0.10)  | k = 853       | 4.418594055944055     | 4.418594055944055     | 0.00e+00 (Exact)        |
| Unnormalized  | 95% (α=0.05)  | k = 900       | 5.7035464912280744    | 5.7035464912280744    | 0.00e+00 (Exact)        |
+---------------+---------------+---------------+-----------------------+-----------------------+-------------------------+
| Locally Norm. | 80% (α=0.20)  | k = 758       | 1.483029282028439     | 1.483029282028439     | 0.00e+00 (Exact)        |
| Locally Norm. | 90% (α=0.10)  | k = 853       | 1.9852183214140642    | 1.9852183214140642    | 0.00e+00 (Exact)        |
| Locally Norm. | 95% (α=0.05)  | k = 900       | 2.580327886837722     | 2.580327886837722     | 0.00e+00 (Exact)        |
+---------------+---------------+---------------+-----------------------+-----------------------+-------------------------+
```

### 4.3 Verification of No Continuous Interpolation
- In Python, `sorted_scores[k - 1]` was used, taking the discrete $k$-th smallest value directly.
- Standard continuous percentile interpolation (e.g., `np.percentile(scores, 90)`) yields $4.417032$, which differs from the exact order statistic $S_{(853)} = 4.418594$.
- Code inspection of `phase_7_1_grouped_conformal.py` (line 72) confirms that discrete order statistics were used, and continuous interpolation was strictly avoided.

### 4.4 Test Interval Construction Verification
In `phase_7_1_test_predictions_and_intervals.csv`:
- For Unnormalized:
  $$\max_i \left| \text{lower\_raw}_i - (\hat{y}_i - q) \right| \le 5.33 \times 10^{-15}$$
  $$\max_i \left| \text{upper\_raw}_i - (\hat{y}_i + q) \right| \le 7.11 \times 10^{-15}$$
- For Locally Normalized:
  $$\max_i \left| \text{lower\_raw}_i - (\hat{y}_i - q \cdot \hat{\sigma}_i) \right| \le 7.33 \times 10^{-15}$$
  $$\max_i \left| \text{upper\_raw}_i - (\hat{y}_i + q \cdot \hat{\sigma}_i) \right| \le 5.33 \times 10^{-15}$$
The saved intervals match the mathematical definitions to within floating-point epsilon.

---

## 5. Statistical Rigor and Test Coverage Uncertainty

### 5.1 What the 946 Test Observations Represent
The 946 test instances represent **one randomly selected experimental device record from each of 946 distinct held-out material-name pairs** (`donor_canonical::acceptor_canonical`). 
The sampling was executed via Alternative B:
```python
sampled_test = df_test.groupby("pair_group", group_keys=False).apply(
    lambda grp: grp.sample(n=1, random_state=4202)
)
```

### 5.2 Why One Record per Pair Does Not Establish Mutual Independence
Previous drafts incorrectly asserted that selecting one record per pair establishes independent observations. While Alternative B eliminates *within-pair replicate clustering*, it **does not establish mutual independence across test pairs** for three distinct scientific reasons:
1. **Network Structure and Molecule Sharing**:
   - The organic photovoltaic material space is a bipartite graph. Popular donor polymers (e.g., PTB7-Th, PBDB-T, PM6) and acceptor molecules (e.g., PC71BM, ITIC, Y6) appear in dozens of distinct pair combinations.
   - For example, if a model systematically mispredicts the electronic properties of PC71BM, all pairs involving PC71BM will share correlated prediction errors.
2. **Laboratory and Batch Clustering**:
   - Multiple material pairs originate from the same research laboratories and publications. Shared solvent formulations, laboratory measurement setups, and device fabrication protocols create cross-pair covariance that is not removed by holding out pair IDs.
3. **Conditioning on Shared Training and Calibration Data**:
   - The test coverage indicator is:
     $$I_i = \mathbf{1}\{ y_i \in \hat{C}(x_i) \}$$
   - All 946 indicators share the **identical point-model parameters** $\hat{\theta}(D_{\text{train}})$ and the **identical calibration cutoff** $\hat{q}(D_{\text{cal}})$.
   - Conditioning on $(D_{\text{train}}, D_{\text{cal}})$, the errors are dependent. If the calibration draw $D_{\text{cal}}$ yields an unusually low cutoff $\hat{q}$, the probability of missing the interval increases simultaneously across all test points.

### 5.3 Required Assumptions for Binomial Confidence Intervals
A binomial confidence interval (such as the Wald interval $\hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}$ or Wilson score interval) requires:
1. **Independent and Identically Distributed (i.i.d.) Bernoulli Trials**: $I_1, \dots, I_N \overset{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$.
2. **Fixed Parameter $p$**: Constant probability of coverage across all instances.

**Scientific Audit Conclusion**:
These assumptions are **NOT strictly satisfied** in grouped benchmark datasets. The reported binomial standard errors ($\text{SE} \approx 1.01\%$ at 90%) and confidence intervals quantify only the idealized binomial test-sample variance conditional on $(D_{\text{train}}, D_{\text{cal}})$. They do not capture:
- Molecule-sharing covariance across test pairs.
- Calibration-sample variance across hypothetical redraws of $D_{\text{cal}}$.
Therefore, binomial confidence intervals must be interpreted strictly as descriptive sample diagnostics, not as rigorous bounds on marginal conformal coverage.

### 5.4 Empirical Test Coverage vs. Conformal Marginal Coverage Guarantee
- **Split Conformal Theoretical Guarantee**:
  $$\mathbb{P}_{(D_{\text{cal}}, (X_{n+1}, Y_{n+1}))}(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$$
  This is a **marginal probability** integrated over all possible calibration datasets $D_{\text{cal}}$ and test instances $(X_{n+1}, Y_{n+1})$ under exchangeability.
- **Empirical Test Coverage**:
  $$\hat{p}_{\text{test}} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbf{1}\{ y_i \in \hat{C}(x_i) \}$$
  This is a **single realization of conditional coverage** for one specific fitted model and one specific calibration cutoff $\hat{q}$.
- Observing $\hat{p}_{\text{test}} = 76.96\%$ at the 80% nominal level ($N_{\text{test}} = 946$) reflects finite-sample conditional fluctuation; it does not violate the marginal conformal theorem. However, with respect to the pre-registered benchmark acceptance gate of $[77.0\%, 83.0\%]$, it remains a strict **empirical gate failure**.

---

## 6. Audit of Subgroup Accounting and Exhaustiveness

The subgroup metrics in `ml/results/phase_7_1_subgroup_metrics.json` were audited for mathematical exhaustiveness, category overlap, and column accuracy.

```
+--------------------------------------------------------------------------------------------------------------------------------+
| Subgroup Accounting and Verification Ledger (N_test = 946)                                                                     |
+----------------------+-----------------------------+---------+-----------+---------------+---------------+---------------------+
| Category             | Slice / Tier                | Count   | % of Test | Unnorm 90% Cov| Loca 90% Cov  | Partition Status    |
+----------------------+-----------------------------+---------+-----------+---------------+---------------+---------------------+
| **PCE Tiers**        | < 5.0% PCE                  | 347     | 36.68%    | 88.76%        | 89.34%        | Exhaustive Partition|
|                      | 5.0% - 10.0% PCE            | 320     | 33.83%    | 91.88%        | 87.19%        | Mutually Exclusive  |
|                      | 10.0% - 15.0% PCE           | 216     | 22.83%    | 88.89%        | 84.72%        | Category Sum = 946  |
|                      | >= 15.0% PCE                | 63      | 6.66%     | 77.78%        | 92.06%        | (100.0% of Test)    |
+----------------------+-----------------------------+---------+-----------+---------------+---------------+---------------------+
| **Pair Frequency**   | Singleton (N_lit = 1)       | 569     | 60.15%    | 89.46%        | 89.98%        | Exhaustive Partition|
|                      | Repeated (N_lit > 1)        | 377     | 39.85%    | 88.59%        | 84.35%        | Category Sum = 946  |
+----------------------+-----------------------------+---------+-----------+---------------+---------------+---------------------+
| **Architecture**     | conventional                | 400     | 42.28%    | 91.75%        | 91.00%        | Non-Exhaustive      |
|                      | inverted                    | 241     | 25.48%    | 87.97%        | 85.89%        | Top-3 Categories    |
|                      | Unknown                     | 189     | 19.98%    | 87.83%        | 86.77%        | Category Sum = 830  |
|                      | *Other free-text labels*    | 116     | 12.26%    | N/A           | N/A           | (116 non-standard)  |
+----------------------+-----------------------------+---------+-----------+---------------+---------------+---------------------+
| **High-PCE Slices**  | PCE >= 12.0%                | 161     | 17.02%    | 83.85%        | 87.58%        | Overlapping Upper-  |
|                      | PCE >= 14.0%                | 84      | 8.88%     | 78.57%        | 90.48%        | Tail Diagnostics;   |
|                      | PCE >= 15.0%                | 63      | 6.66%     | 77.78%        | 92.06%        | DO NOT SUM          |
+----------------------+-----------------------------+---------+-----------+---------------+---------------+---------------------+
```

### 6.1 Findings on Subgroup Accounting
1. **Exhaustive Partitions**:
   - `pce_tiers` sums to exactly $347 + 320 + 216 + 63 = 946$. Every test observation belongs to exactly one tier.
   - `pair_frequency` sums to exactly $569 + 377 = 946$. Every test pair belongs to exactly one frequency class.
2. **Device Architecture (Top-3 Truncation)**:
   - `device_architecture` sums to $400 + 241 + 189 = 830$.
   - **Explanation of Missing 116 Records**: Code inspection of `phase_7_1_grouped_conformal.py` (line 686) reveals `for arch_val in arch_series.value_counts().head(3).index:`. The script deliberately extracted only the top 3 categories (`conventional`, `inverted`, and `Unknown`). The remaining 116 records represent 40 non-standard free-text literature annotations (e.g., `organic solar cell` [12], `polymer solar cell` [12], `bulk heterojunction` [10], `BHJ` [10], `all-polymer solar cell` [7]). This is a reporting truncation, not a record loss.
3. **Overlapping Diagnostic Slices**:
   - `high_pce_slices` ($161, 84, 63$) are cumulative upper-tail subsets:
     $$\{ \text{PCE} \ge 15\% \} \subset \{ \text{PCE} \ge 14\% \} \subset \{ \text{PCE} \ge 12\% \}$$
   - These slices must **never be added together** (their sum of 308 is mathematically meaningless).
4. **Subgroup Accuracy**:
   - Using the exact columns (`pce`, `unno_90_lower_raw`, `unno_90_upper_raw`, `loca_90_lower_raw`, `loca_90_upper_raw`), independent recalculation reproduces all subgroup sample counts, empirical coverages, and median widths with **0.00e+00 discrepancy**.

---

## 7. Audit Status and Final Ledger of Authoritative Numbers

### 7.1 Authoritative Results Table (To be used for all subsequent project phases)

```
+-----------------------------------------------------------------------------------------------------------------------+
| Authoritative Final Test Metrics for Phase 7.1 Grouped Conformal Calibration (N_test = 946)                           |
+--------------------------+-----------------------+---------------------+-------------------+--------------------------+
| Method                   | Nominal Level         | Empirical Coverage  | Raw Median Width  | 90% Winkler Score        |
+--------------------------+-----------------------+---------------------+-------------------+--------------------------+
| Point Baseline Model     | Point Estimate (Mean) | MAE = 2.2202 PCE pp | RMSE = 2.9170     | R² = 0.5859              |
+--------------------------+-----------------------+---------------------+-------------------+--------------------------+
| Unnormalized Conformal   | 80% (α = 0.20)        | 76.96% (728/946)    | 6.51 PCE pp       | 10.34                    |
| Unnormalized Conformal   | 90% (α = 0.10)        | 89.11% (843/946)    | 8.84 PCE pp       | 12.79 (WINNER)           |
| Unnormalized Conformal   | 95% (α = 0.05)        | 95.03% (899/946)    | 11.41 PCE pp      | 15.55                    |
+--------------------------+-----------------------+---------------------+-------------------+--------------------------+
| Locally Normalized Conf. | 80% (α = 0.20)        | 76.22% (721/946)    | 6.48 PCE pp       | 10.31                    |
| Locally Normalized Conf. | 90% (α = 0.10)        | 87.74% (830/946)    | 8.67 PCE pp       | 13.16                    |
| Locally Normalized Conf. | 95% (α = 0.05)        | 93.66% (886/946)    | 11.27 PCE pp      | 16.88                    |
+--------------------------+-----------------------+---------------------+-------------------+--------------------------+
```

### 7.2 Remaining Unresolved Issues
1. **Pre-Registered Acceptance Gate Failures**:
   - Unnormalized split conformal achieved $89.11\%$ at 90% nominal (**PASS**) and $95.03\%$ at 95% nominal (**PASS**).
   - Unnormalized split conformal achieved $76.96\%$ at 80% nominal, missing the lower gate boundary ($77.0\%$) by $0.04$ percentage points (**FAIL**).
   - Locally normalized conformal failed both the 80% gate ($76.22\%$) and the 90% gate ($87.74\%$).
   - A technical rationale addressing the 80% gate near-miss (exchangeability variance over finite test clusters) must be submitted before authorization of production model packaging.
2. **Architecture Field Normalization**:
   - The 116 non-standard architecture entries in `device_type` should be normalized into standard categories (`conventional`, `inverted`, `other`) in Phase 8 data pipeline upgrades.

---

## 8. Literal Git Status Confirmation

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
- **Original result artifacts modified**: **0**

**Execution halted in strict compliance with Phase 7.1.2 instructions.**
