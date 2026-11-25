# OPV-PredictNet — Phase 7: Uncertainty Validation Protocol & Empirical Audit Framework

**Author / Auditor:** Antigravity Agent  
**Date:** November 24, 2026  
**Status:** Completed Validation Protocol Document (Planning Only; Awaiting Execution Approval)  
**Target Model:** Phase 5 Baseline Random Forest Regressor ($n=100, \text{depth}=\text{None}, \text{split}=2, \text{leaf}=1, \text{features}=1.0$)  
**Benchmark Dataset:** OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records, 4,730 material pair clusters)  
**Scope:** Specification of nominal coverage targets, interval efficiency metrics, 6-dimensional sliced auditing, stress-testing against distribution shift, and formal pass/fail criteria.

---

## 1. Protocol Objectives and Scope Restrictions

This protocol establishes the validation procedures to be executed in Phase 7.1 to evaluate the **Locally Scaled Conformal Calibration Engine** designed in `PHASE_7_CALIBRATION_DESIGN.md`.

### 1.1 Strict Pre-Execution Directives
In accordance with user authorization constraints:
- **No Model Replacement:** The Phase 5 Baseline Random Forest point predictor is strictly frozen.
- **No Production Integration:** No uncertainty intervals are exposed to the production API or frontend.
- **No Data Pruning:** Outlier records (e.g. $PCE < 1\%$ or $PCE > 18\%$) must **never be discarded**. All 22,467 records remain in the audit pipeline.
- **No Unapproved Execution:** No calibration fits or test set evaluations will be performed until this validation protocol is formally reviewed and approved.

---

## 2. Nominal Coverage Levels & Evaluation Metrics

Calibration performance will be evaluated across three standard nominal confidence levels:
1. **Nominal 80% Prediction Intervals** ($\alpha = 0.20$): Useful for rapid material screening.
2. **Nominal 90% Prediction Intervals** ($\alpha = 0.10$): Standard industrial risk-mitigation benchmark.
3. **Nominal 95% Prediction Intervals** ($\alpha = 0.05$): Conservative scientific safety threshold.

### 2.1 Quantitative Evaluation Metrics
For an evaluation partition $D_{\text{eval}} = \{(X_i, y_i)\}_{i=1}^{N_{\text{eval}}}$ with prediction intervals $C_{\alpha}(X_i) = [\hat{l}_{\alpha}(X_i), \hat{u}_{\alpha}(X_i)]$:

#### 1. Empirical Coverage Rate (ECR):
$$\text{ECR}_{\alpha} = \frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\left( y_i \in [\hat{l}_{\alpha}(X_i), \hat{u}_{\alpha}(X_i)] \right)$$
*Criterion:* For valid calibration, $\text{ECR}_{\alpha} \ge 1 - \alpha$. Deviations where $\text{ECR}_{\alpha} < 1 - \alpha - 0.02$ constitute severe empirical undercoverage.

#### 2. Mean and Median Prediction Interval Width (MPIW):
$$\text{MPIW}_{\alpha} = \frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} (\hat{u}_{\alpha}(X_i) - \hat{l}_{\alpha}(X_i))$$
$$\text{MedPIW}_{\alpha} = \text{Median}\left( \hat{u}_{\alpha}(X_i) - \hat{l}_{\alpha}(X_i) \right)$$
*Criterion:* Minimizing interval width while maintaining empirical coverage. Unscaled split conformal yields constant widths, while locally scaled conformal achieves adaptive widths.

#### 3. Winkler Interval Score (Penalized Loss):
The Winkler score penalizes both interval width and miscoverage:
$$\text{Score}_{\alpha}(l, u, y) = (u - l) + \frac{2}{\alpha}(l - y)\mathbb{I}(y < l) + \frac{2}{\alpha}(y - u)\mathbb{I}(y > u)$$
$$\text{Mean Winkler Score}_{\alpha} = \frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} \text{Score}_{\alpha}(\hat{l}_{\alpha}(X_i), \hat{u}_{\alpha}(X_i), y_i)$$
*Criterion:* Lower Winkler score indicates superior uncertainty efficiency.

---

## 3. Six-Dimensional Sliced Auditing Framework

Overall marginal coverage can hide severe local undercoverage in specific chemical or physical domains. The validation suite mandates auditing across six distinct dimensions:

```
Six-Dimensional Uncertainty Audit Suite
│
├── Dimension 1: Target Efficiency (PCE) Range
│   ├── Low (< 5%)
│   ├── Medium (5% to 12%)
│   ├── High (12% to 16%)
│   └── Champion (>= 16%) ──> Critical boundary undercoverage check
│
├── Dimension 2: Chemical Family Subgroups
│   ├── Acceptors: Fullerene (PCBM, PC71BM) vs. Non-Fullerene (NFA: Y6, ITIC)
│   └── Donors: Polymer materials vs. Small-molecule donors
│
├── Dimension 3: Device Architecture Subgroups
│   ├── Conventional Architecture (41.32%)
│   ├── Inverted Architecture (22.54%)
│   ├── Architecture Unspecified (36.10%)
│   └── Ambiguous / Dual Structure Reported (9 records, 0.04%)
│
├── Dimension 4: Material-Pair Sampling Frequency (Cluster Size)
│   ├── Rare / Singleton Pairs (n_g = 1)
│   ├── Moderate Frequency (2 <= n_g <= 10)
│   └── High-Frequency Benchmarks (n_g > 50: PM6:Y6, PTB7-Th:PC71BM, P3HT:PCBM)
│
├── Dimension 5: Environmental Split Reality
│   ├── Environment A: In-Domain Record Coverage (Known Material Pairs)
│   └── Environment B: Out-of-Domain Pair Coverage (Unseen Material Pairs)
│
└── Dimension 6: Molecular Scaffold Distance (Applicability Domain Stress Test)
    ├── High Similarity to Training Set (Descriptor Distance < 1st quartile)
    └── Low Similarity / Extrapolation (Descriptor Distance > 3rd quartile)
```

### 3.1 Detailed Slicing Specifications

#### Dimension 1: Performance by PCE Efficiency Tier
- **Low PCE ($PCE < 5.0\%$):** 5,420 records. Check for lower-boundary truncation artifacts at $0.0\%$.
- **Medium PCE ($5.0\% \le PCE < 12.0\%$):** 11,850 records. Bulk of the literature distribution.
- **High PCE ($12.0\% \le PCE < 16.0\%$):** 4,120 records. Advanced non-fullerene systems.
- **Champion Records ($PCE \ge 16.0\%$):** 1,077 records.
  - *Critical Vulnerability Check:* Tree ensembles systematically underestimate champion values due to leaf averaging. We must audit whether prediction intervals for champion devices maintain $90\%$ coverage or suffer catastrophic undercoverage.

#### Dimension 2: Performance by Chemical Family
- **Fullerene vs. Non-Fullerene Acceptors:**
  - Fullerenes (PC61BM, PC71BM, ICBA) represent older literature with lower mean PCE ($6.2\%$) and narrower spread.
  - NFAs (Y6, ITIC, BTP-eC9) represent modern literature with higher mean PCE ($13.8\%$) and greater variance.
  - Check whether the local scaling factor $\hat{\sigma}(X)$ properly broadens NFA intervals and narrows fullerene intervals.
- **Polymer vs. Small-Molecule Donors:**
  - Polymer repeats represented as monomer fragments have distinct descriptor profiles compared to complete small molecules. Auditing ensures both classes achieve nominal coverage.

#### Dimension 3: Performance by Device Architecture
- Verify empirical coverage separately for:
  - Conventional Architecture ($N = 9,284$)
  - Inverted Architecture ($N = 5,064$)
  - Architecture Unspecified ($N = 8,110$)
  - Ambiguous / Dual Structure ($N = 9$)
- *Key Investigation:* Inverted devices have lower target variance ($20.81\%^2$ vs $36.77\%^2$). The audit will verify whether locally scaled conformal prediction produces narrower intervals for Inverted devices while maintaining $\ge 90\%$ coverage.

#### Dimension 4: Material-Pair Sampling Frequency (Cluster Size)
- OPV literature is dominated by "benchmark superstar" blends:
  - High-frequency systems ($n_g > 50$): PM6:Y6, PTB7-Th:PC71BM, PBDB-T:ITIC, P3HT:PCBM.
  - Singleton systems ($n_g = 1$): Novel donor materials synthesized in a single paper.
- Audit whether calibration is dominated by superstar systems, leading to miscoverage on rare pairs.

#### Dimension 5: Environmental Split Reality (Known vs. Unseen Pairs)
- **Environment A (Record-Held-Out Split):** Random 80/20 record split within clusters. Tests ability to predict new device processing variations of known materials.
- **Environment B (Cluster-Held-Out Split):** GroupKFold split (zero material-pair overlap). Tests ability to provide valid prediction intervals for virtual screening of completely novel molecules.
- *Mandatory Reporting:* The degradation in coverage between Environment A and Environment B must be explicitly quantified.

#### Dimension 6: Molecular Scaffold Distance Stress Test
- Measure Euclidean distance in normalized 22-descriptor space from each test query $X_{\text{test}}$ to its nearest neighbor in $D_{\text{train}}$.
- Bin test queries into quartiles of distance ($Q_1$ to $Q_4$).
- Audit whether empirical coverage deteriorates in $Q_4$ (extrapolation domain), exposing the breakdown of exchangeability.

---

## 4. Failure Modes, Stress-Testing, and Diagnostic Criteria

The validation suite implements four explicit stress tests designed to detect calibration failure:

| Stress Test | Implementation Method | Failure Detection Threshold | Consequence if Failed |
| :--- | :--- | :--- | :--- |
| **Champion Cell Undercoverage** | Compute $\text{ECR}_{90\%}$ specifically for records with $PCE \ge 16.0\%$. | $\text{ECR}_{90\%} < 75.0\%$ | Flag champion prediction intervals as unreliable due to leaf truncation; report to user. |
| **Out-of-Distribution Degradation** | Compare $\text{ECR}_{90\%}$ in Distance $Q_1$ vs. Distance $Q_4$. | Coverage drop $> 10.0\%$ (e.g. $92\%$ in $Q_1$ dropping to $< 80\%$ in $Q_4$) | Restrict validity claim: intervals apply only within applicability domain (Distance $< Q_3$). |
| **Interval Width Blow-Up** | Identify queries where $W_{90\%}(X) > 15.0\%$ PCE. | Percentage of queries with width $> 15.0\%$ exceeds $5.0\%$. | Investigate extreme dispersion outliers; adjust numerical stabilizer $\epsilon$. |
| **Cluster Size Bias** | Compare $\text{ECR}_{90\%}$ on singletons ($n_g=1$) vs. high-frequency pairs ($n_g > 50$). | Absolute difference $> 8.0\%$. | Implement cluster-weighted quantile calibration. |

---

## 5. Formal Pass/Fail Decision Criteria for Phase 7.1

Before any uncertainty quantification system can be recommended for user interaction or API integration, it must pass all four Phase 7.1 Gates:

```
Phase 7.1 Approval Gate Check
│
├── Gate 1: Marginal Coverage Fidelity ──────> ECR_90% on held-out test set in [88.0%, 93.0%]
├── Gate 2: Efficiency (Interval Width) ────> MedPIW_90% <= 6.5% PCE across benchmark
├── Gate 3: High-PCE Tier Safety ────────────> ECR_90% on PCE >= 14% must be >= 80.0%
└── Gate 4: Zero Information Leakage ─────────> Automated assertion verifies ZERO cluster overlap
```

1. **Gate 1 (Marginal Coverage Fidelity):** Overall test set empirical coverage at nominal 90% must be between **88.0% and 93.0%**. (Values below 88% indicate undercoverage; values above 93% indicate severe overconservatism).
2. **Gate 2 (Interval Efficiency):** Median 90% prediction interval width across the test partition must not exceed **6.5% PCE**. (Intervals wider than 6.5% PCE provide little screening value in OPV development).
3. **Gate 3 (High-PCE Tier Safety):** Empirical coverage on high-efficiency devices ($PCE \ge 14.0\%$) must not drop below **80.0%** at nominal 90%.
4. **Gate 4 (Leakage Audit):** Zero material pair overlap between model fitting, calibration, and final test sets must be confirmed by automated hash-based assertions.

---

## 6. Execution Plan & Next Steps

When this validation protocol is formally reviewed and approved by the user:
1. **Execution Phase 7.1:** Implement the `GroupConformalCalibrator` engine in a dedicated script `ml/experiments/conformal_calibration_validation.py`.
2. **Results Artifacts:** Save full evaluation metrics, sliced subgroup tables, and interval width distributions to `ml/results/phase_7_*`.
3. **Final Report:** Author `ml/PHASE_7_CALIBRATION_REPORT.md` documenting empirical coverage across all six dimensions.
4. **Governing Policy:** No code changes will be deployed to production, and the Phase 5 Baseline Random Forest will remain the frozen point prediction engine.
