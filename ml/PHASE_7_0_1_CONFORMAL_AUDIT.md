# OPV-PredictNet — Phase 7.0.1: Conformal Calibration Design Audit

**Author / Auditor:** Antigravity Agent  
**Date:** November 24, 2026  
**Status:** Completed Audit Document (Pre-Implementation Architectural Review)  
**Target Model:** Phase 5 Baseline Random Forest Regressor ($n=100, \text{depth}=\text{None}, \text{split}=2, \text{leaf}=1, \text{features}=1.0$)  
**Dataset:** OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records, 4,730 material pair clusters)  
**Scope:** Resolution of core design issues: bootstrap estimand duality, rigorous uncertainty-scale definition, physical constraint policies, and statistical calibration guardrails.

---

## 1. Executive Summary

Phase 7.0.1 provides a critical pre-implementation audit of the uncertainty quantification design formulated in Phase 7. This audit resolves four foundational design and methodological issues before any calibration code is executed:
1. **Bootstrap Estimand Duality:**
   - Both the **10,000-resample record-weighted** ($\bar{\Delta}_{\text{rec}} = -0.0637\text{ PCE pp}$, 95% CI: $[-0.2366, +0.0361]\text{ PCE pp}$, $p = 0.5757$) and **pair-unweighted** ($\bar{\Delta}_{\text{clust}} = +0.0208\text{ PCE pp}$, 95% CI: $[+0.0049, +0.0362]\text{ PCE pp}$, $p = 0.0096$) estimands are preserved.
   - We clarify their distinct scientific roles: record-weighted evaluation represents the primary operational metric for predicting laboratory device realizations, while pair-unweighted evaluation reflects average performance across unique molecular combinations. Neither estimand is universally superior.
2. **Correction of the Uncertainty-Scale Definition:**
   - We eliminate the improper designation of test-point tree disagreement as "out-of-bag (OOB) uncertainty." OOB predictions are defined strictly for training observations. For query observations outside the training set, tree-to-tree dispersion is **ensemble disagreement**, an uncalibrated heuristic proxy.
   - We investigate and specify training-only scale estimation procedures:
     - Baseline 1: Unnormalized Split Conformal (constant width, zero risk of scale misspecification).
     - Option 2: Frozen Tree-Disagreement Scale (ensemble standard deviation of the frozen Phase 5 baseline).
     - Option 3: Training-Only Secondary Residual Regressor (fitted strictly on training OOB residuals).
   - Any scale function must be **frozen strictly on training data** before calibration labels are observed.
3. **Physical Bound Policy (Raw vs. Clipped Intervals):**
   - We reject the blanket post-hoc imposition of an arbitrary 33.0% upper bound without target-domain verification.
   - All conformal prediction intervals will be generated, evaluated, and reported in their **raw, unclipped form** as the primary statistical deliverable. Constrained intervals ($[0.0\%, \text{clip}]$) will be evaluated in parallel to audit possible boundary distortion and empirical coverage loss.

---

## 2. Scientific Interpretation of Bootstrap Estimands

### 2.1 The Two Estimands in Grouped OPV Data
The OPV-DB benchmark exhibits extreme cluster size imbalance across its 4,730 unique donor–acceptor material pairs:
- **Champion Systems:** A handful of prominent systems (e.g. PM6:Y6, PTB7-Th:PC71BM, P3HT:PCBM) account for thousands of device fabrications with varying processing protocols.
- **Singleton Systems:** Over 2,500 material pairs appear only once or twice in published literature.

This structure gives rise to two distinct, legitimate scientific questions:

```
Bootstrap Estimand Duality
│
├── 1. Record-Weighted Estimand (Delta_bar_rec = -0.0637 PCE pp)
│   ├── Target: Average error difference across experimental device fabrications
│   ├── Weighting: Proportional to cluster size (PM6:Y6 heavily weighted)
│   ├── 95% CI (B = 10,000): [-0.2366, +0.0361] PCE pp (crosses zero, p = 0.5757)
│   └── Operational Role: Primary benchmark for device-level experimental prediction
│
└── 2. Pair-Unweighted Estimand (Delta_bar_clust = +0.0208 PCE pp)
    ├── Target: Average error difference across distinct molecular combinations
    ├── Weighting: Equal weight (1 / 4,730) per donor-acceptor pair
    ├── 95% CI (B = 10,000): [+0.0049, +0.0362] PCE pp (strictly positive, p = 0.0096)
    └── Operational Role: Secondary benchmark for broad chemical space screening
```

### 2.2 Mathematical Formulation & 10,000-Resample Empirical Results

| Metric | Record-Weighted Estimand ($\bar{\Delta}_{\text{rec}}$) | Pair-Unweighted Estimand ($\bar{\Delta}_{\text{clust}}$) |
| :--- | :--- | :--- |
| **Formula** | $\bar{\Delta}_{\text{rec}} = \frac{\sum_{g=1}^G \sum_{i \in g} \Delta_i}{\sum_{g=1}^G n_g}$ | $\bar{\Delta}_{\text{clust}} = \frac{1}{G} \sum_{g=1}^G \left( \frac{1}{n_g} \sum_{i \in g} \Delta_i \right)$ |
| **Resampling Estimator** | Ratio Estimator: $\bar{\Delta}^*_{\text{rec}} = \frac{\sum_{g^*} T_{g^*}}{\sum_{g^*} n_{g^*}}$ | Sample Mean of Cluster Means: $\bar{\Delta}^*_{\text{clust}} = \frac{1}{G} \sum_{g^*} \bar{\Delta}_{g^*}$ |
| **Observed Estimate** | **-0.063728 PCE pp** (favors Phase 5 baseline) | **+0.020798 PCE pp** (favors nested regularized model) |
| **Bootstrap Mean ($B=10k$)**| **-0.060014 PCE pp** | **+0.020735 PCE pp** |
| **Bootstrap SE** | **0.078382 PCE pp** | **0.008068 PCE pp** |
| **95% Clustered CI** | **[-0.2366, +0.0361] PCE pp** | **[+0.0049, +0.0362] PCE pp** |
| **Zero Crossing** | **Crosses Zero (Not Significant)** | **Does Not Cross Zero** |
| **Null-Centered p-Value** | **p = 0.5757** | **p = 0.0096** |

### 2.3 Scientific Interpretation & Policy
- **Why They Diverge:** The Phase 5 Baseline Random Forest achieves slightly lower MAE on high-density, heavily optimized material systems like PM6:Y6 (where deep trees capture fine-grained feature interactions). In contrast, the regularized model (`RF_Config_2_FeatureSqrt`) achieves a marginal error reduction ($\approx 0.02\%$) on sparse singleton pairs.
- **Operational Policy:**
  - Neither estimand is universally preferable.
  - The **record-weighted estimand** is retained as the primary operational benchmark for device fabrication modeling because it represents the actual frequency of device records in the literature.
  - The **pair-unweighted estimand** is preserved and reported alongside it as an independent metric of average performance across chemical pairs.
  - We do not claim universal superiority for either model; rather, we document that the choice of weighting determines whether the baseline or regularized model has lower average error.

---

## 3. Conceptual & Mathematical Audit of the Uncertainty Scale

### 3.1 Resolving the "Out-of-Bag (OOB)" Misnomer
In Phase 7 research drafts, the scaling proxy for new observations was described as "out-of-bag ensemble dispersion." This phrasing is technically incorrect:
1. **Strict Definition of OOB:** For a training observation $(X_i, y_i) \in D_{\text{train}}$, an out-of-bag prediction $\hat{y}_{\text{OOB}}(X_i)$ uses *only* the subset of trees $\{t : i \notin \text{bootstrap sample}_t\}$.
2. **Query Observations Outside Training:** For any observation $X \notin D_{\text{train}}$ (e.g. calibration or test points), the observation was not part of the training bootstrap procedure. All 100 trees evaluate $X$.
3. **Correct Nomenclature:** The standard deviation across tree predictions for an external observation $X$ is **ensemble tree disagreement** (or ensemble standard deviation):
   $$\hat{\sigma}_{\text{trees}}(X) = \sqrt{\frac{1}{B} \sum_{b=1}^B (T_b(X) - \bar{T}(X))^2}$$
   It must **never** be called "out-of-bag uncertainty."

### 3.2 Evaluation of Training-Only Scale Estimators
To implement locally scaled conformal prediction without data leakage, the dispersion function $\hat{\sigma}(X)$ must be **fitted strictly on $D_{\text{train}}$ and frozen** before any calibration labels are accessed.

We evaluate three candidate scale architectures:

```
Candidate Scale Estimators (All Frozen on D_train)
│
├── Baseline: Unnormalized Split Conformal
│   └── sigma(X) = 1.0 (Constant width)
│   └── Strengths: Zero parameters, perfectly robust, no scale collapse
│   └── Weaknesses: Fails to adapt to heteroscedasticity across chemical space
│
├── Candidate A: Frozen Ensemble Tree Disagreement
│   └── sigma(X) = sigma_trees(X) + epsilon
│   └── Strengths: Derived directly from the frozen Phase 5 baseline; no extra model
│   └── Weaknesses: Tree disagreement can collapse in sparse extrapolation regions
│
└── Candidate B: Training-Only Secondary Residual Regressor
    └── Train secondary model r_hat(X) on training OOB residuals: |y_i - mu_hat_OOB(X_i)|
    └── sigma(X) = max(r_hat(X), epsilon)
    └── Strengths: Explicitly targets expected absolute residual; smooth predictions
    └── Weaknesses: Adds architectural complexity; requires secondary model tuning
```

#### Selection and Pre-Registration:
- **Primary Method:** **Candidate A (Frozen Ensemble Tree Disagreement)** with numerical stabilizer $\epsilon = 0.10\%$ PCE. It directly leverages the Phase 5 Baseline Random Forest structure without introducing additional auxiliary models.
- **Reference Baseline:** **Unnormalized Split Conformal** ($\hat{\sigma}(X) = 1.0$) must be evaluated in parallel. If Candidate A does not demonstrate superior interval efficiency (lower Winkler score and narrower median width at equal coverage), the unnormalized baseline is preferred due to parsimony.

---

## 4. Physical Interval Constraints and Boundary Handling

### 4.1 Critique of the Universal 33% Upper Bound
In Phase 7 drafts, an upper bound clipping rule at $33.0\%$ was proposed based on the single-junction Shockley–Queisser limit.
- **Methodological Flaw:** OPV-DB contains observational literature records from single-junction, tandem, multi-junction, and measurement artifact devices that cannot be definitively separated from structured metadata.
- Imposing a rigid 33% cutoff without target-domain verification introduces artificial ceiling truncation.
- Furthermore, clipping conformal intervals post-hoc alters the probability distribution of the interval endpoints.

### 4.2 Raw vs. Clipped Reporting Protocol
1. **Primary Deliverable:** **Raw Conformal Prediction Intervals:**
   $$C_{\alpha}^{\text{raw}}(X) = [\hat{\mu}(X) - \hat{q}_{1-\alpha} \hat{\sigma}(X), \;\; \hat{\mu}(X) + \hat{q}_{1-\alpha} \hat{\sigma}(X)]$$
   All formal statistical coverage tests, efficiency metrics, and subgroup audits must be conducted on $C_{\alpha}^{\text{raw}}(X)$.
2. **Physical Feasibility Check (Lower Bound Clipping):**
   PCE physically cannot be negative. Setting $C_{\text{lower}}^{\text{clipped}} = \max(0.0, C_{\text{lower}}^{\text{raw}})$ is physically justified because $P(Y < 0) = 0$. This post-processing can only decrease interval width without reducing coverage.
3. **Upper Bound Clipping Policy:**
   Upper bound clipping at $33.0\%$ will **not** be imposed automatically. If a constrained variant is evaluated, its empirical coverage and truncation frequency must be reported separately.

---

## 5. Summary of Audit Directives for Phase 7.1

| Design Dimension | Phase 7 Research Draft | Phase 7.0.1 Audited Rule |
| :--- | :--- | :--- |
| **Bootstrap Estimands** | Emphasized record-weighted, downplayed cluster-weighted | **Preserve both estimands:** Record-weighted for device fabrication, pair-unweighted for chemical space screening. |
| **Uncertainty Scale** | Referred to test dispersion as "OOB uncertainty" | **Corrected:** OOB is training-only. Test dispersion is **ensemble tree disagreement**, frozen strictly on $D_{\text{train}}$. |
| **Conformal Baseline** | Proposed only normalized conformal | **Mandatory baseline:** Evaluate **Unnormalized Split Conformal** in parallel to prove whether scaling improves efficiency. |
| **Physical Constraints** | Hard-coded $[0.0\%, 33.0\%]$ clipping | **Raw intervals first:** Evaluate raw intervals as primary deliverable; evaluate $[0.0, \infty)$ clipping separately. |
| **Test Set Integrity** | General evaluation plan | **Strict anti-snooping:** Acceptance criteria and method selection are frozen before $D_{\text{test}}$ inspection. |

### 5.1 Final Scope & Statistical Bound Qualifications
1. **Validation Scope Separation:**
   - **Tier 2 (Unseen-Pair Holdout):** The primary focus of Phase 7.1 using a 60/20/20 pair-disjoint split and a single randomly selected record per pair ($S_g = |y_{g, j_g} - \hat{\mu}(x_{g, j_g})| / (\hat{\sigma}(x_{g, j_g}) + \epsilon)$).
   - **Tier 1 (Known-Pair Record Holdout):** Explicitly unvalidated and outside the initial implementation scope for Phase 7.1.
   - **Tier 3 (Scaffold Extrapolation):** Separate exploratory research audit; pair splits do not evaluate scaffold extrapolation.
2. **Finite-Sample Coverage Bounds:**
   - Lower bound: $P(Y_{\text{new}} \in C(X_{\text{new}})) \ge 1 - \alpha$.
   - Upper bound: $1 - \alpha + \frac{1}{n_{\text{cal}} + 1}$ strictly assumes continuous, distinct non-conformity scores (no ties). Discrete descriptor ties may result in empirical coverage exceeding this bound.
   - When $k > n_{\text{cal}}$, returning an infinite interval is theoretically valid, while raising `InsufficientCalibrationDataError` is an explicit implementation policy.
