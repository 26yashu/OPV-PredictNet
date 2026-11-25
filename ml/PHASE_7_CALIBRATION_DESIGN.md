# OPV-PredictNet — Phase 7: Calibration Strategy & Architecture Design

**Author / Auditor:** Antigravity Agent  
**Date:** November 24, 2026  
**Status:** Completed Calibration Design Document (Validation Planning Only)  
**Target Model:** Phase 5 Baseline Random Forest Regressor ($n=100, \text{depth}=\text{None}, \text{split}=2, \text{leaf}=1, \text{features}=1.0$)  
**Scope:** Mathematical architecture of the proposed uncertainty calibration engine, multi-tier leakage prevention protocol, and physical constraint handling.

---

## 1. Design Overview and Objectives

This document specifies the mathematical and architectural design of the uncertainty quantification system for OPV-PredictNet. The design strictly adheres to three fundamental constraints:
1. **Preservation of the Baseline:** The model being calibrated is the existing, frozen **Phase 5 Baseline Random Forest Regressor** trained on 22 molecular descriptors.
2. **Zero Information Leakage:** Calibration records are strictly held out from model fitting, and test evaluation sets are completely isolated from both fitting and calibration.
3. **Adaptive Prediction Intervals:** The system employs **Locally Scaled Conformal Prediction** to ensure that prediction intervals adapt to chemical heteroscedasticity while providing exact finite-sample marginal coverage under exchangeability.

---

## 2. Multi-Tier Leakage Prevention Protocol

To guarantee valid calibration without data snooping or optimistic bias, benchmark records are partitioned using a **strict 3-way group-aware hierarchy**:

```
Full Benchmark Dataset (22,467 records, 4,730 material pair clusters)
│
├── Step 1: Material-Pair Cluster Grouping (donor_canonical + '::' + acceptor_canonical)
│
├── 1. Model Fitting Partition (D_train):   60% of material pairs (~2,838 pairs, ~13,480 records)
│      └── Used EXCLUSIVELY to fit the Phase 5 Baseline Random Forest (mu_hat)
│
├── 2. Calibration Partition (D_cal):       20% of material pairs (~946 pairs, ~4,493 records)
│      └── Used EXCLUSIVELY to compute non-conformity scores and calculate quantile (q_hat)
│
└── 3. Final Validation Partition (D_test):  20% of material pairs (~946 pairs, ~4,494 records)
       └── HELD OUT COMPLETELY; inspected ONLY during final validation audit
```

### 2.1 Formal Group Isolation Constraints
Let $\mathcal{G}_{\text{train}}$, $\mathcal{G}_{\text{cal}}$, and $\mathcal{G}_{\text{test}}$ denote the sets of unique donor–acceptor material pair strings assigned to each partition. The following assertions are programmatically enforced:
$$\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{cal}} = \emptyset$$
$$\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{test}} = \emptyset$$
$$\mathcal{G}_{\text{cal}} \cap \mathcal{G}_{\text{test}} = \emptyset$$

#### Strict Execution Rules:
- **No In-Sample Calibration:** Training records are never passed to the non-conformity scoring function.
- **No Hyperparameter Feedback:** Quantile calculations on $D_{\text{cal}}$ cannot be used to tune or adjust Random Forest hyperparameters.
- **Isolated Evaluation:** $D_{\text{test}}$ is never used to select between dispersion heuristics or calibrate quantiles.

---

## 3. Mathematical Architecture: Locally Scaled Conformal Prediction

### 3.1 Components of the System
The calibration engine consists of three mathematical operators:
1. **Point Estimator ($\hat{\mu}$):** The Phase 5 Baseline Random Forest:
   $$\hat{\mu}(X) = \frac{1}{100} \sum_{t=1}^{100} T_t(X)$$
2. **Local Dispersion Estimator ($\hat{\sigma}$):** A heuristic scaling function that estimates relative prediction difficulty across the 22-dimensional feature space.
   - Primary specification: **Out-of-Bag Ensemble Dispersion** with numerical stabilizer $\epsilon$:
     $$\hat{\sigma}(X) = \sqrt{\frac{1}{100} \sum_{t=1}^{100} (T_t(X) - \hat{\mu}(X))^2} + \epsilon$$
     Where $\epsilon = 0.05\%$ (PCE) prevents division by zero when tree predictions coincide.
   - *Important Scientific Role:* While $\hat{\sigma}(X)$ is uncalibrated on its own, it acts as a valid relative difficulty weighting across chemical families.
3. **Normalized Non-Conformity Score ($S$):** For any calibration sample $(X_i, y_i) \in D_{\text{cal}}$, the non-conformity score measures the scaled residual:
   $$S_i = \frac{|y_i - \hat{\mu}(X_i)|}{\hat{\sigma}(X_i)}$$

### 3.2 Conformal Quantile Computation
For a desired nominal miscoverage level $\alpha \in \{0.20, 0.10, 0.05\}$ (corresponding to 80%, 90%, and 95% prediction intervals):
1. Compute the set of calibration scores $\mathcal{S}_{\text{cal}} = \{S_1, S_2, \dots, S_{n_{\text{cal}}}\}$.
2. Sort the scores in ascending order: $S_{(1)} \le S_{(2)} \le \dots \le S_{(n_{\text{cal}})}$.
3. Compute the finite-sample adjusted conformal quantile index:
   $$k_{\alpha} = \left\lceil (n_{\text{cal}} + 1)(1 - \alpha) \right\rceil$$
4. Extract the empirical conformal multiplier:
   $$\hat{q}_{1-\alpha} = S_{(k_{\alpha})}$$

### 3.3 Prediction Interval Construction
For any new query donor–acceptor system $X_{n+1}$:
1. Compute the baseline point prediction: $\hat{y} = \hat{\mu}(X_{n+1})$.
2. Compute the local dispersion proxy: $\hat{\sigma}_{n+1} = \hat{\sigma}(X_{n+1})$.
3. Compute the half-width:
   $$W_{\alpha}(X_{n+1}) = \hat{q}_{1-\alpha} \cdot \hat{\sigma}_{n+1}$$
4. Construct the raw prediction interval:
   $$C_{\alpha}(X_{n+1}) = [\hat{\mu}(X_{n+1}) - W_{\alpha}(X_{n+1}), \; \hat{\mu}(X_{n+1}) + W_{\alpha}(X_{n+1})]$$

### 3.4 Physical Boundary Constraints (Post-Processing Projection)
Power Conversion Efficiency (PCE) has fundamental physical bounds:
- **Lower Bound:** PCE cannot be negative ($PCE \ge 0.0\%$).
- **Upper Bound:** The theoretical single-junction Shockley–Queisser limit under standard AM1.5G solar illumination is approximately $33.7\%$ (conservatively set to $33.0\%$ for organic single junctions).

The final calibrated prediction interval is projected onto the physically admissible set:
$$C_{\alpha}^{\text{physical}}(X) = \left[ \max(0.0, \; \hat{\mu}(X) - W_{\alpha}(X)), \;\; \min(33.0, \; \hat{\mu}(X) + W_{\alpha}(X)) \right]$$

*Mathematical Property:* Because $y \in [0.0, 33.0]$ almost surely for physical devices, projecting the interval onto the admissible domain can **never decrease coverage**; it strictly improves interval efficiency (narrower average width) without altering coverage validity.

---

## 4. Alternative Specification: Grouped CV+ (Cross-Validation Conformal)

For scenarios where reserving 20% of the dataset for standalone calibration ($D_{\text{cal}}$) reduces model training sample size excessively, the design specifies an alternative **Grouped CV+ Calibration Architecture**:

```
Grouped CV+ Architecture (5-Fold GroupKFold Outer Loop)
│
├── Fold 1: Train on Folds 2-5 ──> Predict out-of-fold on Fold 1 ──> Collect S_1 ... S_n1
├── Fold 2: Train on Folds 1,3-5 ──> Predict out-of-fold on Fold 2 ──> Collect S_n1+1 ... S_n2
├── Fold 3: Train on Folds 1-2,4-5 ──> Predict out-of-fold on Fold 3 ──> Collect S_n2+1 ... S_n3
├── Fold 4: Train on Folds 1-3,5 ──> Predict out-of-fold on Fold 4 ──> Collect S_n3+1 ... S_n4
└── Fold 5: Train on Folds 1-4 ──> Predict out-of-fold on Fold 5 ──> Collect S_n4+1 ... S_22467
│
└── Pooled Out-of-Fold Residuals: Pool all 22,467 strictly out-of-fold scores to compute q_hat
```

### 4.1 Advantages and Trade-Offs of Grouped CV+
- **Data Efficiency:** Utilizes 100% of benchmark records for both model building and calibration.
- **Coverage Bound:** Under exchangeability of folds, Grouped CV+ guarantees coverage of at least $1 - 2\alpha$ theoretically, and empirically achieves approximately $1 - \alpha$.
- **Recommendation:** Standard 3-way Split Conformal ($D_{\text{train}}, D_{\text{cal}}, D_{\text{test}}$) is recommended as the primary architecture for Phase 7 validation due to its clean separation and exact $1-\alpha$ finite-sample coverage proof. Grouped CV+ is specified as a secondary benchmark.

---

## 5. Architectural Implementation Contract

When approved for implementation in Phase 7.1, the calibration engine must implement the following class contract:

```python
class GroupConformalCalibrator:
    """
    Scientifically validated conformal calibration engine for OPV-PredictNet.
    Guarantees zero data leakage and group-aware exchangeability.
    """
    def __init__(self, base_model, epsilon=0.05, pce_min=0.0, pce_max=33.0):
        self.base_model = base_model
        self.epsilon = epsilon
        self.pce_min = pce_min
        self.pce_max = pce_max
        self.quantiles = {} # {alpha: q_hat}
        
    def fit_calibration(self, X_cal, y_true_cal, groups_cal):
        """
        Computes non-conformity scores on strictly held-out calibration partition.
        Asserts zero overlap with training groups.
        """
        pass
        
    def predict_intervals(self, X_query, alpha=0.10):
        """
        Returns point prediction, lower prediction bound, upper prediction bound,
        and interval width for nominal (1 - alpha) coverage.
        """
        pass
```

### 5.1 Programmatic Unit Checks:
1. **Coverage Monotonicity Check:** For any query $X$, assert $C_{80\%}(X) \subseteq C_{90\%}(X) \subseteq C_{95\%}(X)$.
2. **Width Positivity Check:** Assert $W_{\alpha}(X) > 0$ for all valid descriptor vectors.
3. **Physical Admissibility Check:** Assert $C_{\text{lower}}(X) \ge 0.0$ and $C_{\text{upper}}(X) \le 33.0$.
4. **Group Leakage Assertion:** Raise `AssertionError` if any material pair in query set appears in calibration set during out-of-domain validation.
