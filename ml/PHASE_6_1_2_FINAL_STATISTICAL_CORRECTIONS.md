# OPV-PredictNet — Phase 6.1.2: Final Statistical Reporting Corrections

**Author / Auditor:** Antigravity Agent  
**Date:** November 24, 2026  
**Status:** Completed & Independently Verified  
**Dataset:** OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records, 4,730 material pair clusters)  
**Scope:** Resolution of outstanding statistical and metadata reporting issues: bootstrap hypothesis testing correction, observational framing of device architecture performance, and complete empirical audit of additive metadata.

---

## 1. Executive Summary

Phase 6.1.2 resolves three outstanding reporting issues identified in Phase 6.1.1:
1. **Bootstrap Hypothesis Testing Clarification:**
   - The reported value $p = 0.7220$ derived from $2 \times \min(P(\bar{\Delta}^* \le 0), P(\bar{\Delta}^* \ge 0))$ was an uncalibrated, descriptive sign-tail proportion across bootstrap resamples, **not** a formal null-centered hypothesis-test p-value.
   - We implement and document a formal **null-centered cluster-bootstrap test** ($p = 0.5440$), while reaffirming that primary inferential reliance rests on the **material-pair clustered percentile 95% confidence interval** ([-0.2427%, +0.0334%]).
   - **Conclusion:** Both the confidence interval and the null-centered test confirm that the nested-selected model achieves **no statistically significant improvement** over the Phase 5 baseline, and the point estimate favors the Phase 5 baseline ($\bar{\Delta} = -0.0637\%$).
2. **Observational Framing of Device Architecture Performance:**
   - Subgroup metrics across Conventional, Inverted, Unspecified, and Ambiguous architectures are strictly **observational** and do not establish causal physical mechanisms.
   - We report MAE, RMSE, and target variance across all subgroups. We explicitly **refrain from claiming "equivalent performance"** based solely on MAE parity (2.01% vs 2.16%), because Inverted devices exhibit higher RMSE (3.0911% vs 2.8648%), higher relative error (25.4% vs 20.1%), and substantially lower $R^2$ (0.5408 vs 0.7768).
   - Speculations regarding interfacial layers (e.g. ZnO vs PEDOT:PSS) and blend selection are explicitly designated as unverified post-hoc hypotheses. The 9 ambiguous records remain preserved as a distinct category.
3. **Complete Additive Metadata Audit:**
   - An exhaustive inspection of the 347 records previously categorized as "Explicitly Reported Absent" revealed:
     - **3 records** with positive non-zero additive concentrations (e.g. `0.5% v/v`, `1 vol%`, `0.5%`), which were misclassified and are now corrected to **Explicitly Reported Present**.
     - **34 records** with `additive = '-'` (and null ratio), which represent database placeholder dashes, now corrected to **Status Unknown (Unrecorded)**.
     - **2 records** with ambiguous multi-condition text or missing chemical names with bare ratios, corrected to **Status Unknown (Unrecorded)**.
     - **60 records** with a bare digit `'0'` in `additive` (and null ratio). Because a bare `'0'` without chemical identity or ratio represents an inferred missing value rather than explicit confirmation of absence, these are properly classified as **Unknown / Unrecorded**.
     - **248 records (1.10%)** possess genuine, unambiguous source-level confirmation of absence (`'w/o'`, `'without'`, `'no'`, or explicit `'0% [Chemical]'`).

---

## 2. Bootstrap Statistical Reporting Correction

### 2.1 Methodological Critique of the Previous p-Value Calculation
In Phase 6.1 and 6.1.1, the value $p = 0.7220$ was calculated as:
$$p_{\text{sign-tail}} = 2 \times \min\left( \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\bar{\Delta}^*_b \le 0), \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\bar{\Delta}^*_b \ge 0) \right)$$
Where $\bar{\Delta}^*_b$ was obtained by resampling material-pair clusters with replacement from the observed dataset without modification. In this run, 361 out of 1,000 resamples (36.10%) yielded $\bar{\Delta}^*_b \ge 0$, and 639 yielded $\bar{\Delta}^*_b \le 0$, producing $2 \times 0.3610 = 0.7220$.

#### Why This Is Not a Calibrated Hypothesis-Test p-Value:
1. **Resampling Around the Point Estimate:** Ordinary non-parametric bootstrap resampling from the observed sample approximates the distribution of the estimator centered around the sample mean $\bar{\Delta} = -0.0637\%$, **not** under the null hypothesis $H_0: \mathbb{E}[\Delta] = 0$.
2. **Lack of Null Imposition:** A valid hypothesis test requires calculating the probability of observing a test statistic as extreme as or more extreme than the observed value *under the null hypothesis*. Counting how often uncentered bootstrap replications cross zero merely assesses sign stability around the sample estimate.
3. **Correct Label:** This calculation is properly described as a **descriptive bootstrap sign-tail proportion**, not a calibrated hypothesis-test p-value.

### 2.2 Formal Null-Centered Cluster-Bootstrap Procedure
To evaluate the null hypothesis $H_0: \mathbb{E}[\Delta] = 0$ formally using a bootstrap test, we implement a **null-centered cluster-bootstrap test**:

#### Procedure and Assumptions:
1. **Test Statistic:** The observed difference in sample mean absolute error:
   $$T_{\text{obs}} = \bar{\Delta} = \frac{1}{N} \sum_{i=1}^{N} (|y_i - \hat{y}_{\text{baseline}, i}| - |y_i - \hat{y}_{\text{nested}, i}|) = -0.063728\%$$
2. **Resampling Unit:** The material pair cluster ($G = 4,730$ unique donor–acceptor pairs). This accounts for intra-cluster correlation among records sharing the same chemical system.
3. **Null Imposition:** Under $H_0$, the true mean difference is zero. To generate a null distribution, each bootstrap sample mean $\bar{\Delta}^*_b$ is centered by subtracting the observed sample mean $\bar{\Delta}$:
   $$T^*_{0, b} = \bar{\Delta}^*_b - \bar{\Delta}$$
   By construction, $\mathbb{E}^*[T^*_{0}] = 0$, conforming exactly to the null hypothesis.
4. **Two-Sided p-Value Calculation:**
   $$p_{\text{null-centered}} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(|T^*_{0, b}| \ge |T_{\text{obs}}|) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(|\bar{\Delta}^*_b - \bar{\Delta}| \ge |\bar{\Delta}|)$$
5. **Empirical Result ($B = 1,000$ cluster resamples, seed = 42):**
   - Out of 1,000 null-centered replicates, exactly **544 replicates** produced an absolute deviation from zero greater than or equal to $|T_{\text{obs}}| = 0.0637\%$.
   - Formal Null-Centered p-value:
     $$p_{\text{null-centered}} = \frac{544}{1000} = \mathbf{0.5440}$$

### 2.3 Primary Inferential Reliance: Clustered Percentile Confidence Interval
While hypothesis tests depend on specific null-imposition mechanics, estimation via confidence intervals directly quantifies parameter uncertainty without relying on a sharp null distribution:
- **Clustered Percentile 95% CI ($B = 1,000$):** **[-0.2427%, +0.0334%]**
- **Observed Mean Difference:** **-0.0637%** (favors Phase 5 baseline)
- **Observed Median Difference:** **+0.0021%**

#### Limitations and Definitive Verdict:
- The 95% confidence interval spans zero, and both the formal null-centered test ($p = 0.5440$) and descriptive sign-tail proportion ($0.7220$) are substantially greater than $\alpha = 0.05$.
- Limitations include: reliance on $B=1,000$ resamples, assumption of exchangeability across clusters, and sensitivity to cluster size variability.
- **Statistical Verdict:** The nested model has **not demonstrated a statistically significant or reliable improvement** over the Phase 5 Baseline Random Forest. The point estimate indicates that the Phase 5 baseline achieves lower mean absolute error by 0.0637% PCE.

---

## 3. Observational Framing of Device Architecture Performance

### 3.1 Methodological Principle: Observational Data Limitation
All device architecture subgroup analyses are strictly **retrospective and observational**:
- OPV-DB compiles literature from diverse research groups, laboratories, publication years, and materials.
- Subgroup performance differences **do not establish causal explanations** or physical device mechanics.
- Prior references to physical mechanisms (such as interfacial contact properties of ZnO vs PEDOT:PSS, or material blend selections) must be understood solely as **unverified, post-hoc observational hypotheses**, not proven causal drivers.

### 3.2 Comprehensive Subgroup Performance Metrics

| Architecture Category | Sample Count | % of Benchmark | Actual PCE Mean (%) | Actual PCE SD (%) | Target Variance ($\%^2$) | Baseline MAE (%) | Baseline RMSE (%) | Baseline $R^2$ | Nested MAE (%) | Nested RMSE (%) | Nested $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional Architecture** | 9,284 | 41.32% | 9.97 | 6.06 | 36.77 | 2.0069 | 2.8648 | 0.7768 | 2.0929 | 2.8763 | 0.7750 |
| **Architecture Unspecified** | 8,110 | 36.10% | 10.69 | 5.99 | 35.85 | 2.0686 | 2.9697 | 0.7540 | 2.1412 | 2.9720 | 0.7536 |
| **Inverted Architecture** | 5,064 | 22.54% | 8.48 | 4.56 | 20.81 | 2.1569 | 3.0911 | 0.5408 | 2.1652 | 3.0492 | 0.5532 |
| **Ambiguous / Dual Structure** | 9 | 0.04% | 10.81 | 5.19 | 26.89 | 1.4759 | 1.7027 | 0.8922 | 1.7364 | 2.1317 | 0.8310 |

### 3.3 Rejection of "Equivalent Performance" Based Solely on MAE
In earlier drafts, it was suggested that because baseline MAE on Conventional devices (2.0069%) was close to Inverted devices (2.1569%), the model exhibited "equivalent performance."

**This claim is rejected:**
1. **RMSE Divergence:** Despite Inverted devices having a narrower target range ($\text{SD} = 4.56\%$ vs $6.06\%$), their baseline RMSE is **higher** (**3.0911%** vs **2.8648%**). The model produces larger extreme errors on Inverted devices.
2. **Relative Error Disparity:** The mean relative absolute error ($\text{MAE} / \bar{y}$) is **25.4%** for Inverted devices versus **20.1%** for Conventional devices, reflecting poorer relative precision.
3. **Variance and $R^2$ Interaction:** The target variance of Inverted devices ($20.81\%^2$) is 43.4% lower than Conventional devices ($36.77\%^2$). In $R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)}$, lower target variance mathematically depresses $R^2$, but higher MSE ($\text{RMSE}^2 = 9.55$ vs $8.21$) simultaneously drives $R^2$ down.
4. **Summary:** Performance across architectures differs across multiple metrics (RMSE, relative error, $R^2$). These differences reflect observational dataset heterogeneity, and equivalent performance cannot be claimed based on MAE alone.

### 3.4 Preservation of the Nine Ambiguous Records
The 9 records exhibiting conflicting classifications or dual-structure testing (IDs 1321, 1701, 52171–52174, 107192, 109729, 109732) are preserved as an independent category (`Ambiguous / Dual Structure Reported`). They have a mean PCE of 10.81% ($\text{Var} = 26.89\%^2$), baseline MAE of 1.4759%, and baseline RMSE of 1.7027%.

---

## 4. Complete Additive Metadata Audit

### 4.1 Source Fields and Preliminary Classification Rule
The dataset possesses three additive-related columns:
- `additive`: Free-text field intended for additive chemical identity.
- `additive_canonical`: Standardized chemical name.
- `additive_ratio`: Free-text field intended for concentration/volume ratio.

In Phase 6.1, records were categorized as "Explicitly Reported Absent" (yielding 347 records) if `additive` contained strings such as `w/o`, `0`, `no`, `without`, `none`, `-`, or if `additive_ratio` was in `['0', '0%', '0.0']`.

### 4.2 Systematic Audit of the 347 "Absent" Records
An exhaustive item-by-item audit of the 347 records revealed four distinct failure modes in the preliminary rule:

#### 1. Contradictory Positive Additive Concentrations (3 records misclassified as absent):
- **ID 112502** (DOI 10.1039/c8tc06407e): `additive = '-'`, `additive_ratio = '0.5%'`. An additive was used at 0.5% concentration, but the chemical name was entered as a dash.
- **ID 67876** (DOI 10.1039/d3ee04281b): `additive = 'No'`, `additive_ratio = '0.5% v/v'`. The additive ratio confirms 0.5% v/v additive was present.
- **ID 111182** (DOI 10.1039/c4tc02092h): `additive = 'No'`, `additive_ratio = '1 vol%'`. The additive ratio confirms 1 vol% additive was present.
- **Correction:** These 3 records must be classified as **Explicitly Reported Present**.

#### 2. Database Dash Placeholders (34 records):
- `additive = '-'`, `additive_ratio = NaN` (34 records).
- A hyphen/dash `'-'` in curated tables represents an empty or unrecorded entry placeholder, not an explicit scientific confirmation that no additive was used.
- **Correction:** Reclassified as **Status Unknown (Unrecorded)**.

#### 3. Ambiguous Text and Missing Chemical Names (2 records):
- **ID 41017** (DOI 10.1002/adfm.202204871): `additive = '0% CN, 2.5% CN'`, `additive_ratio = '0%'`. Ambiguous entry listing both 0% and 2.5% CN.
- **ID 95951** (DOI 10.1021/acs.macromol.1c00346): `additive = NaN`, `additive_ratio = '0'`. The chemical name is missing, leaving an isolated digit `'0'`.
- **Correction:** Reclassified as **Status Unknown (Unrecorded)**.

#### 4. Bare Digit '0' with Missing Chemical Name and Ratio (60 records):
- `additive = '0'`, `additive_ratio = NaN` (60 records across 28 DOIs).
- In these records, no chemical identity, unit, or ratio is provided. Inferring that a bare digit `'0'` represents an explicit confirmation of additive absence is an unsupported deduction.
- **Correction:** In accordance with the mandate that absence must not be inferred from missing names or ratios, these 60 records are classified as **Status Unknown (Unrecorded)** (or explicitly tagged as ambiguous bare-digit entries).

#### 5. Genuine, Unambiguous Explicit Absence (248 records, 1.10%):
- **Natural Language Negative Confirmations (202 records):** Explicit text confirming absence: `'w/o'` (115 records), `'no'` / `'No'` / `'NO'` (50 records with null or 0 vol% ratio), `'without'` / `'Without'` (23 records), `'W/O'` / `'W/o'` (14 records).
- **Explicit 0% Chemical Controls (46 records):** Explicitly documented 0% control conditions for specific additives: `'DIO 0%'` (13), `'0% DIO'` (9), `'0% CN'` (5), `'0%'` (6), `'DIO (0 v%)'` (3), `'0%DIO'` (2), `'PAE 0%'` (2), `'0% AA'` (1), `'0% DBE'` (1), `'0%CN'` (1), `'PLA 0%'` (1), `'SA9000 0%'` (1), `'DIO'` with ratio `'0%'` (1).

### 4.3 Final Audited Additive Categories & Performance

| Additive Category | Audit Definition | Sample Count | % of Benchmark | Actual Mean PCE (%) | Target Variance ($\%^2$) | Baseline MAE (%) | Baseline RMSE (%) | Baseline $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Explicitly Reported Present** | Confirmed additive chemical or positive concentration | 7,601 | 33.83% | 10.49 ($\sigma = 5.59$) | 31.27 | 2.0220 | 2.7834 | 0.7522 |
| **Status Unknown (Unrecorded)** | Null values (`NaN`), dash placeholders (`'-'`), and bare digits (`'0'`) | 14,618 | 65.06% | 9.59 ($\sigma = 5.86$) | 34.29 | 2.0888 | 3.0466 | 0.7293 |
| **Verified Explicitly Absent** | Unambiguous natural language (`w/o`, `without`, `no`) or explicit `0% [Chemical]` | 248 | 1.10% | 9.58 ($\sigma = 6.36$) | 40.39 | 1.7817 | 2.5137 | 0.8436 |

*Sub-breakdown of Unknown:* The 14,618 unknown records comprise 14,524 raw nulls, 34 dash placeholders (`'-'`), 60 bare-digit `'0'` records, and 2 ambiguous entries.

---

## 5. Model Governance & Final Conclusions

1. **Reference Model:** The **Phase 5 Baseline Random Forest** is retained as the authoritative reference model for OPV-PredictNet.
2. **Nested Model Role:** The nested-selected model is retained strictly as an evaluated alternative demonstrating that hyperparameter regularization under grouped CV does not provide generalizable improvement.
3. **Uncertainty Quantification Policy:** No confidence intervals, prediction intervals, or conformal prediction claims are introduced in this phase. Formal UQ is deferred to Phase 7.
4. **Safety & Stability:** No models have been deployed to production. The 22 molecular descriptor schema is strictly preserved. No Git commits have been created.

---

## 6. Changed Files Inventory

- `ml/PHASE_6_1_2_FINAL_STATISTICAL_CORRECTIONS.md`: Created (this document).
- `ml/PHASE_6_1_1_FINAL_AUDIT.md`: Updated to harmonize bootstrap terminology and additive audit findings.
- `ml/PHASE_6_1_NESTED_VALIDATION_REPORT.md`: Updated where necessary to maintain consistency across the reporting chain.
- Historical experiment scripts and result CSV/JSON files: **Preserved unchanged**.
