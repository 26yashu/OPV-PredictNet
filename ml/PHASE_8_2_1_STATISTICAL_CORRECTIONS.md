# OPV-PredictNet — Phase 8.2.1: Statistical Significance and Cluster Dependence Corrections

**Date:** 2026-11-25  
**Scope:** Forensic audit of all 12 paired model comparisons in `phase_8_2_paired_comparisons.csv`, resolution of the Strategy B `M3 vs M0` p-value vs. confidence interval paradox, multi-margin one-way cluster-bootstrap sensitivity analyses, and inferential downgrades.  
**Constraint Compliance:** Original `phase_8_2_paired_comparisons.csv` preserved untouched. Zero retraining, zero code changes, zero Git commits.

---

## 1. Executive Summary of Statistical Findings

1. **Record-Level Paired Tests (`ttest_rel`, `wilcoxon`) Are Invalid Under Scaffold Clustering:**
   All 12 record-level paired tests reported in [phase_8_2_paired_comparisons.csv](file:///c:/Users/DELL/OPV-PredictNet/ml/results/phase_8_2_paired_comparisons.csv) assume independent, identically distributed (i.i.d.) error differences across rows. Because test partitions contain thousands of devices sharing identical scaffolds, pairs, and publications, these tests treat correlated pseudo-replicates as independent data, artificially deflating p-values by dozens of orders of magnitude.
2. **Resolution of the Strategy B `M3 vs M0` Paradox ($p = 4.29 \times 10^{-47}$ vs. 95% CI `[-0.3781, +0.0386]`):**
   - A single fullerene acceptor scaffold (`PC61BM`) constitutes **2,323 of 3,797 (61.18%)** test observations in Strategy B, where M3 improves on M0 by `-0.4088 pp`.
   - Under `ttest_rel` ($df = 3,796$), those 2,323 correlated observations are treated as independent draws, yielding $t = -14.6134$ ($p = 4.29 \times 10^{-47}$).
   - Under acceptor-scaffold cluster bootstrapping ($G = 163$ clusters, $B = 1,000$), roughly $e^{-1} \approx 36.8\%$ of bootstrap draws exclude the `PC61BM` cluster altogether; in the remaining 162 clusters, several major hubs exhibit positive $\Delta\text{AE}$ (M3 worse than M0 by `+0.1837 pp` and `+0.3136 pp`), driving the 97.5th bootstrap percentile above zero (`+0.0386`).
   - **Verdict:** The record-level p-value is an artifact of pseudo-replication and **must not be used as proof of significance**. The Strategy B M3 vs. M0 difference is **statistically inconclusive** under acceptor-scaffold clustering.
3. **Strategy C1 `scaffold_pair` Clustering Masks Shared Single-Scaffold Dependence:**
   In Strategy C1 ($N=284$), `M2A vs M0` and `M3 vs M0` yielded 95% CIs strictly below zero when clustered by `scaffold_pair` ($G=92$). However, when clustered by `donor_scaffold` ($G=73$) or `acceptor_scaffold` ($G=17$), **both CIs cross zero**. Therefore, Strategy C1 comparisons are **descriptive and statistically inconclusive**.
4. **No Joint Crossed-Cluster Inferential Method Implemented:**
   Because donors, acceptors, and DOIs form an unbalanced, non-nested crossed network, all valid cluster bootstraps here are **separate one-way sensitivity analyses**. Even where all five margins lie below zero (Strategy A `M1`, `M3`; Strategy C2 `M1`, `M2A`, `M3`), conclusions represent **exploratory descriptive evidence on the tested sample** rather than universal parametric proof.

---

## 2. Audit of `cluster_bootstrap_diff` & Parametric Assumptions

Inspection of `cluster_bootstrap_diff` (lines 684–708 of [phase_8_2_scaffold_benchmark.py](file:///c:/Users/DELL/OPV-PredictNet/ml/experiments/phase_8_2_scaffold_benchmark.py)) verified:
- **Test Statistic & Estimand:** Record-weighted mean paired absolute error difference $\bar{\Delta} = \frac{1}{N} \sum_{i=1}^N \left(|y_i - \hat{y}_{M,i}| - |y_i - \hat{y}_{M0,i}|\right)$.
- **Sampling Unit & Repetitions:** Resamples $G$ unique clusters from the strategy's designated `cluster_col` with replacement ($B = 1,000$ repetitions, fixed `seed = 8888`), concatenating all observations belonging to each drawn cluster.
- **Confidence Interval Formula:** Non-parametric percentile interval `[2.5%, 97.5%]` (`np.percentile`).
- **Parametric Tests Audited:** `scipy.stats.ttest_rel(ae_m, ae_m0)` and `scipy.stats.wilcoxon(ae_m, ae_m0)` (default `zero_method='wilcox'`). Both ignore intra-cluster correlation ($\rho > 0$) and cross-cluster dependence ($\text{donor} \times \text{acceptor} \times \text{DOI}$).

---

## 3. Forensic Breakdown: Strategy B `M3 vs M0`

### 3.1 Extreme Cluster Imbalance in Strategy B Test Set ($N = 3,797$, $G = 163$)

Because Strategy B assigned only the rank-1 acceptor scaffold (`Y6`, 7,902 records) to $D_{\text{train}}$, the rank-4 acceptor scaffold (`PC61BM`, 2,323 records) landed in $D_{\text{test}}$ alongside 162 much smaller acceptor scaffolds:

| Cluster Size Statistic / Top 5 Test Acceptor Scaffolds | Cluster Count $n_g$ | Share of Test Set | Cluster Mean $\Delta\text{AE}$ (M3 − M0) | Impact on Bootstrap vs. `ttest_rel` |
| :--- | :---: | :---: | :---: | :--- |
| **#1: `PC61BM` (C60 fullerene derivative)** | **2,323** | **61.18%** | **-0.4088 pp** | Dominates i.i.d. $N=3,797$ sum; omitted in ~36.8% of bootstrap draws |
| **#2: ITIC-like indacenodithiophene variant** | 303 | 7.98% | -0.0132 pp | Near zero difference |
| **#3: Indacenodithiophene-based NFA** | 151 | 3.98% | **+0.1837 pp** | M3 performs *worse* than M0 |
| **#4: Thiophene-bridged rhodanine NFA** | 132 | 3.48% | **+0.3136 pp** | M3 performs *worse* than M0 |
| **#5: Unsubstituted C60 fullerene** | 117 | 3.08% | -0.4896 pp | M3 improves over M0 |
| **Remaining 158 Acceptor Scaffolds** | 771 ($\text{med}=2$) | 20.31% | Mixed | Highly dispersed |

### 3.2 Mathematical Incompatibility Explained

- **Why `ttest_rel` yields $p = 4.29 \times 10^{-47}$:** `ttest_rel` assumes $\text{SE}(\bar{\Delta}) = s_\Delta / \sqrt{3797}$, implicitly pretending 2,323 `PC61BM` devices provide 2,323 independent chemical experiments. Any small systematic difference on `PC61BM` drives $|t| > 14$.
- **Why Acceptor-Scaffold Bootstrap CI is `[-0.3781, +0.0386]`:** Standard cluster-bootstrap asymptotics require no single cluster to dominate the sample ($\max_g n_g / N \to 0$). With one cluster holding $61.18\%$ of all rows, bootstrap draws fluctuate violently depending on whether `PC61BM` is drawn $0, 1, 2,$ or $3$ times. In draws omitting `PC61BM`, clusters #3 (`+0.1837`) and #4 (`+0.3136`) push the sample mean positive, producing a non-significant 95% CI crossing zero.
- **Estimand Sensitivity:** Moreover, the record-weighted average (`-0.2780 pp`) primarily measures how models perform on `PC61BM`, NOT on a typical unseen acceptor scaffold. When resampling by `donor_scaffold`, `scaffold_pair`, `pair_id`, or `doi_norm` instead, `PC61BM` records are split across multiple clusters so their influence never drops out of a bootstrap draw, producing much tighter CIs (`[-0.3756, -0.1272]` by donor scaffold) that contradict the `acceptor_scaffold` CI.

---

## 4. Complete One-Way Cluster-Bootstrap Sensitivity Audit (All 12 Comparisons)

To expose dependence across all structural and bibliographic axes, all 12 paired comparisons were audited across **five separate one-way cluster margins** ($B = 1,000$, `seed = 8888`):

| Strategy ($N$, Test Basis) | Comparison | Mean $\Delta\text{AE}$ | Published `cluster_col` 95% CI | `donor_scaffold` 95% CI | `acceptor_scaffold` 95% CI | `scaffold_pair` 95% CI | `pair_id` 95% CI | `doi_norm` 95% CI | Corrected Statistical Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Strategy A** | **M1 vs M0** | -0.1386 | [-0.2849, -0.0152] ($G=216$) | [-0.2849, -0.0152] ($216$) | [-0.2067, -0.0392] ($71$) | [-0.2727, -0.0133] ($384$) | [-0.2697, -0.0169] ($623$) | [-0.2493, -0.0362] ($513$) | **Consistent Directional Signal** (all 5 one-way CIs < 0; exploratory sensitivity only) |
| ($1,481$) | **M2A vs M0** | -0.0833 | [-0.2387, +0.0456] ($G=216$) | [-0.2387, +0.0456] ($216$) | [-0.1515, +0.0106] ($71$) | [-0.2288, +0.0462] ($384$) | [-0.2190, +0.0446] ($623$) | [-0.1917, +0.0163] ($513$) | **Not Significant** (all 5 one-way CIs cross 0) |
| | **M3 vs M0** | -0.2278 | [-0.3915, -0.0853] ($G=216$) | [-0.3915, -0.0853] ($216$) | [-0.3671, -0.0979] ($71$) | [-0.3822, -0.0875] ($384$) | [-0.3719, -0.0839] ($623$) | [-0.3606, -0.1010] ($513$) | **Consistent Directional Signal** (all 5 one-way CIs < 0; exploratory sensitivity only) |
| **Strategy B** | **M1 vs M0** | -0.0423 | [-0.1191, +0.0443] ($G=163$) | [-0.1579, -0.0002] ($323$) | [-0.1191, +0.0443] ($163$) | [-0.1462, +0.0096] ($603$) | [-0.1192, +0.0329] ($887$) | [-0.1012, +0.0249] ($1163$) | **Not Significant** (4/5 CIs cross 0) |
| ($3,797$) | **M2A vs M0** | -0.1085 | [-0.1471, +0.0168] ($G=163$) | [-0.1786, -0.0719] ($323$) | [-0.1471, +0.0168] ($163$) | [-0.1667, -0.0687] ($603$) | [-0.1676, -0.0336] ($887$) | [-0.1685, -0.0463] ($1163$) | **Inconclusive / Descriptive** (`acceptor_scaffold` CI crosses 0; PC61BM dominates) |
| | **M3 vs M0** | -0.2780 | [-0.3781, +0.0386] ($G=163$) | [-0.3756, -0.1272] ($323$) | [-0.3781, +0.0386] ($163$) | [-0.3698, -0.1190] ($603$) | [-0.3974, -0.1209] ($887$) | [-0.3607, -0.1981] ($1163$) | **Inconclusive / Descriptive** (`acceptor_scaffold` CI crosses 0; t-test $p=4.29\times 10^{-47}$ invalid) |
| **Strategy C1** | **M1 vs M0** | -0.1741 | [-0.4059, +0.0485] ($G=92$) | [-0.4079, +0.0497] ($73$) | [-0.3065, +0.2111] ($17$) | [-0.4059, +0.0485] ($92$) | [-0.4037, +0.0339] ($114$) | [-0.4037, +0.0507] ($116$) | **Not Significant** (all 5 one-way CIs cross 0) |
| ($284$) | **M2A vs M0** | -0.2426 | [-0.4912, -0.0180] ($G=92$) | [-0.5137, +0.0065] ($73$) | [-0.6568, +0.1003] ($17$) | [-0.4912, -0.0180] ($92$) | [-0.4684, -0.0368] ($114$) | [-0.4583, -0.0434] ($116$) | **Downgraded to Descriptive** (`donor_scaffold` & `acceptor_scaffold` CIs cross 0) |
| | **M3 vs M0** | -0.2727 | [-0.5552, -0.0231] ($G=92$) | [-0.6274, +0.0031] ($73$) | [-0.5846, +0.4291] ($17$) | [-0.5552, -0.0231] ($92$) | [-0.5378, -0.0453] ($114$) | [-0.5466, -0.0330] ($116$) | **Downgraded to Descriptive** (`donor_scaffold` & `acceptor_scaffold` CIs cross 0) |
| **Strategy C2** | **M1 vs M0** | -0.1769 | [-0.2684, -0.0858] ($G=565$) | [-0.2582, -0.0975] ($310$) | [-0.2930, -0.0716] ($231$) | [-0.2684, -0.0858] ($565$) | [-0.2570, -0.1061] ($921$) | [-0.2481, -0.1111] ($864$) | **Consistent Directional Signal** (all 5 one-way CIs < 0; exploratory sensitivity only) |
| ($2,067$) | **M2A vs M0** | -0.1727 | [-0.2643, -0.0858] ($G=565$) | [-0.2581, -0.0975] ($310$) | [-0.2807, -0.0678] ($231$) | [-0.2643, -0.0858] ($565$) | [-0.2429, -0.1030] ($921$) | [-0.2465, -0.1036] ($864$) | **Consistent Directional Signal** (all 5 one-way CIs < 0; exploratory sensitivity only) |
| | **M3 vs M0** | -0.1626 | [-0.2818, -0.0397] ($G=565$) | [-0.2588, -0.0426] ($310$) | [-0.2904, -0.0173] ($231$) | [-0.2818, -0.0397] ($565$) | [-0.2640, -0.0580] ($921$) | [-0.2530, -0.0805] ($864$) | **Consistent Directional Signal** (all 5 one-way CIs < 0; exploratory sensitivity only) |

---

## 5. Mandatory Methodological Downgrades

1. **Retire Record-Level `ttest_rel` and `wilcoxon` from Inference Claims:** Those p-values in `phase_8_2_paired_comparisons.csv` are mathematically valid implementations of i.i.d. tests on the numeric vectors, but they are **statistically invalid** for claiming significance across structurally clustered OPV devices.
2. **Downgrade Strategy B & Strategy C1:** Because cluster-level resampling along the held-out structural margins (`acceptor_scaffold` in B; `donor_scaffold` and `acceptor_scaffold` in C1) produces 95% confidence intervals crossing zero, **no statistically significant superiority over M0 is established** in Strategy B or Strategy C1.
3. **Retain Strategy A (`M1`, `M3`) and Strategy C2 (`M1`, `M2A`, `M3`) as Exploratory Directional Evidence:** All five separate one-way bootstrap margins exclude zero, providing coherent descriptive support that processing features reduce error on held-out donor scaffolds and novel scaffold pairings—subject to the limitation that joint crossed-cluster variance estimation remains unimplemented.
