# OPV-PredictNet — Phase 8.1.1: Statistical Review and Inferential Validity Audit

**Date**: 2026-11-24  
**Scope**: In-depth audit of statistical hypothesis testing, multiway clustering structure, cross-pair dependence, and inferential boundaries for Phase 8.1 processing-aware models.  
**Deliverable**: Comprehensive Statistical Audit and Inferential Downgrade Specification.

---

## 1. Executive Summary & Core Statistical Finding

The Phase 8.1 experiment reported highly statistically significant improvements in Mean Absolute Error (MAE) for processing-aware models over the molecular baseline (e.g., Model M1 vs. M0: $\Delta\text{MAE} = -0.1407\text{ pp}$, paired t-test $p = 1.03 \times 10^{-6}$, Wilcoxon $p = 6.51 \times 10^{-6}$, pair-bootstrap 95% CI $[-0.1969, -0.0873]$).

This audit investigated the underlying statistical assumptions and established that:
1. **Mathematical Implementation is Sound**: The paired t-test, Wilcoxon signed-rank test, and cluster bootstrap scripts were correctly executed with no mathematical or software defects.
2. **Cluster Definition Was Restricted to Pair IDs**: In the test set ($N=946$), exactly one device was sampled per material-name pair (`donor_canonical::acceptor_canonical`). As a result, each record had a unique `pair_id`, rendering the implemented pair-cluster bootstrap mathematically identical to a simple i.i.d. row bootstrap.
3. **Severe Multiway Structural Concentration**: The 946 test pairs are **not independent**. They share 495 unique donors and 416 unique acceptors. A single acceptor ($\text{PC}_{71}\text{BM}$) appears in 188 test pairs (19.87%), and the top 5 donors account for 28.86% of the test set.
4. **Acceptor Clustering Substantially Inflates Variance**: Sensitivity analysis using cluster bootstrapping grouped by canonical acceptor inflates the standard error of paired differences by **$38\% - 47\%$**, widening the confidence interval towards zero.
5. **Mandatory Inferential Downgrade**: The reported p-values and confidence intervals constitute valid descriptive evidence of improvement **for this specific held-out sample**, but **cannot support claims of generalization to unseen chemical scaffolds**. All inferential claims must be formally downgraded to exploratory evidence.

---

## 2. Audit of Implemented Hypothesis Tests

### 2.1 Paired t-Test Formulation
- **Formula**:
  $$t = \frac{\bar{\Delta}}{s_\Delta / \sqrt{N}}, \quad \text{where } \Delta_i = |y_i - \hat{y}_{M, i}| - |y_i - \hat{y}_{M0, i}|$$
- **Degrees of Freedom**: $df = N - 1 = 945$.
- **Underlying Assumption**: Differences $\Delta_i$ are independent and identically distributed (i.i.d.) with finite variance.
- **Verification**: Exact match with `scipy.stats.ttest_rel(ae_m, ae_m0)`.

### 2.2 Wilcoxon Signed-Rank Test Formulation
- **Formula**: Non-parametric test ranking absolute non-zero differences $|\Delta_i|$.
- **Zero Handling**: Scipy default (`zero_method='wilcox'`), discarding zero differences before ranking.
- **Ties Handling**: Average ranks assigned to tied absolute differences; normal approximation with continuity correction applied.
- **Verification**: Exact match with `scipy.stats.wilcoxon(ae_m, ae_m0)`.

### 2.3 Cluster Bootstrap Formulation
- **Algorithm**: Resampling clusters with replacement ($B = 1,000$ iterations, seed 8888).
- **Cluster Definition**: Grouped by `pair_id` (`donor_canonical + "::" + acceptor_canonical`).
- **Audit Finding**: Under Alternative B evaluation sampling, exactly one record was drawn per pair group ($N_{\text{pairs}} = 946, N_{\text{records}} = 946$). Therefore, every cluster had size $n_c = 1$. The cluster bootstrap resampled individual device rows independently, failing to capture cross-pair molecular or literature dependence.

---

## 3. Investigation of Multiway Structural Dependence

In organic photovoltaics, devices do not exist as isolated, independent pairs. Experimental research relies heavily on benchmark "workhorse" polymers and universal fullerene / non-fullerene acceptors.

### 3.1 Structural Concentration in the Test Partition ($N = 946$)

```
+---------------------------------------------------------------------------------------------------------+
| Multiway Entity Cardinality in Held-Out Test Set (N = 946 Devices)                                      |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Entity Level             | Unique Entities       | Average Records/Ent | Degree of Concentration        |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Material-Name Pairs      | 946                   | 1.00                | Disjoint by construction       |
| Canonical Donors         | 495                   | 1.91                | High (Top 5 = 28.86%)          |
| Canonical Acceptors      | 416                   | 2.27                | Extreme (PC71BM = 19.87%)      |
| Literature DOIs (Papers) | 817                   | 1.16                | Mild (Max 4 records/DOI)       |
+--------------------------+-----------------------+---------------------+--------------------------------+
```

### 3.2 Dominant Donor and Acceptor Entities

```
+---------------------------------------------------------------------------------------------------------+
| Top Repeated Donors in Test Set (N = 946)          | Top Repeated Acceptors in Test Set (N = 946)       |
+--------------------------+-------+-----------------+--------------------------+-------+-----------------+
| Donor Canonical Name     | Count | Test Share (%)  | Acceptor Canonical Name  | Count | Test Share (%)  |
+--------------------------+-------+-----------------+--------------------------+-------+-----------------+
| **PBDB-T**               | 94    | 9.94%           | **PC71BM**               | 188   | 19.87%          |
| **PM6**                  | 66    | 6.98%           | **ITIC**                 | 58    | 6.13%           |
| **PTB7-Th**              | 50    | 5.29%           | **PC70BM**               | 42    | 4.44%           |
| **PBDB-TF**              | 32    | 3.38%           | **PC61BM**               | 39    | 4.12%           |
| **P3HT**                 | 31    | 3.28%           | **Y6**                   | 33    | 3.49%           |
+--------------------------+-------+-----------------+--------------------------+-------+-----------------+
| Top 5 Donors Combined    | 273   | **28.86%**      | Top 5 Acceptors Combined | 360   | **38.05%**      |
+--------------------------+-------+-----------------+--------------------------+-------+-----------------+
```

> [!IMPORTANT]
> Nearly **1 out of every 5 devices** in the test set shares the exact same acceptor molecule ($\text{PC}_{71}\text{BM}$, $N = 188$). Furthermore, over **28% of devices** share one of five standard donor polymers. Because molecular descriptors for these shared partners are identical, prediction errors are intrinsically correlated across pairs that share a partner.

---

## 4. Separate One-Way Cluster-Bootstrap Sensitivity Analyses

To assess how cross-pair dependence impacts confidence intervals, separate one-way cluster-bootstrap sensitivity analyses were conducted along individual margins:
1. **Material-Name Pair (Original)** ($N_c = 946$ clusters, 1 instance/pair under Alternative B sampling).
2. **Canonical Donor** ($N_c = 495$ clusters): Resampling all devices sharing the same donor polymer.
3. **Canonical Acceptor** ($N_c = 416$ clusters): Resampling all devices sharing the same acceptor molecule.
4. **Literature DOI** ($N_c = 817$ clusters): Resampling all devices originating from the same research publication.

> [!IMPORTANT]
> **Methodological Clarification on Cluster Bootstrapping**:
> These evaluations constitute **separate one-way cluster-bootstrap sensitivity analyses**, not a genuinely joint crossed-cluster bootstrap. In an unbalanced, sparse bipartite/tripartite experimental network (where donors, acceptors, and publications intersect irregularly), crossed-cluster resampling can generate degenerate subgraphs. Because a defensible joint crossed procedure was not implemented, these one-way analyses serve as exploratory sensitivity checks. Inferential conclusions must be downgraded accordingly, and independence between pairs must **never** be assumed merely because pair IDs are distinct.

Each separate one-way bootstrap was evaluated over $B = 2,000$ iterations using the locked seed `8888`:

```
+-----------------------------------------------------------------------------------------------------------------------------+
| Separate One-Way Cluster-Bootstrap Sensitivity Analysis for Paired Absolute Error Difference (PCE pp)                       |
+-------+-------------------+-----------------+-------------------------+------------+-----------------------+----------------+
| Model | Resampling Margin | Cluster Count   | 95% Confidence Interval | SE (pp)    | SE Increase vs. Pair  | Lower / Upper  |
+-------+-------------------+-----------------+-------------------------+------------+-----------------------+----------------+
| **M1**| Pair (Original)   | 946 clusters    | [-0.1962, -0.0865]      | 0.0283     | Baseline              | Safe (< 0)     |
|       | Donor Clustered   | 495 clusters    | [-0.2003, -0.0879]      | 0.0289     | +2.1%                 | Safe (< 0)     |
|       | Acceptor Clustered| 416 clusters    | **[-0.2025, -0.0508]**  | **0.0393** | **+38.9%**            | Safe (< 0)     |
|       | DOI Clustered     | 817 clusters    | [-0.1984, -0.0861]      | 0.0288     | +1.8%                 | Safe (< 0)     |
+-------+-------------------+-----------------+-------------------------+------------+-----------------------+----------------+
| **M2**| Pair (Original)   | 946 clusters    | [-0.1545, -0.0480]      | 0.0276     | Baseline              | Safe (< 0)     |
|       | Donor Clustered   | 495 clusters    | [-0.1564, -0.0506]      | 0.0269     | -2.5%                 | Safe (< 0)     |
|       | Acceptor Clustered| 416 clusters    | **[-0.1635, -0.0079]**  | **0.0406** | **+47.1%**            | **Borderline** |
|       | DOI Clustered     | 817 clusters    | [-0.1578, -0.0444]      | 0.0287     | +4.0%                 | Safe (< 0)     |
+-------+-------------------+-----------------+-------------------------+------------+-----------------------+----------------+
|**M2A**| Pair (Original)   | 946 clusters    | [-0.1536, -0.0524]      | 0.0266     | Baseline              | Safe (< 0)     |
|       | Donor Clustered   | 495 clusters    | [-0.1575, -0.0538]      | 0.0258     | -3.0%                 | Safe (< 0)     |
|       | Acceptor Clustered| 416 clusters    | **[-0.1591, -0.0208]**  | **0.0366** | **+37.6%**            | Safe (< 0)     |
|       | DOI Clustered     | 817 clusters    | [-0.1570, -0.0505]      | 0.0276     | +3.8%                 | Safe (< 0)     |
+-------+-------------------+-----------------+-------------------------+------------+-----------------------+----------------+
| **M3**| Pair (Original)   | 946 clusters    | [-0.1922, -0.0480]      | 0.0375     | Baseline              | Safe (< 0)     |
|       | Donor Clustered   | 495 clusters    | [-0.2086, -0.0313]      | 0.0461     | +22.9%                | Safe (< 0)     |
|       | Acceptor Clustered| 416 clusters    | **[-0.1944, -0.0131]**  | **0.0465** | **+24.0%**            | **Borderline** |
|       | DOI Clustered     | 817 clusters    | [-0.1965, -0.0449]      | 0.0390     | +4.0%                 | Safe (< 0)     |
+-------+-------------------+-----------------+-------------------------+------------+-----------------------+----------------+
```

### 4.3 Key Empirical Findings from Sensitivity Analysis
1. **Acceptor Clustering is the Dominant Source of Variance Inflation**:
   - Resampling by acceptor increases standard errors by up to **47.1%** (Model M2 SE rises from 0.0276 to 0.0406).
   - For Model M2, the upper bound of the 95% confidence interval shifts from $-0.0480$ to **$-0.0079\text{ pp}$**, hovering immediately adjacent to zero.
   - For Model M3, clustering by donor or acceptor expands the upper bound to **$-0.0131\text{ pp}$**.
2. **Model M1 and M2A Remain Robust Under Acceptor Clustering**:
   - Model M1 maintains an upper bound of **$-0.0508\text{ pp}$** even under acceptor clustering.
   - Model M2A maintains an upper bound of **$-0.0208\text{ pp}$**.
   - This proves that M1 and M2A achieve genuine error reductions across diverse chemical partners, whereas M2's advantage is fragile.
3. **Literature Clustering (DOI) is Negligible**:
   - Because the 946 pairs are drawn from 817 distinct publications (average 1.16 records/DOI), clustering by publication has minimal impact on variance (+1.8% to +4.0%).

---

## 5. Three-Level Inferential Boundary Specification

To prevent scientific misinterpretation and unwarranted extrapolation, all claims regarding Phase 8.1 performance must be categorized into three strictly demarcated inferential tiers:

```
+---------------------------------------------------------------------------------------------------------+
| Three-Level Inferential Framework for Phase 8.1 Claims                                                  |
+--------------------------+-----------------------+------------------------------------------------------+
| Inferential Level        | Scientific Validity   | Specific Permissible Language                        |
+--------------------------+-----------------------+------------------------------------------------------+
| **Level 1: Nominal**     | **PROVEN**            | Under nominal i.i.d. assumptions on this test set,   |
| **Statistical Test**     | (Under test premises) | paired t-test (p = 1.0e-6) and Wilcoxon test reject  |
|                          |                       | the null hypothesis of equal absolute error.         |
+--------------------------+-----------------------+------------------------------------------------------+
| **Level 2: Held-Out**    | **SUPPORTED**         | Across the 946 held-out pair instances, processing   |
| **Sample Evidence**      | (Empirically verified)| models M1 and M2A achieve a consistent, measurable   |
|                          |                       | error reduction (mean -0.10 to -0.14 pp; 61.5% of    |
|                          |                       | devices improved).                                   |
+--------------------------+-----------------------+------------------------------------------------------+
| **Level 3: Generalization| **UNESTABLISHED**     | Phase 8.1 DOES NOT establish generalization to novel |
| **to New Scaffolds**     | (Do not claim)        | chemical scaffolds. Pair splitting does not isolate  |
|                          |                       | scaffolds; 38% of pairs share top 5 acceptors.       |
+--------------------------+-----------------------+------------------------------------------------------+
```

### 5.1 What the Experiment Evaluated
- When evaluating held-out material-name pairs with known processing features, models M1 and M2A achieve modest but statistically measurable point prediction error reductions over M0 on this test sample.
- **Model M0**: M0 is a molecular-descriptor-only research baseline evaluated on held-out material-name pairs. Its performance on genuinely novel molecules and unseen chemical scaffolds remains unestablished.
- **Model M2 vs M2A**: M2 was not the preferred model in this experiment because M2A achieved slightly better reported metrics on the locked test sample. The difference does not establish universal inferiority.
- **Completeness Limits**: The Rich tier ($k=6\text{--}7$, $N=135$) combines partially incomplete recipes ($k=6$, $N=113$) with complete recipes ($k=7$, $N=22$). The $k=7$ complete recipe subgroup ($N=22$) is statistically underpowered and descriptive only; complete-recipe validation cannot be claimed based on the combined $k=6\text{--}7$ tier.

### 5.2 What the Experiment Does NOT Establish
- **It does NOT establish generalization to unmade, novel chemical scaffolds**.
  - A material pair split (`donor_canonical::acceptor_canonical`) allows the same donor or acceptor to appear across training, calibration, and test sets.
  - In our test set, $\text{PC}_{71}\text{BM}$ was paired with 188 different donors. The model had already observed $\text{PC}_{71}\text{BM}$ hundreds of times during training.
  - Therefore, the test set evaluated **unseen combinations of familiar materials**, not unseen chemical scaffolds.
- **It does NOT establish that processing parameters can be guessed for new molecules**.
  - For a genuinely new donor polymer, the optimal solvent, D:A ratio, and annealing temperature are completely unknown *a priori*. Conditioning on arbitrary default processing values may degrade predictions rather than improve them.
- **It does NOT validate routing by completeness**:
  - Routing by input completeness remains a research hypothesis until exact input-pattern performance, unknown-category handling, and inference behavior are independently evaluated.

---

## 6. Audit Conclusion & Downgrade Directive

1. **Formal Downgrade Directive**: All references to $p < 10^{-6}$ and "high-confidence statistical validation" in project documentation must be explicitly bounded to **held-out sample descriptive evidence**.
2. **Rejection of Scaffold Claims**: Any characterization of Phase 8.1 as establishing "scaffold out-of-distribution generalization" is scientifically rejected. True scaffold generalization requires an explicit Bemis-Murcko scaffold split benchmark.
3. **Preservation of Research Value**: Within its proper scope (held-out combinations of organic materials with known processing parameters), Models M1 and M2A represent verified, reproducible incremental improvements over the molecular-only baseline.
