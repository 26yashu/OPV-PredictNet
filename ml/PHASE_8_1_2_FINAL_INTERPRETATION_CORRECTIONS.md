# OPV-PredictNet — Phase 8.1.2: Final Interpretation and Routing Corrections

**Date**: 2026-11-25  
**Scope**: Systematic correction of overextended claims, statistical terminology, completeness interpretations, and routing contradictions identified in the Phase 8.1 and Phase 8.1.1 reports.  
**Constraint Compliance**: Zero retraining, zero recalibration, zero test prediction modifications, zero production code changes, zero Git commits. All original result CSVs and calibration outputs are strictly preserved.

---

## 1. Executive Summary & Purpose

Phase 8.1 demonstrated that incorporating processing features yields measurable point prediction error reductions on held-out donor–acceptor material-name pairs. However, the Phase 8.1.1 audit identified several areas where interpretative text and routing rules overextended the experimental evidence or contained internal contradictions:

1. **Bootstrap Terminology**: Calling separate one-way cluster bootstraps a "multiway bootstrap" without implementing a genuinely joint crossed-cluster procedure.
2. **Complete-Recipe Interpretation**: Equating the combined Rich tier ($k=6\text{--}7$, $N=135$) with "complete recipes," when only $k=7$ represents complete formulations ($N=22$, severely underpowered).
3. **Partial-Input Routing Contradiction**: Prohibiting Model M1 for partial inputs despite M1 achieving the lowest MAE ($2.2964\text{ pp}$) on the Low-Partial ($k=2\text{--}3$) subgroup.
4. **Molecular Screening Overreach**: Calling Model M0 the "sole validated model for unmade molecules," conflating held-out material-name pairs with novel chemical scaffolds.
5. **Categorical Model Deprecation**: Claiming Model M2 is "formally deprecated," rather than neutrally noting that M2A achieved slightly better sample metrics.
6. **Production Language**: Proposing operational deployment rules rather than research-bounded candidate designations.

This document formally details and records the corrections applied across the Phase 8 documentation suite.

---

## 2. Correction 1: Bootstrap Terminology and Dependence Structure

### 2.1 The Issue
The Phase 8.1.1 review conducted separate cluster bootstraps grouped by pair ($N_c=946$), canonical donor ($N_c=495$), canonical acceptor ($N_c=416$), and publication DOI ($N_c=817$), but referred to these as a "multiway cluster bootstrap."

### 2.2 The Correction
- A true multiway crossed-cluster bootstrap (e.g., Cameron, Gelbach, & Miller 2011; Owen 2014) requires simultaneously resampling across non-nested, intersecting cluster dimensions. In bipartite/tripartite donor–acceptor experimental graphs with highly unbalanced cluster occupancy, crossed-cluster resampling can generate degenerate subgraphs unless specialized multiway linear variance estimators are used.
- Therefore, the analyses are explicitly designated:
  $$\mathbf{Separate\ One\text{-}Way\ Cluster\text{-}Bootstrap\ Sensitivity\ Analyses}$$
- **Statistical Implication**:
  - The four separate one-way analyses illustrate margin-specific variance sensitivity: clustering by canonical acceptor increases the standard error of paired differences by **$38\% - 47\%$**, demonstrating that pair-level bootstrap variance underestimates uncertainty when generalized across shared acceptors.
  - However, because a joint crossed-dependence procedure was not implemented, all inferential conclusions are **downgraded to exploratory descriptive evidence**.
  - Independence between material-name pairs cannot be assumed simply because pair IDs are distinct.

---

## 3. Correction 2: Complete-Recipe Interpretation ($k=7$ vs. $k=6\text{--}7$)

### 3.1 The Issue
Previous report drafts stated that processing models were "validated for complete recipes" based on the Rich completeness tier ($k=6\text{--}7$, $N=135$).

### 3.2 The Correction
- The seven reported processing fields are:
  1. `d_a_ratio` (Blend ratio)
  2. `solvent_canonical` (Solvent)
  3. `active_layer_thickness` (Film thickness)
  4. `annealing_temp` (Annealing temperature)
  5. `device_type` (Architecture: conventional/inverted)
  6. `additive_canonical` (Solvent additive)
  7. `etl_canonical` and `htl_canonical` (Both interlayers present)
- In the held-out test partition ($N=946$):
  - $k=6$: Exactly 6 fields reported ($N = 113$, 11.95%). These records are **partially incomplete** (one parameter is missing).
  - $k=7$: All 7 fields reported ($N = 22$, 2.33%). Only these 22 records represent truly complete recipes.
- **Corrected Interpretation**:
  - The $k=7$ test subgroup ($N=22$) is **statistically underpowered and descriptive only**.
  - No claim of validated performance for complete formulations can be made based on the combined $k=6\text{--}7$ tier.
  - All statements equating the Rich tier with complete recipes are rescinded.

---

## 4. Correction 3: Reconciliation of Partial-Input Routing Claims

### 4.1 The Contradiction
The Phase 8.1.1 report prohibited Model M1 for partial inputs on theoretical grounds (arguing that missing nominal categories map to zero-vectors or `'missing'` columns). However, empirical evaluation on the locked test sample directly contradicted this prohibition:

```
+---------------------------------------------------------------------------------------------------------------+
| Empirical Test MAE (PCE pp) Across Incomplete-Input Completeness Slices                                       |
+--------------------------+-----+-----------+---------------+-----------+-----------+-----------+--------------+
| Subgroup Slice           | N   | M0        | **M1**        | M2        | M2A       | M3        | Lowest Error |
+--------------------------+-----+-----------+---------------+-----------+-----------+-----------+--------------+
| **Sparse (k=1)**         | 44  | 2.2033 pp | 1.9784 pp     | 1.9336 pp | 1.9044 pp | 1.9908 pp | M2A          |
| **Low-Partial (k=2-3)**  | 283 | 2.4047 pp | **2.2964 pp** | 2.3908 pp | 2.3976 pp | 2.3122 pp | **M1**       |
| **High-Partial (k=4-5)** | 385 | 2.0978 pp | 1.9266 pp     | 1.9363 pp | 1.9306 pp | 1.8894 pp | M3           |
+--------------------------+-----+-----------+---------------+-----------+-----------+-----------+--------------+
```

In the largest incomplete tier ($k=2\text{--}3$, $N=283$), **Model M1 achieved the lowest MAE of any model** ($2.2964\text{ pp}$ vs. M0 2.4047, M2 2.3908, M2A 2.3976, M3 2.3122). Prohibiting M1 while recommending M2A on $k=2\text{--}3$ was empirically indefensible.

### 4.2 The Reconciled Scientific Interpretation
- **Empirical Reality**: On this specific held-out test sample, M1 handled partial processing information effectively, achieving lower error than the molecular baseline across all partial tiers ($k=1$, $k=2\text{--}3$, $k=4\text{--}5$).
- **Theoretical Caveat**: M1's one-hot encoding maps unobserved categories to `'missing'` indicator columns. When the missingness patterns in test instances reflect the distribution of training missingness, this encoding functions adequately. However, M1's response to arbitrary unobserved missingness patterns ($2^7 = 128$ combinations) and novel out-of-vocabulary categorical levels remains untested.
- **Reconciled Directive**:
  - Do NOT prohibit M1 for all partial inputs.
  - Do NOT recommend M1 for arbitrary partial recipes.
  - Formally state:
    > "Routing by input completeness remains a research hypothesis until exact input-pattern performance, unknown-category handling, and inference behavior are independently evaluated."

---

## 5. Correction 4: Molecular-Only Screening Claims

### 5.1 The Issue
The Phase 8.1.1 report described Model M0 as the "sole validated model for unmade molecules" or "validated for virtual screening."

### 5.2 The Correction
- The Phase 8.1 experiment evaluated held-out **material-name pairs** (`donor_canonical::acceptor_canonical`).
- In this benchmark:
  - 38.05% of devices shared one of the top 5 acceptors ($\text{PC}_{71}\text{BM}$, ITIC, $\text{PC}_{70}\text{BM}$, $\text{PC}_{61}\text{BM}$, Y6).
  - 28.86% of devices shared one of the top 5 donors (PBDB-T, PM6, PTB7-Th, PBDB-TF, P3HT).
  - The evaluation measured performance on **unseen combinations of known materials**, not genuinely unmade molecules or novel chemical scaffolds.
- **Standardized Text Applied Across All Reports**:
  > "M0 is a molecular-descriptor-only research baseline evaluated on held-out material-name pairs. Its performance on genuinely novel molecules and unseen chemical scaffolds remains unestablished."

---

## 6. Correction 5: M2 Deprecation Language

### 6.1 The Issue
Previous text stated that Model M2 was "formally deprecated" and "disqualified."

### 6.2 The Correction
- Model M2A (37 features, omitting the 7 missingness indicator flags) achieved slightly better sample metrics on the test partition than Model M2 (44 features):
  - Test MAE: M2A $2.0689\text{ pp}$ vs. M2 $2.0722\text{ pp}$ ($\Delta = -0.0033\text{ pp}$).
  - Test RMSE: M2A $2.8184\text{ pp}$ vs. M2 $2.8289\text{ pp}$ ($\Delta = -0.0105\text{ pp}$).
  - Test Bias: M2A $+0.0387\text{ pp}$ vs. M2 $+0.0474\text{ pp}$.
- A small sample metric difference of $0.0033\text{ pp}$ on $N=946$ indicates that missingness flags do not add demonstrable value on this benchmark, but it does not establish that M2 is universally defective or fundamentally broken.
- **Standardized Text Applied Across All Reports**:
  > "M2 was not the preferred model in this experiment because M2A achieved slightly better reported metrics on the locked test sample. The difference does not establish universal inferiority."
- All M2 results, metrics, and implementation artifacts are preserved.

---

## 7. Summary of Changes Applied to Documentation Suite

```
+---------------------------------------------------------------------------------------------------------+
| Summary of Documentation Revisions across Phase 8.1 and 8.1.1 Files                                     |
+------------------------------------+--------------------------------+-----------------------------------+
| File                               | Previous Language              | Corrected Language                |
+------------------------------------+--------------------------------+-----------------------------------+
| `ml/PHASE_8_1_EXPERIMENT_REPORT.md`| "Complete recipes (k=6-7)"     | "Rich completeness tier (k=6-7;   |
|                                    |                                | k=7 alone is N=22, descriptive)"  |
|                                    | "M0 validated for screening"   | "M0 is a research baseline on     |
|                                    |                                | held-out pairs; scaffold unproven"|
|                                    | "M2 formally deprecated"       | "M2 not preferred; M2A achieved   |
|                                    |                                | slightly better sample metrics"   |
+------------------------------------+--------------------------------+-----------------------------------+
| `ml/PHASE_8_1_MODEL_COMPARISON.md` | "Deprecated Model M2"          | "M2 not preferred in experiment"  |
|                                    | "Operational Deployment Recs"  | "Research Candidate Evaluations"  |
|                                    | "M0 for virtual screening"     | "M0 research baseline; scaffold   |
|                                    |                                | extrapolation unestablished"      |
+------------------------------------+--------------------------------+-----------------------------------+
| `ml/PHASE_8_1_1_INDEPENDENT_AUDIT` | "Complete recipes" in Rich     | "k=7 is N=22, descriptive only;   |
|                                    |                                | Rich tier contains k=6 partial"   |
|                                    | "Multiway cluster bootstrap"   | "Separate one-way cluster         |
|                                    |                                | bootstrap sensitivity analyses"   |
+------------------------------------+--------------------------------+-----------------------------------+
| `ml/PHASE_8_1_1_STATISTICAL_REVIEW`| "Multiway Cluster Bootstrap"   | "Separate One-Way Cluster         |
|                                    |                                | Bootstrap Sensitivity Analyses"   |
|                                    | "Multiway dependence structure"| "Cross-pair structural dependence"|
+------------------------------------+--------------------------------+-----------------------------------+
| `ml/PHASE_8_1_1_ROUTING_READINESS` | "Prohibit M1 for partial input"| "Routing by input completeness is |
|                                    |                                | a research hypothesis; M1 achieved|
|                                    |                                | lowest MAE in k=2-3"              |
|                                    | "M0 sole validated model"      | "M0 research baseline; scaffold   |
|                                    |                                | generalization unestablished"     |
+------------------------------------+--------------------------------+-----------------------------------+
```
