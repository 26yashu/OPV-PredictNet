# OPV-PredictNet — Phase 8.1.2: Research Routing Policy and Evidentiary Boundaries

**Date**: 2026-11-25  
**Scope**: Defensible evidentiary boundaries and research-candidate routing guidelines for Phase 8.1 models (M0, M1, M2, M2A, M3).  
**Status**: Research-use only. Production deployment recommendations are strictly prohibited.

---

## 1. Executive Summary & Policy Overview

Phase 8.1 evaluated five model architectures on 946 held-out donor–acceptor material-name pairs under locked experimental conditions. This policy defines what the experimental evidence actually supports, identifies unvalidated proposed uses, and specifies prohibited claims.

### Key Policy Mandates
1. **No Production Recommendations**: All models are research prototypes. No model is authorized for production deployment in backend APIs or frontend user interfaces.
2. **Four-Tier Evidentiary Distinction**: Every proposed use of these models must be classified into one of four distinct categories:
   - *Experimentally Evaluated*
   - *Supported as a Research Candidate*
   - *Not Yet Validated for Proposed Use*
   - *Prohibited Claims*
3. **Completeness Routing is a Research Hypothesis**: Routing models dynamically based on field count $k$ is an active research hypothesis, not an established operational rule.
4. **Scaffold Generalization Remains Unestablished**: All evaluations were conducted on held-out material-name pairs, not novel chemical scaffolds. Extrapolation to novel chemical families has not been demonstrated.

---

## 2. Master Evidentiary Classification Matrix

```
+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Phase 8.1.2 Master Evidentiary Classification Matrix across Core OPV Modeling Contexts                                                                        |
+--------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------------+
| Operational Context      | Experimentally Evaluated      | Supported as Research Cand.   | Not Yet Validated             | Prohibited Claims                  |
+--------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------------+
| **Context A:**           | • M0 evaluated on 946 held-   | • **Model M0** is a valid     | • Virtual screening on unmade | • Claiming M0 is "validated for    |
| **Molecular-Only**       |   out pairs (MAE 2.1727 pp,   |   research baseline when no   |   molecules with unknown syn- |   unmade molecules."               |
| **Screening**            |   90% coverage 90.06%).       |   processing data is known.   |   thetic accessibility.       | • Claiming M0 generalizes to novel |
|                          | • Evaluated on k=0 tier       | • Compares baseline molecular | • Novel chemical scaffolds    |   chemical scaffolds out-of-dist.  |
|                          |   (N=99, MAE 1.9469 pp).      |   properties across pairs.    |   outside training families.  | • Recommending M0 for production.  |
+--------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------------+
| **Context B:**           | • Rich tier evaluated (k=6-7, | • **Model M1** is an explo-   | • True complete formulations  | • Claiming validated performance   |
| **Complete Recipe**      |   N=135, MAE 1.8777 pp for M1)|   ratory candidate when stan- |   (k=7 has only N=22 instances|   for complete recipes based on the|
| **Supplied**             | • Truly complete recipe (k=7) |   dard solvents/layers known. |   in test set, underpowered). |   combined k=6-7 tier.             |
|                          |   evaluated (N=22 instances). | • **Model M2A** is an explo-  | • Inter-lab reproduction; 90% | • Claiming processing metadata     |
|                          | • Conformal intervals remain  |   ratory candidate for conti- |   intervals remain ~8.5 pp    |   collapses prediction uncertainty.|
|                          |   8.43 - 8.66 pp wide.        |   nuous parameter sweeps.     |   wide due to lab variance.   | • Claiming M2 is universally bad.  |
+--------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------------+
| **Context C:**           | • Low-Partial tier (k=2-3,    | • **Model M1** (achieved      | • Arbitrary missingness combi-| • Prohibiting M1 for all partial   |
| **Partial Recipe**       |   N=283, M1 MAE 2.2964 pp).   |   lowest MAE on k=2-3).       |   nations (2^7 = 128 patterns)|   inputs despite its lowest MAE.   |
| **Supplied**             | • High-Partial tier (k=4-5,   | • **Model M2A** (robust continuous| • Out-of-vocabulary solvents| • Recommending M1 for arbitrary    |
|                          |   N=385, M3 MAE 1.8894 pp).   |   regimes without flags).     |   evaluating to zero-vectors. |   partial recipes without testing. |
|                          | • Sparse tier (k=1, N=44).    | • **Model M3** (native NaN).  | • Routing by completeness tier| • Claiming partial inputs provide  |
|                          | • 90% cov drops to 85.2-87.6%.|                               |   is a research hypothesis.   |   narrow, reliable intervals.      |
+--------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------------+
| **Context D:**           | • Zero scaffold splits were   | • **NO MODEL IS SUPPORTED**   | • All models are unvalidated  | • Claiming scaffold generalization |
| **Scaffold**             |   evaluated in Phase 8.1.     |   for novel chemical scaffold |   for out-of-distribution     |   based on material-pair splits.   |
| **Extrapolation**        | • Benchmark evaluated held-   |   extrapolation.              |   scaffold discovery.         | • Claiming novel acceptor discovery|
|                          |   out pairs, where 38% of pairs|                               | • Bemis-Murcko scaffold split |   capability without scaffold CIs. |
|                          |   shared top 5 acceptors.     |                               |   benchmark is required.      |                                    |
+--------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------------+
| **Champion Cells:**      | • Evaluated on actual PCE     | • Models M1 and M3 reduce     | • Unbiased champion prediction| • Claiming reliable 90% conformal  |
| **Actual PCE >= 15%**    |   >= 15.0% (N=75 devices).    |   champion MAE (2.67-2.72 pp  |   is not achieved by any      |   coverage on champion cells (cov- |
|                          | • Severe negative bias found  |   vs. M0 3.05 pp) as explo-   |   model (all bias <= -2.57 pp)|   erage is only 81.3% - 84.0%).    |
|                          |   (-2.57 to -2.85 pp).        |   ratory research tools.      | • Bounded interval calibration| • Denying regression shrinkage bias|
|                          | • 90% coverage is 81.3-84.0%. |                               |   for high-PCE tail.          |   in tree-based ensembles.         |
+--------------------------+-------------------------------+-------------------------------+-------------------------------+------------------------------------+
```

---

## 3. Detailed Evidentiary Boundaries by Modeling Context

### 3.1 Context A: Molecular-Only Screening ($k = 0$)
- **Experimentally Evaluated**:
  - Model M0 was trained on 22 RDKit molecular descriptors across 14,704 training records and evaluated on 946 held-out pair records ($\text{MAE} = 2.1727\text{ pp}$, $\text{RMSE} = 2.9026\text{ pp}$, $R^2 = 0.6110$).
  - On the subset where no processing features were reported ($k=0$, $N=99$), M0 achieved $\text{MAE} = 1.9469\text{ pp}$ and empirical coverage of $94.95\%$.
- **Supported as a Research Candidate**:
  - Model M0 is supported as an exploratory research baseline for ranking donor–acceptor material-name pairs when device fabrication parameters are completely absent.
- **Not Yet Validated**:
  - Screening genuinely novel molecules with uncharacterized chemical cores, synthetic routes, or steric hindrance profiles.
- **Prohibited Claims**:
  - Do NOT claim Model M0 is the "sole validated model for unmade molecules."
  - Standardized citation:
    > "M0 is a molecular-descriptor-only research baseline evaluated on held-out material-name pairs. Its performance on genuinely novel molecules and unseen chemical scaffolds remains unestablished."

---

### 3.2 Context B: Complete Recipe Supplied
- **Experimentally Evaluated**:
  - The Rich completeness tier ($k=6\text{--}7$, $N=135$) was evaluated: M1 achieved $\text{MAE} = 1.8777\text{ pp}$, M2A achieved $\text{MAE} = 1.9515\text{ pp}$, and M3 achieved $\text{MAE} = 1.9265\text{ pp}$.
  - The truly complete recipe subset ($k=7$) contains only $N = 22$ records in the test set.
  - Across all models, 90% prediction intervals remained $8.43\text{ to } 8.66\text{ pp}$ wide.
- **Supported as a Research Candidate**:
  - Model M1 is supported as an exploratory candidate for formulation optimization where standard solvents (`chloroform`, `chlorobenzene`, `o-DCB`) and interlayers (`PEDOT:PSS`, `ZnO`, `MoO3`) are utilized.
  - Model M2A is supported as an exploratory candidate for continuous parameter sweeps (film thickness, annealing temperature, blend ratio) without discrete categorical boundaries.
- **Not Yet Validated**:
  - Validated performance on complete recipes cannot be established from $N=22$ instances. The $k=7$ results are descriptive and underpowered.
  - Processing metadata does not eliminate inter-laboratory fabrication variability; uncertainty intervals remain wide.
- **Prohibited Claims**:
  - Do NOT claim complete-recipe performance is validated based on the combined $k=6\text{--}7$ tier.
  - Do NOT claim Model M2 is "formally deprecated." Use:
    > "M2 was not the preferred model in this experiment because M2A achieved slightly better reported metrics on the locked test sample. The difference does not establish universal inferiority."

---

### 3.3 Context C: Partial Recipe Supplied ($1 \le k \le 5$)
- **Experimentally Evaluated**:
  - In the Low-Partial tier ($k=2\text{--}3$, $N=283$), Model M1 achieved $\text{MAE} = 2.2964\text{ pp}$, outperforming M0 ($2.4047\text{ pp}$), M2 ($2.3908\text{ pp}$), M2A ($2.3976\text{ pp}$), and M3 ($2.3122\text{ pp}$).
  - In the High-Partial tier ($k=4\text{--}5$, $N=385$), Model M3 achieved $\text{MAE} = 1.8894\text{ pp}$ and M1 achieved $\text{MAE} = 1.9266\text{ pp}$.
  - Conformal 90% coverage on Low-Partial inputs degraded to $85.16\% - 87.63\%$.
- **Supported as a Research Candidate**:
  - Models M1, M2A, and M3 are supported as exploratory research candidates for investigating incomplete device data.
- **Not Yet Validated**:
  - Arbitrary missingness combinations ($2^7 = 128$ distinct patterns) have not been evaluated in a controlled factorial experiment.
  - Handling of unseen, out-of-vocabulary solvents or interlayers (which evaluate to zero-vectors in M1) has not been tested.
- **Prohibited Claims**:
  - Do NOT prohibit Model M1 for all partial inputs (this contradicts its superior empirical performance on $k=2\text{--}3$).
  - Do NOT recommend Model M1 for arbitrary partial inputs without prior testing.
  - Standardized statement:
    > "Routing by input completeness remains a research hypothesis until exact input-pattern performance, unknown-category handling, and inference behavior are independently evaluated."

---

### 3.4 Context D: Chemical Scaffold Extrapolation
- **Experimentally Evaluated**:
  - No scaffold-split benchmark was conducted in Phase 8.1.
  - In the 946 test pairs, 188 pairs (19.87%) shared $\text{PC}_{71}\text{BM}$, and the top 5 acceptors accounted for 38.05% of all records.
  - The experiment evaluated unseen combinations of familiar materials, not novel chemical scaffolds.
  - Separate one-way cluster bootstrap sensitivity analysis revealed that clustering by acceptor inflated standard errors by up to $47.1\%$.
- **Supported as a Research Candidate**:
  - **No model is supported for scaffold extrapolation.**
- **Not Yet Validated**:
  - Any claim of predicting performance for a new chemical class, novel core scaffold, or unprecedented non-fullerene acceptor architecture.
- **Prohibited Claims**:
  - Do NOT equate material-name pair holdout validation with scaffold holdout validation.
  - Do NOT claim that these models can reliably guide the discovery of novel chemical scaffolds without prior training data on that chemical family.

---

### 3.5 Champion-Cell Performance ($\text{PCE} \ge 15.0\%$)
- **Experimentally Evaluated**:
  - Across $N = 75$ test devices with actual $\text{PCE} \ge 15.0\%$:
    - M0: $\text{MAE} = 3.0481\text{ pp}$, $\text{Bias} = -2.8449\text{ pp}$, 90% Coverage = $81.33\%$.
    - M1: $\text{MAE} = 2.7204\text{ pp}$, $\text{Bias} = -2.5723\text{ pp}$, 90% Coverage = $84.00\%$.
    - M2: $\text{MAE} = 2.8286\text{ pp}$, $\text{Bias} = -2.6503\text{ pp}$, 90% Coverage = $81.33\%$.
    - M2A: $\text{MAE} = 2.7987\text{ pp}$, $\text{Bias} = -2.6481\text{ pp}$, 90% Coverage = $82.67\%$.
    - M3: $\text{MAE} = 2.6740\text{ pp}$, $\text{Bias} = -2.6266\text{ pp}$, 90% Coverage = $82.67\%$.
- **Supported as a Research Candidate**:
  - M1 and M3 demonstrate exploratory utility in reducing prediction error on high-efficiency devices relative to M0.
- **Not Yet Validated**:
  - Unbiased point prediction of record champion devices (all models suffer from $\approx -2.6\text{ to } -2.8\text{ pp}$ shrinkage).
  - Reliable 90% prediction intervals on champion cells (empirical coverage is $81.3\% - 84.0\%$).
- **Prohibited Claims**:
  - Do NOT claim 90% conformal coverage holds on champion cells.
  - Mandatory notice: High-PCE predictions are subject to substantial negative shrinkage bias.

---

## 4. Summary of Research Policy Guidelines

1. **Research Candidate Scope**:
   - For molecular-only queries: Model M0 is the baseline candidate on held-out pairs.
   - For formulation tuning sweeps: Models M1 and M2A are exploratory candidates.
   - For gradient-boosted native-missing exploration: Model M3 is an exploratory candidate.
2. **Model Status Preservation**:
   - Model M2 is preserved with its implementation artifacts and metrics; it was simply not preferred over M2A on the test sample.
3. **Generalization Boundaries**:
   - Pair-held-out validation must never be represented as chemical scaffold generalization.
   - True scaffold generalization requires an explicit Bemis-Murcko scaffold benchmark.
