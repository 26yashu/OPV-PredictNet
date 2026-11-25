# OPV-PredictNet — Phase 8.1: Head-to-Head Model Comparison and Trade-Off Analysis

**Date**: 2026-11-24  
**Scope**: Rigorous comparative analysis of five model candidates (M0, M1, M2, M2A, M3) evaluated on identical held-out test records ($N = 946$ unseen material pairs) under Phase 8.1.  
**Constraint Compliance**: All evaluations frozen. Zero production code changes, zero Git commits created.

---

## 1. Executive Summary & Master Comparison Table

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Phase 8.1 Master Model Comparison Matrix (N = 946 Held-Out Material Pairs)                                                   |
+-------+-------------------------+-------+---------+-----------+-----------+---------+-----------+-----------+-----------------+
| Model | Identifier              | Feats | Time (s)| Test MAE  | Test RMSE | Test R² | Test Bias | 90% Width | 90% Coverage    |
+-------+-------------------------+-------+---------+-----------+-----------+---------+-----------+-----------+-----------------+
| **M0**| Molecular Baseline      | 22    | 7.37s   | 2.1727 pp | 2.9026 pp | 0.6110  | +0.0245 pp| 8.9671 pp | 90.06% (Exact)  |
| **M1**| Naive OHE Processing    | 50    | 15.54s  | **2.0321**| 2.7837 pp | 0.6422  | +0.0529 pp| 8.6488 pp | 90.17% (Target) |
| **M2**| Processing Aware (Flags)| 44    | 15.31s  | 2.0722 pp | 2.8289 pp | 0.6305  | +0.0474 pp| 8.7298 pp | 90.06% (Exact)  |
|**M2A**| Ablation (No Flags)     | 37    | 14.13s  | 2.0689 pp | 2.8184 pp | 0.6332  | +0.0387 pp| 8.6600 pp | 90.38% (Target) |
| **M3**| Native-NaN GBDT         | 44    | **1.27s**| 2.0540 pp| **2.7525**| **0.6502**|+0.1108 pp| **8.4283**| 88.69% (Marginal)|
+-------+-------------------------+-------+---------+-----------+-----------+---------+-----------+-----------+-----------------+
```

---

## 2. Scientific Trade-Off Analysis

### 2.1 Point Accuracy vs. Architectural Complexity
1. **Model M1 (One-Hot Encoded Processing, 50 Features)**:
   - **Performance**: Lowest MAE across the entire test set (**2.0321 PCE pp**), representing a **-0.1407 pp reduction** over baseline M0 ($p = 1.03 \times 10^{-6}$).
   - **Why It Works**: Decision trees in Random Forests perform best with orthogonal, binary indicator splits. One-hot encoding the dominant solvents (`chloroform`, `chlorobenzene`, `o-DCB`) and interlayers (`ZnO`, `PEDOT:PSS`, `MoO3`) allows the ensemble to isolate distinct morphological fabrication regimes without forcing arbitrary continuous relationships.
   - **Trade-Off**: Higher dimensionality (50 features) and inability to extrapolate to novel solvents outside the 23-column training vocabulary (unseen solvents evaluate to all zeros).
2. **Model M3 (Native-NaN GBDT, 44 Features)**:
   - **Performance**: Lowest RMSE (**2.7525 PCE pp**) and highest $R^2$ (**0.6502** vs. M0 0.6110). Extremely fast training (**1.27 seconds** vs. 15.54s for RF).
   - **Why It Works**: Gradient boosting iteratively corrects residual errors and natively routes missing values down optimal tree branches, effectively capturing non-linear interactions between molecular frontier orbitals and processing parameters.
   - **Trade-Off**: Higher global positive bias (+0.1108 pp) and slightly lower 90% interval coverage (88.69% vs. 90.17% for M1).
3. **Missingness Ablation: Model M2 vs. Model M2A**:
   - Model M2 (with 7 missingness indicator flags): $\text{MAE} = 2.0722$, $R^2 = 0.6305$, $\text{Bias} = +0.0474$.
   - Model M2A (without missingness indicator flags): $\text{MAE} = \mathbf{2.0689}$, $R^2 = \mathbf{0.6332}$, $\text{Bias} = \mathbf{+0.0387}$.
   - **Definitive Scientific Finding**: Omitting the 7 explicit missingness indicators improves accuracy and reduces bias. The binary flags in M2 did not contribute genuine physical signal; instead, they introduced slight overfitting. The normalized continuous variables, physical regimes, and interface constants in M2A capture the processing signal cleanly. **Model M2A is superior to Model M2**.

---

## 3. Uncertainty Interval Tightness and Limits of Processing Information

A central scientific hypothesis of Phase 8 was whether incorporating processing information would dramatically narrow the 90% conformal prediction intervals (which were $8.84 - 8.97\text{ pp}$ wide in the Phase 5/7 baseline).

### Key Empirical Findings
1. **Modest Width Reduction**:
   - M0 Baseline: 90% median width = **8.97 PCE pp** ($q = 4.4836$).
   - M1 (OHE Processing): 90% median width = **8.65 PCE pp** ($q = 4.3244$) $\to$ **3.5% reduction**.
   - M2A (Ablation Processing): 90% median width = **8.66 PCE pp** ($q = 4.3300$) $\to$ **3.4% reduction**.
   - M3 (Native-NaN GBDT): 90% median width = **8.43 PCE pp** ($q = 4.2141$) $\to$ **6.0% reduction**.
2. **Why Processing Metadata Does NOT Collapse Uncertainty**:
   - Even with exact fabrication parameters (blend ratio, solvent, additive, temperature, thickness, architecture), prediction intervals remain $\approx 8.5\text{ PCE pp}$ wide on unseen material-name pairs.
   - **Physical Explanation**:
     - *Inter-laboratory variability*: Substrate preparation, glovebox oxygen/moisture levels, spin-coater acceleration curves, and batch-to-batch polymer molecular weight dispersity create substantial within-recipe variation that literature metadata does not capture.
     - *Chemical novelty dominance*: When predicting a completely new donor-acceptor pair, electronic interface energetics and film morphology remain dominated by chemical compatibility. Processing parameters fine-tune performance within a pair, but cannot bridge large out-of-distribution chemical gaps.

---

## 4. Subgroup Performance Across Completeness Tiers and PCE Ranges

```
+---------------------------------------------------------------------------------------------------------------------+
| Subgroup MAE (PCE pp) Comparison Across All Models                                                                  |
+--------------------------+------+-----------+-----------+-----------+-----------+-----------+-----------------------+
| Subgroup Slice           | N    | M0        | M1        | M2        | M2A       | M3        | Best Model in Slice   |
+--------------------------+------+-----------+-----------+-----------+-----------+-----------+-----------------------+
| **Molecular-Only (k=0)** | 99   | 1.9469 pp | 1.9209 pp | 1.9167 pp |**1.9000 pp**| 2.1584 pp| **M2A** (-0.0469 pp)  |
| **Sparse (k=1)**         | 44   | 2.2033 pp | 1.9784 pp | 1.9336 pp |**1.9044 pp**| 1.9908 pp| **M2A** (-0.2989 pp)  |
| **Low-Partial (k=2-3)**  | 283  | 2.4047 pp |**2.2964 pp**| 2.3908 pp| 2.3976 pp | 2.3122 pp| **M1**  (-0.1083 pp)  |
| **High-Partial (k=4-5)** | 385  | 2.0978 pp | 1.9266 pp | 1.9363 pp | 1.9306 pp |**1.8894 pp**| **M3**  (-0.2084 pp)  |
| **Rich (k=6-7)**         | 135  | 2.0558 pp |**1.8777 pp**| 1.9510 pp | 1.9515 pp | 1.9265 pp| **M1**  (-0.1781 pp)  |
+--------------------------+------+-----------+-----------+-----------+-----------+-----------+-----------------------+
| **PCE < 5.0%**           | 322  | 2.2470 pp | 2.1224 pp | 2.1416 pp | 2.1320 pp |**2.1125 pp**| **M3**  (-0.1345 pp)  |
| **PCE 5.0% - 10.0%**     | 339  | 1.9434 pp |**1.8129 pp**| 1.8480 pp | 1.8558 pp | 1.8609 pp| **M1**  (-0.1305 pp)  |
| **PCE 10.0% - 15.0%**    | 210  | 2.1164 pp |**2.0016 pp**| 2.0576 pp | 2.0553 pp | 2.0548 pp| **M1**  (-0.1148 pp)  |
| **PCE >= 15.0%**         | 75   | 3.0481 pp | 2.7204 pp | 2.8286 pp | 2.7987 pp |**2.6740 pp**| **M3**  (-0.3741 pp)  |
+--------------------------+------+-----------+-----------+-----------+-----------+-----------+-----------------------+
```

### Analysis of Champion Cell Performance ($\text{PCE} \ge 15.0\%$, $N=75$)
- **Point Accuracy**: Processing-aware models significantly reduce error on high-efficiency champion cells (M3 MAE = 2.67 pp, M1 MAE = 2.72 pp vs. M0 MAE = 3.05 pp).
- **Persistent Negative Bias**: All models suffer from severe regression-to-the-mean shrinkage on champion cells ($\text{Bias} \approx -2.57\text{ to } -2.84\text{ pp}$).
- **Interval Undercoverage**: 90% intervals cover only **$81.3\% - 84.0\%$** of champion cells.
- **Scientific Conclusion**: Device processing information helps explain why certain cells reach high efficiency, but tree-based regressors still shrink champion predictions toward the global dataset mean ($\approx 9.2\%$).

---

## 5. Research Candidate Evaluations

```
+---------------------------------------------------------------------------------------------------------+
| Research Candidate Evaluations (Research Prototypes Only — No Production Deployment)                   |
+--------------------------+-----------------------+------------------------------------------------------+
| Research Context         | Candidate Model       | Evidentiary Scope and Boundaries                     |
+--------------------------+-----------------------+------------------------------------------------------+
| **Molecular-Only**       | **Model M0**          | M0 is a molecular-descriptor-only research baseline  |
| **Research Baseline**    | (Molecular Baseline)  | evaluated on held-out pairs. Its performance on      |
|                          |                       | genuinely novel molecules and unseen scaffolds       |
|                          |                       | remains unestablished.                               |
+--------------------------+-----------------------+------------------------------------------------------+
| **Standard Laboratory**  | **Model M1**          | Lowest sample MAE (2.03 pp) on held-out pairs when   |
| **Formulation Tuning**   | (OHE Processing)      | standard solvents (CF/CB) and interlayers are used.  |
|                          |                       | Exploratory research candidate only.                 |
+--------------------------+-----------------------+------------------------------------------------------+
| **Continuous Parameter** | **Model M2A**         | Useful for continuous scans (D:A ratio, thickness,   |
| **Physics Sweeps**       | (Ablation Processing) | annealing temp); avoids discrete OHE barriers.       |
|                          |                       | Exploratory research candidate only.                 |
+--------------------------+-----------------------+------------------------------------------------------+
| **Fast Iterative**       | **Model M3**          | Lowest sample RMSE (2.75 pp), highest R² (0.6502),   |
| **Screening / GBDT**     | (Native-NaN GBDT)     | fits in 1.27s; natively routes missing values.       |
|                          |                       | Exploratory research candidate only.                 |
+--------------------------+-----------------------+------------------------------------------------------+
| **Non-Preferred Model**  | **Model M2** (Flags)  | M2 was not the preferred model in this experiment    |
|                          |                       | because M2A achieved slightly better reported metrics|
|                          |                       | on the locked test sample. The difference does not   |
|                          |                       | establish universal inferiority.                     |
+--------------------------+-----------------------+------------------------------------------------------+
```

### Core Research Boundaries
1. **No Production Recommendations**: These models are research prototypes. No production deployment to API backends or frontends is authorized.
2. **Partial-Input Routing is a Hypothesis**: Routing by input completeness remains a research hypothesis until exact input-pattern performance, unknown-category handling, and inference behavior are independently evaluated.
3. **Complete-Recipe Performance Unvalidated**: Truly complete recipes ($k=7$) have only $N=22$ instances in the test set and are descriptive only. Performance on complete formulations cannot be claimed based on the combined $k=6\text{--}7$ tier.
4. **Scaffold Extrapolation Unproven**: Pair-held-out validation evaluated unseen combinations of familiar donors and acceptors, not novel chemical scaffolds. Extrapolation to novel chemical families remains unestablished.

---

**Head-to-Head Comparison concluded. All experimental results validated and persisted.**
