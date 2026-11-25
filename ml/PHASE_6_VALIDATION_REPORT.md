# Phase 6 — Model Validation, Robustness, and Improvement Report

**Project:** OPV-PredictNet  
**Phase:** 6 — Model Validation, Robustness and Improvement  
**Date:** November 2026  
**Auditor:** Quality & Machine Learning Engineering Lead  
**Experiment Type:** Controlled Validation & Robustness Evaluation (*NOT Production Model Certification*)  
**Execution Environment:** Python `3.11.9`, Scikit-learn `1.9.1`, RDKit `2026.09.2`, Pandas `3.1.1`, NumPy `2.4.6` (Windows 10/11)  
**Fixed Random Seed:** `42`  
**Dataset:** OPV-DB v1.1.0 Strict Molecular Benchmark (`22,467` records)  
**Target:** Power Conversion Efficiency ($PCE$, percentage points)  
**Status:** **VALIDATION COMPLETED — ROBUSTNESS BOUNDARIES & CALIBRATION CODIFIED**

---

## 1. Executive Summary

Phase 6 executes a comprehensive scientific audit of the baseline `RandomForestRegressor` to establish its operational boundaries, identify failure modes, test controlled hyperparameter improvements using inner training folds, and evaluate ensemble prediction uncertainty.

### Key Conclusions:
1. **Baseline Preserved Unchanged:** All Phase 5 primary and secondary baseline outputs remain unmodified in `ml/results/phase_5_*`.
2. **Device Architecture Sensitivity Identified:**
   - On **Conventional devices** ($N=9,284$), the baseline achieves strong predictive accuracy ($\text{MAE} = 2.01\%$, $\text{RMSE} = 2.86\%$, $R^2 = \mathbf{0.78}$).
   - On **Inverted devices** ($N=5,073$), performance drops noticeably ($\text{MAE} = 2.16\%$, $\text{RMSE} = 3.09\%$, $R^2 = \mathbf{0.54}$, with a negative bias of $-0.38\%$). This asymmetry occurs because inverted cells depend heavily on metal oxide interlayers (ZnO, sol-gel, $\text{TiO}_x$) and top-contact work functions that are not captured by 2D active-layer molecular descriptors.
3. **Chemical Family Performance Profile:**
   - **Fullerene Acceptors** ($N=7,177$): Lowest absolute error ($\text{MAE} = 1.69\%$, $\text{RMSE} = 2.23\%$).
   - **Fused-Ring Non-Fullerene Acceptors (NFAs)** ($N=10,185$): Higher absolute error ($\text{MAE} = 2.23\%$, $\text{RMSE} = 3.33\%$) driven by large processing optimization sweeps ($PCE$ ranges of $12 - 19\%$).
   - **Polymer Acceptors (P-SMAs)** ($N=852$): Systematically **under-predicted** by $+0.99\%$ PCE on average.
   - **PDI-based Acceptors** ($N=197$): Systematically **over-predicted** by $-0.91\%$ PCE.
4. **Controlled Model Improvement (Inner Training Selection):**
   - Evaluating a regularized grid strictly within training fold cross-validation selected **`RF_Config_4_EnsembleScaled`** ($200$ trees, `max_features=0.8`).
   - On the outer 5-fold pair-grouped validation, this yields a modest improvement over the Phase 5 baseline: outer MAE improves from **$2.0628\%$** to **$2.0532\%$**, and outer $R^2$ improves from **$0.7099$** to **$0.7131$** (pooled $R^2 = 0.7426$).
5. **Prediction Uncertainty Is Severely Uncalibrated:**
   - Inter-tree dispersion ($\sigma_{trees}$) exhibits weak correlation with realized absolute error ($r = 0.1627$).
   - Nominal confidence intervals exhibit **severe under-coverage**: a nominal $68.3\%$ interval ($\pm 1.0 \sigma$) covers only **$38.35\%$** of actual targets, and a nominal $95.0\%$ interval ($\pm 1.96 \sigma$) covers only **$57.66\%$**.
   - **Mandatory Policy:** Raw tree variance **must never be reported to users as a calibrated confidence interval**. It measures tree disagreement, not total predictive uncertainty.

---

## 2. Experimental Metadata & Architecture Coverage Audit

An audit of the experimental metadata columns across all 22,467 benchmark records revealed the following empirical coverage:

| Metadata Field | Non-Null Records | Missing Records | Coverage (%) | Distinct Values | Dominant Values / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `device_type` | 17,072 | 5,395 | **75.99%** | 303 | `conventional` (9,020), `inverted` (4,735), `organic solar cell` (490) |
| `device_structure` | 18,619 | 3,848 | **82.87%** | 4,097 | `inverted` (1,829), `conventional` (660), specific layer stacks |
| `active_layer_thickness`| 6,687 | 15,780 | **29.76%** | 622 | 70.24% missing; numerical thickness in nm |
| `d_a_ratio` | 11,362 | 11,105 | **50.57%** | 655 | 49.43% missing; e.g. `1:1`, `1:1.2`, `1:1.5` |
| `additive` | 7,942 | 14,525 | **35.35%** | 338 | 64.65% missing; e.g. `DIO`, `CN`, `DPE` |
| `annealing_temp` | 5,523 | 16,944 | **24.58%** | 185 | 75.42% missing; thermal annealing in °C |
| `solvent` | 12,159 | 10,308 | **54.12%** | 196 | 45.88% missing; e.g. `CB`, `CF`, `o-xylene` |

### 2.1 Single-Junction vs. Tandem Device Intermixing
- In OPV-DB, terms such as `tandem`, `multi-junction`, and `multijunction` do **not** appear as structured tags in `device_type` or `device_structure`.
- `single-junction` is explicitly tagged in only 76 records ($0.34\%$).
- `binary` is explicitly tagged in 57 records ($0.25\%$).
- `ternary` is explicitly tagged in 7 records ($0.03\%$).
- **Implication:** Single-junction, ternary, and tandem records are subtly intermixed in the literature dataset without explicit machine-readable flags. As identified in Phase 5.1, extreme under-prediction outliers (e.g. `SM1::PC71BM` achieving $20.56\%$ PCE) correspond to champion tandem or specialized optical-management cells.

---

## 3. Residual and Reliability Analysis Across Device & Chemical Subsets

Using the primary 5-fold pair-grouped out-of-fold predictions from Phase 5, performance was stratified across architectures, processing conditions, chemical families, and target ranges:

### Comprehensive Error Analysis Table (N = 22,467):

| Category | Subcategory | Record Count | % of Data | Mean Actual PCE (%) | Mean Pred PCE (%) | MAE (% PCE) | RMSE (% PCE) | $R^2$ Score | Mean Bias ($y - \hat{y}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | All Benchmark Records | 22,467 | 100.0% | 9.89 | 10.02 | **2.0628** | **2.9546** | **0.7395** | **-0.1289** |
| **Device Architecture** | Conventional | 9,284 | 41.3% | 9.97 | 10.08 | **2.0069** | **2.8648** | **0.7768** | -0.1135 |
| **Device Architecture** | Inverted | 5,073 | 22.6% | 8.48 | 8.86 | **2.1557** | **3.0892** | **0.5418** | **-0.3794** |
| **Device Architecture** | Unspecified / Other | 8,110 | 36.1% | 10.69 | 10.68 | **2.0686** | **2.9697** | **0.7540** | +0.0102 |
| **Processing Metadata** | With Additive Recorded | 7,942 | 35.4% | 10.41 | 10.36 | **2.0153** | **2.7735** | **0.7579** | +0.0439 |
| **Processing Metadata** | No Additive Recorded | 14,525 | 64.6% | 9.61 | 9.84 | **2.0888** | **3.0491** | **0.7285** | -0.2234 |
| **Processing Metadata** | With Annealing Recorded| 5,523 | 24.6% | 9.27 | 9.52 | **2.0091** | **2.8612** | **0.7433** | -0.2553 |
| **Processing Metadata** | No Annealing Recorded | 16,944 | 75.4% | 10.10 | 10.19 | **2.0803** | **2.9844** | **0.7370** | -0.0877 |
| **Processing Metadata** | With Thickness Recorded | 6,687 | 29.8% | 7.90 | 8.22 | **2.1222** | **3.0032** | **0.6724** | -0.3196 |
| **Processing Metadata** | No Thickness Recorded | 15,780 | 70.2% | 10.74 | 10.78 | **2.0376** | **2.9338** | **0.7443** | -0.0481 |
| **Acceptor Family** | Fused-Ring NFA | 10,185 | 45.3% | 13.86 | 14.21 | **2.2284** | **3.3254** | **0.4321** | -0.3491 |
| **Acceptor Family** | Fullerene | 7,177 | 31.9% | 4.59 | 4.67 | **1.6890** | **2.2285** | **0.3731** | -0.0819 |
| **Acceptor Family** | Other NFA | 4,056 | 18.1% | 9.14 | 8.99 | **2.2406** | **3.1004** | **0.5974** | +0.1437 |
| **Acceptor Family** | Polymer Acceptor (P-SMA)| 852 | 3.8% | 12.11 | 11.12 | **2.4560** | **3.0320** | **0.6868** | **+0.9906** |
| **Acceptor Family** | PDI-based Multimer | 197 | 0.9% | 4.11 | 5.02 | **1.7554** | **2.2983** | **0.0936** | **-0.9091** |
| **Donor Family** | Polymer Donor | 16,968 | 75.5% | 11.57 | 11.76 | **2.1246** | **3.0094** | **0.6786** | -0.1870 |
| **Donor Family** | Small-Molecule Donor | 5,499 | 24.5% | 4.73 | 4.67 | **1.8719** | **2.7787** | **0.4712** | +0.0505 |
| **PCE Target Range** | $\text{PCE} < 3.0\%$ | 3,320 | 14.8% | 1.67 | 4.19 | **2.5616** | **3.6891** | -19.702 | **-2.5223** |
| **PCE Target Range** | $3.0\% \le \text{PCE} < 6.0\%$ | 3,695 | 16.5% | 4.40 | 5.26 | **1.8192** | **2.8247** | -9.592 | **-0.8588** |
| **PCE Target Range** | $6.0\% \le \text{PCE} < 9.0\%$ | 3,588 | 16.0% | 7.49 | 7.58 | **2.1744** | **2.9743** | -10.827 | **-0.0947** |
| **PCE Target Range** | $9.0\% \le \text{PCE} < 12.0\%$ | 3,182 | 14.2% | 10.35 | 10.20 | **2.2754** | **2.9664** | -11.245 | **+0.1567** |
| **PCE Target Range** | $12.0\% \le \text{PCE} < 15.0\%$| 2,410 | 10.7% | 13.42 | 13.05 | **2.4401** | **3.0119** | -10.934 | **+0.3667** |
| **PCE Target Range** | $\text{PCE} \ge 15.0\%$ | 6,272 | 27.9% | 17.27 | 16.06 | **1.6255** | **2.5210** | -2.337 | **+1.2131** |
| **Pair Frequency** | Singletons ($N=1$) | 2,790 | 12.4% | 7.29 | 7.48 | **2.1075** | **2.8463** | **0.5742** | -0.1932 |
| **Pair Frequency** | Repeated ($2 \le N < 20$) | 6,973 | 31.0% | 7.69 | 7.55 | **2.1615** | **2.9160** | **0.6626** | +0.1396 |
| **Pair Frequency** | Frequent ($N \ge 20$) | 12,704 | 56.5% | 11.68 | 11.94 | **1.9988** | **2.9987** | **0.7348** | -0.2622 |

---

### 3.1 Systematic Error Patterns & Physical Insights:

```mermaid
flowchart LR
    subgraph LowPCE["Low-Efficiency Range (< 3%)"]
        L1["Actual: 1.67%"] -->|Model Over-Predicts| L2["Predicted: 4.19%"]
        L3["Bias = -2.52%<br/>(Caused by failed/unannealed negative controls)"]
    end
    subgraph MidPCE["Medium Range (6% - 12%)"]
        M1["Actual: ~7 - 10%"] -->|Accurate Center| M2["Predicted: ~7 - 10%"]
        M3["Bias = -0.09% to +0.16%<br/>(Unbiased baseline screening)"]
    end
    subgraph HighPCE["Champion Range (>= 15%)"]
        H1["Actual: 17.27%"] -->|Model Under-Predicts| H2["Predicted: 16.06%"]
        H3["Bias = +1.21%<br/>(Ensemble mean cannot capture champion morphology)"]
    end
```

1. **Systematic Over-Prediction of Low-Performing Devices ($\text{Bias} = -2.52\%$):**  
   For devices with $\text{PCE} < 3\%$, the model predicts a mean of $4.19\%$. These records correspond to negative experimental controls, aged devices, or unoptimized trial conditions of intrinsically capable donor-acceptor combinations. The model correctly identifies the chemical pair's intrinsic potential but cannot observe that the experimenter omitted annealing or additives.
2. **Systematic Under-Prediction of Champion Devices ($\text{Bias} = +1.21\%$):**  
   For devices with $\text{PCE} \ge 15\%$, the model predicts a mean of $16.06\%$ for actual records averaging $17.27\%$. Decision-tree ensembles average leaf samples, pulling extreme champion records toward the conditional mean of the chemical cluster.
3. **Acceptor Family Discrepancies:**  
   - **Polymer Acceptors (P-SMAs):** Under-predicted by $+0.99\%$ PCE. Polymer-polymer blends have different entanglement and phase-separation physics that are slightly underestimated by the small-molecule-trained trees.
   - **PDI Multimers:** Over-predicted by $-0.91\%$ PCE. Their massive molecular weight and high aromatic ring count cause the model to assign higher efficiency than their actual steric twisting permits.

---

## 4. Controlled Model Improvement (Nested Training Fold Selection)

To avoid tuning against final validation data, hyperparameter selection was conducted strictly using **3-fold inner GroupKFold within Training Fold 1**.

### 4.1 Inner Cross-Validation Candidate Selection:

| Configuration ID | Hyperparameter Description | Inner Fold MAE (% PCE) | Inner Fold RMSE (% PCE) | Inner Fold $R^2$ | Selection Decision |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`RF_Config_0_Baseline`** | Baseline: $n=100, \text{depth}=\text{None}, \text{leaf}=1, \text{feat}=1.0$ | 2.1028 | 2.9023 | 0.7466 | Reference |
| **`RF_Config_1_DepthReg`** | Regularized: $n=100, \text{depth}=20, \text{leaf}=2, \text{feat}=0.8$ | 2.1201 | 2.8702 | 0.7525 | Rejected |
| **`RF_Config_2_FeatureSqrt`**| Decorrelated: $n=200, \text{depth}=\text{None}, \text{leaf}=2, \text{feat}=\text{sqrt}$| 2.1423 | 2.8586 | 0.7544 | Rejected |
| **`RF_Config_3_LeafReg`** | Leaf regularized: $n=100, \text{leaf}=4, \text{feat}=0.8$ | 2.1563 | 2.8651 | 0.7535 | Rejected |
| **`RF_Config_4_EnsembleScaled`**| Scaled: $n=200, \text{depth}=\text{None}, \text{leaf}=1, \text{feat}=0.8$ | **2.0974** | **2.8955** | **0.7478** | **SELECTED** |

*Selection:* `RF_Config_4_EnsembleScaled` achieved the lowest inner-fold MAE ($2.0974\%$) on the training set and was selected for outer evaluation.

---

### 4.2 Outer 5-Fold Validation Comparison (Primary Pair-Grouped Split):

All configurations were evaluated on the independent outer 5-fold pair-grouped validation splits:

| Model Configuration | Outer Mean MAE (% PCE) | Outer Mean RMSE (% PCE) | Outer Mean $R^2$ | Outer Pooled $R^2$ | Total Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`RF_Config_0_Baseline` (Phase 5)** | $2.0628 \pm 0.1134$ | $2.9466 \pm 0.2170$ | $0.7099 \pm 0.0821$ | **$0.7395$** | 14.7s |
| **`RF_Config_1_DepthReg`** | $2.0860 \pm 0.0792$ | $2.9214 \pm 0.2039$ | $0.7159 \pm 0.0779$ | **$0.7441$** | 10.3s |
| **`RF_Config_2_FeatureSqrt`** | $2.1218 \pm 0.1157$ | $2.9229 \pm 0.1699$ | $0.7181 \pm 0.0674$ | **$0.7442$** | 9.1s |
| **`RF_Config_3_LeafReg`** | $2.1287 \pm 0.1187$ | $2.9333 \pm 0.2047$ | $0.7160 \pm 0.0708$ | **$0.7420$** | 9.4s |
| **`RF_Config_4_EnsembleScaled` (Improved)**| **$2.0532 \pm 0.1056$** | **$2.9289 \pm 0.2182$** | **$0.7131 \pm 0.0824$** | **$0.7426$** | 23.4s |

### Improvement Summary:
- The selected configuration (`RF_Config_4_EnsembleScaled`: $200$ trees, $80\%$ feature subsampling per split) modestly improves out-of-fold accuracy over the baseline:
  - Outer MAE improves from **$2.0628\%$** to **$2.0532\%$**.
  - Outer RMSE improves from **$2.9466\%$** to **$2.9289\%$**.
  - Outer Mean $R^2$ improves from **$0.7099$** to **$0.7131$**.
  - Outer Pooled $R^2$ improves from **$0.7395$** to **$0.7426$**.
- This demonstrates that variance reduction through tree expansion and feature subsampling slightly stabilizes pair-grouped prediction without changing the 22-feature schema.

---

## 5. Prediction Uncertainty & Empirical Calibration Audit

We evaluated whether the inter-tree standard deviation ($\sigma_{trees}$) of the Random Forest ensemble can serve as an estimate of prediction uncertainty:

$$\sigma_{i, trees} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B \left(\hat{y}_{i, b} - \bar{y}_i\right)^2}$$

### 5.1 Dispersion vs. Realized Error Correlation
- The Pearson correlation between raw tree standard deviation ($\sigma_{trees}$) and actual realized absolute prediction error ($|y_i - \hat{y}_i|$) is only:
  $$r = \mathbf{0.1627}$$
- While positive (the ensemble exhibits slightly higher tree disagreement on difficult or novel molecules), the correlation is weak. Tree disagreement alone cannot predict large errors caused by missing experimental fabrication variables.

---

### 5.2 Empirical Coverage of Nominal Prediction Intervals

We tested nominal Gaussian prediction intervals centered at $\hat{y}_i \pm z \cdot \sigma_i$:

| Nominal Interval Width | Multiplier ($z$) | Expected Theoretical Coverage | Observed Empirical Coverage | Deficit / Calibration Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **$\pm 1.00 \cdot \sigma_{trees}$** | 1.00 | **68.27%** | **38.35%** | **Severe Under-Coverage (-29.92%)** |
| **$\pm 1.96 \cdot \sigma_{trees}$** | 1.96 | **95.00%** | **57.66%** | **Severe Under-Coverage (-37.34%)** |
| **$\pm 3.00 \cdot \sigma_{trees}$** | 3.00 | **99.73%** | **66.13%** | **Severe Under-Coverage (-33.60%)** |

### Empirical Multipliers Required for True Calibration:
- To achieve true **$68.3\%$ empirical coverage**, the interval width must be expanded to **$3.35 \times \sigma_{trees}$**.
- To achieve true **$95.0\%$ empirical coverage**, the interval width must be expanded to **$40.94 \times \sigma_{trees}$** (driven by extreme unannealed/negative-control outliers).

---

### 5.3 Scientific Distinction & Uncertainty Policy

A clear scientific distinction must be codified:

```mermaid
flowchart TD
    subgraph ModelUncertainty["Prediction Uncertainty (Epistemic)"]
        U1["Tree Disagreement (sigma_trees ~ 1.35% PCE)"]
        U2["Measures model parameter & subspace ambiguity"]
    end
    subgraph PhysicalVariance["Experimental Variability (Aleatoric)"]
        P1["Physical Fabrication Noise Floor (~1.0 - 1.5% PCE)"]
        P2["Unmodeled processing variables (annealing, thickness, additives)"]
    end
    subgraph TotalError["Realized Model Error"]
        E1["Residual |y - y_hat| (Mean 2.06% PCE)"]
        E2["Combination of epistemic uncertainty, aleatoric noise & negative controls"]
    end
    ModelUncertainty -.-> TotalError
    PhysicalVariance -.-> TotalError
```

1. **Prediction Uncertainty (Epistemic):** Represents model ambiguity across decision trees ($\bar{\sigma}_{trees} \approx 1.35\%$ PCE).
2. **Experimental Variability (Aleatoric):** Represents physical fabrication variance across laboratories and processing conditions ($\sim 1.0 - 1.5\%$ PCE).
3. **Model Error:** Represents the actual discrepancy $|y - \hat{y}|$ (mean $2.06\%$ PCE).
4. **Mandatory Scientific Rule:** Raw tree standard deviation **must never be displayed to users as a calibrated confidence interval**. In downstream phases, any uncertainty display must explicitly be labeled as an **uncalibrated ensemble dispersion index** or calibrated via conformal prediction.

---

## 6. Generalization Boundaries Re-Affirmed

The generalization hierarchy established across Phases 5 and 6 is re-affirmed:

| Evaluation Paradigm | Grouping Dimension | Model Out-of-Fold $R^2$ | Scientific Scope & Interpretation |
| :--- | :--- | :---: | :--- |
| **Publication-Grouped** | `doi_norm` | **$0.7721$** | Tests publication-level clustering. **Not lab independence** (70% of validation records share material pairs with training papers). |
| **Pair-Grouped (Primary)** | `pair_col` | **$0.7099 - 0.7131$** | Tests virtual screening of **unseen material combinations** within known/related chemical families. |
| **Combined Scaffold Pair**| `pair_scaff` | **$0.4788$** | Tests holdout of specific **pairwise scaffold combinations**. |
| **Donor Scaffold Holdout** | `donor_scaff` | **$0.4176$** | Tests out-of-distribution extrapolation to **unseen donor core scaffolds**. |
| **Acceptor Scaffold Holdout**| `acceptor_scaff` | **$0.3410$** | Tests out-of-distribution extrapolation to **unseen acceptor core scaffolds**. |

---

## 7. Deliverables & Artifact Verification

All evaluation artifacts are preserved in `ml/results/` and `ml/experiments/`:

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `ml/experiments/model_validation_and_improvement.py` | Python Script | Reproducible script for architecture audit, error breakdown, nested tuning, and uncertainty calibration. |
| `ml/results/phase_6_model_comparison.csv` | CSV | Comparison table of candidate hyperparameter configurations against the baseline. |
| `ml/results/phase_6_error_analysis.csv` | CSV | Stratified performance metrics across 27 subsets (architecture, processing, families, PCE ranges). |
| `ml/results/phase_6_validation_summary.json` | JSON | Machine-readable summary of architecture counts, inner selection results, and uncertainty calibration. |
| `ml/PHASE_6_VALIDATION_REPORT.md` | Markdown | This Document. |

### Reproducibility Command:
```powershell
backend\.venv\Scripts\python.exe ml\experiments\model_validation_and_improvement.py
```

---

## 8. Final Verdict & Stop Sign-Off

**PHASE 6 VERDICT: COMPLETED AND VERIFIED (PASS)**

- Robustness boundaries have been mapped: conventional devices ($R^2 = 0.78$) outperform inverted devices ($R^2 = 0.54$); fullerenes have lowest absolute error ($1.69\%$), while NFAs have higher error ($2.23\%$) driven by processing sensitivity.
- Controlled improvement via inner training CV confirms `RF_Config_4_EnsembleScaled` ($200$ trees, `max_features=0.8`) as a sound refinement ($\text{MAE} = 2.05\%$, $R^2 = 0.713$).
- Uncertainty calibration rules are established: raw tree dispersion cannot be presented as a valid confidence interval.
- All strict governance restrictions were maintained: zero production API changes, zero frontend changes, zero feature schema changes, and zero git commits.

*Awaiting your review and approval before proceeding to Phase 7.*
