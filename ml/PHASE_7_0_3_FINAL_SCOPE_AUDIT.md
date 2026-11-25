# OPV-PredictNet — Phase 7.0.3: Final Statistical Scope and Consistency Audit

**Author / Auditor:** Antigravity Agent  
**Date:** November 24, 2026  
**Status:** Completed Final Scope Audit (Pre-Implementation Lock)  
**Target Model:** Phase 5 Baseline Random Forest Regressor ($n=100, \text{depth}=\text{None}, \text{split}=2, \text{leaf}=1, \text{features}=1.0$)  
**Dataset:** OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records, 4,730 unique canonical material pair groups)  
**Scope:** Resolution of statistical scope, within-pair feature variance analysis, separation of Tier 1 / Tier 2 / Tier 3 validation protocols, qualified finite-sample coverage bounds, literal Git repository audit, and pre-implementation freeze.

---

## 1. Executive Summary & Audit Mandate

Phase 7.0.3 establishes the final statistical scope, coverage boundaries, and repository accounting required before authorizing Phase 7.1 implementation. In strict compliance with the Phase 7.0.3 charter:
1. **Documentation-Only Enforcement:** No ML models were fitted, no prediction intervals were generated, no test split labels were accessed or audited, and no Git commits were created.
2. **Corrected Coverage Language:** The Tier 2 coverage guarantee has been stripped of misleading "protocol-uniform" phrasing and restricted to its precise mathematical estimand: marginal coverage for a randomly selected experimental record from a randomly selected new material pair under exchangeable compound sampling.
3. **Empirical Feature Variance Discovery:** Audit of the 22-feature production pipeline revealed that **21 of 22 molecular features vary across records within the same canonical material pair** due to multiple SMILES representations (monomer vs. repeat unit, oligomeric degree, side-chain alkyl isomers). The non-conformity score formula is formally corrected to use the selected record's actual feature vector $x_{g, j_g}$.
4. **Decoupled Validation Protocols:** Tier 1 (known-pair record holdout) is formally decoupled from Tier 2 (unseen-pair holdout). Tier 1 is explicitly designated as **UNVALIDATED AND OUTSIDE INITIAL SCOPE** for Phase 7.1, as the 60/20/20 pair-disjoint design cannot evaluate known-pair re-fabrication. Tier 3 (scaffold extrapolation) is confirmed as a separate research evaluation.
5. **Qualified Finite-Sample Bounds:** The theoretical upper bound $1 - \alpha + \frac{1}{n_{\text{cal}} + 1}$ is qualified by the requisite no-tie condition; the effect of tied scores causing overcoverage is documented. The infinite cutoff is recognized as a theoretically valid treatment for $k > n_{\text{cal}}$, while `InsufficientCalibrationDataError` is codified as an explicit engineering policy.
6. **Literal Repository Status:** `git status --short` is reported literally, accounting for all untracked research markdown files, experiment scripts, and results, confirming zero modified tracked files.

---

## 2. Corrected Tier 2 Random-Record Coverage Guarantee

### 2.1 Formal Coverage Statement
The conformal coverage claim for Tier 2 Alternative B (single randomly sampled record per pair) is formally defined as:

> *"The intended guarantee is marginal coverage for a randomly selected experimental record from a randomly selected new material pair, provided calibration and future material-pair clusters are exchangeable under the same pair-selection and within-pair record-sampling mechanisms."*

Mathematically:
$$\mathbb{P}\left(Y_{g_{\text{new}}, \text{random}} \in C_{\alpha}^{\text{single}}(x_{g_{\text{new}}, \text{random}})\right) \ge 1 - \alpha$$

### 2.2 Explicit Documentation of Sampling Mechanisms & Boundaries

$$\begin{array}{l|p{11cm}}
\hline
\textbf{Sampling Dimension} & \textbf{Rigorous Statistical Specification} \\
\hline
\textbf{Material-Pair Population} & The sampling population $\mathcal{G}$ consists of donor–acceptor material combinations curated from the scientific literature into the OPV-DB v1.1.0 Strict Molecular Benchmark. A candidate query pair $g_{\text{new}}$ is assumed to be an exchangeable draw from the same chemical distribution of literature-synthesized OPV donor–acceptor pairs. \\
\hline
\textbf{Within-Pair Record Selection} & For each pair cluster $g$ containing $n_g$ observed records $\mathcal{R}_g = \{r_{g, 1}, \dots, r_{g, n_g}\}$, an integer index $j_g$ is drawn uniformly at random:
$$P(j_g = i) = \frac{1}{n_g}, \quad \forall i \in \{1, \dots, n_g\}$$ \\
\hline
\textbf{Uniformity over Observed Records Only} & Selection is strictly uniform over the **observed database records**, NOT over the unobserved physical space of experimental processing parameters (spin speeds, solvent ratios, annealing temperatures) or literature publications. Due to publication bias, literature records are strongly skewed toward high-efficiency devices, champion cells, and optimized donor:acceptor blend ratios. \\
\hline
\textbf{Target Independence} & The record index $j_g$ is drawn using a pseudo-random index generator initialized with a pre-registered seed (`seed = 42`). The draw is strictly independent of the target value $y_{g, i} = \text{PCE}_{g, i}$, the model prediction $\hat{\mu}$, and the residual magnitude. \\
\hline
\textbf{No Protocol-Specific Coverage Guarantee} & **This guarantee does NOT apply to a specific user-chosen experimental protocol.** If an experimentalist fabricates the new pair using a deliberate non-random protocol (e.g., choosing an unannealed control device, an untested high-boiling solvent additive, or an inverted architecture when literature predominately used conventional), the resulting outcome $y_{\text{user}}$ is NOT an exchangeable draw from the empirical within-pair distribution. \\
\hline
\textbf{No Arbitrary Laboratory Guarantee} & **This guarantee does NOT apply to an arbitrary future laboratory distribution.** Differences in local glovebox environmental controls (oxygen/water ppm), solar simulator spectral mismatch, or spin-coating dynamics constitute systematic distribution shifts that violate calibration exchangeability. \\
\hline
\end{array}$$

**Conclusion:** Protocol-specific and laboratory-conditional coverage claims are explicitly rejected and prohibited. The conformal prediction interval bounds a typical literature realization drawn under the identical compound sampling process.

---

## 3. Empirical Feature Variance Analysis: Record-Level vs. Pair-Level Descriptors

### 3.1 Empirical Findings on the 22-Feature Production Pipeline
An exhaustive audit of all 22,467 records and 4,730 unique canonical material pairs (`donor_canonical::acceptor_canonical`) in the OPV-DB v1.1.0 Strict Molecular Benchmark was conducted using `ml/src/features/descriptors.py`:

```
Dataset Analysis Summary:
├── Total Records: 22,467
├── Total Unique Canonical Material Pairs: 4,730
├── Total Unique (donor_smiles, acceptor_smiles) Pairs: 4,832
├── Pairs with Multiple Donor SMILES: 174 pairs (max 6 SMILES per pair)
├── Pairs with Multiple Acceptor SMILES: 107 pairs (max 5 SMILES per pair)
└── Molecular Feature Invariance across Records of Same Pair:
    ├── Features Strictly Constant: 1 / 22 (acceptor_hbd)
    └── Features Varying Within Pairs: 21 / 22 (95.5% of features vary!)
```

### 3.2 Breakdown of Feature Variation Within Pairs

| Feature Name | Pairs with Varying Values | Max Within-Pair Std Dev | Primary Physical / Chemical Cause |
| :--- | :---: | :---: | :--- |
| `donor_mw` | 141 pairs | $999.63\text{ Da}$ | Polymer repeat unit representation (monomer vs dimer vs oligomer). |
| `donor_logp` | 142 pairs | $22.90$ | Varying repeat unit count or solubilizing alkyl chain lengths. |
| `donor_heavy_atoms` | 141 pairs | $58.00\text{ atoms}$ | Extended repeat-unit representations in literature curations. |
| `donor_fraction_csp3` | 141 pairs | $0.324$ | Side-chain truncation vs fully detailed alkyl substituents. |
| `donor_rotatable_bonds` | 44 pairs | $21.02$ | Alkyl chain flexibility variations in curated SMILES. |
| `donor_tpsa` | 28 pairs | $173.64\text{ \AA}^2$ | Heteroatom count shifts across repeat unit notations. |
| `donor_aromatic_rings` | 30 pairs | $11.31$ | Number of conjugated core units in repeat string. |
| `donor_hba` | 33 pairs | $18.38$ | Additional acceptor moieties in oligomeric representation. |
| `donor_hbd` | 1 pair | $6.26$ | Rare end-group protonation / phenolic end-cap difference. |
| `acceptor_mw` | 24 pairs | $277.72\text{ Da}$ | Non-fullerene acceptor alkyl chain isomerism (e.g. 2-ethylhexyl vs 4-octyl). |
| `acceptor_logp` | 26 pairs | $7.56$ | Branching and side-chain length differences. |
| `acceptor_fraction_csp3`| 23 pairs | $0.189$ | Varying side-chain structures for identical acceptor names. |
| `acceptor_heavy_atoms` | 23 pairs | $19.80\text{ atoms}$ | Side-chain variations across publications. |
| `acceptor_rotatable_bonds`| 17 pairs | $18.38$ | Flexible alkyl chain differences. |
| `acceptor_tpsa` | 5 pairs | $64.62\text{ \AA}^2$ | End-capping cyano / rhodanine variations in literature. |
| `acceptor_aromatic_rings`| 5 pairs | $4.24$ | Fused-ring core notation differences. |
| `acceptor_hba` | 4 pairs | $1.41$ | Cyano / carbonyl count differences. |
| `acceptor_hbd` | **0 pairs** | **0.00** | Strictly zero (both donors and NFAs lack hydrogen bond donors). |
| `mw_diff` | 161 pairs | $697.21\text{ Da}$ | Composite difference reflects donor and acceptor variation. |
| `logp_diff` | 163 pairs | $14.46$ | Composite difference reflects donor and acceptor variation. |
| `tpsa_total` | 33 pairs | $173.64\text{ \AA}^2$ | Composite sum reflects structural variation. |
| `aromatic_rings_total` | 35 pairs | $11.31$ | Composite sum reflects structural variation. |

### 3.3 Missing Data and Zero-Imputation Check
- **Missing Descriptors:** 0 NaNs and 0 Infs across all 22,467 records.
- **Imputation:** No zero-imputation is applied. Every descriptor is physically computed from the RDKit `Mol` object corresponding to that specific record's curated `donor_smiles` and `acceptor_smiles`.

### 3.4 Mathematical Score Formulation Correction
Because features vary within canonical pairs, defining the non-conformity score with an assumed common pair vector $x_g$ is invalid. The score function is formally updated to evaluate the selected record's actual feature vector $x_{g, j_g}$:

$$S_g^{\text{single}} = \frac{|y_{g, j_g} - \hat{\mu}(x_{g, j_g})|}{\hat{\sigma}(x_{g, j_g}) + \epsilon}$$

### 3.5 Implications for Virtual Screening Applications
- **Inference Requirement:** When querying the model for an unobserved material pair in virtual screening, the user must provide the **exact chemical SMILES** for both donor and acceptor.
- **Structural Specificity:** If a polymer donor can be represented as a monomer repeat unit, dimer, or end-capped oligomer, each representation generates a distinct feature vector $x$, resulting in distinct point predictions $\hat{\mu}(x)$ and calibrated prediction intervals $[\hat{\mu}(x) - \hat{q}\hat{\sigma}(x), \hat{\mu}(x) + \hat{q}\hat{\sigma}(x)]$.
- **Calibrator Integrity:** Evaluating $x_{g, j_g}$ ensures that the calibration non-conformity score precisely measures the model's error on the specific chemical structure paired with that experimental device outcome.

---

## 4. Rigorous Separation of Tier 1, Tier 2, and Tier 3 Validation Protocols

A pair-disjoint split tests generalization to unseen material pairs. It does **not** test prediction of new devices made from known material pairs, nor does it guarantee validity on novel chemical scaffolds. The protocols are formally decoupled:

```
Validation Protocol Decoupling in OPV-PredictNet
│
├── Tier 1: Known-Pair Record Holdout (Device Re-fabrication)
│   ├── Target: Predicting a new device outcome for a pair already in the database
│   ├── Data Split: Record-level partition (material pairs shared between train and test)
│   ├── Coverage Claim: P(Y_new ∈ C(X_new)) ≥ 1 - α for re-fabrications of known pairs
│   └── Phase 7.1 Status: UNVALIDATED & OUTSIDE INITIAL SCOPE
│
├── Tier 2: Unseen-Pair Holdout (Virtual Screening — Primary Scope)
│   ├── Target: Predicting experimental PCE of an unseen donor-acceptor combination
│   ├── Data Split: Strict 60/20/20 pair-disjoint partition (zero pair overlap)
│   ├── Calibration Unit: Single randomly sampled record per pair (Alternative B)
│   ├── Coverage Claim: Marginal coverage for a random literature record of a new pair
│   └── Phase 7.1 Status: PRIMARY VALIDATION SCOPE
│
└── Tier 3: Scaffold Extrapolation (Novel Conjugated Chemotypes)
    ├── Target: Predicting pairs containing core frameworks absent from training
    ├── Data Split: Bemis-Murcko scaffold disjoint evaluation (Gate 5)
    ├── Coverage Claim: None (distribution-free conformal exchangeability fails)
    └── Phase 7.1 Status: SEPARATE EXPLORATORY RESEARCH AUDIT
```

### 4.1 Detailed Protocol Specifications

$$\begin{array}{l|p{4.5cm}|p{4.5cm}|p{4.5cm}}
\hline
\textbf{Protocol Feature} & \textbf{Tier 1: Known-Pair Holdout} & \textbf{Tier 2: Unseen-Pair Holdout} & \textbf{Tier 3: Scaffold Extrapolation} \\
\hline
\textbf{Operational Question} & "If I make another device with PM6:Y6, what will its PCE be?" & "If I synthesize a new donor and combine it with Y6, what will its PCE be?" & "If I design a radically new chromophore backbone, what will its PCE be?" \\
\hline
\textbf{Pair Overlap} & **Permitted:** Same pair exists in train, cal, and test. & **Strictly Zero:** No material pair in test appears in train or cal. & **Zero Pair & Zero Scaffold:** Core frameworks absent from train. \\
\hline
\textbf{Calibration Unit} & Individual device records $(x_i, y_i)$. & Single record per pair cluster $(x_{g, j_g}, y_{g, j_g})$. & Cluster-level non-conformity scores under scaffold shift. \\
\hline
\textbf{Exchangeability} & Device records exchangeable under standard experimental draws. & Material-pair clusters and within-pair draws exchangeable. & **Exchangeability systematically violated.** \\
\hline
\textbf{Coverage Target} & $P(Y_{\text{new}} \in C(X_{\text{new}})) \ge 1 - \alpha$ for known pairs. & Marginal coverage for a random record of a new pair. & Diagnostic only: audit coverage degradation (Gate 5). \\
\hline
\textbf{Phase 7.1 Scope} & **UNVALIDATED / OUT OF SCOPE** & **PRIMARY VALIDATION TARGET** & **DIAGNOSTIC RESEARCH AUDIT** \\
\hline
\end{array}$$

---

## 5. Qualified Finite-Sample Coverage Bounds and Edge Cases

### 5.1 Lower Bound
Under exchangeability of calibration and test observations and using the finite-sample adjusted rank $k = \lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil$, split conformal prediction guarantees that the marginal coverage satisfies:
$$\mathbb{P}\left(Y_{\text{new}} \in C(X_{\text{new}})\right) \ge 1 - \alpha$$
This lower bound holds universally, regardless of whether non-conformity scores contain ties.

### 5.2 Upper Bound Qualification & The No-Tie Condition
The standard split conformal upper bound:
$$\mathbb{P}\left(Y_{\text{new}} \in C(X_{\text{new}})\right) \le 1 - \alpha + \frac{1}{n_{\text{cal}} + 1}$$
**strictly requires that the non-conformity scores $\{S_1, \dots, S_{n_{\text{cal}}}, S_{n+1}\}$ are almost surely distinct (no-tie condition)**. This assumption typically holds when the data distribution is continuous and the non-conformity function produces continuous outputs.

**Effect of Tied Scores:**
In molecular informatics benchmarks, non-conformity scores can contain discrete ties due to identical computed descriptors (e.g. isomers with identical 22-feature vectors), identical rounded target values in publications (e.g. PCE reported to one decimal place), or clamped predictions. When ties occur:
$$P(S_i = \hat{q}_{1-\alpha}) > 0$$
The discrete calibration quantile includes all tied calibration scores, placing excess probability mass in the interval. As a consequence:
$$\mathbb{P}\left(Y_{\text{new}} \in C(X_{\text{new}})\right) \text{ may strictly exceed } 1 - \alpha + \frac{1}{n_{\text{cal}} + 1}$$
*Empirical coverage exceeding the theoretical upper bound is conservative overcoverage, never undercoverage, and does not violate the fundamental validity guarantee.*

### 5.3 Quantile Edge-Case Protocol ($k > n_{\text{cal}}$)
When the requested confidence level $1 - \alpha$ and calibration sample size $n_{\text{cal}}$ result in $k = \lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil > n_{\text{cal}}$ (e.g., $n_{\text{cal}} < \lceil 1/\alpha \rceil - 1$):
- **Theoretical Conformal Treatment:** In standard conformal theory (Vovk et al. 2005, Lei et al. 2018), the $(n_{\text{cal}} + 1)$-th order statistic of an exchangeable sequence is defined as $+\infty$. Returning an infinite interval $(-\infty, +\infty)$ is a theoretically valid finite-sample treatment that trivially guarantees $100\% \ge 1 - \alpha$ coverage.
- **Engineering Implementation Policy:** Returning an infinite interval in an automated virtual screening pipeline silently passes vacuous intervals to downstream filters, causing silent operational failures. Therefore, as an explicit **implementation policy**, the `GroupConformalCalibrator` raises an explicit `InsufficientCalibrationDataError`. Silent fallback to unadjusted empirical percentiles (e.g. `np.percentile`) is strictly prohibited.

---

## 6. Literal Repository Status Audit

In compliance with Phase 7.0.3 requirements, the literal output of `git status --short` was executed and is reported below.

### 6.1 Literal Command Output
```text
$ git status --short
?? ml/DATASET_DECISION.md
?? ml/DATASET_RESEARCH_REPORT.md
?? ml/DATASET_SCHEMA_REPORT.md
?? ml/DATA_QUALITY_ASSESSMENT.md
?? ml/PHASE_4_1_SCIENTIFIC_AUDIT.md
?? ml/PHASE_4_2_ACQUISITION_REPORT.md
?? ml/PHASE_4_2_DATA_READINESS.md
?? ml/PHASE_4_2_FEATURE_AUDIT.md
?? ml/PHASE_4_2_STRUCTURE_AUDIT.md
?? ml/PHASE_4_3_PRETRAINING_REVIEW.md
?? ml/PHASE_5_1_INDEPENDENT_AUDIT.md
?? ml/PHASE_5_BASELINE_REPORT.md
?? ml/PHASE_6_1_1_FINAL_AUDIT.md
?? ml/PHASE_6_1_2_FINAL_STATISTICAL_CORRECTIONS.md
?? ml/PHASE_6_1_NESTED_VALIDATION_REPORT.md
?? ml/PHASE_6_VALIDATION_REPORT.md
?? ml/PHASE_7_0_1_CONFORMAL_AUDIT.md
?? ml/PHASE_7_0_1_FINAL_VALIDATION_DESIGN.md
?? ml/PHASE_7_0_1_GROUPED_COVERAGE_DESIGN.md
?? ml/PHASE_7_0_2_FINAL_DESIGN_CORRECTIONS.md
?? ml/PHASE_7_CALIBRATION_DESIGN.md
?? ml/PHASE_7_UNCERTAINTY_RESEARCH.md
?? ml/PHASE_7_VALIDATION_PROTOCOL.md
?? ml/experiments/
?? ml/results/
```

### 6.2 Working Tree Accounting & Categorization
The working tree contains **zero modified tracked files** (no ` M` or `M ` entries) and **zero staged files**. However, because untracked files are present, **the working tree is NOT clean in the Git sense**. Every untracked file is classified and accounted for below:

1. **Phase Research & Audit Markdown Documents (23 files in `ml/`):**
   - Phase 4.1–4.3: Data readiness, dataset selection, schema, feature, and pre-training review documents.
   - Phase 5–5.1: Baseline training report and independent audit report.
   - Phase 6–6.1.2: Validation, nested CV, statistical corrections, and annealing metadata reports.
   - Phase 7–7.0.3: Uncertainty research, calibration design, validation protocols, conformal audits, and scope reviews.
2. **Experiment Execution Scripts (3 files in `ml/experiments/`):**
   - `ml/experiments/train_baseline.py` (Phase 5 baseline training script)
   - `ml/experiments/model_validation_and_improvement.py` (Phase 6 exploratory validation script)
   - `ml/experiments/nested_cross_validation.py` (Phase 6.1 nested cross-validation script)
3. **Reproducible Experimental Results (10 files in `ml/results/`):**
   - Phase 5: `phase_5_fold_metrics.csv`, `phase_5_predictions.csv`, `phase_5_summary.json`, `phase_5_1_audit_summary.json`
   - Phase 6: `phase_6_model_comparison.csv`, `phase_6_error_analysis.csv`, `phase_6_validation_summary.json`
   - Phase 6.1: `phase_6_1_nested_fold_metrics.csv`, `phase_6_1_paired_comparison.csv`, `phase_6_1_summary.json`
4. **Temporary Update Scripts Location:**
   - All scratch scripts used for metric audits, bootstrap evaluations, and text updates are stored in `<appDataDir>/brain/<conversation-id>/scratch/` outside the repository tree, preventing clutter in the project root.
5. **Unintended or Orphaned Files:**
   - None. All files in `ml/` represent authorized, deliberate deliverables from approved project phases. No files should be deleted.

---

## 7. Remaining Limitations & Explicit Phase 7.1 Scope

### 7.1 Explicit Scope Boundaries for Phase 7.1
When authorized, Phase 7.1 will implement and evaluate **only**:
1. **Model:** The frozen Phase 5 Baseline Random Forest point predictor ($n=100, \text{depth}=\text{None}, \text{split}=2, \text{leaf}=1, \text{features}=1.0$) on 22 molecular descriptors.
2. **Data Partitioning:** A strict 60/20/20 donor–acceptor pair-disjoint split ($D_{\text{train}}, D_{\text{cal}}, D_{\text{test}}$).
3. **Scale Selection:** 5-fold grouped internal CV on $D_{\text{train}}$ comparing Unnormalized ($\hat{\sigma}_0 = 1.0$) vs. Frozen Tree Disagreement ($\hat{\sigma}_{\text{trees}}(x) + 0.10$). Winning scale locked before accessing $D_{\text{cal}}$.
4. **Calibration Score:** Tier 2 Alternative B using a single random record per pair ($j_g \sim \text{Uniform}(\{1, \dots, n_g\})$, seed 42) with its actual feature vector $x_{g, j_g}$.
5. **Quantiles:** Finite-sample ranks $k = \lceil (N_{\text{cal}} + 1)(1 - \alpha) \rceil$ for $\alpha \in \{0.20, 0.10, 0.05\}$.
6. **Validation:** Evaluating locked prediction intervals against the 5 pre-registered gates on $D_{\text{test}}$, testing raw intervals first, followed by lower-bound clipped intervals ($C \ge 0.0\%$).

### 7.2 Explicitly Excluded from Phase 7.1 Scope
1. **Tier 1 (Known-Pair Holdout):** Unvalidated and excluded.
2. **Hyperparameter Tuning:** Baseline model hyperparameters are frozen; no tuning will occur.
3. **Feature Engineering:** The 22-feature schema remains strictly frozen.
4. **Test-Set Tuning:** Calibration parameters will not be adjusted based on $D_{\text{test}}$ outcomes.
5. **Backend / Frontend Integration:** No changes to `backend/` or `frontend/` will occur during Phase 7.1.

---

## 8. Verification of Phase Restrictions & Gate Sign-Off

The following restrictions were strictly enforced throughout Phase 7.0.3:
- [x] **No models fitted.**
- [x] **No prediction intervals generated.**
- [x] **No final test labels inspected.**
- [x] **No production backend or frontend code modified.**
- [x] **Frozen Phase 5 baseline unchanged.**
- [x] **No deployment or integration attempted.**
- [x] **No Git commits created.**

Phase 7.0.3 is complete. The design, scope, and statistical boundaries are fully audited and frozen. We await your explicit authorization before proceeding to **Phase 7.1: Conformal Calibration Implementation and Validation**.
