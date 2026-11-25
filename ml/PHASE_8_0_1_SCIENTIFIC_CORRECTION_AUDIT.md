# OPV-PredictNet — Phase 8.0.1: Scientific Protocol Correction Audit

**Date**: 2026-11-24  
**Audit Scope**: Forensic review, methodological correction, and design locking for Phase 8.0 documentation prior to Phase 8.1 experiment authorization.  
**Audit Decision**: **CONDITIONALLY APPROVED FOR PHASE 8.1 (Pending Execution Safeguards)**  
**Constraint Enforcement**: Strictly read-only audit. No models trained, no calibration intervals fitted, no test metrics generated, no production code altered, no Git commits created.

---

## 1. Executive Summary & Audit Mandate

Phase 8.0 proposed extending the Phase 5 / Phase 7.1 baseline by incorporating organic photovoltaic (OPV) fabrication and processing metadata. This audit inspects the three Phase 8.0 foundation documents:
- `ml/PHASE_8_0_PROCESSING_METADATA_AUDIT.md`
- `ml/PHASE_8_0_FEATURE_STRATEGY.md`
- `ml/PHASE_8_0_EXPERIMENTAL_PROTOCOL.md`

### Required Corrections Identified
1. **Missingness Epistemology**: Previous drafts claimed missingness was proven to be Missing at Random (MAR) or Missing Not at Random (MNAR) based solely on observed efficiency differences across completeness tiers. This is statistically unproven; the observed differences are empirical associations confounded by literature era and automated extraction limits.
2. **Missingness Indicator Contradiction**: The Phase 8.0 audit cautioned against using completeness as a predictive feature, yet Configuration C introduced seven binary missingness flags. If tree models split on `is_thickness_missing`, they risk learning an unscientific shortcut: *missing processing tables $\implies$ recent literature $\implies$ high PCE*. An explicit ablation study (features with indicators vs. features without indicators) is mandatory.
3. **Deployment Preset Fallacy**: Pre-configured standard processing parameters (e.g. 1:1 ratio, 0.5% DIO, 100 nm thickness) must never silently substitute for unknown real-world processing in screening. Predictions must be partitioned into two distinct operational modes: *Molecular-Only Screening* (using M0) and *Processing-Conditional Prediction* (using M1/M2/M3).
4. **Partition Contamination**: Reusing the Phase 7.1 final test set ($N=946$) to benchmark Phase 8 would violate the non-reuse contract. A fresh, deterministic, locked Phase 8 partition (`seed=8888`) must be established, with baseline M0 recalibrated on that identical split.
5. **Categorical Terminology**: The term "embedding" was incorrectly applied to tabular categorical encodings. Encodings are strictly ordinal integer mappings or one-hot dummy matrices.

---

## 2. Audit of Missingness Methodology and Shortcut Learning

### 2.1 Distinguishing Observed Associations from Missingness Mechanisms (MCAR, MAR, MNAR)
- **Correction**: In `PHASE_8_0_PROCESSING_METADATA_AUDIT.md`, the higher mean PCE of sparse-metadata records (11.61% vs. 7.85% for rich records) was characterized as proof of MNAR/MAR.
- **Scientific Clarification**: Formally proving Missing Not at Random (MNAR) requires knowledge of the values of the unobserved variables themselves, which are absent by definition. Proving Missing at Random (MAR) requires that missingness is fully explained by observable covariates conditional on which missingness is conditionally independent of the missing values.
- **Audited Reality**: We observe an **empirical association** between metadata sparsity and higher PCE. This association is driven by two confounding historical factors:
  1. *Historical Literature Era*: Pre-2017 papers focused on fullerene blends (P3HT, PTB7) and exhaustively tabulated fabrication sweeps. Post-2019 papers focused on non-fullerene champion blends (PM6:Y6, D18) where publications frequently relegated processing details to supplementary PDF tables that automated text mining failed to capture.
  2. *Extraction Failure*: PDF extraction tools reliably captured primary summary tables (which contain donor, acceptor, and PCE) but frequently failed on complex fabrication footnotes and multi-column supplementary methods.

### 2.2 The Shortcut Learning Hazard
If machine learning models (Random Forest, LightGBM) are provided with binary indicators:
$$M_j = \mathbf{1}\{ X_j \text{ is missing} \}$$
the decision trees can split directly on $M_j$. For example:
$$\text{If } \text{is\_thickness\_missing} == 1 \implies \text{Predict High Efficiency}$$
This would constitute **shortcut learning (cheating on extraction artifacts)** rather than learning physical device relationships.

### 2.3 Resolution: Prohibition of Completeness Count & Mandatory Ablation
1. **Total Completeness Count Banned**: Using the raw number of reported fields (`n_proc_fields`) as a feature is **strictly prohibited**.
2. **Mandatory Missingness Ablation Study in Phase 8.1**:
   To isolate whether processing parameters provide genuine physical signal or merely shortcut extraction flags, Phase 8.1 must benchmark:
   - **Configuration C (Full)**: Molecular + Processing Values + Binary Missingness Flags.
   - **Configuration C-NoFlags (Ablation)**: Molecular + Processing Values **without** binary missingness flags (NaNs handled via default physical constants in RF, or native split routing in GBDT).
   - If Configuration C outperforms C-NoFlags by a large margin while collapsing on records where flags are absent, shortcut learning is confirmed and Configuration C must be disqualified.
3. **Evaluation Across Pre-Defined Completeness Tiers**:
   All candidate models must be evaluated on three held-out test tiers:
   - **Sparse Tier (0–1 fields reported)** ($N = 127$ test pairs)
   - **Partial Tier (2–4 fields reported)** ($N = 405$ test pairs)
   - **Rich Tier (5–7 fields reported)** ($N = 414$ test pairs)

---

## 3. Correction of Deployment Preset Design

Previous drafts suggested that when an end-user provides only donor and acceptor SMILES, the system could automatically apply a "champion preset" (1:1 ratio, 0.5% DIO, 100°C annealing, 100 nm thickness) to predict performance.

### 3.1 The Scientific Fallacy of Silent Imputation
Silently substituting a standard recipe when fabrication conditions are unknown is scientifically misleading:
- A user evaluating an uncharacterized material blend would receive a prediction that assumes optimal champion processing, masking the reality that the blend might perform poorly if unannealed or cast from a different solvent.
- It falsely creates the impression that the processing-aware model is predicting intrinsic molecular potential, when it is actually predicting a hypothetical conditional experiment.

### 3.2 Locked Definition: Two Distinct Operational Modes
The system architecture must enforce two mutually exclusive prediction interfaces:

```
+---------------------------------------------------------------------------------------------------------+
| Two Distinct Operational Prediction Modes                                                               |
+-------------------------------+-----------------------+---------------------+---------------------------+
| Prediction Mode               | User Inputs Required  | Model Invoked       | Output Contract           |
+-------------------------------+-----------------------+---------------------+---------------------------+
| **Mode 1: Molecular-Only**    | Donor SMILES &        | Model M0            | Unconditioned estimate of |
| **Virtual Screening**         | Acceptor SMILES only  | (Phase 5 / 8.1      | intrinsic molecular blend |
|                               |                       | 22-feat RF)         | potential; standard bounds|
+-------------------------------+-----------------------+---------------------+---------------------------+
| **Mode 2: Processing-**       | Donor SMILES,         | Model M2 / M3       | Explicitly labeled as     |
| **Conditional Prediction**    | Acceptor SMILES, plus | (Processing-Aware   | conditional on the exact  |
|                               | specific formulation  | 44-feat Predictor)  | user-specified processing |
|                               | parameters            |                     | parameters                |
+-------------------------------+-----------------------+---------------------+---------------------------+
```

### 3.3 Strict Labeling Rule for Screening Presets
If a user explicitly requests a preset query (e.g. *"Evaluate this blend under standard champion inverted processing"*):
- The UI and API contract must explicitly tag the output:  
  `Prediction: 14.2% PCE [Conditional on Formulation Preset: 1:1 ratio, CF/0.5% DIO, 100°C, 100nm, inverted]`.
- Unknown processing parameters in historical datasets must **never** be back-filled with preset values.

---

## 4. Exact Feature Registry Audit

The feature spaces of M0, M1, M2, and M3 were audited to eliminate mathematical ambiguity and clarify categorical encodings:

### 4.1 Mathematical Feature Count Reconciliation
- **M0 (Molecular Baseline)**: Exactly **22 features**.
- **M1 (Naive Processing Integration)**: Exactly $22 + 10 = \mathbf{32\text{ features}}$.
  - 22 molecular descriptors.
  - 3 continuous processing parameters: `da_ratio_numeric`, `thickness_nm`, `annealing_temp_c`.
  - 5 categorical integer-encoded parameters: `solvent_code`, `additive_code`, `device_architecture_code`, `htl_code`, `etl_code`.
  - 2 naive binary flags: `has_additive`, `has_thermal_annealing`.
- **M2 & M3 (Principled Processing Architecture)**: Exactly $22 + 4 + 7 + 7 + 4 = \mathbf{44\text{ features}}$.
  - 22 molecular descriptors.
  - 4 normalized continuous features: `donor_blend_fraction`, `thickness_log10`, `annealing_temp_norm`, `additive_concentration_pct`.
  - 7 explicit missingness indicators: `missing_da_ratio`, `missing_thickness`, `missing_annealing`, `missing_solvent`, `missing_additive`, `missing_architecture`, `missing_interlayers`.
  - 7 domain-aware physical regime indicators: `regime_as_cast_room_temp`, `regime_thermal_annealed`, `regime_solvent_additive_used`, `regime_chlorinated_solvent`, `regime_green_solvent`, `regime_inverted_stack`, `regime_conventional_stack`.
  - 4 physical interaction features: `thickness_per_donor_mw`, `solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`.

### 4.2 Categorical Encoding Precision
- The Phase 8.0 text loosely used the word "embedding." **This is corrected**:
  - Tree-based models (Random Forest, LightGBM) in Phase 8 do not use dense neural learned embedding layers.
  - Encodings are strictly **ordinal integer index mappings** (for GBDT native categorical support) or **explicit binary dummy columns** (0 or 1).

### 4.3 Semantic Audit of Specific Processing Fields
1. **Donor:Acceptor Ratio Semantics**:
   - Raw strings (e.g. `1:1`, `1:1.2`, `1:1.5 w/w`) represent weight/mass ratios $D:A$.
   - The transformation:
     $$f_{\text{donor}} = \frac{D}{D + A}$$
     normalizes the parameter into $(0, 1)$, with $0.5$ representing 1:1 blend.
   - Non-standard ternary mixtures (`1:1:0.2`) and ranged text (`1:2-1:2.5`) cannot be converted to a binary scalar; they must be mapped to NaN and flagged with `missing_da_ratio = 1`.
2. **Additive Concentration Basis**:
   - 93% of reported additive concentrations are in volume percentage (`vol%` or `v/v`).
   - Parsed into continuous float `additive_concentration_pct` bounded to $[0.0, 10.0]\%$.
   - Missing concentration is mapped to $0.0\%$ with `missing_additive = 1`.
3. **Active Layer Thickness Positivity**:
   - Logarithmic transformation $\log_{10}(\text{thickness})$ requires $\text{thickness} > 0$.
   - Exactly 4 benchmark records report non-positive thickness ($\le 0.0$).
   - Rule: Any thickness $\le 0.0$ or $>1000\text{ nm}$ is physically invalid and must be mapped to NaN prior to log transformation. Valid range: $[15\text{ nm}, 600\text{ nm}]$.
4. **Annealing Temperature vs. Room-Temperature Processing**:
   - Temperatures $\le 35^\circ\text{C}$ (e.g. 0°C, 25°C) signify room-temperature / as-cast processing. They must set `regime_as_cast_room_temp = 1` and `regime_thermal_annealed = 0`.
   - Temperatures in $[40^\circ\text{C}, 250^\circ\text{C}]$ represent thermal annealing (`regime_thermal_annealed = 1`).
   - Temperatures $> 250^\circ\text{C}$ represent sol-gel sintering, not organic active layer annealing, and must be mapped to NaN.
5. **Additive Absence vs. Unknown Status**:
   - Only 267 records contain explicit keywords for absence (`none`, `neat`, `without`, `w/o`, `0`). These set `regime_solvent_additive_used = 0` with `missing_additive = 0`.
   - For all other 14,526 null records, status is unknown (`missing_additive = 1`). We do not infer that null means neat blend.
6. **Redundancy Between `device_structure` and `device_type`**:
   - `device_structure` contains 4,096 raw layer strings. It is highly redundant with `device_type`, `etl_canonical`, and `htl_canonical`.
   - Rule: Do not use raw `device_structure` strings directly. Use `device_type` for architecture classification (`conventional` vs. `inverted`), and use `etl_canonical` and `htl_canonical` for interlayer work functions.

---

## 5. Uncontaminated Phase 8 Validation Protocol

### 5.1 Prohibition of Phase 7.1 Test Set Reuse
Reusing the Phase 7.1 test set ($N=946$) to benchmark Phase 8 would violate the non-reuse principle, as decisions in Phase 7.1.3 and 7.1.4 were made after observing test metrics on those specific pairs.

### 5.2 Deterministic Phase 8 Pair Partition (`seed=8888`)
An independent, locked 60 / 20 / 20 pair-group partition of the 4,730 material-name pairs was verified:
- **Partition Random Seed**: `8888`
- **$D_{\text{train}}$ Partition**: Exactly **2,838 material-name pairs** (14,704 raw records).
- **$D_{\text{cal}}$ Partition**: Exactly **946 material-name pairs** (3,349 raw records; Alternative B sampled to exactly **946 records**, `cal_seed=8201`).
- **$D_{\text{test}}$ Partition**: Exactly **946 material-name pairs** (4,414 raw records; Alternative B sampled to exactly **946 records**, `test_seed=8202`).
- **Mutual Exclusivity**:
  $$\text{Pairs}(D_{\text{train}}) \cap \text{Pairs}(D_{\text{cal}}) = \emptyset, \quad \text{Pairs}(D_{\text{train}}) \cap \text{Pairs}(D_{\text{test}}) = \emptyset, \quad \text{Pairs}(D_{\text{cal}}) \cap \text{Pairs}(D_{\text{test}}) = \emptyset$$
- **Test Independence**: The Phase 8 test partition shares only 184 pairs (19.5%) with the Phase 7.1 test partition, consistent with an independent random draw.

### 5.3 Parallel Calibration of Baseline M0 on Phase 8 Partitions
- To ensure unbiased head-to-head comparison, the molecular baseline **M0 will be newly fitted on $D_{\text{train}}$ (Phase 8) and calibrated on $D_{\text{cal}}$ (Phase 8)**.
- Models M0, M1, M2, and M3 will be evaluated on the **identical, frozen Phase 8 test records** ($N_{\text{test}} = 946$).
- The historical Phase 7.1 unnormalized width ($8.84\text{ PCE pp}$) is treated as background context; the primary point of comparison is the concurrent M0 model on the Phase 8 partition.

---

## 6. Uncertainty Research and Acceptance Gates

1. **Pre-Registered Acceptance Gates (Unchanged)**:
   - 80% Marginal Coverage Gate: $[77.0\%, 83.0\%]$.
   - 90% Marginal Coverage Gate: $[88.0\%, 92.5\%]$.
   - 95% Marginal Coverage Gate: $[93.5\%, 96.5\%]$.
   - 90% Median Interval Width Gate: **Maximum 5.80 PCE pp**.
2. **Conditional / Subgroup Guarantees Disclaimed**:
   - Marginal split conformal prediction guarantees coverage on average over the population of material pairs. It does **not** provide mathematical guarantees for specific subgroups.
   - Subgroup coverages (PCE tiers, completeness tiers, architectures) are empirical diagnostic audits, not formal theoretical guarantees.
3. **High-Prediction Subgroup Reporting**:
   - High-prediction cells ($\hat{y} \ge 14.0\%$) will be reported descriptively (coverage, count, bias, median width).
   - Independent-binomial significance tests on conditional slices are avoided due to molecule-sharing network covariance.

---

## 7. Git and Artifact Integrity Ledger

### 7.1 Untracked Files Status
A literal `git status --short` confirms that all project research documentation and experiment files remain strictly untracked:
- `ml/DATASET_DECISION.md` through `ml/PHASE_7_1_4_DETERMINISM_AND_CHAMPION_AUDIT.md` (all historical research documents preserved intact).
- `ml/experiments/phase_7_1_grouped_conformal.py` (preserved).
- `ml/results/phase_7_1_*` (all Phase 7.1 result files preserved without modification).
- Zero tracked repository files have been modified. Zero Git commits created.

### 7.2 Authorized Phase 8.1 Experiment Deliverables
When authorized, Phase 8.1 is permitted to create strictly:
1. `ml/experiments/phase_8_1_processing_aware_models.py` (experiment script).
2. `ml/results/phase_8_1_partition_metadata.json`
3. `ml/results/phase_8_1_selected_records.csv`
4. `ml/results/phase_8_1_point_metrics.json`
5. `ml/results/phase_8_1_calibration_quantiles.json`
6. `ml/results/phase_8_1_test_predictions_and_intervals.csv`
7. `ml/results/phase_8_1_final_test_metrics.json`
8. `ml/results/phase_8_1_subgroup_metrics.json`
9. `ml/PHASE_8_1_PROCESSING_IMPROVEMENT_REPORT.md`

---

## 8. Final Audit Decision

### **CONDITIONALLY APPROVED FOR PHASE 8.1**

### Prerequisites for Phase 8.1 Experiment Authorization:
1. Complete the exact Feature Registry in `ml/PHASE_8_0_1_FEATURE_REGISTRY.md`.
2. Lock the updated validation protocol in `ml/PHASE_8_0_1_LOCKED_VALIDATION_PROTOCOL.md`.
3. Synchronize Phase 8.0 documents to reflect the corrected missingness epistemology, two-mode prediction contract, and Phase 8 partition seeds.
4. Obtain explicit authorization from the user before executing `phase_8_1_processing_aware_models.py`.

**Audit completed. Proceed to authoring Feature Registry and Locked Validation Protocol.**
