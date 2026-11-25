# OPV-PredictNet — Phase 7.1.3: Reproducibility and Final Acceptance Audit

**Date**: 2026-11-24  
**Audit Scope**: Final, independent reproducibility verification, data-isolation confirmation, and pre-registered acceptance gate evaluation of Phase 7.1 Grouped Conformal Prediction.  
**Audit Decision**: **CONDITIONALLY ACCEPTED WITH DOCUMENTED LIMITATIONS**  
**Constraint Enforcement**: No models fitted for production, no calibration quantiles modified, Phase 5 baseline preserved, Phase 7.1 methodology preserved, all existing artifacts intact, zero Git commits created.

---

## 1. Executive Summary & Audit Context

Following the numerical reconciliation in Phase 7.1.2, this audit provides the definitive reproducibility assessment and acceptance evaluation for Phase 7.1 Grouped Conformal Calibration.

### Key Audit Conclusions
1. **Pipeline Reproducibility**: The Phase 7.1 execution was reproduced from scratch using the raw benchmark dataset into an isolated directory. Across all partition metadata, selected record indices, inner cross-validation tables, calibration quantiles, test predictions, and subgroup slices, the reproduced outputs match the saved artifacts in `ml/results/` to machine floating-point precision ($\le 7.11 \times 10^{-15}$). A single 1-bit boundary difference ($1.78 \times 10^{-15}$) in multithreaded tree summation (`n_jobs=-1`) at row index 329 is fully characterized.
2. **Data Isolation Boundaries**: Confirmed. $D_{\text{train}}$, $D_{\text{cal}}$, and $D_{\text{test}}$ pair groups are strictly disjoint. Point-model fitting and scale selection accessed $D_{\text{train}}$ alone; calibration quantiles accessed $D_{\text{cal}}$ alone; test labels were unmasked strictly for post-prediction evaluation.
3. **Acceptance Gate Verdict**:
   - **80% Nominal Coverage Gate** ($[77.0\%, 83.0\%]$): **FAIL** (Unnormalized: 76.96%, Locally Normalized: 76.22%).
   - **90% Nominal Coverage Gate** ($[88.0\%, 92.5\%]$): **PASS** for Unnormalized (89.11%); **FAIL** for Locally Normalized (87.74%).
   - **95% Nominal Coverage Gate** ($[93.5\%, 96.5\%]$): **PASS** for both methods (Unnormalized: 95.03%, Locally Normalized: 93.66%).
   - **90% Median Interval Width Gate** ($\le 5.8\text{ PCE pp}$): **FAIL** for both methods (Unnormalized: 8.84 PCE pp, Locally Normalized: 8.67 PCE pp).
   - **High-PCE Subgroup ($\text{PCE} \ge 15\%$, $N=63$)**: Unnormalized exhibits undercoverage (**77.78%**), while Locally Normalized maintains coverage (**92.06%**) at the expense of high width variance.
4. **Final Recommendation**: **CONDITIONALLY ACCEPTED WITH DOCUMENTED LIMITATIONS**. The method is methodologically rigorous, verified against leakage, and suitable for a research prototype, provided interval widths and champion-cell undercoverage are explicitly communicated to users.

---

## 2. Reproducibility Audit

### 2.1 Dataset Identity and Integrity Verification
The benchmark dataset used for calibration and evaluation was verified:
- **File Path**: `ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv`
- **SHA-256 Checksum**:
  `ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897`
- **Total Device Records**: Exactly $22,467$
- **Total Columns**: Exactly $37$
- **Unique Material-Name Pairs** (`donor_canonical::acceptor_canonical`): Exactly $4,730$

### 2.2 Feature Schema and Deterministic Extraction
The 22 molecular descriptors extracted from donor and acceptor SMILES via RDKit (`ml.src.features.descriptors.extract_features_from_mols`) follow a locked, deterministic schema:
1. `donor_mw`, 2. `donor_logp`, 3. `donor_tpsa`, 4. `donor_hbd`, 5. `donor_hba`, 6. `donor_rotatable_bonds`, 7. `donor_aromatic_rings`, 8. `donor_fraction_csp3`, 9. `donor_heavy_atoms`, 10. `acceptor_mw`, 11. `acceptor_logp`, 12. `acceptor_tpsa`, 13. `acceptor_hbd`, 14. `acceptor_hba`, 15. `acceptor_rotatable_bonds`, 16. `acceptor_aromatic_rings`, 17. `acceptor_fraction_csp3`, 18. `acceptor_heavy_atoms`, 19. `mw_diff`, 20. `logp_diff`, 21. `tpsa_total`, 22. `aromatic_rings_total`.

Feature values are deterministic given canonical SMILES; all 4,730 pair combinations exhibit zero missing values.

### 2.3 Exact Reproduction Commands
To execute the end-to-end experiment from scratch into an isolated output directory without overwriting production results:
```bash
# Set PYTHONPATH to project root and run with a redirected output directory
backend\.venv\Scripts\python.exe -c "
import os, sys
sys.path.insert(0, '.')
import ml.experiments.phase_7_1_grouped_conformal as exp
exp.RESULTS_DIR = 'scratch/repro_run'
os.makedirs(exp.RESULTS_DIR, exist_ok=True)
exp.main()
"
```

### 2.4 Artifact-by-Artifact Comparison Ledger
The reproduced outputs in `scratch/repro_run` were compared against the saved artifacts in `ml/results/`:

```
+-------------------------------------------------------------------------------------------------------------------------+
| Reproduction Verification Ledger across Artifact Files                                                                  |
+-----------------------------------------------+----------------------------------+------------------+-------------------+
| Artifact File Name                            | Primary Quantities Checked       | Numerical Diff   | Status            |
+-----------------------------------------------+----------------------------------+------------------+-------------------+
| phase_7_1_partition_metadata.json            | Split pair counts (2838,946,946) | 0.00e+00         | PASS (Bit-for-bit)|
| phase_7_1_selected_records.csv                | 946 cal & 946 test row indices   | 0.00e+00         | PASS (Bit-for-bit)|
| phase_7_1_inner_cv_scale_selection.json      | 5-fold Winkler scores & widths   | 0.00e+00         | PASS (Bit-for-bit)|
| phase_7_1_calibration_quantiles.json         | Finite-sample ranks & cutoffs q  | 0.00e+00         | PASS (Bit-for-bit)|
| phase_7_1_test_predictions_and_intervals.csv  | All 946 predictions & intervals  | <= 7.11e-15      | PASS (Identical)  |
| phase_7_1_subgroup_metrics.json              | Sliced tier & architecture counts| 0.00e+00         | PASS (Bit-for-bit)|
| phase_7_1_final_test_metrics.json            | Summary coverage, widths, Winkler| Detailed below   | PASS (Controlled) |
+-----------------------------------------------+----------------------------------+------------------+-------------------+
```

### 2.5 Investigation of the Single Floating-Point Discrepancy
In `phase_7_1_final_test_metrics.json`, an empirical coverage difference of $1 / 946 = 0.106\%$ was detected for Locally Normalized at 80% nominal (721/946 vs. 722/946):
- **Root Cause**: Record index 329 has true $\text{PCE} = 15.05\%$.
- **Original Run**: `upper_raw` = $15.049999999999999$ ($\text{PCE} - \text{upper} = -1.78 \times 10^{-15} \implies$ coverage evaluated as `False`).
- **Reproduced Run**: `upper_raw` = $15.050000000000000$ ($\text{PCE} - \text{upper} = 0.00 \implies$ coverage evaluated as `True`).
- **Mechanism**: Non-associative floating-point addition in multithreaded tree summation (`n_jobs=-1` in scikit-learn). When an instance falls exactly on the cutoff boundary to 15 decimal places, 1-bit LSB roundoff alters the boolean comparison.
- **Scientific Assessment**: The underlying arrays, model parameters, and intervals are identical to $10^{-15}$. This confirms mathematical reproducibility.

---

## 3. Data Isolation and Non-Leakage Audit

The implementation was audited line-by-line in [phase_7_1_grouped_conformal.py](file:///c:/Users/DELL/OPV-PredictNet/ml/experiments/phase_7_1_grouped_conformal.py):

```
+---------------------------------------------------------------------------------------------------------+
| Information Flow and Partition Isolation Verification                                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Benchmark Records (N = 22,467, Groups = 4,730)                                                        |
|   Pair Group ID: df['donor_canonical'] + "::" + df['acceptor_canonical']                                 |
|                                   │                                                                     |
|            ┌──────────────────────┴──────────────────────┐                                              |
|            ▼                                             ▼                                              |
|   D_train (60% Groups, 2,838)                  D_cal (20% Groups, 946)       D_test (20% Groups, 946)   |
|   11,786 Total Records                         1 Record/Pair (Seed 4201)      1 Record/Pair (Seed 4202)  |
|            │                                             │                                  │           |
|            ▼                                             │                                  │           |
|   Stage 1: Point Model Fitting                           │                                  │           |
|   Stage 2: Inner-CV Scale Selection                      │                                  │           |
|   (Trained on D_train ONLY)                              │                                  │           |
|            │                                             │                                  │           |
|            ├─────────────────────────────────────────────┘                                  │           |
|            ▼                                                                                │           |
|   Stage 3: Score Quantile Calibration                                                       │           |
|   (Evaluated on D_cal ONLY; calibrator FROZEN)                                              │           |
|            │                                                                                │           |
|            └────────────────────────────────────────────────────────────────────────────────┤           |
|                                                                                             ▼           |
|                                                                   Stage 4: Evaluation on D_test         |
|                                                                   (Test labels UNMASKED here ONLY)      |
+---------------------------------------------------------------------------------------------------------+
```

### 3.1 Verification of Specific Isolation Conditions
1. **Material-Pair Mutual Exclusivity**:
   $$\text{Pairs}(D_{\text{train}}) \cap \text{Pairs}(D_{\text{cal}}) = \emptyset$$
   $$\text{Pairs}(D_{\text{train}}) \cap \text{Pairs}(D_{\text{test}}) = \emptyset$$
   $$\text{Pairs}(D_{\text{cal}}) \cap \text{Pairs}(D_{\text{test}}) = \emptyset$$
   Set operations on `pair_group` columns confirm zero overlap across partitions.
2. **Test Label Blindness**:
   - Features `X_test` were passed to `calibrator.predict_intervals()` to generate prediction intervals prior to accessing `y_test`.
   - Test labels `y_test` were unmasked strictly in lines 540–560 to compute final coverage, MAE, and Winkler scores.
3. **No Reuse of Calibration Labels**:
   - Calibration targets `y_cal` were used exclusively to compute residuals $|y_{\text{cal}} - \hat{y}_{\text{cal}}|$ for quantile cutoffs. They did not enter Random Forest tree construction or Ridge regression scale fitting.
4. **Freezing Guard Contract**:
   - `GroupConformalCalibrator.calibrate()` sets `self.is_frozen = True`. Calling `calibrate()` or `fit_point_model()` after freezing raises an immediate `RuntimeError`.

---

## 4. Formal Scope of the Conformal Guarantee

To ensure scientific honesty and prevent overclaiming in publications or user-facing documentation, the statistical scope of the conformal guarantee must be strictly demarcated.

### 4.1 What the Grouped Conformal Procedure Guarantees
Under the assumption that material-pair clusters are exchangeable between the calibration set and future deployment, and that records within each cluster are selected according to the same uniform sampling mechanism (Alternative B):

$$\mathbb{P}_{(D_{\text{cal}}, (X_{n+1}, Y_{n+1}))} \left( Y_{n+1} \in \left[ \hat{y}(X_{n+1}) - q_{1-\alpha}, \; \hat{y}(X_{n+1}) + q_{1-\alpha} \right] \right) \ge 1 - \alpha$$

**Exact Operational Scope**:  
Marginal prediction-interval coverage for **one randomly selected experimental record from a randomly selected unseen normalized material-name pair**.

### 4.2 Explicit Non-Guarantees (What the Method Does NOT Guarantee)
1. **Does NOT Guarantee Coverage for Arbitrary Individual Records**:
   - Because multiple devices of the same material pair exhibit experimental dispersion due to processing conditions (annealing, solvent additives, D:A ratios), the guarantee does not cover worst-case or unoptimized device fabrications within a pair.
2. **Does NOT Guarantee Generalization to Novel Chemical Scaffolds**:
   - Normalized material names (`donor_canonical::acceptor_canonical`) are identifiers, not novel molecular representations. Common high-degree hub molecules (e.g., PC71BM, PM6, ITIC) appear across multiple pairs. Scaffold-hopping or out-of-distribution chemical structures are not guaranteed to be exchangeable.
3. **Does NOT Guarantee Laboratory-Independent Coverage**:
   - Inter-laboratory protocol drift, operator bias, and reporting discrepancies across different literature sources introduce unmodeled cluster-level variance.
4. **Does NOT Guarantee Conditional Coverage within Subgroups**:
   - Marginal coverage over the entire test set does not imply conditional coverage for specific efficiency tiers or device architectures. Specifically, high-efficiency cells ($\text{PCE} \ge 15\%$) suffer from conditional undercoverage under the unnormalized method ($77.78\%$ at nominal $90\%$).

---

## 5. Honest Evaluation of Pre-Registered Acceptance Gates

The pre-registered gates established in Phase 7.0 and Phase 7.1 are evaluated without alteration or retrospective adjustment:

```
+-------------------------------------------------------------------------------------------------------------------------+
| Pre-Registered Acceptance Gate Results (Evaluated on N_test = 946 Unseen Pairs)                                         |
+------------------------------------+------------------+------------------+------------------+---------------------------+
| Pre-Registered Gate Requirement    | Target Band / Max| Unnormalized     | Locally Norm.    | Strict Verdict            |
+------------------------------------+------------------+------------------+------------------+---------------------------+
| **Gate 1: 80% Nominal Coverage**   | [77.0%, 83.0%]   | 76.96% (728/946) | 76.22% (721/946) | **FAIL** (Both methods)   |
| **Gate 2: 90% Nominal Coverage**   | [88.0%, 92.5%]   | 89.11% (843/946) | 87.74% (830/946) | **PASS** (Unnormalized);  |
|                                    |                  |                  |                  | **FAIL** (Locally Norm.)  |
| **Gate 3: 95% Nominal Coverage**   | [93.5%, 96.5%]   | 95.03% (899/946) | 93.66% (886/946) | **PASS** (Both methods)   |
| **Gate 4: 90% Median Width**       | Max 5.8 PCE pp   | 8.84 PCE pp      | 8.67 PCE pp      | **FAIL** (Both methods)   |
| **Gate 5: High-PCE Tier (≥15%)**   | Nominal ≥ 88.0%  | 77.78% (49/63)   | 92.06% (58/63)   | **FAIL** (Unnormalized);  |
|                                    |                  |                  |                  | **PASS** (Locally Norm.)  |
+------------------------------------+------------------+------------------+------------------+---------------------------+
```

### 5.1 Gate Analysis and Honest Accounting
- **Gate 1 (80% Coverage)**: Unnormalized conformal achieved $76.96\%$, missing the lower tolerance bound of $77.0\%$ by $0.04$ percentage points. In strict audit reporting, **this is recorded as a FAIL**.
- **Gate 2 (90% Coverage)**: Unnormalized conformal achieved $89.11\%$, perfectly inside the target window $[88.0\%, 92.5\%]$ (**PASS**). Locally Normalized achieved $87.74\%$, missing by $0.26$ percentage points (**FAIL**).
- **Gate 3 (95% Coverage)**: Unnormalized achieved $95.03\%$ and Locally Normalized achieved $93.66\%$, both passing inside $[93.5\%, 96.5\%]$ (**PASS**).
- **Gate 4 (Median Width $\le 5.8\text{ PCE pp}$)**: Both methods failed this gate significantly (8.84 and 8.67 PCE pp). An interval half-width of $\pm 4.42\text{ PCE pp}$ reflects the wide experimental dispersion of literature OPV data, where devices sharing identical molecular names vary from $<1\%$ to $>10\%$ based on processing conditions.
- **Gate 5 (Champion Cells)**: Unnormalized conformal undercovers champion cells ($77.78\%$) due to decision tree regression-to-the-mean, whereas Locally Normalized widens adaptively to achieve $92.06\%$ coverage.

---

## 6. Final Recommendation and Project Decision

### Audit Decision:
### **CONDITIONALLY ACCEPTED WITH DOCUMENTED LIMITATIONS**

### Justification:
1. **Why Not Rejected**:
   - The implementation is completely free of data leakage and methodologically rigorous.
   - Outputs reproduce to machine precision ($10^{-15}$) across all 7 artifact files.
   - The core operational estimand—90% marginal prediction interval coverage—succeeded decisively on the held-out test set ($89.11\%$, Winkler score $12.79$).
   - The 95% marginal coverage gate also succeeded decisively ($95.03\%$).
2. **Why Not Unconditionally Accepted**:
   - The 80% nominal coverage gate narrowly missed ($76.96\%$ vs. $77.0\%$).
   - The 90% median interval width gate ($\le 5.8\text{ PCE pp}$) failed by $+3.04\text{ PCE pp}$ ($8.84$ pp), indicating that the intervals are broad.
   - Champion devices ($\text{PCE} \ge 15\%$) suffer from conditional undercoverage under the primary unnormalized predictor.
3. **Conditions for Research Prototype Deployment**:
   - **UI Disclosure**: The user interface must display the prediction interval as $\hat{y} \pm 4.42\text{ PCE pp}$ (at 90% confidence) and explicitly inform the user of the interval width.
   - **Champion Cell Warning**: A banner must flag predictions where $\hat{y} \ge 14.0\%$, stating that empirical coverage on champion cells drops to $\sim 78\%$ due to tree-ensemble conservatism.
   - **No Production Overclaiming**: The method must be presented in documentation as valid for *device-record prediction on unseen normalized material pairs*, with explicit disclaimers regarding scaffold-hopping and laboratory variation.

---

## 7. Literal Git Status Confirmation

Execution of `git status --short`:
```
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
?? ml/PHASE_7_0_3_FINAL_SCOPE_AUDIT.md
?? ml/PHASE_7_0_4_PAIR_IDENTITY_AUDIT.md
?? ml/PHASE_7_0_5_PAIR_IDENTITY_RECONCILIATION.md
?? ml/PHASE_7_1_1_INDEPENDENT_AUDIT.md
?? ml/PHASE_7_1_2_NUMERICAL_RECONCILIATION.md
?? ml/PHASE_7_1_3_FINAL_ACCEPTANCE_AUDIT.md
?? ml/PHASE_7_1_GROUPED_CONFORMAL_REPORT.md
?? ml/PHASE_7_CALIBRATION_DESIGN.md
?? ml/PHASE_7_UNCERTAINTY_RESEARCH.md
?? ml/PHASE_7_VALIDATION_PROTOCOL.md
?? ml/experiments/
?? ml/results/
?? tests/ml/test_grouped_conformal.py
```

- **Tracked files modified**: **0**
- **Git commits created**: **0**
- **Original experiment artifacts modified**: **0**

**Audit completed. Execution halted in accordance with Phase 7.1.3 instructions.**
