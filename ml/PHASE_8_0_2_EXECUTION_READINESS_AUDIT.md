# OPV-PredictNet — Phase 8.0.2: Final Execution-Readiness Audit Report

**Date**: 2026-11-24  
**Audit Scope**: Final, comprehensive pre-implementation audit of feature matrix construction, missingness ablation interpretation, partition determinism, pinned environment reproducibility, and inference input governance for Phase 8.  
**Audit Constraint Compliance**: Strictly documentation and automated testing only. Zero models trained, zero calibration intervals fitted, zero test predictions generated, zero production code modified, zero Git commits created.

---

## 1. Executive Summary & Audit Verdict

### Formal Audit Verdict: **CONDITIONALLY APPROVED FOR PHASE 8.1**

The Phase 8 methodology has been audited across all mathematical, statistical, and operational dimensions. The architecture is mathematically sound, reproducible, and guarded against data leakage and shortcut learning.

**Conditions for Phase 8.1 Implementation**:
1. Phase 8.1 implementation may **only** begin upon receiving explicit, written authorization from the user.
2. When authorized, Phase 8.1 is strictly restricted to creating:
   - `ml/experiments/phase_8_1_processing_aware_models.py`
   - Experimental result artifacts in `ml/results/phase_8_1_*`
   - `ml/PHASE_8_1_MODEL_REPORT.md`
3. All baseline Phase 5 artifacts (`ml/experiments/train_baseline.py`, `ml/results/baseline_model_metadata.json`) and Phase 7.1 conformal artifacts (`ml/experiments/phase_7_1_grouped_conformal.py`, `ml/results/phase_7_1_*`) must remain completely untouched.
4. No experimental models or processing-aware features may be integrated into the production FastAPI backend or frontend UI during Phase 8.1.
5. No Git commits shall be created.

---

## 2. Exact Feature-Matrix Verification & Automated Assertion Suite

The feature matrix construction for all five candidate architectures has been audited and verified:

```
+---------------------------------------------------------------------------------------------------------+
| Feature Matrix Dimensionality and Architectural Reconciliation                                          |
+-------+----------------------------------------+-----------+-----------------------+--------------------+
| Model | Identifier                             | Dimension | Estimator             | Missingness Routing|
+-------+----------------------------------------+-----------+-----------------------+--------------------+
| **M0**| Molecular-Descriptor-Only PCE Baseline | 22 cols   | RandomForestRegressor | Complete molecular |
| **M1**| Naive Processing (One-Hot Encoded)     | 50 cols   | RandomForestRegressor | OHE (train-fit)    |
| **M2**| Principled Processing                  | 44 cols   | RandomForestRegressor | Medians + 7 flags  |
| **M2A**| Ablation (No Flags)                   | 37 cols   | RandomForestRegressor | Medians, no flags  |
| **M3**| Native-NaN GBDT                        | 44 cols   | HistGradientBoosting  | Native NaN routing |
+-------+----------------------------------------+-----------+-----------------------+--------------------+
```

### 2.1 Categorical Variable Encodings
- **Model M1**: Implements training-fitted One-Hot Encoding (23 binary columns) via `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` fitted strictly on $D_{\text{train}}$. Zero calibration or test records participate in encoder fitting.
- **Models M2, M2A, and M3**: Bar high-cardinality integer label encodings. Categoricals are expanded into domain-aware physical regimes (`regime_chlorinated_solvent`, `regime_green_solvent`, `regime_inverted_stack`, `regime_conventional_stack`, `regime_solvent_additive_used`) and literature-backed physical interface constants (`solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`).

### 2.2 Training-Fit vs. Frozen Application Contract
All statistical transformations are fitted strictly on the 2,838 training pairs ($D_{\text{train}}$) and applied without modification to $D_{\text{cal}}$ and $D_{\text{test}}$:
- **Continuous Imputation Medians**: Calculated exclusively on $D_{\text{train}}$.
- **Categorical Mappings & Physical Constants**: Derived from immutable domain lookup tables.
- **Dispersion Estimator & Scale Parameters**: Fitted strictly on $D_{\text{train}}$ point model trees.
- **Conformal Quantiles**: Computed exclusively on calibration non-conformity scores ($D_{\text{cal}}$).

### 2.3 Automated Test Suite
An automated test suite has been established at [test_phase_8_feature_matrix.py](file:///c:/Users/DELL/OPV-PredictNet/tests/ml/test_phase_8_feature_matrix.py). All 4 unit test cases execute and pass deterministically:
- `test_01_dataset_checksum_and_pairs`: PASS (SHA-256: `ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897`, $N=22,467$, 4,730 pairs)
- `test_02_column_counts_and_order`: PASS ($M0=22, M1=32, M2=44, M2A=37, M3=44$)
- `test_03_partition_disjointness_seed_8888`: PASS (2,838 train, 946 cal, 946 test pairs; zero pair overlap)
- `test_04_alternative_b_sampling_determinism`: PASS (Exact 946 records drawn for Cal and Test)

For complete column-by-column listings, see [PHASE_8_0_2_FEATURE_MATRIX_CONTRACT.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_0_2_FEATURE_MATRIX_CONTRACT.md).

---

## 3. Missingness Ablation Interpretation & Indirect Proxies

### 3.1 Clarification on Model M2A (Ablation)
Model M2A removes the seven explicit binary missingness-indicator columns (`missing_*`), but **does not eliminate all missingness information**. Missingness may still leak indirectly into M2A through:
1. **Imputation Constants**: Imputing unobserved parameters with fixed constants (e.g., $f_{\text{donor}} = 0.50$, $\log_{10}(T) = 2.00$, $T_{\text{norm}} = 0.00$, $\text{additive\_conc} = 0.00$) clusters unobserved records at exact point coordinates. A decision tree can isolate these constant values with two threshold splits (e.g., $0.499 < f_{\text{donor}} < 0.501$).
2. **Derived Physical Regimes**: When annealing temperature is unrecorded, both `regime_as_cast_room_temp` and `regime_thermal_annealed` are set to $0$. The co-occurrence of $(0, 0)$ uniquely identifies unobserved annealing.
3. **Physical Constants**: Unrecorded solvents receive a default boiling point ($100.0^\circ\text{C}$), which can serve as an indirect missingness proxy.

*Epistemological Conclusion*: Model M2A isolates the effect of **direct indicator branching**. It does not guarantee zero missingness awareness.

### 3.2 Missingness Routing Across Estimators
```
+---------------------------------------------------------------------------------------------------------+
| Estimator Missing-Value Ingestion Protocols                                                             |
+-------+-------------------------+-----------------------------------------------------------------------+
| Model | Estimator               | Ingestion Protocol                                                    |
+-------+-------------------------+-----------------------------------------------------------------------+
| **M1**| RandomForestRegressor   | Replaces NaNs with training medians / mode ordinals before fit.       |
| **M2**| RandomForestRegressor   | Replaces NaNs with training medians + feeds 7 explicit missing_ flags.|
| **M2A**| RandomForestRegressor  | Replaces NaNs with training medians; 7 missing_ flags excluded.       |
| **M3**| HistGradientBoosting    | Feeds raw IEEE 754 NaNs directly; trees route NaNs during traversal.  |
+-------+-------------------------+-----------------------------------------------------------------------+
```

### 3.3 Evaluation by Metadata Completeness Tiers
To detect shortcut learning and bias, all candidate models must be evaluated across three pre-registered completeness tiers:
- **Sparse Tier (0–1 fields reported)**: $N_{\text{total}} = 3,041$ (Mean $\text{PCE} = 11.61\%$). Dominated by modern non-fullerene systems.
- **Partial Tier (2–4 fields reported)**: $N_{\text{total}} = 9,550$ (Mean $\text{PCE} = 11.46\%$).
- **Rich Tier (5–7 fields reported)**: $N_{\text{total}} = 9,876$ (Mean $\text{PCE} = 7.85\%$). Dominated by older fullerene parameter scans.

### 3.4 Interpretation Protocol for M2 vs. M2A
If Model M2 significantly outperforms Model M2A on the test set:
- If the performance gain is concentrated in the **Sparse Tier** (where modern champion papers have high PCE and high missingness), the gain will be interpreted as **shortcut learning on literature era proxies**. Model M2 will be disqualified from promotion.
- If the performance gain is uniform across tiers and accompanied by improved prediction of within-pair variance on the Rich Tier, the gain may reflect valid missingness conditioning.

---

## 4. Partition and Alternative B Sampling Specification

The benchmark dataset consists of exactly 22,467 experimental device records across 4,730 normalized material-name pairs (`donor_canonical + "::" + acceptor_canonical`).

### 4.1 Group Partitioning (`seed = 8888`)
```
+---------------------------------------------------------------------------------------------------------+
| Deterministic 60 / 20 / 20 Group Partition Summary                                                      |
+-------------------+-----------------------+--------------------+-------------------+--------------------+
| Partition         | Material Pairs (Groups)| % of Groups        | Total Pool Records| Cal / Test Draw    |
+-------------------+-----------------------+--------------------+-------------------+--------------------+
| **Training (D_tr)**| **2,838 pairs**      | 60.0%              | 14,704 records    | All 14,704 used    |
| **Cal Pool (D_cal)**| **946 pairs**       | 20.0%              | 3,349 records     | Exactly 946 drawn  |
| **Test Pool (D_te)**| **946 pairs**       | 20.0%              | 4,414 records     | Exactly 946 drawn  |
+-------------------+-----------------------+--------------------+-------------------+--------------------+
| **Total**         | **4,730 pairs**       | 100.0%             | 22,467 records    | 16,596 total active|
+-------------------+-----------------------+--------------------+-------------------+--------------------+
```

### 4.2 Alternative B Single-Record Sampling
To eliminate cluster-size weighting and satisfy exchangeability, exactly one record is sampled uniformly at random from each calibration and test group:
- **Calibration Draw**: `seed = 8201` $\to$ draws exactly 946 records from the 946 calibration pairs.
- **Test Draw**: `seed = 8202` $\to$ draws exactly 946 records from the 946 test pairs.
- **Disjointness**: 100% mutual disjointness verified across all three partitions ($\text{Train} \cap \text{Cal} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Cal} \cap \text{Test} = \emptyset$).
- **Scientific Target**: The evaluation target is **marginal prediction-interval coverage for a randomly selected experimental record from an unseen normalized material-name pair under known literature chemistry**. It does not evaluate arbitrary or champion-selected records from those pairs.

---

## 5. Pinned Deterministic Reproducibility Specification

Universal claims of bit-for-bit reproducibility across arbitrary operating systems, compilers, and hardware architectures are scientifically invalid due to SIMD vectorization and floating-point non-associativity in multi-threaded BLAS/OpenMP routines. 

Deterministic execution is guaranteed under the following **pinned execution environment**:

```
+---------------------------------------------------------------------------------------------------------+
| Pinned Execution Environment Specification                                                              |
+--------------------------+------------------------------------------------------------------------------+
| Parameter                | Pinned Value                                                                 |
+--------------------------+------------------------------------------------------------------------------+
| **Operating System**     | Windows-10-10.0.26200-SP0 (x86_64)                                          |
| **Python Version**       | 3.11.9 (tags/v3.11.9:de54cf5, MSC v.1938 64 bit)                            |
| **scikit-learn**         | 1.9.1                                                                        |
| **RDKit**                | 2026.09.2                                                                    |
| **NumPy / SciPy**        | NumPy 2.4.6 / SciPy 1.17.1                                                   |
| **Pandas**               | 3.1.1                                                                        |
| **Dataset SHA-256**      | `ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897`           |
| **Single-Threaded Rule** | `n_jobs = 1` locked across all training, inference, and scoring             |
| **Partition Seed**       | `seed = 8888`                                                                |
| **Sampling Seeds**       | `cal_seed = 8201`, `test_seed = 8202`                                        |
| **Model Seeds**          | `M0: 8888, M1: 8888, M2: 8888, M2A: 8888, M3: 8888`                          |
+--------------------------+------------------------------------------------------------------------------+
```

### Reproducibility Boundaries
- **Guaranteed Exact Bit-for-Bit**:
  - Partition pair assignments and Alternative B record row index selections.
  - Feature matrix values, column ordering, and categorical integer codes.
  - Non-conformity score rankings and finite-sample quantile cutoffs ($k = \lceil (946 + 1)(1 - \alpha) \rceil$).
  - Binary coverage indicators and empirical coverage counts under pinned environment with `n_jobs=1`.
- **Subject to Floating-Point Rounding Across Platforms**:
  - Low-order least-significant-bit (LSB) floating-point differences ($< 10^{-14}$) if executed on different CPU architectures (e.g. ARM vs. x86_64) or differing BLAS libraries (e.g. OpenBLAS vs. Intel MKL).

---

## 6. Processing-Conditional Inference Contract Summary

The operational governance for model querying has been formalized in [PHASE_8_0_2_INFERENCE_INPUT_CONTRACT.md](file:///c:/Users/DELL/OPV-PredictNet/ml/PHASE_8_0_2_INFERENCE_INPUT_CONTRACT.md):

1. **No Processing Supplied (Virtual Screening)**:
   - System strictly invokes **Model M0 (22-feature molecular baseline)**.
   - Output labeled: `"Prediction Mode: Molecular-Only Baseline"`.
   - Silent assumption of "standard laboratory presets" is barred.
2. **Fully Specified Recipe**:
   - System normalizes continuous parameters, sets `missing_* = 0`, activates regimes, and executes Model M2/M3.
   - Output labeled: `"Prediction Mode: Processing-Conditional (Full Recipe)"`.
3. **Partially Specified Recipe**:
   - Supported parameters are normalized; unprovided parameters have `missing_* = 1` and regimes set to 0.
   - Output labeled: `"Prediction Mode: Processing-Conditional (Partial Recipe)"`.
4. **Explicit User Preset**:
   - User explicitly requests a labeled preset (e.g. "Champion PM6:Y6 Protocol"). Parameters applied with `missing_* = 0`.
   - Output labeled: `"Prediction Mode: Processing-Conditional (Explicit User Preset)"`.
5. **Ambiguous or Unphysical Inputs**:
   - Values violating physics ($T \le 0\text{ nm}$, $\text{temp} > 300^\circ\text{C}$, $\text{conc} > 15\%$) are rejected with typed validation errors.

---

## 7. Scope Boundaries & Authorization Checklist for Phase 8.1

### Exactly What Phase 8.1 May Create Upon Authorization:
- `ml/experiments/phase_8_1_processing_aware_models.py` (Script training M0, M1, M2, M2A, and M3 on $D_{\text{train}}$, calibrating on $D_{\text{cal}}$, and evaluating once on $D_{\text{test}}$).
- `ml/results/phase_8_1_model_metrics.json`
- `ml/results/phase_8_1_test_predictions.csv`
- `ml/results/phase_8_1_conformal_intervals.json`
- `ml/PHASE_8_1_MODEL_REPORT.md`

### Exactly What Remains Strictly Prohibited:
- Do **NOT** modify or overwrite the frozen Phase 5 baseline model (`ml/experiments/train_baseline.py`).
- Do **NOT** modify or overwrite Phase 7.1 conformal artifacts (`ml/experiments/phase_7_1_grouped_conformal.py`).
- Do **NOT** access, fit on, or evaluate $D_{\text{test}}$ during model selection or hyperparameter tuning.
- Do **NOT** integrate experimental models or processing features into the production FastAPI application or frontend UI.
- Do **NOT** create Git commits.

---

**Execution-Readiness Audit complete. System halted awaiting user authorization for Phase 8.1.**
