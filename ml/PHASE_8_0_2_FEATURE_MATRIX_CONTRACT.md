# OPV-PredictNet — Phase 8.0.2: Locked Feature Matrix Specification and Transformation Contract

**Date**: 2026-11-24  
**Scope**: Definitive specification of feature column names, exact column ordering, categorical encodings, training-fit vs. frozen transformations, automated validation assertions, and mathematical dimension reconciliations across models M0, M1, M2, M2A, and M3.  
**Deliverable**: Locked Feature Matrix Contract for Phase 8.

---

## 1. Executive Summary & Architectural Integrity

To guarantee mathematical consistency and eliminate ambiguity prior to Phase 8.1 implementation, this document specifies the exact feature matrix constructed for every candidate model. 

```
+---------------------------------------------------------------------------------------------------------+
| Feature Space Dimensions and Architecture Summary                                                       |
+-------+-------------------------+-----------+-------------------------+---------------------------------+
| Model | Identifier              | Columns   | Estimator               | Missingness Treatment           |
+-------+-------------------------+-----------+-------------------------+---------------------------------+
| **M0**| Molecular-Descriptor-Only PCE Baseline| **22**| RandomForestRegressor   | Complete molecular descriptors  |
| **M1**| Naive Processing (OHE)  | **50**    | RandomForestRegressor   | OHE (train-fit) + medians       |
| **M2**| Principled Processing   | **44**    | RandomForestRegressor   | Medians + 7 missingness flags   |
| **M2A**| Ablation (No Flags)    | **37**    | RandomForestRegressor   | Medians without missingness flag|
| **M3**| Native-NaN GBDT         | **44**    | HistGradientBoosting    | Native tree split routing (NaNs)|
+-------+-------------------------+-----------+-------------------------+---------------------------------+
```

No model shall be described as having 22, 37, 44, or 50 features unless the actual feature matrix passed to `fit()` and `predict()` contains exactly that number of numeric columns.

---

## 2. Exact Column Ordering Across All Model Matrices

### 2.1 Model M0: Molecular-Descriptor-Only PCE Baseline (22 Columns)
All 22 features are derived deterministically from `donor_smiles` and `acceptor_smiles` using RDKit (`ml.src.features.descriptors.extract_features_from_mols`). Molecular descriptors are 100% complete across all 22,467 records. Model M0 predicts measured PCE from molecular descriptors without explicit processing inputs, reflecting the empirical distribution of fabrication conditions present in the training set.

```
+----+----------------------------+-----------+-------+-----------------------+---------------------------------------+
| #  | Column Name                | Data Type | Units | Source Variable       | Formula / Transformation              |
+----+----------------------------+-----------+-------+-----------------------+---------------------------------------+
| 1  | `donor_mw`                 | Float64   | g/mol | `donor_smiles`        | Descriptors.MolWt(donor)              |
| 2  | `donor_logp`               | Float64   | None  | `donor_smiles`        | Descriptors.MolLogP(donor)            |
| 3  | `donor_tpsa`               | Float64   | Å²    | `donor_smiles`        | Descriptors.TPSA(donor)               |
| 4  | `donor_hbd`                | Float64   | Count | `donor_smiles`        | Lipinski.NumHDonors(donor)            |
| 5  | `donor_hba`                | Float64   | Count | `donor_smiles`        | Lipinski.NumHAcceptors(donor)         |
| 6  | `donor_rotatable_bonds`    | Float64   | Count | `donor_smiles`        | Lipinski.NumRotatableBonds(donor)     |
| 7  | `donor_aromatic_rings`     | Float64   | Count | `donor_smiles`        | Lipinski.NumAromaticRings(donor)      |
| 8  | `donor_fraction_csp3`      | Float64   | Ratio | `donor_smiles`        | Lipinski.FractionCSP3(donor)          |
| 9  | `donor_heavy_atoms`        | Float64   | Count | `donor_smiles`        | Descriptors.HeavyAtomCount(donor)     |
| 10 | `acceptor_mw`              | Float64   | g/mol | `acceptor_smiles`     | Descriptors.MolWt(acceptor)           |
| 11 | `acceptor_logp`            | Float64   | None  | `acceptor_smiles`     | Descriptors.MolLogP(acceptor)         |
| 12 | `acceptor_tpsa`            | Float64   | Å²    | `acceptor_smiles`     | Descriptors.TPSA(acceptor)            |
| 13 | `acceptor_hbd`             | Float64   | Count | `acceptor_smiles`     | Lipinski.NumHDonors(acceptor)         |
| 14 | `acceptor_hba`             | Float64   | Count | `acceptor_smiles`     | Lipinski.NumHAcceptors(acceptor)      |
| 15 | `acceptor_rotatable_bonds` | Float64   | Count | `acceptor_smiles`     | Lipinski.NumRotatableBonds(acceptor)  |
| 16 | `acceptor_aromatic_rings`  | Float64   | Count | `acceptor_smiles`     | Lipinski.NumAromaticRings(acceptor)   |
| 17 | `acceptor_fraction_csp3`   | Float64   | Ratio | `acceptor_smiles`     | Lipinski.FractionCSP3(acceptor)       |
| 18 | `acceptor_heavy_atoms`     | Float64   | Count | `acceptor_smiles`     | Descriptors.HeavyAtomCount(acceptor)  |
| 19 | `mw_diff`                  | Float64   | g/mol | Derived (1, 10)       | abs(donor_mw - acceptor_mw)           |
| 20 | `logp_diff`                | Float64   | None  | Derived (2, 11)       | abs(donor_logp - acceptor_logp)       |
| 21 | `tpsa_total`               | Float64   | Å²    | Derived (3, 12)       | donor_tpsa + acceptor_tpsa           |
| 22 | `aromatic_rings_total`     | Float64   | Count | Derived (7, 16)       | donor_aromatic + acceptor_aromatic    |
+----+----------------------------+-----------+-------+-----------------------+---------------------------------------+
```

---

### 2.2 Model M1: Naive Processing Integration with Training-Fitted One-Hot Encoding (50 Columns)
Columns 1–22 are identical to M0. Columns 23–27 append continuous numerical and presence indicators. Columns 28–50 append the 23 one-hot encoded columns fitted strictly on $D_{\text{train}}$ via `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.

```
+----+-------------------------------+-----------+-------+-----------------------+---------------------------------------+
| #  | Column Name                   | Data Type | Units | Source Variable       | Encoding / Imputation Specification   |
+----+-------------------------------+-----------+-------+-----------------------+---------------------------------------+
| 1–22| (Features 1 to 22 of M0)     | Float64   | Mixed | (Molecular features)  | Exact identical columns and order     |
| 23 | `da_ratio_numeric`            | Float64   | Ratio | `d_a_ratio`           | Numeric D/A scalar; missing -> 1.0    |
| 24 | `thickness_nm`                | Float64   | nm    | `active_layer_thick`  | Numeric nm scalar; missing -> 100.0   |
| 25 | `annealing_temp_c`            | Float64   | °C    | `annealing_temp`      | Numeric °C scalar; missing -> 100.0   |
| 26 | `has_additive`                | Int64     | {0,1} | `additive_canonical`  | 1 if reported additive, 0 if null/neat|
| 27 | `has_thermal_annealing`       | Int64     | {0,1} | `annealing_temp`      | 1 if temp > 35°C, 0 if null or room T |
|    | **One-Hot Encoded Categorical**|           |       |                       | (Fitted on D_train only, ignore unk)  |
| 28 | `solvent_chlorobenzene`       | Float64   | {0,1} | `solvent_canonical`   | 1 if chlorobenzene, 0 otherwise       |
| 29 | `solvent_chloroform`          | Float64   | {0,1} | `solvent_canonical`   | 1 if chloroform, 0 otherwise          |
| 30 | `solvent_missing`             | Float64   | {0,1} | `solvent_canonical`   | 1 if null, 0 otherwise                |
| 31 | `solvent_o-dcb`               | Float64   | {0,1} | `solvent_canonical`   | 1 if o-DCB, 0 otherwise               |
| 32 | `solvent_other`               | Float64   | {0,1} | `solvent_canonical`   | 1 if other reported solvent, 0 otherw |
| 33 | `additive_cn`                 | Float64   | {0,1} | `additive_canonical`  | 1 if 1-chloronaphthalene, 0 otherwise |
| 34 | `additive_dio`                | Float64   | {0,1} | `additive_canonical`  | 1 if 1,8-diiodooctane, 0 otherwise    |
| 35 | `additive_missing`            | Float64   | {0,1} | `additive_canonical`  | 1 if null / neat blend, 0 otherwise   |
| 36 | `additive_other`              | Float64   | {0,1} | `additive_canonical`  | 1 if other additive, 0 otherwise      |
| 37 | `architecture_conventional`   | Float64   | {0,1} | `device_type`         | 1 if conventional, 0 otherwise        |
| 38 | `architecture_inverted`       | Float64   | {0,1} | `device_type`         | 1 if inverted, 0 otherwise            |
| 39 | `architecture_missing`        | Float64   | {0,1} | `device_type`         | 1 if null, 0 otherwise                |
| 40 | `architecture_other`          | Float64   | {0,1} | `device_type`         | 1 if non-standard label, 0 otherwise  |
| 41 | `htl_missing`                 | Float64   | {0,1} | `htl_canonical`       | 1 if null, 0 otherwise                |
| 42 | `htl_moo3`                    | Float64   | {0,1} | `htl_canonical`       | 1 if MoO3, 0 otherwise                |
| 43 | `htl_other`                   | Float64   | {0,1} | `htl_canonical`       | 1 if other HTL, 0 otherwise           |
| 44 | `htl_pedot_pss`               | Float64   | {0,1} | `htl_canonical`       | 1 if PEDOT:PSS, 0 otherwise           |
| 45 | `etl_ca`                      | Float64   | {0,1} | `etl_canonical`       | 1 if Ca, 0 otherwise                  |
| 46 | `etl_missing`                 | Float64   | {0,1} | `etl_canonical`       | 1 if null, 0 otherwise                |
| 47 | `etl_other`                   | Float64   | {0,1} | `etl_canonical`       | 1 if other ETL, 0 otherwise           |
| 48 | `etl_pdinn`                   | Float64   | {0,1} | `etl_canonical`       | 1 if PDINN, 0 otherwise               |
| 49 | `etl_pfn_br`                  | Float64   | {0,1} | `etl_canonical`       | 1 if PFN-Br, 0 otherwise              |
| 50 | `etl_zno`                     | Float64   | {0,1} | `etl_canonical`       | 1 if ZnO, 0 otherwise                 |
+----+-------------------------------+-----------+-------+-----------------------+---------------------------------------+
```

---

### 2.3 Model M2: Principled Processing with Explicit Missingness (44 Columns)
Columns 1–22 are identical to M0. Columns 23–44 represent continuous physical normalizations, 7 explicit binary missingness flags, 7 domain physical regimes, and 4 physical interface constants.

```
+----+-------------------------------+-----------+-------+-----------------------+--------------------------------------+
| #  | Column Name                   | Data Type | Units | Source Variable       | Formulation / Physical Derivation    |
+----+-------------------------------+-----------+-------+-----------------------+--------------------------------------+
| 1–22| (Features 1 to 22 of M0)     | Float64   | Mixed | (Molecular features)  | Exact identical columns and order    |
|    | **Continuous Normalized**     |           |       |                       |                                      |
| 23 | `donor_blend_fraction`        | Float64   | Ratio | `d_a_ratio`           | f_donor = D / (D + A); missing -> 0.5|
| 24 | `thickness_log10`             | Float64   | log10 | `active_layer_thick`  | log10(T) for T in [15,600]; imp 2.0  |
| 25 | `annealing_temp_norm`         | Float64   | norm  | `annealing_temp`      | (T - 100) / 50; as-cast -> -1.3;imp 0|
| 26 | `additive_concentration_pct`  | Float64   | % vol | `additive_ratio`      | Parsed vol% in [0, 10]; missing -> 0 |
|    | **Missingness Indicators**    |           |       |                       |                                      |
| 27 | `missing_da_ratio`            | Int64     | {0,1} | `d_a_ratio`           | 1 if null / unparseable, 0 if valid  |
| 28 | `missing_thickness`           | Int64     | {0,1} | `active_layer_thick`  | 1 if null / <=0 / >600 nm, 0 if valid|
| 29 | `missing_annealing`           | Int64     | {0,1} | `annealing_temp`      | 1 if null / unphysical, 0 if valid   |
| 30 | `missing_solvent`             | Int64     | {0,1} | `solvent_canonical`   | 1 if null, 0 if reported             |
| 31 | `missing_additive`            | Int64     | {0,1} | `additive_canonical`  | 1 if null, 0 if reported / neat      |
| 32 | `missing_architecture`        | Int64     | {0,1} | `device_type`         | 1 if null / unclassified, 0 if conv/i|
| 33 | `missing_interlayers`         | Int64     | {0,1} | `etl`/`htl`           | 1 if either ETL or HTL null, 0 if both|
|    | **Physical Regimes**          |           |       |                       |                                      |
| 34 | `regime_as_cast_room_temp`    | Int64     | {0,1} | `annealing_temp`      | 1 if confirmed T <= 35°C; 0 otherwise|
| 35 | `regime_thermal_annealed`      | Int64     | {0,1} | `annealing_temp`      | 1 if confirmed 40 <= T <= 250°C; 0 oth|
| 36 | `regime_solvent_additive_used` | Int64     | {0,1} | `additive_canonical`  | 1 if non-null and not 'neat'; 0 oth  |
| 37 | `regime_chlorinated_solvent`  | Int64     | {0,1} | `solvent_canonical`   | 1 if CF, CB, or o-DCB; 0 otherwise   |
| 38 | `regime_green_solvent`        | Int64     | {0,1} | `solvent_canonical`   | 1 if toluene, o-xylene, etc.; 0 oth  |
| 39 | `regime_inverted_stack`       | Int64     | {0,1} | `device_type`         | 1 if 'inverted'; 0 otherwise         |
| 40 | `regime_conventional_stack`   | Int64     | {0,1} | `device_type`         | 1 if 'conventional'; 0 otherwise     |
|    | **Interface & Constants**     |           |       |                       |                                      |
| 41 | `thickness_per_donor_mw`      | Float64   | Ratio | Derived (24, 1)       | thickness_log10 / log10(donor_mw)    |
| 42 | `solvent_boiling_point_c`     | Float64   | °C    | `solvent_canonical`   | CF(61.2), CB(131.6), o-DCB(180.5); missing/unmapped -> D_train median (131.6°C) for M2; NaN for M3|
| 43 | `etl_workfunction_ev`         | Float64   | eV    | `etl_canonical`       | ZnO(4.30), PDINN(4.00), Ca(2.87), PFN-Br(4.20); missing/unmapped -> D_train median (4.30 eV) for M2; NaN for M3|
| 44 | `htl_workfunction_ev`         | Float64   | eV    | `htl_canonical`       | PEDOT:PSS(5.00), MoO3(5.30), 2PACz(5.20); missing/unmapped -> D_train median (5.00 eV) for M2; NaN for M3|
+----+-------------------------------+-----------+-------+-----------------------+--------------------------------------+
```

---

### 2.4 Model M2A: Ablation Processing Model (37 Columns)
Model M2A contains **exactly** the feature set of Model M2 with the seven explicit missingness-indicator columns (`missing_*`, columns 27–33) removed.

```
M2A Column Order (Exactly 37 Columns):
Columns 1–22: Features 1–22 of M0 (Molecular descriptors)
Columns 23–26: Continuous normalized processing (donor_blend_fraction, thickness_log10, annealing_temp_norm, additive_concentration_pct)
Columns 27–33: Physical regimes (regime_as_cast_room_temp, regime_thermal_annealed, regime_solvent_additive_used, regime_chlorinated_solvent, regime_green_solvent, regime_inverted_stack, regime_conventional_stack)
Columns 34–37: Physical interface constants (thickness_per_donor_mw, solvent_boiling_point_c, etl_workfunction_ev, htl_workfunction_ev)
```

---

### 2.5 Model M3: Native-NaN Gradient Boosted Decision Tree (44 Columns)
Model M3 uses the **exact same 44 column names and column order as Model M2**.
- *Difference from M2*: Continuous variables (`donor_blend_fraction`, `thickness_log10`, `annealing_temp_norm`, `solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`) **retain IEEE 754 NaN values** when unobserved or missing, instead of being imputed with training medians.
- *Estimator*: `sklearn.ensemble.HistGradientBoostingRegressor` (which implements native split-direction routing for missing values during tree traversal).

---

## 3. Categorical Encodings: Source Variables vs. Expanded Columns

Every categorical variable is handled via one of two audited protocols:

```
+---------------------------------------------------------------------------------------------------------+
| Categorical Processing Protocols                                                                        |
+--------------------------+-----------------------+-----------------------+------------------------------+
| Source Column            | Distinct Raw Strings  | Model M1 Encoding     | Model M2 / M2A / M3 Encoding |
+--------------------------+-----------------------+-----------------------+------------------------------+
| `solvent_canonical`      | 372                   | One-Hot (5 columns)   | 2 Regimes (Chlorinated/Green)|
|                          |                       | (CF, CB, o-DCB,oth,nan| + Solvent Boiling Point (°C) |
+--------------------------+-----------------------+-----------------------+------------------------------+
| `additive_canonical`     | 1,162                 | One-Hot (4 columns)   | 1 Regime (Additive Used)     |
|                          |                       | (DIO, CN, other, nan) | + Missingness Flag           |
+--------------------------+-----------------------+-----------------------+------------------------------+
| `device_type`            | 302                   | One-Hot (4 columns)   | 2 Regimes (Conv / Inverted)  |
|                          |                       | (conv, inv, oth, nan) | + Missingness Flag           |
+--------------------------+-----------------------+-----------------------+------------------------------+
| `etl_canonical`          | 1,164                 | One-Hot (6 columns)   | Work Function Constant (eV)  |
|                          |                       | (ZnO, PDINN, Ca, etc) | + Missingness Flag           |
+--------------------------+-----------------------+-----------------------+------------------------------+
| `htl_canonical`          | 724                   | One-Hot (4 columns)   | Work Function Constant (eV)  |
|                          |                       | (PEDOT, MoO3, oth, nan| + Missingness Flag           |
+--------------------------+-----------------------+-----------------------+------------------------------+
```
*Note*: For Model M1, all One-Hot Encoders are fitted strictly on $D_{\text{train}}$ with `handle_unknown='ignore'`. No calibration or test records participate in encoder fitting.

---

## 4. Training-Fit vs. Frozen Application Contract

To prevent data leakage across partitions, all parametric transformations and statistical summaries must be fitted **exclusively on the 2,838 training pairs ($D_{\text{train}}$)** and then applied unchanged to calibration and test partitions.

```
+---------------------------------------------------------------------------------------------------------+
| Transformation Provenance and Freezing Rules                                                            |
+------------------------------+---------------------------+----------------------------------------------+
| Transformation / Parameter   | Fitting Source            | Execution on D_cal and D_test                |
+------------------------------+---------------------------+----------------------------------------------+
| Molecular Descriptor Formulae| Pure Deterministic Code   | Exact RDKit descriptors computed per SMILES  |
| Continuous Imputation Medians| Computed on D_train only  | Frozen scalar applied to missing test entries|
| Categorical Mapping Tables   | Pre-registered dictionary | Exact immutable lookup table                 |
| Physical Constant Lookups    | Physical literature table | Exact immutable lookup table                 |
| Tree Dispersion Estimator    | Fitted on D_train RF only | Frozen tree ensemble evaluates test variance |
| Conformal Quantiles          | Computed on D_cal only    | Frozen cutoffs applied to test intervals     |
+------------------------------+---------------------------+----------------------------------------------+
```

*Strict Guarantee*: No mean, median, standard deviation, quantile, or vocabulary shall be computed or updated using $D_{\text{cal}}$ or $D_{\text{test}}$.

---

## 5. Automated Validation Assertions

Prior to training or evaluating any model in Phase 8.1, the feature construction pipeline must pass the following automated assertions (implemented in `tests/ml/test_phase_8_feature_matrix.py`):

1. **Exact Shape Assertion**:
   $$\text{Shape}(X_{\text{M0}}) = (N, 22), \quad \text{Shape}(X_{\text{M1}}) = (N, 50), \quad \text{Shape}(X_{\text{M2}}) = (N, 44)$$
   $$\text{Shape}(X_{\text{M2A}}) = (N, 37), \quad \text{Shape}(X_{\text{M3}}) = (N, 44)$$
2. **Column Name and Order Assertion**:
   $$\text{list}(X_m.\text{columns}) == \text{LOCKED\_COLUMNS}_m \quad \forall m \in \{\text{M0}, \text{M1}, \text{M2}, \text{M2A}, \text{M3}\}$$
3. **Finite Value Assertion (M0, M1, M2, M2A)**:
   $$\sum \mathbf{1}\{\text{isnan}(X)\} == 0, \quad \sum \mathbf{1}\{\text{isinf}(X)\} == 0$$
4. **Controlled NaN Assertion (M3)**:
   - NaNs permitted *only* in continuous unobserved columns (`thickness_log10`, `annealing_temp_norm`, `additive_concentration_pct`, `donor_blend_fraction`, `solvent_boiling_point_c`, `etl_workfunction_ev`, `htl_workfunction_ev`).
   - Zero NaNs permitted in molecular descriptors (Columns 1–22), missingness indicators (Columns 27–33), or regime indicators (Columns 34–40).
5. **Strict Binary Value Assertion**:
   Every column with prefix `missing_`, `regime_`, or `has_` must contain values strictly in $\{0, 1\}$.

---

**Feature Matrix Contract locked and verified against unit test suite.**
