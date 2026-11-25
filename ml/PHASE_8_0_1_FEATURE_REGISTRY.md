# OPV-PredictNet — Phase 8.0.1: Comprehensive Feature Registry

**Date**: 2026-11-24  
**Scope**: Precise specification of all feature definitions, transformations, units, categorical encodings, missingness treatments, and prediction-time contracts across candidate models M0, M1, M2, and M3.  
**Deliverable**: Locked Feature Registry for Phase 8.

---

## 1. Feature Count Mathematical Reconciliation

The candidate model architectures evaluate systematically expanding feature spaces:

```
+---------------------------------------------------------------------------------------------------------+
| Feature Count Reconciliation across Model Candidates                                                    |
+-------+-------------------------+-----------+-----------------------------------------------------------+
| Model | Name                    | Count     | Mathematical Composition                                  |
+-------+-------------------------+-----------+-----------------------------------------------------------+
| **M0**| Frozen Molecular Baseline| **22**    | 22 Intrinsic RDKit Molecular Descriptors                  |
| **M1**| Naive Processing RF     | **32**    | 22 Molecular + 10 Naive Processing Features               |
| **M2**| Principled Processing RF| **44**    | 22 Molecular + 4 Norm Processing + 7 Missingness Flags +  |
|       |                         |           | 7 Physical Regimes + 4 Physical Interface Constants       |
| **M2A**| Ablation (No Flags)    | **37**    | M2 minus 7 Missingness Flags (Testing Shortcut Learning)  |
| **M3**| Native-NaN GBDT         | **44**    | Identical 44 features to M2, evaluated via GBDT Native NaNs|
+-------+-------------------------+-----------+-----------------------------------------------------------+
```

---

## 2. Universal Feature Definitions: Configuration A (Features 1–22)

All models (M0–M3) share the identical, frozen 22 molecular descriptors computed from donor and acceptor SMILES via RDKit (`ml.src.features.descriptors.extract_features_from_mols`). These descriptors are 100% complete across all 4,730 material-name pairs in the benchmark dataset.

```
+-------------------------------------------------------------------------------------------------------------------------+
| Molecular Baseline Feature Matrix (Features 1 to 22)                                                                    |
+----+----------------------------+-----------+-------+-----------------------+--------------------+----------------------+
| #  | Feature Name               | Type      | Units | Source Column         | Derivation Formula | Prediction Available |
+----+----------------------------+-----------+-------+-----------------------+--------------------+----------------------+
| 1  | `donor_mw`                 | Float64   | g/mol | `donor_smiles`        | RDKit MolWt        | Yes (from SMILES)    |
| 2  | `donor_logp`               | Float64   | None  | `donor_smiles`        | RDKit MolLogP      | Yes (from SMILES)    |
| 3  | `donor_tpsa`               | Float64   | Å²    | `donor_smiles`        | RDKit TPSA         | Yes (from SMILES)    |
| 4  | `donor_hbd`                | Float64   | Count | `donor_smiles`        | RDKit NumHDonors   | Yes (from SMILES)    |
| 5  | `donor_hba`                | Float64   | Count | `donor_smiles`        | RDKit NumHAcceptors| Yes (from SMILES)    |
| 6  | `donor_rotatable_bonds`    | Float64   | Count | `donor_smiles`        | RDKit NumRotBonds  | Yes (from SMILES)    |
| 7  | `donor_aromatic_rings`     | Float64   | Count | `donor_smiles`        | RDKit NumAromatic  | Yes (from SMILES)    |
| 8  | `donor_fraction_csp3`      | Float64   | Ratio | `donor_smiles`        | RDKit FractionCSP3 | Yes (from SMILES)    |
| 9  | `donor_heavy_atoms`        | Float64   | Count | `donor_smiles`        | RDKit HeavyAtomCnt | Yes (from SMILES)    |
| 10 | `acceptor_mw`              | Float64   | g/mol | `acceptor_smiles`     | RDKit MolWt        | Yes (from SMILES)    |
| 11 | `acceptor_logp`            | Float64   | None  | `acceptor_smiles`     | RDKit MolLogP      | Yes (from SMILES)    |
| 12 | `acceptor_tpsa`            | Float64   | Å²    | `acceptor_smiles`     | RDKit TPSA         | Yes (from SMILES)    |
| 13 | `acceptor_hbd`             | Float64   | Count | `acceptor_smiles`     | RDKit NumHDonors   | Yes (from SMILES)    |
| 14 | `acceptor_hba`             | Float64   | Count | `acceptor_smiles`     | RDKit NumHAcceptors| Yes (from SMILES)    |
| 15 | `acceptor_rotatable_bonds` | Float64   | Count | `acceptor_smiles`     | RDKit NumRotBonds  | Yes (from SMILES)    |
| 16 | `acceptor_aromatic_rings`  | Float64   | Count | `acceptor_smiles`     | RDKit NumAromatic  | Yes (from SMILES)    |
| 17 | `acceptor_fraction_csp3`   | Float64   | Ratio | `acceptor_smiles`     | RDKit FractionCSP3 | Yes (from SMILES)    |
| 18 | `acceptor_heavy_atoms`     | Float64   | Count | `acceptor_smiles`     | RDKit HeavyAtomCnt | Yes (from SMILES)    |
| 19 | `mw_diff`                  | Float64   | g/mol | Derived (1, 10)       | donor_mw - acc_mw  | Yes (Calculated)     |
| 20 | `logp_diff`                | Float64   | None  | Derived (2, 11)       | donor_logp - acc_p | Yes (Calculated)     |
| 21 | `tpsa_total`               | Float64   | Å²    | Derived (3, 12)       | donor_tpsa + acc_t | Yes (Calculated)     |
| 22 | `aromatic_rings_total`     | Float64   | Count | Derived (7, 16)       | donor_ar + acc_ar  | Yes (Calculated)     |
+----+----------------------------+-----------+-------+-----------------------+--------------------+----------------------+
```

---

## 3. Configuration B Feature Additions: Naive Processing (Features 23–32)

Model M1 appends 10 naive processing features directly with standard median/mode imputation:

```
+-------------------------------------------------------------------------------------------------------------------------+
| Configuration B Processing Additions (Features 23 to 32)                                                                |
+----+----------------------------+-----------+-------+-----------------------+--------------------+----------------------+
| #  | Feature Name               | Type      | Units | Source Column         | Missingness Value  | Encoding Method      |
+----+----------------------------+-----------+-------+-----------------------+--------------------+----------------------+
| 23 | `da_ratio_numeric`         | Float64   | Ratio | `d_a_ratio`           | Imputed to 1.0     | Numeric scalar D/A   |
| 24 | `thickness_nm`             | Float64   | nm    | `active_layer_thick`  | Imputed to 100.0   | Numeric scalar nm    |
| 25 | `annealing_temp_c`         | Float64   | °C    | `annealing_temp`      | Imputed to 100.0   | Numeric scalar °C    |
| 26 | `solvent_code`             | Int (Ord) | Index | `solvent_canonical`   | Imputed to 4 (null)| Ordinal integer      |
| 27 | `additive_code`            | Int (Ord) | Index | `additive_canonical`  | Imputed to 3 (null)| Ordinal integer      |
| 28 | `device_architecture_code` | Int (Ord) | Index | `device_type`         | Imputed to 2 (null)| Ordinal integer      |
| 29 | `htl_code`                 | Int (Ord) | Index | `htl_canonical`       | Imputed to 3 (null)| Ordinal integer      |
| 30 | `etl_code`                 | Int (Ord) | Index | `etl_canonical`       | Imputed to 5 (null)| Ordinal integer      |
| 31 | `has_additive`             | Binary    | {0,1} | `additive_canonical`  | 0 (if null)        | Indicator flag       |
| 32 | `has_thermal_annealing`    | Binary    | {0,1} | `annealing_temp`      | 0 (if null)        | Indicator flag       |
+----+----------------------------+-----------+-------+-----------------------+--------------------+----------------------+
```
*Note on Categorical Ordinal Mappings for M1*:
- `solvent_code`: `chloroform` = 0, `chlorobenzene` = 1, `o-DCB` = 2, `other` = 3, `missing` = 4.
- `additive_code`: `DIO` = 0, `CN` = 1, `other` = 2, `missing` = 3.
- `device_architecture_code`: `conventional` = 0, `inverted` = 1, `missing/other` = 2.
- `htl_code`: `PEDOT:PSS` = 0, `MoO3` = 1, `other` = 2, `missing` = 3.
- `etl_code`: `ZnO` = 0, `PDINN` = 1, `Ca` = 2, `PFN-Br` = 3, `other` = 4, `missing` = 5.

---

## 4. Configuration C Feature Additions: Principled Processing (Features 23–44)

Models M2, M2A (Ablation), and M3 utilize the complete, physically normalized feature matrix.

### 4.1 Normalized Continuous Processing Features (Features 23–26)
```
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| #  | Feature Name                  | Type      | Units | Source Column      | Physical Transformation & Valid Bounds    |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| 23 | `donor_blend_fraction`        | Float64   | Ratio | `d_a_ratio`        | f_donor = D / (D + A) in (0, 1); imp 0.50 |
| 24 | `thickness_log10`             | Float64   | log10 | `active_layer_thk` | log10(T) for T in [15, 600] nm; imp 2.00  |
| 25 | `annealing_temp_norm`         | Float64   | norm  | `annealing_temp`   | (T - 100) / 50 for T in [40, 250]; imp 0.0|
| 26 | `additive_concentration_pct`  | Float64   | % vol | `additive_ratio`   | Parsed vol% in [0.0, 10.0]; imp 0.00      |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
```
*Transformation Rules*:
- `donor_blend_fraction`: Parsed via regular expression `^([0-9\.]+)\s*[:/]\s*([0-9\.]+)` stripping weight descriptors (`w/w`, `wt:wt`, `mass ratio`). If $D$ and $A$ are positive, $f_{\text{donor}} = D / (D + A)$. If unparseable or ternary, mapped to NaN.
- `thickness_log10`: Raw string parsed to numeric float $T$. If $T \le 0$ or $T > 1000\text{ nm}$, mapped to NaN. If valid, $\log_{10}(T)$ is computed.
- `annealing_temp_norm`: If $T \le 35^\circ\text{C}$, mapped to $0.0$ (signified by `regime_as_cast_room_temp`). If $40^\circ\text{C} \le T \le 250^\circ\text{C}$, normalized as $(T - 100) / 50$. If $T > 250^\circ\text{C}$ (sol-gel sintering), mapped to NaN.
- `additive_concentration_pct`: Extracted numerical float from volume percentage strings. If absent, set to $0.0$.

### 4.2 Explicit Missingness Indicators (Features 27–33)
*(Included in M2 and M3; removed in M2A Ablation)*
```
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| #  | Feature Name                  | Type      | Units | Source Column      | Indicator Logic                           |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| 27 | `missing_da_ratio`            | Binary    | {0,1} | `d_a_ratio`        | 1 if null or unparseable, 0 if valid      |
| 28 | `missing_thickness`           | Binary    | {0,1} | `active_layer_thk` | 1 if null or physically invalid, 0 if val |
| 29 | `missing_annealing`           | Binary    | {0,1} | `annealing_temp`   | 1 if null or physically invalid, 0 if val |
| 30 | `missing_solvent`             | Binary    | {0,1} | `solvent_canonical`| 1 if null, 0 if reported                  |
| 31 | `missing_additive`            | Binary    | {0,1} | `additive_canonical`| 1 if null, 0 if reported / explicit neat  |
| 32 | `missing_architecture`        | Binary    | {0,1} | `device_type`      | 1 if null or generic 'OSC/BHJ', 0 if valid|
| 33 | `missing_interlayers`         | Binary    | {0,1} | `etl`/`htl`        | 1 if either ETL or HTL is null, 0 if both |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
```

### 4.3 Physical Regime Binary Indicators (Features 34–40)
```
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| #  | Feature Name                  | Type      | Units | Source Column      | Physical Verification Rule                |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| 34 | `regime_as_cast_room_temp`    | Binary    | {0,1} | `annealing_temp`   | 1 if explicitly T <= 35°C; 0 otherwise    |
| 35 | `regime_thermal_annealed`      | Binary    | {0,1} | `annealing_temp`   | 1 if confirmed 40 <= T <= 250°C; 0 otherw |
| 36 | `regime_solvent_additive_used` | Binary    | {0,1} | `additive_canonical`| 1 if non-null and not 'neat/none'; 0 other|
| 37 | `regime_chlorinated_solvent`  | Binary    | {0,1} | `solvent_canonical`| 1 if CF, CB, or o-DCB; 0 otherwise        |
| 38 | `regime_green_solvent`        | Binary    | {0,1} | `solvent_canonical`| 1 if toluene, o-xylene, etc.; 0 otherwise |
| 39 | `regime_inverted_stack`       | Binary    | {0,1} | `device_type`      | 1 if 'inverted'; 0 otherwise              |
| 40 | `regime_conventional_stack`   | Binary    | {0,1} | `device_type`      | 1 if 'conventional'; 0 otherwise          |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
```
*Physical Integrity Safeguards*:
- If `annealing_temp` is null, **both** `regime_as_cast_room_temp` and `regime_thermal_annealed` are set to $0$ (the state is unknown; do not infer room temperature).
- If `additive_canonical` is null without explicit absence keywords, `regime_solvent_additive_used` is set to $0$ with `missing_additive = 1`.

### 4.4 Physical Interaction and Energy Level Constants (Features 41–44)
```
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| #  | Feature Name                  | Type      | Units | Source Entity      | Physical Definition                       |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
| 41 | `thickness_per_donor_mw`      | Float64   | Ratio | Derived (24, 1)    | log10(thickness) / log10(donor_mw)        |
| 42 | `solvent_boiling_point_c`     | Float64   | °C    | `solvent_canonical`| Literature BP: CF(61.2), CB(131.6), o-DCB |
|    |                               |           |       |                    | (180.5), o-xylene(144.4), toluene(110.6)  |
| 43 | `etl_workfunction_ev`         | Float64   | eV    | `etl_canonical`    | Literature WF: ZnO(4.3), PDINN(4.0),      |
|    |                               |           |       |                    | Ca(2.9), PFN-Br(4.2), other(4.0)          |
| 44 | `htl_workfunction_ev`         | Float64   | eV    | `htl_canonical`    | Literature WF: PEDOT:PSS(5.0), MoO3(5.3), |
|    |                               |           |       |                    | 2PACz(5.2), other(5.1)                    |
+----+-------------------------------+-----------+-------+--------------------+-------------------------------------------+
```

---

## 5. Prediction-Time Availability and Operational Contracts

```
+---------------------------------------------------------------------------------------------------------+
| Prediction-Time Operational Interface Matrix                                                            |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Feature Block            | Molecular Screening   | Processing-Condition| Operational Governance         |
|                          | Mode (Mode 1)         | Mode (Mode 2)       | at Prediction Time             |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Features 1–22            | **Available**         | **Available**       | Computed from user SMILES via  |
| (Molecular Descriptors)  | (Evaluated via M0)    | (Evaluated via M2)  | RDKit deterministic pipeline   |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Features 23–26           | **NOT APPLICABLE**    | **User-Specified**  | In Mode 1: M0 invoked directly.|
| (Continuous Processing)  | (M0 invoked)          | (Formulation Input) | In Mode 2: user input values   |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Features 27–33           | **NOT APPLICABLE**    | **Calculated**      | In Mode 1: M0 invoked directly.|
| (Missingness Indicators) | (M0 invoked)          | (Flags active)      | In Mode 2: 1 if null, 0 if set |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Features 34–40           | **NOT APPLICABLE**    | **Calculated**      | In Mode 1: M0 invoked directly.|
| (Physical Regimes)       | (M0 invoked)          | (From Inputs)       | In Mode 2: from user conditions|
+--------------------------+-----------------------+---------------------+--------------------------------+
| Features 41–44           | **NOT APPLICABLE**    | **Calculated**      | In Mode 1: M0 invoked directly.|
| (Physical Constants)     | (M0 invoked)          | (From Inputs)       | In Mode 2: literature constants|
+--------------------------+-----------------------+---------------------+--------------------------------+
```
*Operational Rule*: Unknown processing must never trigger silent substitution of arbitrary "standard presets" with missingness flags forced to 0. When processing is unchosen, the system routes directly to Model M0 (22 molecular features). When processing is specified or when the user explicitly selects a labeled recipe preset, the system routes to Model M2 or M3.

---

**Feature registry locked and mathematically verified.**
