# OPV-PredictNet — Phase 8.0: Processing-Aware Feature Strategy

**Date**: 2026-11-24  
**Scope**: Formulation of candidate feature configurations, imputation architectures, target-leakage controls, and inference-time deployment contracts for Phase 8.  
**Deliverable**: Feature Strategy Specification for Processing-Aware OPV Modeling.

---

## 1. Executive Summary & Design Principles

The metadata audit in `ml/PHASE_8_0_PROCESSING_METADATA_AUDIT.md` demonstrated that processing information is both highly influential and non-randomly missing. To evaluate the impact of fabrication parameters scientifically, three separate feature configurations are defined:

1. **Configuration A (Molecular Baseline)**: 22 intrinsic physicochemical descriptors (the locked Phase 5 baseline).
2. **Configuration B (Naive Processing Integration)**: 22 molecular descriptors + available processing parameters with conventional median/mode imputation and no missingness indicators.
3. **Configuration C (Principled Processing with Explicit Missingness)**: 22 molecular descriptors + domain-normalized processing features + explicit binary missingness flags + categorical regime encodings + native tree missing-value routing.

### Core Methodological Rules
- **Zero Target Leakage**: Post-fabrication photovoltaic outputs ($J_{\text{sc}}, V_{\text{oc}}, \text{FF}$, diode ideality factor, series/shunt resistances, dark currents) are strictly barred.
- **Inference Availability**: Every feature in Configuration B and C must be either controllable by the experimenter before synthesis or resolvable via standard screening presets.
- **No Inferred Absence**: Missing metadata must never be silently converted to physical absence without an explicit indicator.

---

## 2. Feature Configuration Specifications

### 2.1 Configuration A: Frozen 22-Feature Molecular Baseline
This configuration represents the benchmark model from Phase 5 and Phase 7.1, capturing purely intrinsic chemical potential from donor and acceptor structures alone.

```
+---------------------------------------------------------------------------------------------------------+
| Configuration A: Pure Molecular Descriptor Matrix (22 Features, 100% Complete)                          |
+----+----------------------------+-----------------------+-----------------------------------------------+
| #  | Feature Identifier         | Source Entity         | Scientific Description                        |
+----+----------------------------+-----------------------+-----------------------------------------------+
| 1  | `donor_mw`                 | Donor SMILES          | Molecular weight of donor repeating unit (g/mol) |
| 2  | `donor_logp`               | Donor SMILES          | Wildman-Crippen octanol-water partition coeff |
| 3  | `donor_tpsa`               | Donor SMILES          | Topological polar surface area (Å²)           |
| 4  | `donor_hbd`                | Donor SMILES          | Hydrogen bond donor count                     |
| 5  | `donor_hba`                | Donor SMILES          | Hydrogen bond acceptor count                  |
| 6  | `donor_rotatable_bonds`    | Donor SMILES          | Number of rotatable single bonds              |
| 7  | `donor_aromatic_rings`     | Donor SMILES          | Count of aromatic ring systems                |
| 8  | `donor_fraction_csp3`      | Donor SMILES          | Fraction of sp3-hybridized carbons            |
| 9  | `donor_heavy_atoms`        | Donor SMILES          | Total non-hydrogen atom count                 |
| 10 | `acceptor_mw`              | Acceptor SMILES       | Molecular weight of acceptor molecule (g/mol) |
| 11 | `acceptor_logp`            | Acceptor SMILES       | Octanol-water partition coefficient           |
| 12 | `acceptor_tpsa`            | Acceptor SMILES       | Topological polar surface area (Å²)           |
| 13 | `acceptor_hbd`             | Acceptor SMILES       | Hydrogen bond donor count                     |
| 14 | `acceptor_hba`             | Acceptor SMILES       | Hydrogen bond acceptor count                  |
| 15 | `acceptor_rotatable_bonds` | Acceptor SMILES       | Number of rotatable single bonds              |
| 16 | `acceptor_aromatic_rings`  | Acceptor SMILES       | Count of aromatic ring systems                |
| 17 | `acceptor_fraction_csp3`   | Acceptor SMILES       | Fraction of sp3-hybridized carbons            |
| 18 | `acceptor_heavy_atoms`     | Acceptor SMILES       | Total non-hydrogen atom count                 |
| 19 | `mw_diff`                  | Derived Interface     | donor_mw - acceptor_mw                        |
| 20 | `logp_diff`                | Derived Interface     | donor_logp - acceptor_logp                    |
| 21 | `tpsa_total`               | Derived Interface     | donor_tpsa + acceptor_tpsa                    |
| 22 | `aromatic_rings_total`     | Derived Interface     | donor_aromatic_rings + acceptor_aromatic_rings|
+----+----------------------------+-----------------------+-----------------------------------------------+
```

---

### 2.2 Configuration B: Naive Processing Integration (32 Features)
Configuration B evaluates whether standard tabular processing features improve accuracy without specialized missing-data engineering.

```
+---------------------------------------------------------------------------------------------------------+
| Configuration B: Molecular Baseline + Direct Processing Features (32 Features)                         |
+----+----------------------------+-----------+-----------------------------------------------------------+
| #  | Feature Identifier         | Type      | Imputation & Encoding Protocol                            |
+----+----------------------------+-----------+-----------------------------------------------------------+
| 1-22| (All Configuration A features)| Float64 | (Standard scaling / median imputation on D_train)          |
| 23 | `da_ratio_numeric`         | Float64   | Parsed D/A ratio; missing imputed to median (1.0)         |
| 24 | `thickness_nm`             | Float64   | Parsed thickness (nm); missing imputed to median (100.0)  |
| 25 | `annealing_temp_c`         | Float64   | Parsed temperature (°C); missing imputed to median (100.0)|
| 26 | `solvent_code`             | Int (Cat) | Categorical: CF (0), CB (1), o-DCB (2), other (3), null (4)|
| 27 | `additive_code`            | Int (Cat) | Categorical: DIO (0), CN (1), other (2), null (3)          |
| 28 | `device_architecture_code` | Int (Cat) | Categorical: conventional (0), inverted (1), null (2)     |
| 29 | `htl_code`                 | Int (Cat) | Categorical: PEDOT:PSS (0), MoO3 (1), other (2), null (3)  |
| 30 | `etl_code`                 | Int (Cat) | Categorical: ZnO (0), PDINN (1), Ca (2), other (3), null(4)|
| 31 | `has_additive`             | Binary    | 1 if additive is non-null, 0 if null                      |
| 32 | `has_thermal_annealing`    | Binary    | 1 if annealing_temp is non-null, 0 if null                |
+----+----------------------------+-----------+-----------------------------------------------------------+
```

---

### 2.3 Configuration C: Principled Processing with Explicit Missingness (44 Features)
Configuration C treats missingness as an informative, structural property of scientific literature while normalizing physical variables into scale-invariant, physically bounded forms.

```
+---------------------------------------------------------------------------------------------------------+
| Configuration C: Principled Feature Architecture (44 Features)                                          |
+----+-------------------------------+-----------+--------------------------------------------------------+
| #  | Feature Identifier            | Type      | Physical Definition & Missingness Strategy             |
+----+-------------------------------+-----------+--------------------------------------------------------+
| 1-22| (All Configuration A features)   | Float64   | Intrinsic chemical baseline (100% complete)            |
+----+-------------------------------+-----------+--------------------------------------------------------+
|    | **Normalized Numerical Features**                                                                  |
| 23 | `donor_blend_fraction`        | Float64   | f_donor = D / (D + A) in (0, 1); imputed to 0.50       |
| 24 | `thickness_log10`             | Float64   | log10(thickness in nm); bounded [1.3, 2.7]; imp. 2.00  |
| 25 | `annealing_temp_norm`         | Float64   | (temp - 100) / 50; bounded [-1.2, 3.0]; imputed to 0.0 |
| 26 | `additive_concentration_pct`  | Float64   | Continuous vol/wt % in [0.0, 10.0]; imputed to 0.0     |
+----+-------------------------------+-----------+--------------------------------------------------------+
|    | **Explicit Missingness Indicators (M)**                                                            |
| 27 | `missing_da_ratio`            | Binary    | 1 if d_a_ratio was null or unparseable, 0 otherwise    |
| 28 | `missing_thickness`           | Binary    | 1 if thickness was null or unparseable, 0 otherwise   |
| 29 | `missing_annealing`           | Binary    | 1 if annealing_temp was null, 0 otherwise              |
| 30 | `missing_solvent`             | Binary    | 1 if solvent was null, 0 otherwise                     |
| 31 | `missing_additive`            | Binary    | 1 if additive was null, 0 otherwise                    |
| 32 | `missing_architecture`        | Binary    | 1 if device_type was null or non-standard, 0 otherwise |
| 33 | `missing_interlayers`         | Binary    | 1 if either ETL or HTL was null, 0 otherwise           |
+----+-------------------------------+-----------+--------------------------------------------------------+
|    | **Domain-Aware Physical Regimes**                                                                  |
| 34 | `regime_as_cast_room_temp`    | Binary    | 1 if explicitly unannealed / room temperature (temp=0) |
| 35 | `regime_thermal_annealed`      | Binary    | 1 if thermal annealing >= 50°C was reported            |
| 36 | `regime_solvent_additive_used` | Binary    | 1 if an additive (DIO, CN, etc.) was confirmed used    |
| 37 | `regime_chlorinated_solvent`  | Binary    | 1 if solvent is CF, CB, or o-DCB                       |
| 38 | `regime_green_solvent`        | Binary    | 1 if solvent is non-halogenated (toluene, xylene, etc) |
| 39 | `regime_inverted_stack`       | Binary    | 1 if inverted architecture, 0 otherwise                |
| 40 | `regime_conventional_stack`   | Binary    | 1 if conventional architecture, 0 otherwise            |
+----+-------------------------------+-----------+--------------------------------------------------------+
|    | **Interaction & Ratio Features**                                                                   |
| 41 | `thickness_per_donor_mw`      | Float64   | log10(thickness) / log10(donor_mw)                     |
| 42 | `solvent_boiling_point_c`     | Float64   | Solvent BP: CF (61.2), CB (131.6), o-DCB (180.5)       |
| 43 | `etl_workfunction_ev`         | Float64   | Nominal ETL WF: ZnO (4.3), PDINN (4.0), Ca (2.9)       |
| 44 | `htl_workfunction_ev`         | Float64   | Nominal HTL WF: PEDOT:PSS (5.0), MoO3 (5.3)            |
+----+-------------------------------+-----------+--------------------------------------------------------+
```

---

### 2.4 Configuration C-Ablation: Processing Features Without Missingness Flags (37 Features)
To strictly audit whether Configuration C gains predictive power through physical processing dynamics or merely exploits missingness flags as shortcut proxies for modern high-PCE literature eras, an explicit ablation configuration is mandated:

```
+---------------------------------------------------------------------------------------------------------+
| Configuration C-Ablation (Model M2A): 37 Features (Omits 7 Missingness Flags)                          |
+---------------------------------------------------------------------------------------------------------+
| Features 1–22: Intrinsic chemical baseline (Configuration A)                                            |
| Features 23–26: Continuous normalized processing (f_donor, log_thickness, norm_temp, additive_conc)      |
| Features 27–33: Regime indicators (as-cast, annealed, additive used, chlorinated, green, inverted, conv)|
| Features 34–37: Domain interaction features (thickness/MW, solvent BP, ETL WF, HTL WF)                 |
| EXCLUDED: All 7 missing_* binary indicators (Features 27-33 of Configuration C)                        |
+---------------------------------------------------------------------------------------------------------+
```

If Model M2 significantly outperforms Model M2A solely because of the 7 missingness flags, the model fails the shortcut learning check.

A critical vulnerability in scientific machine learning is the inadvertent inclusion of variables that can only be measured after the experiment is completed.

```
+---------------------------------------------------------------------------------------------------------+
| Target Leakage Classification Matrix                                                                    |
+------------------------------+--------------------+-----------------------------------------------------+
| Candidate Variable           | Status             | Scientific Rationale                                |
+------------------------------+--------------------+-----------------------------------------------------+
| `d_a_ratio`                  | **PERMITTED**      | Pre-fabrication solution formulation parameter      |
| `solvent_canonical`          | **PERMITTED**      | Pre-fabrication solvent selection                   |
| `additive_canonical`         | **PERMITTED**      | Pre-fabrication solution additive                   |
| `annealing_temp`             | **PERMITTED**      | Pre-measurement thermal processing step            |
| `active_layer_thickness`     | **PERMITTED**      | Target fabrication specification (spin-coat target) |
| `device_type` (stack)        | **PERMITTED**      | Pre-fabrication substrate/interlayer choice         |
+------------------------------+--------------------+-----------------------------------------------------+
| `voc` (Open-Circuit Voltage) | **STRICTLY BANNED**| Post-fabrication I-V curve electrical measurement   |
| `jsc` (Short-Circuit Current)| **STRICTLY BANNED**| Post-fabrication I-V curve electrical measurement   |
| `ff` (Fill Factor)           | **STRICTLY BANNED**| Post-fabrication I-V curve electrical measurement   |
| `dark_current_density`       | **STRICTLY BANNED**| Post-fabrication diode electrical measurement       |
| `film_roughness_rms`         | **STRICTLY BANNED**| Post-fabrication AFM morphological measurement      |
| `domain_size_tem`            | **STRICTLY BANNED**| Post-fabrication TEM nanoscale measurement          |
+------------------------------+--------------------+-----------------------------------------------------+
```

---

## 4. Inference-Time Deployment Strategy

To be practically useful in automated screening and active learning, the model must be executable when an experimentalist evaluates candidate molecules that have never been synthesized or fabricated.

### 4.1 Two Operational Deployment Modes (Audited Governance)
1. **Mode 1: Molecular-Only Virtual Screening (Unchosen Processing)**:
   - *Use Case*: High-throughput virtual screening of novel candidate donor-acceptor combinations prior to any laboratory synthesis or formulation decisions.
   - *Model Invocation*: **Model M0 (22-feature molecular baseline)**.
   - *Methodological Rule*: Unknown processing must **never** be silently filled with arbitrary "standard presets" (e.g. 1:1 ratio, 0.5% DIO, 100°C) with missingness flags forced to 0. Doing so misleads researchers by asserting high conditional performance under fictitious processing.
   - *Output*: Predicted baseline intrinsic chemical potential and conformal uncertainty interval.

2. **Mode 2: Processing-Conditional Recipe Evaluation (Controlled Processing)**:
   - *Use Case*: An experimentalist has synthesized a candidate pair and either provides specific formulation parameters (e.g. D:A ratio = 1:1.2, solvent = chlorobenzene, annealing = 120°C) or explicitly requests an optimization sweep across candidate processing recipes.
   - *Model Invocation*: **Model M2 (44 features) or Model M3 (Native-NaN GBDT)**.
   - *Inputs*: Donor SMILES, Acceptor SMILES, plus user-specified formulation parameters. Any unprovided parameters have their corresponding missingness flags set to 1.
   - *Output*: Conditional predicted PCE and uncertainty interval for that specific, transparently documented fabrication recipe.

---

**Feature strategy finalized. Proceed to Experimental Protocol formulation.**
