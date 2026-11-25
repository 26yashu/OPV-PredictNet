# OPV-PredictNet — Phase 8.0.3: Final Feature and Inference Corrections

**Date**: 2026-11-24  
**Audit Scope**: Final pre-experiment corrections resolving M1 categorical encoding, physical property lookup missingness, partial-input validation, and physical validation threshold categorization.  
**Constraint Compliance**: Strictly documentation and automated testing only. Zero models trained, zero calibration intervals fitted, zero test predictions generated, zero production code modified, zero Git commits created.

---

## 1. M1 Categorical Encoding Correction

### 1.1 Defect in Prior Ordinal-Coding Design
In the preliminary Phase 8.0 design, Model M1 assigned arbitrary integer codes to nominal categories:
`solvent_code` (0–4), `additive_code` (0–3), `device_architecture_code` (0–2), `htl_code` (0–3), `etl_code` (0–5).
Treating nominal variables as ordinal integers imposes an artificial mathematical distance and rank ordering (e.g., asserting $\text{chloroform} < \text{chlorobenzene} < \text{o-DCB}$) that decision trees in `RandomForestRegressor` evaluate as continuous split thresholds. This distorts tree topology and introduces spurious inductive bias.

### 1.2 Replacement: Training-Fitted One-Hot Encoding
The ordinal integer encoding for Model M1 is completely replaced with **training-fitted One-Hot Encoding**:
- **Estimator**: `sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.
- **Fitting Provenance**: Fitted strictly on the 14,704 training records ($D_{\text{train}}$) across the 2,838 training pairs.
- **Strict Anti-Leakage Contract**: Zero calibration or test records ($D_{\text{cal}}$, $D_{\text{test}}$) are passed to `fit()`.
- **Unknown-Category Handling**: Any unseen category appearing in calibration, test, or inference evaluates to all zeros across the one-hot columns for that variable (`handle_unknown='ignore'`).

### 1.3 Recalculated M1 Feature Count (Exactly 50 Columns)
The five nominal categorical variables in $D_{\text{train}}$ expand into exactly **23 binary one-hot columns**:
1. `solvent` (5 columns): `solvent_chlorobenzene`, `solvent_chloroform`, `solvent_missing`, `solvent_o-dcb`, `solvent_other`
2. `additive` (4 columns): `additive_cn`, `additive_dio`, `additive_missing`, `additive_other`
3. `architecture` (4 columns): `architecture_conventional`, `architecture_inverted`, `architecture_missing`, `architecture_other`
4. `htl` (4 columns): `htl_missing`, `htl_moo3`, `htl_other`, `htl_pedot_pss`
5. `etl` (6 columns): `etl_ca`, `etl_missing`, `etl_other`, `etl_pdinn`, `etl_pfn_br`, `etl_zno`

```
+---------------------------------------------------------------------------------------------------------+
| Recalculated Model M1 Feature Matrix Composition                                                        |
+----+----------------------------------------+-----------+-----------------------------------------------+
| #  | Feature Block                          | Columns   | Details                                       |
+----+----------------------------------------+-----------+-----------------------------------------------+
| 1  | Molecular Descriptors (Baseline)       | **22**    | Features 1–22 of Model M0 (RDKit)             |
| 2  | Direct Numerical Processing Features   | **3**     | da_ratio_numeric, thickness_nm, temp_c        |
| 3  | Binary Presence Indicators             | **2**     | has_additive, has_thermal_annealing           |
| 4  | One-Hot Encoded Categoricals           | **23**    | OneHotEncoder fitted on D_train               |
+----+----------------------------------------+-----------+-----------------------------------------------+
|    | **Total M1 Feature Columns**           | **50**    | Exactly 50 numeric columns                    |
+----+----------------------------------------+-----------+-----------------------------------------------+
```

---

## 2. Audit and Correction of Physical Lookups

### 2.1 Elimination of Arbitrary Imputation
In preliminary scripts, missing solvents were arbitrarily assigned a boiling point of $100.0^\circ\text{C}$ (the boiling point of water). Assigning water's boiling point to an organic solar cell device processed from organic solvents distorts the physical scale. 

*Rule*: A physical value must **never** be assigned merely because a source field is missing. If a physical parameter is unrecorded or unavailable, its missing state must be preserved as `NaN` or handled via a principled training-fitted strategy.

### 2.2 Provenance, Context, and Mapping Tables
```
+-----------------------------------------------------------------------------------------------------------------------------+
| Physical Property Lookup Provenance and Measurement Context                                                                 |
+--------------------------+-----------+-------+-----------------------------+------------------------------------------------+
| Feature Column           | Physical Property | Units | Standard Measurement Context| Primary Literature Source                      |
+--------------------------+-----------+-------+-----------------------------+------------------------------------------------+
| `solvent_boiling_point_c`| Boiling Point     | °C    | 1 atm (101.325 kPa)         | CRC Handbook of Chem & Phys; PubChem           |
| `etl_workfunction_ev`    | Conduction/WF     | eV    | UPS / Kelvin Probe in vacuo | Adv. Mater. / ACS Energy Lett. OPV reviews    |
| `htl_workfunction_ev`    | Valence/WF        | eV    | UPS / Kelvin Probe in vacuo | Adv. Mater. / ACS Energy Lett. OPV reviews    |
+--------------------------+-----------+-------+-----------------------------+------------------------------------------------+
```

```
+---------------------------------------------------------------------------------------------------------+
| Material-to-Value Mapping Tables                                                                        |
+------------------------------+---------------------------+----------------------------------------------+
| Material Category            | Normalized Key Identifier | Literature Property Value                    |
+------------------------------+---------------------------+----------------------------------------------+
| **Host Solvents**            | `chloroform` (CF)         | 61.2 °C                                      |
|                              | `chlorobenzene` (CB)      | 131.6 °C                                     |
|                              | `o-DCB` (dichlorobenzene) | 180.5 °C                                     |
|                              | `toluene`                 | 110.6 °C                                     |
|                              | `o-xylene`                | 144.4 °C                                     |
|                              | `tetrahydrofuran` (THF)   | 66.0 °C                                      |
+------------------------------+---------------------------+----------------------------------------------+
| **Electron Transport Layer** | `ZnO` (zinc oxide)        | 4.30 eV (nominal sol-gel / nanoparticle film)|
|                              | `PDINN`                   | 4.00 eV (aliphatic amine-functionalized PDI) |
|                              | `Ca` (calcium metal)      | 2.87 eV (vacuum-evaporated thin cathode)     |
|                              | `PFN-Br`                  | 4.20 eV (conjugated polyelectrolyte dipole)  |
+------------------------------+---------------------------+----------------------------------------------+
| **Hole Transport Layer**     | `PEDOT:PSS`               | 5.00 eV (standard aqueous dispersion spin)   |
|                              | `MoO3` (molybdenum oxide) | 5.30 eV (thermally evaporated interlayer)    |
|                              | `2PACz`                   | 5.20 eV (carbazole self-assembled monolayer) |
+------------------------------+---------------------------+----------------------------------------------+
```

### 2.3 Missing-Value Handling Strategy
1. **Unmapped / Custom Materials**:
   - If a user inputs an unrecognized solvent (e.g. proprietary fluorinated green solvent) or a novel interlayer, the property is **genuinely unavailable**. It is mapped to `NaN`.
2. **Model M3 (HistGradientBoostingRegressor)**:
   - Preserves `NaN` directly. The gradient boosted tree natively routes missing values during split traversal.
3. **Model M2 and M2A (RandomForestRegressor)**:
   - Because `RandomForestRegressor` requires numeric floats, unobserved entries are imputed using the **training set median of observed values computed strictly on $D_{\text{train}}$**:
     - `solvent_boiling_point_c`: Observed median on $D_{\text{train}}$ = **131.6 °C** (the exact boiling point of chlorobenzene, the median solvent in OPV literature).
     - `etl_workfunction_ev`: Observed median on $D_{\text{train}}$ = **4.30 eV** (the exact work function of ZnO).
     - `htl_workfunction_ev`: Observed median on $D_{\text{train}}$ = **5.00 eV** (the exact work function of PEDOT:PSS).
   - In Model M2, this imputation is strictly accompanied by active missingness indicators (`missing_solvent = 1` or `missing_interlayers = 1`).

---

## 3. Validation and Governance of Partial-Input Patterns

### 3.1 Empirical Evidence in the Benchmark Dataset
Empirical analysis of the 14,704 records in $D_{\text{train}}$ provides the mutually exclusive and exhaustive distribution of reported processing fields (reconciled in `ml/PHASE_8_0_4_FINAL_STATISTICAL_CLARIFICATIONS.md`):

```
+---------------------------------------------------------------------------------------------------------------------+
| Supported Input Patterns and Empirical Training Evidence (D_train, N = 14,704)                                      |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
| ID | Input Pattern Description      | D_train Count | % of Data | Supported Route       | Operational State         |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
| P1 | **Fully Specified Processing** | 457 records   | 3.11%     | Model M2 or M3        | Fully Specified Recipe    |
|    | (All 7 main fields provided)   |               |           | (missing_* = 0)       | Observed in training data |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
| P2 | **Rich Processing**            | 1,750 records | 11.90%    | Model M2 or M3        | Partial Conditioned       |
|    | (6 of 7 fields present)        |               |           | (active flags = 1)    | Observed in training data |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
| P3 | **Moderate Partial Processing**| 4,804 records | 32.67%    | Model M2 or M3        | Partial Conditioned       |
|    | (4 to 5 fields present)        |               |           | (active flags = 1)    | Observed in training data |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
| P4 | **Low Partial Processing**     | 4,512 records | 30.69%    | Model M2 or M3        | Partial Conditioned       |
|    | (2 to 3 fields present)        |               |           | (active flags = 1)    | Observed in training data |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
| P5 | **Minimal Processing**         | 1,633 records | 11.11%    | Model M2 or M3        | Partial Conditioned       |
|    | (Exactly 1 field present)      |               |           | (active flags = 1)    | Observed in training data |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
| P6 | **Molecular-Descriptor-Only**  | 1,548 records | 10.53%    | **Model M0**          | **Molecular Baseline**    |
|    | (Zero processing fields)       |               |           | (Molecular descriptors| Observed in training data |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
|    | **Total (Exhaustive Sum)**     | 14,704 records| 100.0%    | Mutually exclusive    | Zero unclassified records |
+----+--------------------------------+---------------+-----------+-----------------------+---------------------------+
```

### 3.2 Non-Inference Rule for Physical Regimes
When processing variables are omitted, the pipeline **must never silently represent missing conditions as known physical regimes**:
- If `annealing_temp` is missing: `regime_thermal_annealed = 0` **and** `regime_as_cast_room_temp = 0`. The system must not assume room temperature as-cast processing.
- If `additive_canonical` is missing: `regime_solvent_additive_used = 0` with `missing_additive = 1`. The system must not assume neat blend processing.
- If `device_type` is missing: `regime_conventional_stack = 0` **and** `regime_inverted_stack = 0` with `missing_architecture = 1`.

---

## 4. Categorization of Physical Validation Rules

To prevent arbitrary data rejection, validation rules are partitioned into four distinct operational tiers:

```
+---------------------------------------------------------------------------------------------------------------------+
| Physical Validation Rule Taxonomy                                                                                   |
+--------------------------+-----------------------+-----------------------+------------------------------------------+
| Category                 | Example Input         | System Action         | Physical Justification                   |
+--------------------------+-----------------------+-----------------------+------------------------------------------+
| **Tier 1**               | `"abc nm"`, `"NaN"`,  | **Hard Rejection**    | Malformed strings cannot be mapped to    |
| Syntax & Unit Errors     | negative Kelvin       | (Validation Error)    | mathematical representations.            |
+--------------------------+-----------------------+-----------------------+------------------------------------------+
| **Tier 2**               | Thickness $T \le 0$,  | **Hard Rejection**    | Violates fundamental physics: negative   |
| Physically Impossible    | $T_{\text{ann}} > 350^\circ\text{C}$,| (Physical Error)      | thickness violates spatial geometry;     |
| Values                   | $f_{\text{donor}} \le 0$ or $\ge 1$, |                       | polymer carbonizes/degrades $>350^\circ\text{C}$;|
|                          | Additive Conc $>100\%$|                       | blend requires both donor and acceptor.  |
+--------------------------+-----------------------+-----------------------+------------------------------------------+
| **Tier 3**               | Thickness $300–800$ nm| **Permitted with**    | Physically valid and observed in thick-  |
| Valid but Unusual Values | Temp $150–250^\circ\text{C}$,| **Warning**           | junction literature, but deviates from   |
|                          | Additive Conc $5–15\%$| (`ExtrapolationWarn`) | standard spin-coating laboratory norms.  |
+--------------------------+-----------------------+-----------------------+------------------------------------------+
| **Tier 4**               | Thickness $>600$ nm,  | **Permitted with**    | Outside 99th percentile of empirical     |
| Outside Observed Training| Temp $>250^\circ\text{C}$,    | **Warning**           | benchmark distribution; tree models may  |
| Distribution             | Donor MW $>3000$ g/mol| (`DomainWarn`)        | extrapolate flat leaf values.            |
+--------------------------+-----------------------+-----------------------+------------------------------------------+
```

---

## 5. Verification Checklist

- [x] **M1 Categorical Encoding**: Replaced ordinal integer IDs with 23 one-hot columns fitted strictly on $D_{\text{train}}$. M1 feature count verified at exactly 50 columns. Zero leakage to Cal/Test.
- [x] **Physical Lookup Missingness**: Arbitrary 100°C default eliminated. Missing values preserve `NaN` (for M3) or use training observed medians (131.6°C, 4.30 eV, 5.00 eV for M2) accompanied by explicit missingness indicators.
- [x] **Partial-Input Validation**: Supported input patterns tabulated with empirical frequencies in $D_{\text{train}}$. Silent inference of room-temperature or neat regimes when unobserved is strictly barred.
- [x] **Physical Validation Rules**: Divided into syntax errors, physical impossibilities (hard rejection), and unusual/extrapolation values (warnings with full justification).
- [x] **Automated Tests**: Unit test suite `tests/ml/test_phase_8_feature_matrix.py` updated with 5 test cases, all passing deterministically.
- [x] **Scope Boundaries Preserved**: Zero models trained, zero calibration quantiles calculated, zero final-test predictions generated, zero production code touched, zero Git commits created.

---

**All Phase 8.0.3 pre-experiment corrections are complete and locked.**
