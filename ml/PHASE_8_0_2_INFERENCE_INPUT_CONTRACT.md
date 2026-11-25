# OPV-PredictNet — Phase 8.0.2: Processing-Conditional Inference Input Contract

**Date**: 2026-11-24  
**Scope**: Operational contracts, input validation rules, model dispatch logic, and reporting transparency across molecular-only and processing-conditional inference modes.  
**Deliverable**: Locked Inference Input Contract for Phase 8.

---

## 1. Executive Summary & Non-Negotiable Governance Principles

In real-world OPV research, users query models under varying states of laboratory knowledge—from virtual screening of unmade polymers to fine-tuning multi-parameter spin-coating formulations.

### Non-Negotiable Principles
1. **Zero Silent Presets**: An unobserved or missing processing parameter must **never** be silently filled with an assumed standard laboratory value (e.g., assuming 1:1 ratio or 100 nm thickness) while masquerading as a high-confidence processing-aware prediction.
2. **Explicit Conditioning Transparency**: Every inference response must explicitly declare:
   - The exact model identifier invoked (`M0`, `M1`, `M2`, `M2A`, `M3`).
   - The operational prediction mode (`Molecular-Only` vs. `Processing-Conditional`).
   - The exact parameter list supplied, assumed, or flagged as missing.
3. **No Improvisation on Invalid Data**: If input parameters contain unphysical values or ambiguous units that violate domain physics, the request must be rejected or explicitly flagged—never silently coerced to a plausible default.

---

## 2. Definitive Input Scenario Contracts

```
+---------------------------------------------------------------------------------------------------------+
| Inference Dispatch and Governance Matrix                                                                |
+--------------------------+-----------------------+-----------+------------------------------------------+
| User Input Scenario      | Provided Data         | Model Run | Operational Behavior & Output Label      |
+--------------------------+-----------------------+-----------+------------------------------------------+
| **Scenario 1**           | Donor SMILES +        | **M0**    | Routes strictly to 22-feature molecular  |
| No Processing Supplied   | Acceptor SMILES only  |           | baseline. Processing models NOT invoked. |
| (Virtual Screening)      |                       |           | Label: `Molecular-Only Baseline`         |
+--------------------------+-----------------------+-----------+------------------------------------------+
| **Scenario 2**           | SMILES + Complete     | **M2** or | Normalizes all features. Missingness     |
| Fully Specified Recipe   | Fabrication Parameters| **M3**    | flags set to 0. Physical regimes active. |
| (Controlled Recipe)      |                       |           | Label: `Processing-Conditional (Full)`   |
+--------------------------+-----------------------+-----------+------------------------------------------+
| **Scenario 3**           | SMILES + Partial      | **M2** or | Parsed parameters normalized; unobserved |
| Partially Specified      | Fabrication Conditions| **M3**    | fields flagged with `missing_* = 1`.     |
| (Exploratory Sweep)      | (e.g. ratio + solvent)|           | Label: `Processing-Conditional (Partial)`|
+--------------------------+-----------------------+-----------+------------------------------------------+
| **Scenario 4**           | SMILES + Explicit     | **M2** or | User explicitly requests named recipe.   |
| Named Recipe Preset      | Preset Selection      | **M3**    | Parameters applied with flags = 0.       |
| (Benchmark Recipe)       | (e.g. 'Champion PM6') |           | Label: `Processing-Conditional (Preset)` |
+--------------------------+-----------------------+-----------+------------------------------------------+
| **Scenario 5**           | Unphysical / Ambiguous| **REJECT**| Strict validation error raised.          |
| Invalid / Malformed Data | Values (T <= 0, etc.) | (No run)  | Never silently coerced or guessed.       |
+--------------------------+-----------------------+-----------+------------------------------------------+
```

---

## 3. Detailed Specification by Input Scenario

### Scenario 1: No Processing Conditions Supplied (Molecular-Only Screening)
- **Applicable Use Case**: High-throughput virtual screening of thousands of candidate donor or acceptor molecular candidates prior to synthesis.
- **Inputs**: `donor_smiles` (string), `acceptor_smiles` (string).
- **Execution Contract**:
  - The inference engine routes **exclusively to Model M0** (22 intrinsic RDKit molecular descriptors).
  - Models M1, M2, M2A, and M3 are **not invoked**.
  - No dummy processing values, default ratios, or missingness indicators are constructed.
- **Output Schema**:
  ```json
  {
    "prediction_mode": "molecular_descriptor_only_baseline",
    "model_id": "M0",
    "model_description": "Molecular-Descriptor-Only PCE Baseline (Random Forest on 22 RDKit descriptors)",
    "point_prediction_pce": 9.42,
    "conformal_interval_90": {
      "lower_pce": 4.98,
      "upper_pce": 13.86,
      "width_pce": 8.88
    },
    "processing_context": "None. Prediction reflects measured PCE under the empirical processing and experimental condition distribution of the literature training dataset."
  }
  ```

---

### Scenario 2: Fully Specified Processing Recipe
- **Applicable Use Case**: Laboratory validation where all primary device fabrication parameters are known and controlled.
- **Inputs**: `donor_smiles`, `acceptor_smiles`, plus:
  - `d_a_ratio`: e.g. `"1:1.2"`
  - `solvent`: e.g. `"chlorobenzene"`
  - `additive`: e.g. `"DIO"`
  - `additive_ratio`: e.g. `"0.5 vol%"`
  - `annealing_temp`: e.g. `100.0` (°C)
  - `active_layer_thickness`: e.g. `100.0` (nm)
  - `device_type`: e.g. `"inverted"`
  - `etl`: e.g. `"ZnO"`
  - `htl`: e.g. `"MoO3"`
- **Execution Contract**:
  - Continuous parameters are normalized ($f_{\text{donor}} = 0.455$, $\log_{10}(T) = 2.0$, $T_{\text{norm}} = 0.0$, $\text{conc} = 0.5\%$).
  - All seven missingness indicators are set to **0** (`missing_* = 0`).
  - Physical regimes are activated (`regime_thermal_annealed = 1`, `regime_chlorinated_solvent = 1`, `regime_solvent_additive_used = 1`, `regime_inverted_stack = 1`).
  - Interface constants are populated from verified lookups ($\text{WF}_{\text{ZnO}} = 4.3\text{ eV}$, $\text{WF}_{\text{MoO3}} = 5.3\text{ eV}$, $\text{BP}_{\text{CB}} = 131.6^\circ\text{C}$).
  - Model M2 (or M3) is executed.
- **Output Schema**:
  ```json
  {
    "prediction_mode": "processing_conditional_full",
    "model_id": "M2",
    "point_prediction_pce": 14.85,
    "conformal_interval_90": {
      "lower_pce": 11.95,
      "upper_pce": 17.75,
      "width_pce": 5.80
    },
    "processing_context": {
      "status": "fully_specified",
      "missing_fields": [],
      "active_regimes": ["thermal_annealed", "chlorinated_solvent", "solvent_additive_used", "inverted_stack"]
    }
  }
  ```

---

### Scenario 3: Partially Specified Processing Conditions
- **Applicable Use Case**: An experimentalist has chosen a blend ratio and host solvent, but has not yet decided on thermal annealing or interlayers.
- **Inputs**: `donor_smiles`, `acceptor_smiles`, with partial fabrication keys (e.g., `d_a_ratio = "1:1"`, `solvent = "chloroform"`).
- **Execution Contract**:
  - Provided fields are normalized into continuous or regime representations.
  - Omitted fields have their corresponding missingness flags set to **1** (`missing_thickness = 1`, `missing_annealing = 1`, `missing_interlayers = 1`).
  - Continuous omitted fields receive frozen $D_{\text{train}}$ medians in Model M2, or retain `NaN` in Model M3.
  - Regimes associated with unobserved parameters remain **0** (`regime_thermal_annealed = 0`, `regime_as_cast_room_temp = 0`).
- **Output Schema**:
  ```json
  {
    "prediction_mode": "processing_conditional_partial",
    "model_id": "M2",
    "point_prediction_pce": 12.10,
    "conformal_interval_90": {
      "lower_pce": 8.60,
      "upper_pce": 15.60,
      "width_pce": 7.00
    },
    "processing_context": {
      "status": "partially_specified",
      "provided_fields": ["d_a_ratio", "solvent"],
      "unobserved_fields": ["active_layer_thickness", "annealing_temp", "additive", "device_type", "etl", "htl"],
      "missingness_flags_active": ["missing_thickness", "missing_annealing", "missing_additive", "missing_architecture", "missing_interlayers"]
    }
  }
  ```

---

### Scenario 4: Explicitly Selected Recipe Presets
- **Applicable Use Case**: A researcher wishes to simulate how a new pair would perform under standard high-efficiency laboratory protocols without manually entering nine parameters.
- **Inputs**: `donor_smiles`, `acceptor_smiles`, `preset_id = "CHAMPION_INVERTED_DIO"`.
- **Execution Contract**:
  - The preset applies verified parameters:
    - Blend ratio: `1:1.2`
    - Solvent: `chloroform`
    - Additive: `DIO` at `0.5 vol%`
    - Thermal annealing: `100.0 °C`
    - Thickness: `100.0 nm`
    - Architecture: `inverted` (`ZnO` ETL, $\text{MoO}_3$ HTL)
  - Missingness flags are set to **0** because the recipe is fully specified.
  - The prediction output is explicitly labeled as reflecting this specific preset.

---

### Scenario 5: Physical Validation Rules and Error Taxonomy
Validation rules are categorized into four distinct operational tiers:

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

## 4. Supported Partial-Input Patterns and Empirical Training Evidence

Empirical analysis of the 14,704 training records in $D_{\text{train}}$ provides the mutually exclusive and exhaustive distribution of reported processing fields (reconciled in `ml/PHASE_8_0_4_FINAL_STATISTICAL_CLARIFICATIONS.md`):

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

*Non-Inference Rule for Regimes*: Missing conditions must never be silently coded as active physical regimes. If annealing temperature is omitted, `regime_thermal_annealed = 0` and `regime_as_cast_room_temp = 0`, with `missing_annealing = 1`.

---

## 5. Pipeline Rejection Boundary

If an incoming inference request attempts to query a model with unsupported combinations (e.g. attempting to run Model M1 with missing interlayers when M1 requires complete ordinal codes, or submitting unparseable SMILES), the system raises a typed exception:

```python
class UnsupportedInferenceInputError(ValueError):
    """Raised when user inputs cannot be mapped to the requested model's contract."""
    pass
```

The system will never improvise missing values, never fill in arbitrary numbers, and never return an unflagged prediction.

---

**Inference Input Contract locked and approved.**
