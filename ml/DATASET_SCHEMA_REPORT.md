# OPV-PredictNet — Phase 4: Dataset Schema & Data Dictionary Report

**Document Version:** 1.0.0  
**Phase:** 4 (Dataset Discovery, Source Verification, and Quality Assessment)  
**Date:** November 2026  
**Status:** Canonical Schema Specification  

---

## 1. Overview & Data Architecture

This report provides the exhaustive schema specification, column taxonomy, and data dictionary for the candidate datasets evaluated for OPV-PredictNet, focusing primarily on the primary candidate: **OPV-DB v1.1.0** (Jiangjie Qiu et al. / Tsinghua University; Version DOI: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960); Concept DOI: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542); Archive: `opvdb_2026_q3.zip`), with structural comparisons against **HOPV15** and **CEPDB**.

The primary candidate's release packaging (`opvdb_2026_q3.zip`) comprises two core relational tables:
1. **`data/opv_devices.csv`**: The master observation table recording individual experimental device fabrication runs, literature citations, device stack architectures, and measured photovoltaic figures of merit.
2. **`data/materials_reference.csv`**: The entity reference table mapping material identifiers, common names, aliases, material classifications, and validated canonical SMILES representations.

---

## 2. OPV-DB Master Schema: `data/opv_devices.csv`

### 2.1 Schema Definition & Data Dictionary

| Column Identifier | SQL Data Type | Unit | Nullable | Description & Scientific Semantics | Provenance Label |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `device_id` | `VARCHAR(64)` | None | No | Unique primary key identifying the experimental device run. | Verified from source |
| `doi` | `VARCHAR(128)` | None | No | Digital Object Identifier of the original peer-reviewed publication. | Verified from source |
| `publication_year` | `INTEGER` | Year | No | Year of publication (temporal tracking for dataset splitting). | Verified from source |
| `donor_name` | `VARCHAR(128)` | None | No | Standardized chemical label or common acronym of the donor. | Verified from source |
| `donor_smiles` | `TEXT` | None | Yes | Canonical SMILES string of the donor material (or repeat unit). | Verified from source |
| `acceptor_name` | `VARCHAR(128)` | None | No | Standardized chemical label or common acronym of the acceptor. | Verified from source |
| `acceptor_smiles` | `TEXT` | None | Yes | Canonical SMILES string of the acceptor molecule. | Verified from source |
| `d_a_ratio` | `VARCHAR(32)` | Ratio | Yes | Donor-to-acceptor weight stoichiometry (e.g., `'1:1'`, `'1:1.2'`). | Reported by source |
| `active_layer_thickness_nm`| `FLOAT` | nm | Yes | Film thickness of the active blend layer (typically 70–300 nm). | Reported by source |
| `host_solvent` | `VARCHAR(64)` | None | Yes | Primary processing solvent (e.g. Chlorobenzene, Chloroform). | Reported by source |
| `solvent_additive` | `VARCHAR(64)` | None | Yes | Morphology-directing additive (e.g. 1,8-diiodooctane [DIO], CN).| Reported by source |
| `additive_conc_vol_pct` | `FLOAT` | vol % | Yes | Volume percentage of the solvent additive in host solvent (0–5%).| Reported by source |
| `thermal_annealing_temp_c`| `FLOAT` | °C | Yes | Post-deposition thermal annealing temperature (20–200 °C). | Reported by source |
| `thermal_annealing_time_min`|`FLOAT` | min | Yes | Duration of thermal annealing process (typically 5–60 min). | Reported by source |
| `device_architecture` | `VARCHAR(32)` | None | Yes | Stack polarity: `'conventional'` (ITO/HTL/AL/ETL/Metal) or `'inverted'`.| Reported by source |
| `htl` | `VARCHAR(64)` | None | Yes | Hole transport layer material (e.g. `'PEDOT:PSS'`, `'MoO3'`). | Reported by source |
| `etl` | `VARCHAR(64)` | None | Yes | Electron transport layer material (e.g. `'PDINN'`, `'ZnO'`, `'PFN-Br'`).| Reported by source |
| `voc_v` | `FLOAT` | V | Yes | Measured Open-Circuit Voltage (AM1.5G, 100 mW/cm² inferred). | Reported by source (Testing conditions inferred) |
| `jsc_ma_cm2` | `FLOAT` | $\text{mA/cm}^2$ | Yes | Measured Short-Circuit Current Density (AM1.5G inferred). | Reported by source (Testing conditions inferred) |
| `ff_pct` | `FLOAT` | % | Yes | Measured Fill Factor (normalized to 0–100% scale). | Reported by source (Unit conversion applied) |
| `pce_pct` | `FLOAT` | % | No | Reported Power Conversion Efficiency (AM1.5G inferred). | Reported by source (Testing conditions inferred) |
| `pce_calc_pct` | `FLOAT` | % | Yes | Internally recomputed PCE: $V_{OC} \cdot J_{SC} \cdot FF / 100$. | Verified from source (Internal heuristic) |
| `pce_rel_diff_pct` | `FLOAT` | % | Yes | Relative disparity: $|PCE_{rep} - PCE_{calc}| / PCE_{rep} \times 100$.| Verified from source |
| `in_strict_performance_benchmark`| `BOOLEAN` | None | No | Quality flag: complete $V_{OC}, J_{SC}, FF, PCE$ & $|PCE_{diff}| < 10\%$.| Reported by source (Internal heuristic) |
| `in_strict_molecular_benchmark` | `BOOLEAN` | None | No | Quality flag: strict performance satisfied + valid SMILES for both.| Reported by source (Internal heuristic) |

---

## 3. OPV-DB Materials Reference Schema: `data/materials_reference.csv`

### 3.1 Schema Definition & Data Dictionary

| Column Identifier | SQL Data Type | Nullable | Description & Chemical Semantics | Provenance Label |
| :--- | :--- | :--- | :--- | :--- |
| `material_id` | `VARCHAR(64)` | No | Unique canonical material key (e.g., `'mat_pm6'`, `'mat_y6'`). | Verified from source |
| `preferred_name` | `VARCHAR(128)` | No | Primary IUPAC or standardized literature acronym (e.g. `'PM6'`). | Verified from source |
| `aliases_json` | `TEXT` | Yes | JSON array of documented literature aliases (e.g. `["PBDB-T-2F"]`).| Reported by source |
| `material_type` | `VARCHAR(32)` | No | Classification: `'polymer_donor'`, `'small_molecule_donor'`, `'fullerene_acceptor'`, `'nfa'`. | Verified from source |
| `canonical_smiles` | `TEXT` | No | RDKit-standardized canonical SMILES representation. | Verified from source |
| `molecular_formula`| `VARCHAR(64)` | Yes | Hill notation formula for monomer repeat unit or small molecule. | Inferred from SMILES |
| `monomer_molecular_weight` | `FLOAT` | Yes | Molecular weight (g/mol) of the finite repeat unit / molecule. | Inferred from SMILES |
| `reported_bandgap_ev` | `FLOAT` | Yes | Experimentally reported optical bandgap from UV-Vis absorption edge.| Reported by source |
| `reported_homo_ev` | `FLOAT` | Yes | Experimentally reported HOMO energy level from UPS / CV (eV). | Reported by source |
| `reported_lumo_ev` | `FLOAT` | Yes | Experimentally reported LUMO energy level from IPES / CV (eV). | Reported by source |

---

## 4. Photovoltaic Target Metric Representation & Conversion Standards

Different publications and historical datasets report photovoltaic figures of merit using varying unit conventions. The schema harmonizes these into SI and IUPAC standards:

### 4.1 Target Metric Standards

```
+------------------------------------+--------------------------+-----------------------+-----------------------------+
| Metric                             | Primary IUPAC / SI Unit  | Raw Literature Units  | Normalization Factor        |
+------------------------------------+--------------------------+-----------------------+-----------------------------+
| Power Conversion Efficiency (PCE)  | %                        | %, fraction (0-0.25)  | If val < 1.0 -> val * 100.0 |
| Open-Circuit Voltage (Voc)         | Volts (V)                | V, mV                 | If val > 100 -> val / 1000  |
| Short-Circuit Current (Jsc)        | mA/cm²                   | mA/cm², A/m²          | If val > 100 -> val / 10.0  |
| Fill Factor (FF)                   | % (0 - 100%)             | %, fraction (0 - 1.0) | If val <= 1.0 -> val * 100.0|
+------------------------------------+--------------------------+-----------------------+-----------------------------+
```

### 4.2 Recomputation Consistency Equation
In single-junction photovoltaic testing under standard test conditions (AM1.5G, $P_{in} = 100\text{ mW/cm}^2$), Power Conversion Efficiency is physically constrained by the product of $V_{OC}, J_{SC}$, and $FF$:
$$PCE_{calc} = \frac{V_{OC}\text{ (V)} \cdot J_{SC}\text{ (mA/cm}^2\text{)} \cdot FF\text{ (fraction)}}{P_{in}\text{ (mW/cm}^2\text{)}} \cdot 100\% = \frac{V_{OC} \cdot J_{SC} \cdot FF_{\%}}{100}$$

In OPV-DB, any record where $|PCE_{reported} - PCE_{calc}| / PCE_{reported} > 0.10$ ($10\%$ relative discrepancy) is flagged for rounding or typographical discrepancy and excluded from the `in_strict_performance_benchmark`.

---

## 5. Comparative Schema Mapping Across Candidate Datasets

| Schema Dimension | OPV-DB (Primary) | HOPV15 | Harvard CEPDB |
| :--- | :--- | :--- | :--- |
| **Primary Key** | Device Run ID | InChIKey / Entry ID | InChIKey / Compound ID |
| **Donor Identifiers** | Name, Aliases, SMILES | InChI, Canonical SMILES | Generated Virtual SMILES |
| **Acceptor Identifiers** | Name, Aliases, SMILES | Fixed ($PC_{61}BM, PC_{71}BM$) | Fixed theoretical acceptor |
| **PCE Field** | `pce_pct` (Real lab) | `pce` (Real lab) | `pce_sch` (Simulated Scharber) |
| **Voc Field** | `voc_v` (Real lab) | `voc` (Real lab) | None (Derived intermediate) |
| **Jsc Field** | `jsc_ma_cm2` (Real lab) | `jsc` (Real lab) | None (Derived intermediate) |
| **FF Field** | `ff_pct` (Real lab) | `ff` (Real lab) | None (Fixed at 65%) |
| **Processing Conditions** | 8 dedicated columns | Unstructured notes | None (Gas-phase single mol) |
| **Quantum Chemistry** | None (Empirical focus) | Matched DFT (B3LYP, PBE) | 100% DFT (B3LYP/6-31G*) |
| **Publication DOI** | Mandatory on every row | Mandatory | None (In-house computational) |
| **Data Integrity Flag** | Embedded boolean filters | None (Manual curation) | Convergence error codes |

---

## 6. Verification Status Summary

- **Verified directly from source:**
  - OPV-DB v1.1.0 column naming conventions, boolean benchmark filter definitions (`in_strict_performance_benchmark`, `in_strict_molecular_benchmark`), and unit representations from official Zenodo release records (Version DOI: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960); Concept DOI: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542); archive: `opvdb_2026_q3.zip`).
  - HOPV15 column keys and data types from Figshare repository (`10.6084/m9.figshare.1610049`).
  - CEPDB SQL schema and Scharber formulation from EES publication (`10.1039/c1ee01463e`).
- **Reported by source:**
  - Processing metadata distributions (thickness ranges, additive concentrations).
- **Inferred from metadata:**
  - Standard measurement conditions (AM1.5G, 100 mW/cm², 25°C): Inferred from publication conventions; **NOT** explicitly documented or sensor-verified per record in the database tables.
  - Normalization rules converting fractional fill factor ($0.75$) to percentage ($75.0\%$) and mV to V.
- **Not available / unverified:**
  - Full-dataset descriptor extraction: **Pending empirical verification** against `opvdb_2026_q3.zip` upon authorized download.
  - Real-time environmental humidity and glovebox oxygen concentration ($<0.1\text{ ppm}$) during spin-coating (seldom reported uniformly across historical literature).
