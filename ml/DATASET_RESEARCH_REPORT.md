# OPV-PredictNet — Phase 4: Candidate Dataset Research Report

**Document Version:** 1.0.0  
**Phase:** 4 (Dataset Discovery, Source Verification, and Quality Assessment)  
**Date:** November 2026  
**Status:** Verification & Provenance Complete (Pre-Training Phase)  

---

## 1. Executive Summary & Objective

The objective of Phase 4 is to identify, verify, and critically evaluate genuine experimental organic photovoltaic (OPV) datasets for training the OPV-PredictNet machine learning screening engine.

In accordance with strict scientific integrity rules:
- **No data was downloaded or staged without verification.**
- **No synthetic or fabricated experimental records are admitted.**
- **Computational quantum-chemical simulations are explicitly separated from physical laboratory measurements.**
- **Candidate datasets are evaluated on provenance, experimental rigor, chemical completeness, and task alignment.**

Five candidate dataset families were systematically investigated:
1. **OPV-DB (Primary Candidate)**: Literature-mined organic photovoltaic device performance database (Jiangjie Qiu et al. / Tsinghua University; extended builds).
2. **Harvard Clean Energy Project (CEP / CEPDB)**: High-throughput computational DFT screening dataset (Johannes Hachmann, Alán Aspuru-Guzik et al.).
3. **Harvard Organic Photovoltaic Dataset 2015 (HOPV15)**: Curated literature calibration set with matched experimental and DFT data (Steven A. Lopez et al.).
4. **Nagasawa & Saeki Conjugated Polymer Dataset**: Polymer donor classification set compiled from literature (Shinji Nagasawa, Akinori Saeki et al.).
5. **Zhao, del Cueto & Troisi Small-Molecule OPV Dataset**: Small-molecule donor-acceptor database (Zhi-Wen Zhao, Marcos del Cueto, Alessandro Troisi et al.).

---

## 2. Categorization Framework & Scientific Distinctions

To prevent methodological errors common in materials informatics, candidate datasets are categorized across five distinct data archetypes:

```
+---------------------------------------------------------------------------------------------------+
|                                  DATASET ARCHETYPE TAXONOMY                                       |
+------------------------------------+--------------------------------------------------------------+
| 1. Experimental Device Records     | Physical laboratory devices fabricated and measured under     |
|    (Highest Priority for OPV)      | simulated solar irradiance (AM1.5G, 100 mW/cm², 25°C).        |
|                                    | Reports: PCE, Voc, Jsc, FF, blend ratio, solvent, annealing. |
+------------------------------------+--------------------------------------------------------------+
| 2. Literature-Mined Records        | Manually or semi-automatically extracted from peer-reviewed  |
|    (Valuable, High Heterogeneity)  | publications. Reflects real lab variance across world groups.|
|                                    | Requires deduplication, curation, and unit standardization.  |
+------------------------------------+--------------------------------------------------------------+
| 3. Computational / DFT Sets        | First-principles quantum chemistry (HOMO, LUMO, gap).        |
|    (Zero Device Realism)           | Calculated on isolated gas-phase conformers or oligomers.     |
|                                    | Lacks film morphology, packing, and interface physics.       |
+------------------------------------+--------------------------------------------------------------+
| 4. Simulated Device Properties     | Empirical or drift-diffusion models (e.g. Scharber model).   |
|    (Synthetic Efficiencies)        | Estimates theoretical upper limits, not actual lab devices.  |
+------------------------------------+--------------------------------------------------------------+
| 5. Chemical / Structural Bases     | Molecular repositories (PubChem, ChEMBL, Cambridge CSD).    |
|    (Structure Only)                | Molecular connectivity only; no device performance figures.  |
+------------------------------------+--------------------------------------------------------------+
```

---

## 3. Systematic Investigation of Candidate Datasets

### 3.1 Candidate 1: OPV-DB (Literature-Mined Experimental OPV Database)

- **Primary Investigators:** Jiangjie Qiu, Xuan Yang, et al. (Tsinghua University).
- **Core Publication / Dataset Reference:** *"OPV-DB: a literature-mined organic photovoltaic device performance database"*.
- **Dataset Version:** OPV-DB v1.1.0.
- **Version-Specific DOI:** Zenodo DOI: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960) (permanently identifies the frozen v1.1.0 release snapshot).
- **Concept DOI:** Zenodo DOI: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542) (umbrella identifier resolving to the latest iteration across all releases).
- **Official Distribution Archive:** `opvdb_2026_q3.zip` (contains structured tables including `data/opv_devices.csv` and `data/materials_reference.csv`).
- **Derivative / Extended Build Reference:** GitHub `jongsu-king/opv-db-based-on-literature` / Zenodo concept DOI: [`10.5281/zenodo.23682463`](https://doi.org/10.5281/zenodo.23682463).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Archetype:** Literature-Mined Experimental Device Measurements.
- **Data Provenance & Traceability:** Every device record is tagged with its original source publication DOI, journal name, publication year, and specific table/figure reference.

#### Available Data Dimensions:
- **Photovoltaic Figures of Merit:** Power Conversion Efficiency (PCE, %), Open-Circuit Voltage ($V_{OC}$, V), Short-Circuit Current Density ($J_{SC}$, $\text{mA/cm}^2$), Fill Factor (FF, % or ratio 0–1).
- **Chemical Identifiers:** Donor chemical name/acronym, Acceptor chemical name/acronym, Donor canonical SMILES, Acceptor canonical SMILES.
- **Materials Reference Table (`materials_reference.csv`):** Mapping of donor and acceptor material classes (polymer donor, small-molecule donor, fullerene acceptor, non-fullerene acceptor), CAS/common aliases, and curated molecular structures.
- **Fabrication & Architecture Metadata:**
  - Active layer thickness (nm).
  - Donor:Acceptor stoichiometry (weight ratio).
  - Processing host solvent (e.g. Chlorobenzene, Chloroform, o-Dichlorobenzene).
  - Solvent additives (e.g. 1,8-diiodooctane [DIO], 1-chloronaphthalene [CN]) and concentration (vol %).
  - Thermal annealing temperature (°C) and duration (min).
  - Device stack architecture: Conventional (`ITO/PEDOT:PSS/Active Layer/Metal`) vs. Inverted (`ITO/Metal Oxide/Active Layer/Top Anode`).
- **Illumination & Measurement Condition Assessment:**
  - *Standard Test Conditions (STC: AM1.5G, $100\text{ mW/cm}^2$, $25^\circ\text{C}$)* are **inferred** from general literature conventions, but are **NOT explicitly documented or sensor-verified per record in the database tables**.
  - In primary literature, active temperature control at 25°C is rarely maintained during continuous measurement; cell temperatures often reach 30–40°C under 1-sun irradiance.
  - Spectral mismatch corrections between laboratory solar simulators and the reference ASTM G173-03 spectrum are seldom reported in primary synthesis papers, introducing an estimated ±5–10% systematic experimental variance across laboratories.
- **Quality Subsets (Embedded Flags):**
  - `in_strict_performance_benchmark`: Complete core parameters ($V_{OC}, J_{SC}, FF, PCE$) with internal recomputed PCE consistency ($PCE_{calc} = V_{OC} \cdot J_{SC} \cdot FF / 100$).
  - `in_strict_molecular_benchmark`: Satisfies strict performance criteria and provides curated SMILES strings for both donor and acceptor. Note: These flags represent the dataset authors' internal filtering heuristics, not external physical proof of experimental validity.

- **Verification Status:**
  - *Version DOI:* Verified directly from Zenodo record [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960).
  - *Concept DOI:* Verified directly from Zenodo concept [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542).
  - *Archive Name:* Verified as `opvdb_2026_q3.zip`.
  - *License:* Verified directly from source (CC BY 4.0).
  - *Measurement Conditions:* Inferred from literature conventions; not verified per record.

---

### 3.2 Candidate 2: Harvard Clean Energy Project (CEP / CEPDB)

- **Primary Investigators:** Johannes Hachmann, Roberto Olivares-Amaya, Alán Aspuru-Guzik, et al. (Harvard University / University at Buffalo / IBM World Community Grid).
- **Core Publications:**
  - Hachmann, J., et al., *"The Harvard Clean Energy Project: Large-scale computational screening and design of organic photovoltaics on the World Community Grid"*, *Energy & Environmental Science*, 2011, 4, 3271-3281, DOI: [`10.1039/c1ee01463e`](https://doi.org/10.1039/c1ee01463e).
  - Hachmann, J., et al., *"Lead candidates for high-performance organic photovoltaics from high-throughput quantum chemistry"*, *Physical Chemistry Chemical Physics*, 2014, 16, 22449-22457, DOI: [`10.1039/C4CP03157J`](https://doi.org/10.1039/C4CP03157J).
- **Repository:** Clean Energy Project Database (CEPDB), IBM World Community Grid archives.
- **License:** Open Access research data (academic use).
- **Archetype:** **Computationally Generated Molecular Data / Simulated Properties (100% In-Silico DFT)**.

#### Critical Scientific Assessment:
- **Scale:** Over 2.3 million candidate organic molecular structures screened through $>150$ million quantum-chemical calculations.
- **Computed Properties:** Gas-phase DFT electronic properties at the B3LYP/6-31G* level of theory (HOMO energy, LUMO energy, fundamental bandgap, chemical hardness).
- **Photovoltaic "PCE" Metric:** The PCE values reported in CEP are **not experimentally measured**. They are theoretical estimations derived from the semi-empirical **Scharber model** (Scharber et al., *Adv. Mater.* 2006):
  $$PCE_{sim} = \frac{V_{OC}^{est} \cdot J_{SC}^{est} \cdot FF^{est}}{P_{in}} = \frac{(E_{LUMO}^{Acceptor} - E_{HOMO}^{Donor} - 0.3) \cdot J_{SC}^{est} \cdot 0.65}{100}$$
- **Absence of Experimental Reality:**
  - Zero fabricated physical devices.
  - Zero experimental I-V curve measurements.
  - No active layer thin-film morphology.
  - No solvent or processing effects.
  - Assumes an idealized, fixed fullerene acceptor ($PC_{61}BM$) and fixed 65% fill factor.
- **Suitability Verdict:** **UNSUITABLE as an experimental training ground-truth.** Using CEP as training targets would cause OPV-PredictNet to predict theoretical Scharber model approximations rather than real physical device performance.

---

### 3.3 Candidate 3: Harvard Organic Photovoltaic Dataset 2015 (HOPV15)

- **Primary Investigators:** Steven A. Lopez, Edward O. Pyzer-Knapp, Alán Aspuru-Guzik, et al. (Harvard University).
- **Core Publication:** Lopez, S. A., et al., *"The Harvard organic photovoltaic dataset"*, *Scientific Data*, 2016, 3, 160086, DOI: [`10.1038/sdata.2016.86`](https://doi.org/10.1038/sdata.2016.86).
- **Repository / Download Location:** Figshare DOI: [`10.6084/m9.figshare.1610049`](https://doi.org/10.6084/m9.figshare.1610049).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Archetype:** Curated Experimental Literature Measurements Paired with DFT Quantum Chemistry.

#### Available Data Dimensions:
- **Sample Size:** 350 unique experimentally characterized donor-acceptor photovoltaic records.
- **Experimental Figures of Merit:** Real literature-extracted PCE (%), $V_{OC}$ (V), $J_{SC}$ ($\text{mA/cm}^2$), and Fill Factor (%).
- **Matched DFT Calculations:** Frontier orbital energies and geometries computed across multiple functionals (B3LYP, PBE, BP86, M06-2X) and basis sets (def2-SVP, 6-31G*).
- **Molecular Representations:** InChI, SMILES, and 3D coordinate files (XYZ, PDB).

#### Critical Scientific Limitations for OPV-PredictNet:
1. **Sample Scarcity:** $N \approx 350$ is insufficient for robust multi-descriptor cross-validation and scaffold generalization without severe statistical variance.
2. **Acceptor Homogeneity:** Almost all entries ($>95\%$) are paired with fullerene derivatives ($PC_{61}BM$ or $PC_{71}BM$). Modern state-of-the-art non-fullerene acceptors (NFAs) such as ITIC, Y6, and their halogenated derivatives are entirely absent.
3. **Limited Process Conditions:** Thin-film processing parameters (thickness, additives, thermal treatment) are inconsistently reported across the 350 entries.
- **Suitability Verdict:** Suitable as a historical cross-validation benchmark or sanity control, but **insufficient as the primary training baseline for modern non-fullerene OPV architectures**.

---

### 3.4 Candidate 4: Nagasawa & Saeki Conjugated Polymer Dataset (2018)

- **Primary Investigators:** Shinji Nagasawa, Eman Al-Naamani, Akinori Saeki (Osaka University).
- **Core Publication:** Nagasawa, S., Al-Naamani, E., & Saeki, A., *"Computer-Aided Screening of Conjugated Polymers for Organic Solar Cell: Classification by Random Forest"*, *The Journal of Physical Chemistry Letters*, 2018, 9, 10, 2639-2646, DOI: [`10.1021/acs.jpclett.8b00635`](https://doi.org/10.1021/acs.jpclett.8b00635).
- **Repository Location:** Supporting Information of ACS publication (`acs.jpclett.8b00635`).
- **License:** ACS Journal Author Rights / Academic Research Use.
- **Archetype:** Literature-Mined Polymer Donor Dataset.

#### Available Data Dimensions:
- **Sample Size:** ~1,200 conjugated polymer donor records compiled from ~500 literature papers.
- **Reported Features:** Polymer molecular weight ($M_n$), optical bandgap ($E_g$), HOMO/LUMO levels, and observed device PCE.
- **Acceptor Identity:** Dominated by fullerene acceptors ($PC_{61}BM$ and $PC_{71}BM$).

#### Critical Scientific Limitations for OPV-PredictNet:
1. **Target Formulation:** Data was processed primarily for **binary classification** (e.g. PCE $\ge 4\%$ or PCE $\ge 8\%$) rather than multi-target regression ($PCE, V_{OC}, J_{SC}, FF$).
2. **Missing Photovoltaic Quadrant:** $V_{OC}, J_{SC}$, and Fill Factor are not systematically tabulated alongside PCE across the full cohort.
3. **Acceptor Diversity:** Lacks modern non-fullerene acceptor diversity.
- **Suitability Verdict:** Highly informative for conjugated polymer donor descriptors, but **unsuitable as a primary multi-target regression dataset**.

---

### 3.5 Candidate 5: Zhao, del Cueto & Troisi Small-Molecule OPV Dataset (2020)

- **Primary Investigators:** Zhi-Wen Zhao, Marcos del Cueto, Yun Geng, Alessandro Troisi (University of Liverpool / Northeast Normal University).
- **Core Publication:** Zhao, Z.-W., del Cueto, M., Geng, Y., & Troisi, A., *"Effect of Increasing the Descriptor Set on Machine Learning Prediction of Small Molecule-Based Organic Solar Cells"*, *Chemistry of Materials*, 2020, 32, 18, 7777–7787, DOI: [`10.1021/acs.chemmater.0c02325`](https://doi.org/10.1021/acs.chemmater.0c02325).
- **Repository Location:** GitHub [`marcosdelcueto/MachineLearning_AcceptorDonor`](https://github.com/marcosdelcueto/MachineLearning_AcceptorDonor).
- **License:** Open Source MIT / Academic.
- **Archetype:** Curated Small-Molecule Experimental Literature Records.

#### Available Data Dimensions:
- **Sample Size:** ~500 small-molecule donor entries paired with $PC_{71}BM$.
- **Reported Features:** Complete experimental photovoltaic metrics ($PCE, V_{OC}, J_{SC}, FF$), molecular structures with calculated descriptors.
- **Limitation:** Restricted entirely to small-molecule donors; polymers are intentionally excluded. Single acceptor ($PC_{71}BM$).
- **Suitability Verdict:** Valuable for small-molecule vs. polymer sensitivity analysis, but **too domain-restricted to serve as the general OPV engine dataset**.

---

## 4. Comprehensive Candidate Comparison Matrix

| Evaluation Dimension | Candidate 1: OPV-DB | Candidate 2: Harvard CEP | Candidate 3: HOPV15 | Candidate 4: Saeki (2018) | Candidate 5: Troisi (2020) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Data Nature** | Real Experimental Devices | Purely Simulated DFT | Real Experimental + DFT | Real Experimental Devices | Real Experimental Devices |
| **Source Type** | Literature-Mined ($>1,000$ papers) | High-Throughput DFT ($>2\text{M}$ mols) | Curated Literature ($350$ pairs) | Literature-Mined ($500$ papers) | Curated Literature |
| **DOI / Citation** | Version: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960)<br>Concept: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542)<br>Archive: `opvdb_2026_q3.zip` | [`10.1039/c1ee01463e`](https://doi.org/10.1039/c1ee01463e) | [`10.1038/sdata.2016.86`](https://doi.org/10.1038/sdata.2016.86) | [`10.1021/acs.jpclett.8b00635`](https://doi.org/10.1021/acs.jpclett.8b00635) | [`10.1021/acs.chemmater.0c02325`](https://doi.org/10.1021/acs.chemmater.0c02325) |
| **License** | CC BY 4.0 | Academic Research | CC BY 4.0 | ACS Author Rights | MIT / Open Academic |
| **Material Diversity** | Polymers + Small Molecules + NFAs | Virtual Donor Library | Fullerene-dominant ($PC_{61}BM$) | Polymers only | Small molecules only |
| **NFA Coverage (ITIC, Y6)** | Extensive Non-Fullerene Records | Zero (Pre-dates modern NFAs) | Negligible ($<2\%$) | Negligible | Zero ($PC_{71}BM$ only) |
| **PCE Availability** | Real Experimental (%) | Simulated Scharber Model (%) | Real Experimental (%) | Real Experimental (%) | Real Experimental (%) |
| **Voc, Jsc, FF Metrics** | Complete in Strict Benchmark | None (Synthetic approximations) | Complete for all entries | Incomplete / Binned | Complete for all entries |
| **SMILES Completeness** | Validated in Molecular Benchmark | Full SMILES (virtual) | Full SMILES / InChI | Partial SMILES | Full SMILES |
| **Device Processing Data** | Thickness, Solvents, Additives, Temp | None (Vacuum / gas-phase) | Unstandardized | Minimal ($M_n$ only) | Solvent / additive annotations |
| **Primary Limitation** | Publication clustering & variance | **Zero experimental devices** | Small sample size ($N \approx 350$) | Targets binned for classification | Small molecules only ($PC_{71}BM$) |
| **Suitability for Project** | **Recommended Candidate** | **Unsuitable** | **Secondary Reference Only** | **Supplementary Reference** | **Supplementary Reference** |

---

## 5. Verification Provenance Classification

All dataset metadata in this report is strictly designated according to the required four-tier verification taxonomy:

- **Verified directly from source:**
  - OPV-DB v1.1.0 Zenodo release: Version DOI [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960), Concept DOI [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542), archive `opvdb_2026_q3.zip`, and CC BY 4.0 license terms.
  - Harvard CEP original publication [`10.1039/c1ee01463e`](https://doi.org/10.1039/c1ee01463e) and documentation of IBM World Community Grid computational methodology.
  - HOPV15 Nature Scientific Data publication [`10.1038/sdata.2016.86`](https://doi.org/10.1038/sdata.2016.86) and Figshare archive.
  - Saeki dataset publication [`10.1021/acs.jpclett.8b00635`](https://doi.org/10.1021/acs.jpclett.8b00635) in J. Phys. Chem. Lett.
  - Troisi dataset publication [`10.1021/acs.chemmater.0c02325`](https://doi.org/10.1021/acs.chemmater.0c02325) in Chem. Mater.
- **Reported by original publication:**
  - OPV-DB literature mining pipeline and criteria for `in_strict_performance_benchmark` and `in_strict_molecular_benchmark`.
  - CEP DFT simulation count ($>150\text{ million}$ evaluations) and 2.3 million virtual candidates.
  - HOPV15 calibration errors and DFT functional comparisons.
- **Inferred from available metadata:**
  - Standard measurement conditions (AM1.5G, 100 mW/cm², 25°C): Inferred from general literature reporting conventions; **NOT** explicitly logged per record or sensor-verified by a reference cell in the database tables.
  - That combining OPV-DB's strict molecular benchmark with scaffold-based cross-validation will reduce train/test analog leakage while preserving non-fullerene donor-acceptor diversity.
- **Not available or not verified:**
  - Batch-specific polymer molecular weight distribution ($M_w, M_n, PDI$) for individual OPV-DB rows where authors omitted GPC characterization in their original experimental papers.
  - Full-dataset 22-feature descriptor extraction success rate: **Pending empirical verification** against `opvdb_2026_q3.zip` upon authorized download.
