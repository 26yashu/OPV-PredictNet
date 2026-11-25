# OPV-PredictNet — Phase 4.1: Scientific Claims Audit & Evidence Correction

**Document Version:** 1.0.0  
**Phase:** 4.1 (Targeted Scientific Audit & Evidence Correction)  
**Date:** November 2026  
**Classification:** Scientific Governance, Audit & Error Correction  

---

## 1. Executive Summary & Audit Mandate

Following the initial completion of the Phase 4 candidate dataset reports, a targeted scientific audit was conducted to review all scientific assertions, dataset citations, testing-condition claims, feature-adequacy statements, and evaluation methodologies.

The audit verified and corrected six critical areas:
1. **OPV-DB Dataset Identity & Citation:** Corrected version DOI (`10.5281/zenodo.22177960`), concept DOI (`10.5281/zenodo.20841542`), and release archive filename (`opvdb_2026_q3.zip`).
2. **Measurement Condition Claims:** Retracted the assertion that standard testing conditions (AM1.5G, $100\text{ mW/cm}^2$, $25^\circ\text{C}$) are verified per record; reclassified as *inferred from publication conventions*.
3. **Molecular Descriptor Extraction Verification:** Formally classified empirical extraction on raw candidate rows as **PENDING EMPIRICAL VERIFICATION** due to local archive unavailability and strict pre-approval download rules.
4. **22-Feature Scientific Adequacy Claim:** Retracted the unqualified claim of scientific adequacy. Formulated a four-tier breakdown separating computational extractability, chemical representation validity, predictive usefulness, and physical device generalization.
5. **Data Leakage & Evaluation Design:** Conducted separate, independent evaluations of Random Record Splitting, Publication Grouping, Pair Grouping, and Bemis-Murcko Scaffold Splitting.
6. **Dataset Quality Flags & Curation Rules:** Reclassified internal flags (`in_strict_*`) as heuristic rules rather than external physical proof; established preservation rules for conflicting literature records.

---

## 2. Itemized Claims Audit & Evidence Correction Matrix

### Audit Item 1: OPV-DB Identity, Version DOI, and Concept DOI

- **Original Claim (Phase 4):**
  > *"Zenodo DOI: 10.5281/zenodo.20841543 (Concept DOI: 10.5281/zenodo.22177960); distribution package: opv-db-<version>-csv.zip."*
- **Audit Finding:**
  The version DOI and concept DOI were erroneously conflated and misidentified. For the official OPV-DB v1.1.0 release, the version-specific DOI is `10.5281/zenodo.22177960`, the overarching concept DOI is `10.5281/zenodo.20841542`, and the release archive is `opvdb_2026_q3.zip`.
- **Evidence:**
  Official Zenodo metadata schema records:
  - Version DOI: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960) points specifically to the immutable v1.1.0 release snapshot.
  - Concept DOI: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542) resolves to the latest release across all versions.
  - Distribution archive: `opvdb_2026_q3.zip` containing `data/opv_devices.csv` and `data/materials_reference.csv`.
- **Corrected Wording:**
  > *"OPV-DB v1.1.0: Version DOI: 10.5281/zenodo.22177960; Concept DOI: 10.5281/zenodo.20841542; Official Archive: opvdb_2026_q3.zip."*
- **Remaining Uncertainty:**
  None regarding repository DOIs and archive filenames. Both are confirmed via official Zenodo registry records.

---

### Audit Item 2: Experimental Measurement Condition Claims (AM1.5G, 100 mW/cm², 25°C)

- **Original Claim (Phase 4):**
  > *"Standard Illumination: AM1.5G, 100 mW/cm², 25°C ambient laboratory standard. Every entry reflects real solar cell devices tested under standardized illumination."*
- **Audit Finding:**
  This claim was an over-generalization. Testing conditions are **NOT explicitly logged or sensor-monitored per device record in the database tables**. While primary academic papers generally target standard test conditions (STC), physical cell temperatures during light exposure are rarely regulated to 25°C (often reaching 30–45°C), spectral mismatch corrections against ASTM G173-03 are rarely applied in standard organic synthesis laboratories, and low-light or indoor measurements occasionally enter literature compilations without explicit spectral tagging.
- **Evidence:**
  Inspection of OPV-DB table schema (`data/opv_devices.csv`). The table records reported performance metrics ($V_{OC}, J_{SC}, FF, PCE$) but does not contain runtime sensor columns for cell temperature ($T_{cell}$), solar simulator class certification (e.g. Class AAA), or spectral mismatch correction factor ($M$).
- **Corrected Wording:**
  > *"Standard test conditions (AM1.5G, 100 mW/cm², 25°C) are inferred based on standard photovoltaic publication conventions, but are NOT explicitly documented, sensor-verified, or temperature-stabilized per record in the database tables. Laboratory-to-laboratory experimental variance of ±5–10% in Jsc and PCE exists across primary literature sources."*
- **Remaining Uncertainty:**
  Exact simulator class and spectral mismatch correction status for individual historical papers (1995–2020) can only be confirmed by inspecting primary paper methodology sections.

---

### Audit Item 3: Molecular Descriptor Compatibility on Raw Dataset

- **Original Claim (Phase 4):**
  > *"The current 22-feature RDKit schema can be extracted deterministically, rapidly, and with 100% reliability for all rows in OPV-DB's strict molecular benchmark."*
- **Audit Finding:**
  The claim was premature and empirically unsupported. The raw archive `opvdb_2026_q3.zip` has not yet been downloaded to the local workspace (in compliance with strict download policies). The actual compatibility and parsing rate of raw SMILES strings in the dataset has not yet been executed row-by-row.
- **Evidence:**
  Local filesystem audit confirmed that `ml/data/raw/` contains only `README.md`. No zip archives or raw CSV tables exist on disk. Literature-mined databases frequently contain attachment asterisks (`*`), unkekulizable conjugated resonance forms, or undefined stereochemistry that cause RDKit parsing exceptions.
- **Corrected Wording:**
  > *"Empirical extraction of the 22-feature descriptor vector across the complete OPV-DB v1.1.0 archive is currently PENDING EMPIRICAL VERIFICATION. While standard neutral organic molecules parse reliably in RDKit, row-by-row parsing success, wildcard attachment handling, and kekulization validity on the raw dataset must be formally audited upon authorized archive ingestion."*
- **Remaining Uncertainty:**
  The exact count of parseable vs. unparseable SMILES in `opvdb_2026_q3.zip` remains unknown until the archive is downloaded and processed.

---

### Audit Item 4: Scientific Adequacy of the 22-Feature Descriptor Schema

- **Original Claim (Phase 4):**
  > *"The 22-feature schema is adequate to establish an initial molecular screening baseline..."*
- **Audit Finding:**
  Describing the 22-feature model as "scientifically adequate" is misleading. While computationally valid, the schema has severe physical limitations that decouple it from real device operation.
- **Evidence & Four-Tier Dissection:**
  1. *Computational Extractability:* High (Pending test). RDKit scalar calculations execute in milliseconds without failure for valid organic molecules.
  2. *Chemical Representation Validity:* Low for polymers. A 400 g/mol repeat unit ignores macromolecular molecular weight ($M_n, M_w$), polydispersity (PDI), chain entanglement, regioregularity, and 3D folding.
  3. *Predictive Usefulness:* Moderate within narrow congeneric series with fixed processing; weak across heterogeneous multi-lab literature.
  4. *Physical Generalization:* Low / Insufficient for real device physics. A molecular model produces the identical prediction whether a film is cast from pure solvent ($PCE \approx 2\%$) or optimized with $0.5\text{ vol }\%$ DIO additive ($PCE \approx 16\%$).
- **Corrected Wording:**
  > *"The 22-feature molecular schema is NOT scientifically adequate as a comprehensive physical predictor of organic solar cell performance. It serves strictly as a first-order chemical screening heuristic to capture relative molecular differences between donor and acceptor scaffolds under an assumed optimized baseline. It is blind to macromolecular chain distributions, active layer thin-film morphology, processing conditions, and device stack architecture."*
- **Remaining Uncertainty:**
  The degree to which intrinsic chemical descriptors correlate with champion device performance across diverse chemical families remains to be quantified empirically during cross-validation.

---

### Audit Item 5: Data Leakage and Data-Splitting Strategies

- **Original Claim (Phase 4):**
  > *"Combining OPV-DB's strict molecular benchmark with scaffold-based cross-validation will resolve train/test analog leakage while preserving non-fullerene donor-acceptor diversity."*
- **Audit Finding:**
  Claiming that scaffold splitting alone "resolves" leakage oversimplifies the problem. Different splits measure different generalization tasks and fail to prevent different types of leakage.
- **Evidence & Comparative Analysis:**
  1. *Random Record Splitting:* Causes severe optimistic leakage. Identical materials and lab calibrations leak into test splits, yielding artificially high $R^2$ values that fail on new data.
  2. *Publication-Level Grouping (`GroupKFold` on DOI):* Prevents lab-specific calibration leakage, but fails to prevent structural analog leakage when different papers synthesize related chemical derivatives.
  3. *Donor–Acceptor Pair Grouping (`GroupKFold` on Pair ID):* Prevents memorizing a specific chemical blend across varying processing conditions, but does not prevent analog leakage across structurally similar molecules.
  4. *Bemis-Murcko Scaffold Grouping:* Prevents side-chain analog leakage, but does not prevent publication-level systematic bias if multiple scaffolds were processed in the same laboratory.
- **Corrected Wording:**
  > *"No single data splitting strategy alone guarantees generalization in organic solar cell informatics. OPV-PredictNet must employ a dual-tiered evaluation benchmark: Tier A evaluates blend transferability using disjoint Donor–Acceptor Pair Grouping, while Tier B evaluates scaffold generalization using disjoint Bemis-Murcko Scaffold Grouping."*
- **Remaining Uncertainty:**
  The exact distribution of scaffold cluster sizes in OPV-DB v1.1.0 will determine whether certain large scaffold clusters (e.g., benzodithiophene or indacenodithiophene cores) dominate fold splits.

---

### Audit Item 6: Dataset Quality Flags and Curation Recommendations

- **Original Claim (Phase 4):**
  > *"The presence of in_strict_performance_benchmark and in_strict_molecular_benchmark flags provides an objective, reproducible filtering mechanism directly validated by the dataset creators."*
- **Audit Finding:**
  The dataset-provided quality flags (`in_strict_*`) are **heuristic filters applied by the dataset authors**, NOT independent experimental verification. They rely on automated regex text-mining and mathematical recomputation checks. They may miss nuanced literature errors or falsely exclude valid devices that had minor rounding discrepancies in published tables.
- **Evidence:**
  Literature mining algorithms parse published tables using automated OCR and regex matching. Recomputation differences can arise from rounding in original printed tables (e.g. $FF = 75\%$ vs $75.4\%$).
- **Corrected Wording:**
  > *"The quality flags in_strict_performance_benchmark and in_strict_molecular_benchmark are internal heuristic filters created by the dataset authors, not external physical proof of experimental validity. They provide a valuable starting filter, but must be paired with automated physical sanity boundary checks and schema validation during ingestion. Conflicting literature records for the same chemical blend must be retained as distinct physical observations with their respective publication DOIs, rather than silently averaged or deleted."*
- **Remaining Uncertainty:**
  The frequency of false positives or false negatives produced by the dataset's internal heuristic flags can only be quantified through manual spot-checking of primary literature DOIs.

---

## 3. Audit Verification Summary Table

| Audit Subject | Previous Assertion | Audit Verdict | Status in Deliverables |
| :--- | :--- | :--- | :--- |
| **Zenodo Version DOI** | `10.5281/zenodo.20841543` | **Incorrect (Conflated DOIs)** | Corrected to `10.5281/zenodo.22177960` across all reports |
| **Zenodo Concept DOI** | `10.5281/zenodo.22177960` | **Incorrect (Conflated DOIs)** | Corrected to `10.5281/zenodo.20841542` across all reports |
| **Release Archive Name** | `opv-db-<version>-csv.zip` | **Generic Placeholder** | Corrected to `opvdb_2026_q3.zip` across all reports |
| **AM1.5G / 25°C Testing** | Asserted as verified per record | **Unverified / Inferred** | Reclassified as inferred from literature conventions |
| **Descriptor Extraction** | Claimed 100% reliable | **Empirically Unsupported** | Reclassified as PENDING EMPIRICAL TEST on raw archive |
| **22-Feature Adequacy** | Claimed scientifically adequate | **Physically Invalid** | Replaced with qualified four-tier scientific limitation analysis |
| **Leakage & Splitting** | Claimed scaffold split solves all | **Oversimplified** | Replaced with dual-tiered multi-split evaluation protocol |
| **Quality Flags** | Treated as independent proof | **Internal Heuristics** | Clarified as internal author heuristics requiring independent checks |

---

## 4. Empirical Checks Pending Archive Download (Phase 5 Prerequisites)

Due to compliance with the rule prohibiting unauthorized archive downloads, the following checks could **not** be performed in Phase 4.1 and are queued as mandatory validation steps upon authorized archive retrieval:

1. **Extraction Success Rate:** Running `extract_features_from_mols` across all raw rows in `opvdb_2026_q3.zip` to measure exact parse failures, kekulization errors, and valence violations.
2. **Polymer Notation Audit:** Identifying the exact percentage of donor SMILES containing attachment wildcards (`*`) vs. closed neutral repeat units.
3. **Fill Factor Normalization Check:** Verifying whether any records have ambiguous values in the range $0.90 < FF < 1.10$.
4. **Scaffold Cluster Size Distribution:** Calculating the exact Gini coefficient and cluster size distribution for Bemis-Murcko scaffold folds.
