# OPV-PredictNet — Phase 4: Final Dataset Selection & Decision Report

**Document Version:** 1.0.0  
**Phase:** 4 (Dataset Discovery, Source Verification, and Quality Assessment)  
**Date:** November 2026  
**Status:** Formal Recommendation & Pre-Training Governance  

---

## 1. Executive Decision Summary

Based on rigorous scientific evaluation, provenance verification, schema analysis, and data-quality assessment across all five candidate dataset families:

### Primary Recommendation:
> **OPV-DB v1.1.0 (Version DOI: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960); Concept DOI: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542); Archive: `opvdb_2026_q3.zip`; CC BY 4.0 license) is recommended as SUITABLE ONLY AFTER ADDITIONAL CURATION for the initial experimental machine learning baseline of OPV-PredictNet.**

### Disqualification & Secondary Classifications:
1. **Harvard Clean Energy Project (CEP / CEPDB):** Classed as **UNSUITABLE FOR THIS PROJECT**. It consists entirely of simulated quantum-chemical (DFT) calculations using an empirical Scharber model with zero physical experimental devices, zero lab-measured I-V curves, and zero fabrication conditions.
2. **HOPV15 (Nature Scientific Data 2016):** Classed as **SUITABLE AS A SECONDARY VALIDATION BENCHMARK ONLY**. While scientifically rigorous and paired with DFT, its sample size ($N \approx 350$) and near-complete lack of modern non-fullerene acceptors (NFAs) make it insufficient as the primary engine.
3. **Nagasawa & Saeki Dataset (2018):** Classed as **SUITABLE AS A SUPPLEMENTARY REFERENCE ONLY**. Its binned classification targets (PCE classes) and lack of non-fullerene diversity limit its utility for continuous multi-target regression ($PCE, V_{OC}, J_{SC}, FF$).
4. **Zhao, del Cueto & Troisi Small-Molecule Dataset (2020):** Classed as **SUITABLE AS A SUPPLEMENTARY REFERENCE ONLY**. Restricted exclusively to small-molecule donors paired with $PC_{71}BM$, excluding polymer donors.

---

## 2. Multi-Criteria Evidence Matrix

| Evaluation Dimension | Weight | OPV-DB | Harvard CEP | HOPV15 | Saeki (2018) | Troisi (2020) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Experimental Authenticity** | Critical | **High (Real labs)** | **Zero (100% DFT)** | **High (Real labs)** | **High (Real labs)** | **High (Real labs)** |
| **Source Traceability (DOIs)**| High | **100% per record** (v1.1.0: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960)) | Computational run ID | **100% per record** | Literature citation | **100% per record** |
| **Donor-Acceptor Diversity** | High | **Polymers + NFAs** | Virtual donors only | Fullerene-heavy | Polymers only | Small-mols only |
| **Target Metric Completeness**| High | **Complete in benchmark**| Simulated PCE only | **Complete (4 targets)**| Incomplete/binned | **Complete (4 targets)**|
| **SMILES Resolvability** | Critical | **Pending empirical test on raw archive**| Virtual SMILES | **Verified InChI/SMILES**| Partial SMILES | **Verified SMILES** |
| **Device Metadata Coverage** | Medium | **High (8 parameters)**| None (Gas-phase) | Unstructured | Minimal ($M_n$) | Moderate |
| **Sample Size for ML** | High | **Thousands of records** | $>2\text{M}$ (synthetic) | ~350 records | ~1,200 records | ~500 records |
| **License Compatibility** | Mandatory| **CC BY 4.0** | Academic open | **CC BY 4.0** | ACS author rights | MIT Open |
| **Overall Recommendation** | — | **Suitable with Curation**| **Unsuitable** | **Secondary Reference** | **Supplementary** | **Supplementary** |

---

## 3. Detailed Justification for OPV-DB Selection

OPV-DB is the only candidate dataset meeting the project's foundational requirements:
1. **Real Experimental Physics:** Entries reflect physical solar cell devices. Note: Testing under standard conditions (AM1.5G, $100\text{ mW/cm}^2$, 25°C) is inferred from publication conventions, not explicitly verified or sensor-monitored per record.
2. **Modern Non-Fullerene Coverage:** Unlike historical datasets created before 2018, OPV-DB contains modern high-efficiency non-fullerene acceptor systems (e.g., Y6, ITIC, and their fluorinated/chlorinated derivatives) alongside classical fullerene systems.
3. **Traceability to Literature:** Each row references its original peer-reviewed publication DOI, enabling strict group-level partitioning to prevent optimistic publication leakage.
4. **Embedded Quality Tiering:** The presence of `in_strict_performance_benchmark` and `in_strict_molecular_benchmark` flags provides an objective starting filter created by the authors, though these are internal heuristics rather than independent physical proof.
5. **Permissive Open-Access License:** Released under the Creative Commons Attribution 4.0 International license (CC BY 4.0), allowing full academic and open-source derivative computation.

---

## 4. Why "Suitable Only After Additional Curation"?

OPV-DB cannot be ingested blindly into an automated training loop. It requires six mandatory curation safeguards:

### Curation Safeguard 1: Strict Molecular Benchmark Subsetting
Only rows where `in_strict_molecular_benchmark == True` are admitted. This guarantees:
- Non-empty, RDKit-valid SMILES for both donor and acceptor.
- Complete reported $V_{OC}, J_{SC}, FF$, and $PCE$.
- Physical recomputation agreement: $|PCE_{reported} - PCE_{calc}| / PCE_{reported} \le 0.10$.

### Curation Safeguard 2: Physical Sanity Bounds Enforcement
Filter out anomalous historical literature typographical errors:
- $0.0\% < PCE \le 34.0\%$ (Shockley-Queisser ceiling)
- $0.0\text{ V} < V_{OC} \le 2.0\text{ V}$
- $0.0\text{ mA/cm}^2 < J_{SC} \le 45.0\text{ mA/cm}^2$
- $0.0\% < FF \le 100.0\%$

### Curation Safeguard 3: Fill Factor Normalization
In literature mining, Fill Factor is inconsistently reported as a fraction ($0.75$) or a percentage ($75.0\%$). All fill factor values must be harmonized onto the unified percentage scale ($0.0 - 100.0\%$).

### Curation Safeguard 4: Deduplication of Identical Rows
Entries sharing the identical publication DOI, identical donor and acceptor SMILES, and identical device parameters must be deduplicated to a single observation.

### Curation Safeguard 5: Preservation of Inter-Laboratory Variance
Distinct publications measuring the same material pair (e.g. independent groups optimizing PM6:Y6) must **not be naively averaged**. They reflect real physical variance under differing laboratory conditions and should be preserved as distinct records.

### Curation Safeguard 6: Leakage-Aware Splitting
Data must be partitioned using **Bemis-Murcko scaffold splitting** and **publication-grouped cross-validation (`GroupKFold` on DOI)**, strictly prohibiting random shuffling.

---

## 5. Download, Extraction, and Inspection Protocol (For Phase 5)

In strict adherence to the project's download and safety policy:
- **No dataset was downloaded during Phase 4.**
- When approved by the user for Phase 5 ingestion:
  1. The dataset archive `opvdb_2026_q3.zip` must be retrieved from its authoritative Zenodo source (Version DOI: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960); Concept DOI: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542)).
  2. The SHA256 checksum must be verified against official repository release manifests.
  3. The raw, uncompressed archive must be preserved untouched under `ml/data/raw/`.
  4. Extracted tables must reside exclusively in isolated working directories.
  5. Curation transformations must write exclusively to `ml/data/processed/`, accompanied by automated provenance manifests.

---

## 6. Pre-Training Outstanding Tasks & Roadmap

Before model training can be authorized in subsequent phases:
1. **User Approval:** Explicit user authorization to fetch and curate the recommended OPV-DB v1.1.0 release.
2. **Deterministic Ingestion & Checksum Verification:** Retrieval of `opvdb_2026_q3.zip` into `ml/data/raw/` with verified checksums.
3. **Empirical Descriptor Extraction Verification:** Execution of `descriptors.py` on 100% of the raw candidate rows to empirically record parse successes, valence errors, and unkekulizable structures.
4. **Chemical Structure Sanitization Pipeline:** Automated RDKit standardization, salt removal, canonicalization, and neutral form verification.
5. **Scaffold & Pair Clustering Module:** RDKit Bemis-Murcko scaffold extraction and pair grouping to generate multi-tiered cross-validation folds.
6. **Baseline Benchmark Training Protocol:** Training of the primary `RandomForestRegressor` on the curated molecular benchmark subset using nested cross-validation and evaluation across all four target metrics ($PCE, V_{OC}, J_{SC}, FF$).
