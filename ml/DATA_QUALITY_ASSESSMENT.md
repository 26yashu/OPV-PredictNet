# OPV-PredictNet — Phase 4: Data Quality & Feature Compatibility Assessment

**Document Version:** 1.0.0  
**Phase:** 4 (Dataset Discovery, Source Verification, and Quality Assessment)  
**Date:** November 2026  
**Status:** In-Depth Quality Audit & Scientific Limitations  

---

## 1. Executive Summary

This report performs a comprehensive data-quality assessment of literature-mined experimental organic photovoltaic datasets, centered on the primary candidate: **OPV-DB v1.1.0** (Jiangjie Qiu et al. / Tsinghua University; Version DOI: [`10.5281/zenodo.22177960`](https://doi.org/10.5281/zenodo.22177960); Concept DOI: [`10.5281/zenodo.20841542`](https://doi.org/10.5281/zenodo.20841542); archive: `opvdb_2026_q3.zip`). 

In physical materials informatics, experimental datasets suffer from distinct failure modes that do not exist in synthetic or purely computational benchmarks:
1. **High Literature Heterogeneity**: Measurements originate from hundreds of academic laboratories worldwide using varying spin-coating rates, solvent purities, glovebox environments, and solar simulators.
2. **Extreme Class / Material Imbalance**: Popular high-efficiency pairs (e.g. `PM6:Y6`, `PTB7-Th:PC71BM`, `P3HT:PCBM`) are published dozens of times under varying device optimizations, while unpromising pairs are published once or abandoned ("file-drawer" publication bias).
3. **Publication-Level & Congeneric Scaffold Leakage**: Random train/test partitioning causes severe optimistic bias because analog molecules with identical core chromophores appear simultaneously in train and test splits.
4. **Finite Monomer vs. Macromolecular Reality**: RDKit 2D descriptors computed on finite monomeric repeat units fail to capture polymer molecular weight ($M_n, M_w$), chain polydispersity (PDI), conjugation lengths, solid-state thin-film morphology, and nanoscale phase separation.
5. **Quality Flags Are Internal Heuristics**: The dataset-provided quality flags (`in_strict_performance_benchmark` and `in_strict_molecular_benchmark`) represent internal regex and consistency rules applied by the dataset curators, NOT independent experimental proof of laboratory validity.

---

## 2. Missingness Patterns & Proposed Inclusion Criteria

### 2.1 Observed Missingness Taxonomy

```
+----------------------------------------------------------------------------------------------------+
|                                    MISSINGNESS STRATIFICATION                                       |
+------------------------------+---------------------------+-----------------------------------------+
| Category                     | Observed Data Columns     | Missingness Mechanism & Scientific Risk  |
+------------------------------+---------------------------+-----------------------------------------+
| Mandatory Core Metrics       | PCE, Voc, Jsc, FF         | If missing, record cannot be physically |
| (Must be complete)           |                           | verified for thermodynamic consistency. |
+------------------------------+---------------------------+-----------------------------------------+
| Mandatory Chemical Identity  | Donor SMILES, Acc SMILES  | If missing or unparseable, RDKit feature|
| (Must be complete)           |                           | extraction is strictly impossible.      |
+------------------------------+---------------------------+-----------------------------------------+
| Fabrication Processing Data  | Active layer thickness    | Missing at Random (MAR). Frequently     |
| (Frequently Unreported)      | D:A weight ratio          | omitted in papers focused purely on new |
|                              | Solvent additives (DIO)   | chemical synthesis rather than device   |
|                              | Annealing temp / time     | physics optimization.                   |
+------------------------------+---------------------------+-----------------------------------------+
| Interfacial Stack Metadata   | HTL, ETL, Contact work    | Often standardized (e.g. PEDOT:PSS/Al)   |
| (Partially Documented)       | function                  | but omitted in main paper abstracts.    |
+------------------------------+---------------------------+-----------------------------------------+
```

### 2.2 Proposed Inclusion Criteria
- **Inclusion Criterion 1 (Physical Sanity & Completeness):** Rows must have non-null $V_{OC}, J_{SC}, FF$, and $PCE$, satisfying the physical bounds:
  - $0.0\% < PCE \le 34.0\%$
  - $0.0\text{ V} < V_{OC} \le 2.0\text{ V}$
  - $0.0\text{ mA/cm}^2 < J_{SC} \le 45.0\text{ mA/cm}^2$
  - $0.0\% < FF \le 100.0\%$
- **Inclusion Criterion 2 (Chemical Resolvability):** Both donor and acceptor must parse successfully in RDKit with zero valence or kekulization errors.
- **Inclusion Criterion 3 (Recomputation Consistency):** The reported PCE must not deviate from the recomputed theoretical product by more than $10\%$:
  $$\frac{|PCE_{reported} - PCE_{calc}|}{PCE_{reported}} \le 0.10 \quad \text{where } PCE_{calc} = \frac{V_{OC} \cdot J_{SC} \cdot FF}{100}$$
- **Handling of Missing Process Parameters:**
  - Records must **NOT** be discarded merely because processing parameters (such as annealing time or additive concentration) are absent.
  - In the baseline molecular model, missing processing variables do not affect the 22-descriptor vector.
  - When expanding to future multi-modal device models (Pillar 3), missing values must be encoded using an explicit `Missing` indicator flag rather than naive zero-imputation.

---

## 3. Duplication, Repeated Measurements, and Variance Analysis

### 3.1 Exact Duplicate Records
- **Observation:** In literature databases, identical rows occasionally arise from multiple citations of the same primary paper (e.g., in review articles or meta-analyses).
- **Quality Protocol:** Exact duplicates (same DOI, identical donor, acceptor, processing conditions, and measured values) must be identified and deduplicated to a single authoritative entry.

### 3.2 Repeated Measurements Across Different Laboratories
- **Observation:** Benchmark systems like `PM6:Y6` have been fabricated by dozens of research groups worldwide, with reported PCEs ranging from $14.5\%$ to $>18.2\%$ depending on film thickness, morphology, and solvent optimization.
- **Scientific Risk of Naive Averaging:**
  - Naive averaging across papers destroys critical variance and creates artificial target values that were never experimentally observed in any real laboratory device.
  - Different records for the same chemical blend represent **distinct physical devices** processed under differing lab conditions.
- **Proposed Protocol:** Retain distinct device entries as independent observations, preserving their unique publication DOIs and processing metadata. Only collapse entries that share identical fabrication conditions within the same publication.

---

## 4. Leakage Prevention: Evaluation of Splitting Strategies

In materials informatics and chemoinformatics, evaluation metrics depend heavily on data splitting. Different partitioning schemes measure distinct generalization regimes and prevent different forms of data leakage.

### 4.1 Comparative Evaluation of Splitting Strategies

```
+-----------------------------------------------------------------------------------------------------------------+
|                                       DATA SPLITTING STRATEGY EVALUATION                                        |
+--------------------------+------------------------------+---------------------------+---------------------------+
| Splitting Strategy       | What It Measures             | Leakage Prevented         | Leakage FAILED to Prevent |
+--------------------------+------------------------------+---------------------------+---------------------------+
| 1. Random Record Split   | In-distribution record       | None. Catastrophic leakage| • Duplicate lab runs leak |
|    (Standard 80/20)      | interpolation. Highly        | of identical materials    | • Congeneric analogs leak |
|                          | optimistic; unrepresentative.| across train/test splits. | • Publication bias leaks  |
+--------------------------+------------------------------+---------------------------+---------------------------+
| 2. Publication Grouping  | Transferability to           | Prevents lab-specific     | Structural analog leakage |
|    (`GroupKFold` on DOI) | independent research         | calibration bias, tool    | across different papers   |
|                          | groups and equipment.        | drift, and paper series.  | synthesizing derivatives. |
+--------------------------+------------------------------+---------------------------+---------------------------+
| 3. Donor-Acceptor Pair   | Generalization to untested   | Completely prevents       | Chemical analog leakage   |
|    Grouping (`Pair ID`)  | combinations of known        | memorizing a specific     | where pairs share similar |
|                          | materials.                   | chemical blend.           | conjugated cores.         |
+--------------------------+------------------------------+---------------------------+---------------------------+
| 4. Molecular Scaffold    | Generalization to genuinely  | Prevents side-chain       | Publication-level bias if |
|    Grouping (Bemis-      | novel chromophore frameworks | analog leakage (e.g.      | multiple scaffolds were   |
|    Murcko Splitting)     | (out-of-distribution).       | alkyl chain variations).  | processed in one lab.     |
+--------------------------+------------------------------+---------------------------+---------------------------+
```

### 4.2 Critical Audit Finding: No Single Split Guarantees Generalization
No single partitioning scheme solves all leakage in OPV literature:
- Scaffold splitting isolates chemical frameworks, but if two different scaffolds are reported in the same paper with an identical uncalibrated solar simulator, publication leakage persists.
- Publication grouping prevents lab bias, but does not prevent identical molecular scaffolds from leaking across papers.
- **Scientifically Rigorous Recommendation:** OPV-PredictNet must employ a **dual-tiered cross-validation evaluation protocol**:
  1. *Tier A (Blend Transferability):* Disjoint Donor–Acceptor Pair Grouping (`GroupKFold` on material pair).
  2. *Tier B (Chemical Discovery):* Disjoint Bemis-Murcko Scaffold Grouping on donor and acceptor backbones.

---

## 5. Feature Compatibility Assessment: 22-Descriptor Schema

### 5.1 Empirical Verification Status: PENDING
In compliance with strict download policies, the raw archive `opvdb_2026_q3.zip` has not yet been downloaded to the local workspace. Therefore:
> **The claim that all strict molecular benchmark records support 100% reliable extraction of the 22 features is EMPIRICALLY UNVERIFIED on the raw dataset and must be treated as PENDING until tested directly against the extracted archive.**

Literature-mined SMILES strings frequently contain attachment wildcards (e.g., `*`, `[R]`), non-kekulizable quinoidal resonances, mixture notations, or undefined stereochemistry that can cause RDKit sanitization exceptions. Syntactic parsing success must be verified row-by-row upon archive ingestion.

### 5.2 Four-Tier Scientific Assessment Framework

To evaluate the 22-feature descriptor schema rigorously, we distinguish four levels of validity:

```
+----------------------------------------------------------------------------------------------------+
|                                FOUR-TIER SCIENTIFIC ASSESSMENT MATRIX                              |
+-----------------------------+----------------------------------------------------------------------+
| 1. Computational            | High (Pending test). Standard RDKit scalar calculations execute in   |
|    Extractability           | milliseconds for valid, neutral organic valence structures.          |
+-----------------------------+----------------------------------------------------------------------+
| 2. Chemical Representation  | SEVERELY LIMITED for polymers. A 400 g/mol repeat unit ignores real  |
|    Validity                 | macromolecular molecular weight (Mn, Mw), polydispersity (PDI),     |
|                             | regioregularity (head-to-tail), and conformational folding.          |
+-----------------------------+----------------------------------------------------------------------+
| 3. Predictive Usefulness    | MODERATE within narrow congeneric series where processing is fixed.  |
|                             | Weak across heterogeneous multi-lab literature records.              |
+-----------------------------+----------------------------------------------------------------------+
| 4. Experimental             | LOW / INSUFFICIENT for real physical devices. Blind to thin-film     |
|    Generalization           | morphology, solvent additives (DIO), thermal annealing, and stack.  |
+-----------------------------+----------------------------------------------------------------------+
```

### 5.3 Detailed Descriptor Extraction & Limitation Matrix

| Feature Index & Name | Chemical Role | Computational Status | Scientific & Physical Limitations in OPV |
| :--- | :--- | :--- | :--- |
| `0: donor_mw` | Molecular Size | Syntactically computable | Represents monomer repeat unit only; blind to true polymer chain length ($M_n = 30k-100k$). |
| `1: donor_logp` | Lipophilicity / Polarity | Syntactically computable | Wildman-Crippen 2D model; ignores solid-state film dielectric environment and packing. |
| `2: donor_tpsa` | Polar Surface Area | Syntactically computable | 2D topological approximation; ignores 3D dihedral backbone twisting and aggregation. |
| `3: donor_hbd` | Hydrogen Donors | Syntactically computable | OPV materials typically have 0 HBD; near-zero variance across dataset. |
| `4: donor_hba` | Hydrogen Acceptors | Syntactically computable | Counts heteroatoms; does not differentiate weak vs. strong coordination. |
| `5: donor_rotatable_bonds`| Chain Flexibility | Syntactically computable | Sensitive to alkyl representation; ignores backbone conformational rigidity. |
| `6: donor_aromatic_rings` | $\pi$-Conjugation Proxy | Syntactically computable | Counts rings; does not capture effective conjugation length ($L_{conj}$) along the polymer. |
| `7: donor_fraction_csp3` | Aliphatic Fraction | Syntactically computable | Proxy for side-chain ratio; blind to side-chain branching position (e.g. 2-ethylhexyl). |
| `8: donor_heavy_atom_count`| Structural Magnitude | Syntactically computable | Monomer size indicator only. |
| `9: donor_ring_count` | Ring Topology | Syntactically computable | Total rings; ignores fused vs. isolated electronic coupling. |
| `10–19: acceptor_*` | Same 10 for Acceptor | Syntactically computable | Valid for small molecules & NFAs; monomer approximation applies to polymer acceptors. |
| `20: delta_mw` | Size Disparity $|MW_D - MW_A|$| Syntactically computable | Physically meaningless when donor is a polymer and acceptor is a small molecule. |
| `21: delta_logp`| Polarity Disparity $|LogP_D - LogP_A|$| Syntactically computable | Crude proxy for blend miscibility; does not account for Flory-Huggins parameter $\chi$. |

---

## 6. Scientific Adequacy Audit: Rejection of Unsupported Claims

### 6.1 Rejection of General Adequacy Claim
The previous assertion that the 22-feature schema is "scientifically adequate" is **retracted**. 
The 22-feature molecular descriptor model is **NOT scientifically adequate as a comprehensive physical predictor of organic solar cell performance**.

### 6.2 Key Physical Blind Spots:
1. **Macromolecular Reality:** Polymer active layers are polydisperse systems. Variations in molecular weight distribution ($M_w, M_n, PDI$) and regioregularity can alter hole mobility and device Fill Factor by $>30\%$ without any change in the monomer SMILES.
2. **Thin-Film Morphology:** Bulk-heterojunction operation depends on nanoscale bicontinuous phase separation (~10–20 nm). Purely molecular descriptors cannot distinguish an unoptimized, phase-separated film ($PCE = 2\%$) from an optimized morphology ($PCE = 16\%$) using the exact same chemical blend.
3. **Processing Parameters:** Film thickness, host solvent volatility, solvent additives (e.g., $0.5\text{ vol }\%$ DIO), and thermal annealing govern crystallization and phase separation. A molecular-only model is completely blind to these levers.
4. **Device Stack Architecture:** Conventional vs. inverted architectures, interfacial buffer layers (PEDOT:PSS, $\text{MoO}_3$, ZnO, PDINN), and electrode work functions directly modulate $V_{OC}$ and contact resistance.

### 6.3 Qualified Scientific Role for Phase 5 Baseline
The 22-feature schema serves strictly as a **first-order molecular compatibility screening heuristic** to capture intrinsic chemical and polarity differences between donor and acceptor pairs. It must be explicitly communicated to researchers that predictions represent statistical expectations under optimized baseline conditions, and cannot replace physical morphological or device optimization.
