# OPV-PredictNet — Phase 8.2: Molecular Structure and Scaffold Mapping Audit

**Date**: 2026-11-25  
**Scope**: Pre-execution audit of dataset molecular representations, RDKit Bemis–Murcko scaffold behavior on conjugated polymer repeat-units, empty/invalid scaffold resolution, and name-to-structure mapping cardinality.  
**Benchmark File**: `ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv`  
**Dataset Checksum**: SHA-256 `ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897` ($N = 22,467$).

---

## 1. Executive Summary

This audit establishes the chemical and computational foundation for the Phase 8.2 structure-level generalization benchmark. Prior phases (Phases 5 through 8.1) partitioned data by **normalized material-name pairs** (`donor_canonical::acceptor_canonical`).

### Critical Audit Findings
1. **Material-Name Grouping Does NOT Prevent Structure Leakage**:
   - 79 donor canonical names and 68 acceptor canonical names map to multiple distinct molecular structures (SMILES).
   - Conversely, 30 donor structures and 29 acceptor structures are associated with multiple synonymous canonical material names.
   - Partitioning by material name allows identical or nearly identical chemical structures to leak across train and test partitions under different aliases. Grouping must be enforced strictly at the **canonical molecular structure and scaffold levels**.
2. **Bemis–Murcko Scaffolds Successfully Abstract Conjugated Cores**:
   - The dataset contains 1,984 unique RDKit canonical donor SMILES and 1,462 unique acceptor SMILES.
   - Bemis–Murcko scaffold abstraction trims solubilizing alkyl chains (e.g., 2-ethylhexyl, octyl, dodecyl) while preserving the conjugated heteroaromatic core, reducing donors to **1,073 unique scaffolds** and acceptors to **812 unique scaffolds**.
3. **Empty Scaffolds Identified and Deterministically Resolved**:
   - Exactly 2 donor structures (8 records) and 3 acceptor structures (4 records) lack ring systems, producing empty Bemis–Murcko scaffolds (`""`).
   - Rather than silently discarding these 12 records, they are deterministically assigned to pseudo-scaffolds `"[NO_RING]:" + canonical_smiles`, ensuring 100% record retention ($N = 22,467$).
4. **Graph-Theoretic Proof of Severe Bipartite Connectivity**:
   - The donor–acceptor scaffold bipartite graph forms a **single giant connected component containing 22,455 records (99.95% of the dataset)**.
   - Because experimentalists test novel donors against established hub acceptors ($\text{PC}_{71}\text{BM}$, Y6, ITIC) and novel acceptors against established hub donors (PM6, PBDB-T), a strict global dual-scaffold holdout isolates only **77 records across 30 scaffold pairs**.

---

## 2. Molecular Structure Representation and Entity Cardinality

The benchmark dataset ($N = 22,467$) contains four parallel levels of entity representation for both donor and acceptor molecules:

```
+---------------------------------------------------------------------------------------------------------+
| Molecular Entity Cardinality Across the Benchmark Dataset (N = 22,467 Devices)                          |
+--------------------------+-----------------------+-----------------------+------------------------------+
| Entity Level             | Donor Entities        | Acceptor Entities     | Pair Combinations            |
+--------------------------+-----------------------+-----------------------+------------------------------+
| Raw Extracted Name       | 2,192 strings         | 1,616 strings         | —                            |
| Normalized Canonical Name| 1,920 names           | 1,435 names           | 4,730 material-name pairs    |
| Raw Extracted SMILES     | 1,997 SMILES          | 1,480 SMILES          | 4,832 raw SMILES pairs       |
| RDKit Canonical SMILES   | **1,984 SMILES**      | **1,462 SMILES**      | **4,807 unique chem. pairs** |
| Bemis–Murcko Scaffolds   | **1,073 scaffolds**   | **812 scaffolds**     | **2,819 scaffold pairs**     |
| Generic (Carbon) Scaff.  | 827 frameworks        | 626 frameworks        | 2,214 framework pairs        |
+--------------------------+-----------------------+-----------------------+------------------------------+
```

### 2.1 Name-to-Structure Mapping Ambiguities
- **Multiple SMILES per Canonical Name**:
  - Donors: 79 canonical names map to 2 or more distinct RDKit SMILES.
    - Example: `D18` has 2 structural variants in literature representing slightly different end-group terminations; `BTR-Cl` and `FTAZ` have multiple side-chain and stereochemical notations.
  - Acceptors: 68 canonical names map to multiple SMILES.
    - Example: `AQx`, `ABP4T-4F`, and non-fullerene acceptor variants with differing regioisomeric alkyl attachments.
- **Multiple Canonical Names per Structure (Synonymy)**:
  - Donors: 30 canonical SMILES map to multiple material names (e.g., author lab codes such as `PBDTT-FTTE` vs. systematic nomenclature).
  - Acceptors: 29 canonical SMILES map to multiple names.

> [!CRITICAL]
> **Structure Leakage in Material-Name Splits**:
> If a split is defined solely by `donor_canonical`, an author using the alias `PBDTT-FTTE` in training while another record uses `PBDB-T` in test would leak the identical chemical structure across partitions. Phase 8.2 partitions must be constructed strictly by **canonical structure and scaffold hashes**.

---

## 3. Bemis–Murcko Scaffold Behavior on Conjugated Polymers

### 3.1 Algorithmic Mechanics on Polymer Repeat-Units
The standard RDKit Bemis–Murcko algorithm (`MurckoScaffold.MurckoScaffoldSmiles`):
1. Identifies all ring systems within the molecule.
2. Identifies all linkers connecting two or more ring systems.
3. Strips all acyclic substituents attached to rings that do not connect to other rings.

In OPV materials science, donors and non-fullerene acceptors are typically constructed from conjugated aromatic / heteroaromatic cores decorated with branched aliphatic side chains (e.g., 2-ethylhexyl, 2-butyloctyl, 4-octylphenyl) added strictly to impart solution processability:
- **What is Preserved**: Fused aromatic rings (e.g., benzo[1,2-b:4,5-b']dithiophene [BDT], benzothiadiazole [BT], thieno[3,4-b]thiophene [TT]), connecting conjugated vinylene or ethynylene linkers, and ring-fused heterocycles.
- **What is Removed**: Solubilizing aliphatic side chains. The resulting scaffold represents the **conjugated electronic backbone**.

### 3.2 Scaffold Frequency and Concentration
The distribution of scaffolds is heavily skewed toward a small number of landmark chemical discoveries:

```
+---------------------------------------------------------------------------------------------------------+
| Top 5 Most Frequent Donor Scaffolds in Dataset (N = 22,467)                                             |
+---+--------------------------------------------------------------+-------+------------+-----------------+
| # | Bemis–Murcko Scaffold SMILES Fragment                        | Count | Percentage | Chemical Family |
+---+--------------------------------------------------------------+-------+------------+-----------------+
| 1 | `O=C1c2cscc2C(=O)c2c(-c3ccc(-c4cc5c...` (BDT-BDD core)       | 9,251 | 41.18%     | PBDB-T / PM6    |
| 2 | `c1ccsc1` (Single thiophene ring)                            | 1,877 | 8.35%      | P3HT family     |
| 3 | `c1csc(-c2c3cc(-c4scc5sccc45)sc3...` (BDT-TT core)           | 1,800 | 8.01%      | PTB7 / PTB7-Th  |
| 4 | `c1csc(-c2cc3c4nsnc4c4cc(-c5ccc...` (BDT-BT-DPP core)        | 1,415 | 6.30%      | PCDTBT / PffBT  |
| 5 | `c1cc2cc3sc(-c4scc5ccsc45)cc3cc2s1` (DNTT fused thiophene)   | 651   | 2.90%      | Fused Acene     |
+---+--------------------------------------------------------------+-------+------------+-----------------+
|   | Top 5 Donor Scaffolds Combined                               | 14,994| **66.74%** | Two-thirds data |
+---+--------------------------------------------------------------+-------+------------+-----------------+
```

```
+---------------------------------------------------------------------------------------------------------+
| Top 5 Most Frequent Acceptor Scaffolds in Dataset (N = 22,467)                                           |
+---+--------------------------------------------------------------+-------+------------+-----------------+
| # | Bemis–Murcko Scaffold SMILES Fragment                        | Count | Percentage | Chemical Family |
+---+--------------------------------------------------------------+-------+------------+-----------------+
| 1 | `C=C1C(=Cc2cc3sc4c([nH]c5c6[nH]c7c8sc(C=C9...` (Y6 Core)     | 7,902 | 35.17%     | Y6 / BTP-4F NFA |
| 2 | `c1ccc(C2C34c5c6ccc7c8cc9c%10c%11...` (PC71BM Fullerene C70)| 3,639 | 16.20%     | PC71BM Fullerene|
| 3 | `C=C1C(=Cc2cc3sc4c(c3s2)C(c2ccccc2)...` (ITIC Core)          | 2,345 | 10.44%     | ITIC NFA family |
| 4 | `c1ccc(C2C34c5c6c7c8c9c%10c(c%11...` (PC61BM Fullerene C60) | 2,323 | 10.34%     | PC61BM Fullerene|
| 5 | `C1=Cc2cccc3cccc4ccc5ccccc6cccc7...` (PDI Dimer Core)        | 965   | 4.30%      | Perylene Diimide|
+---+--------------------------------------------------------------+-------+------------+-----------------+
|   | Top 5 Acceptor Scaffolds Combined                            | 17,174| **76.44%** | Three-fourths   |
+---+--------------------------------------------------------------+-------+------------+-----------------+
```

---

## 4. Empty and Uninformative Scaffolds Audit

The Bemis–Murcko algorithm strictly requires at least one ring system. Molecules lacking rings evaluate to an empty string (`""`).

### 4.1 Audit of Empty Scaffolds
All 1,984 donor SMILES and 1,462 acceptor SMILES were evaluated:
- **Donors with Empty Scaffolds (2 structures, 8 records total)**:
  1. `N#C[S-].[Cu+]` (Copper(I) thiocyanate, 7 records): Inorganic transport layer salt mislabeled in raw extraction as a donor material.
  2. `CC(C)CC(C(=O)NC(CC(=O)N)C(=O)N...` (Acyclic oligopeptide `PR1`, 1 record).
- **Acceptors with Empty Scaffolds (3 structures, 4 records total)**:
  1. `C(CC(=O)NC(CS(=O)(=O)O)C(=O)NC...` (Acyclic peptide `GT-s`, 2 records).
  2. `[O]=[Ti]=[O]` (Titanium dioxide, 1 record): Inorganic nanoparticle mislabeled as acceptor.
  3. `C=C(CC(=O)O)C(=O)O` (Itaconic acid `ITN`, 1 record).

### 4.2 Non-Discarding Handling Policy
- **Prohibition on Silent Discarding**: Discarding records alters the benchmark population and breaks dataset traceability.
- **Resolution**:
  $$\text{scaffold}(S) = \begin{cases} \text{MurckoScaffoldSmiles}(S) & \text{if } \text{MurckoScaffoldSmiles}(S) \neq "" \\ \text{"[NO\_RING]:"} + \text{CanonicalSmiles}(S) & \text{otherwise} \end{cases}$$
- This policy ensures all 12 records are assigned distinct, non-overlapping pseudo-scaffolds. They are isolated into specific groups and cannot leak across train and test.

---

## 5. Bipartite Graph Structure and Dual-Scaffold Holdout Feasibility

A formal graph-theoretic analysis of the donor–acceptor scaffold bipartite graph $G = (V_D, V_A, E)$ was performed, where $V_D$ is the set of 1,073 donor scaffolds, $V_A$ is the set of 812 acceptor scaffolds, and $E$ is the set of 2,819 observed scaffold pairings.

### 5.1 Connected Component Decomposition
```
+---------------------------------------------------------------------------------------------------------+
| Bipartite Scaffold Graph Connectivity                                                                   |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Component                | Donor Scaffolds       | Acceptor Scaffolds  | Record Count (% of Dataset)    |
+--------------------------+-----------------------+---------------------+--------------------------------+
| **Giant Component #1**   | 1,066 scaffolds       | 803 scaffolds       | **22,455 records (99.95%)**    |
| Component #2 (CuSCN)     | 1 pseudo-scaffold     | 4 scaffolds         | 7 records (0.03%)              |
| Component #3 (GT-s)      | 2 scaffolds           | 1 pseudo-scaffold   | 2 records (0.01%)              |
| Components #4–7          | 4 scaffolds           | 4 scaffolds         | 3 records (0.01%)              |
+--------------------------+-----------------------+---------------------+--------------------------------+
```

### 5.2 Mathematical Proof of Dual-Scaffold Holdout Sparsity
1. **The Hub Effect**: In published solar cell literature, virtually every new donor scaffold is co-fabricated with standard $\text{PC}_{71}\text{BM}$, $\text{PC}_{61}\text{BM}$, or $\text{Y6}$. Conversely, virtually every new acceptor scaffold is co-fabricated with $\text{PM6}$ or $\text{PTB7-Th}$.
2. **The Deletion Theorem**:
   - Let $D_{\text{test}} \subset V_D$ be held-out donor scaffolds and $A_{\text{test}} \subset V_A$ be held-out acceptor scaffolds.
   - For a partition to be a **strict global dual-scaffold holdout**, no test donor scaffold may appear in training ($S_D^{\text{test}} \cap S_D^{\text{train}} = \emptyset$) AND no test acceptor scaffold may appear in training ($S_A^{\text{test}} \cap S_A^{\text{train}} = \emptyset$).
   - Any cross-edge $(d, a)$ where $d \in D_{\text{test}}$ and $a \in A_{\text{train}}$ **cannot be assigned to train** (it would leak donor $d$) and **cannot be assigned to test** (it would leak acceptor $a$).
   - Every cross-edge **must be deleted**.
3. **Empirical Consequence**:
   - When holding out a standard 20% random selection of donor scaffolds and 20% of acceptor scaffolds, deleting cross-edges leaves only **77 surviving records across 30 scaffold pairs**.
   - These 77 records represent obscure studies where an author paired an unconventional non-standard donor with an unconventional non-standard acceptor.
   - Constructing an independent calibration partition $D_{\text{cal}}$ under strict dual disjointness is mathematically impossible on this dataset (requires a tripartite component split of a single giant component).

### 5.3 Distinction between Strategy C Variants
- **Strategy C1: Strict Global Dual-Scaffold Holdout**:
  - $S_D^{\text{test}} \cap S_D^{\text{train}} = \emptyset$ AND $S_A^{\text{test}} \cap S_A^{\text{train}} = \emptyset$.
  - Sample size is severely underpowered ($N = 77$). Documented as an exploratory, underpowered boundary.
- **Strategy C2: Scaffold-Pair Holdout (Unseen Combinations)**:
  - $(S_D, S_A)_{\text{test}} \cap (S_D, S_A)_{\text{train}} = \emptyset$.
  - Unseen combination of scaffolds ($N = 3,604$ records across 565 scaffold pairs).
  - Explicitly documents that individual donor (39.3%) and acceptor (44.4%) scaffolds overlap with training.

This pre-execution audit provides the verified mathematical specifications required to execute Phase 8.2.
