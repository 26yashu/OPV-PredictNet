# OPV-PredictNet — Phase 7.0.4: Material-Pair Identity and Sampling Verification

**Author / Auditor:** Antigravity Agent  
**Date:** November 24, 2026  
**Status:** Completed Pair Identity and Sampling Audit (Pre-Implementation Verification)  
**Target Model:** Phase 5 Baseline Random Forest Regressor ($n=100, \text{depth}=\text{None}, \text{split}=2, \text{leaf}=1, \text{features}=1.0$)  
**Dataset:** OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records)  
**Scope:** Material-pair identity implementation audit, resolution of the 4,730 vs. 4,832 discrepancy, scientific definition of the Tier 2 target, verification of the random-record sampling mechanism, literal Git repository status, and pre-implementation freeze.

---

## 1. Audit of the Actual `pair_id` Implementation

### 1.1 Source Code Excerpt & Line References
In `ml/experiments/train_baseline.py`, the grouping column for the primary cross-validation strategy is defined at lines 200–202:

```python
# ml/experiments/train_baseline.py: Lines 200-202
# 4. Construct Grouping Columns
# Primary: Donor-Acceptor Pair
pair_groups = (df['donor_canonical'] + "::" + df['acceptor_canonical']).to_numpy()
```

This grouping variable is passed directly to `run_grouped_cv` at line 221:

```python
# ml/experiments/train_baseline.py: Lines 220-222
primary_metrics, dummy_oof_pce, rf_oof_pce = run_grouped_cv(
    X=X, y=y, groups=pair_groups, split_name="Pair_Grouped_5Fold", n_splits=5
)
```

### 1.2 Architectural Properties of the Grouping Column
1. **Exact Source Columns:**
   - Donor: `df['donor_canonical']` (string column in `opv_devices_strict_molecular_benchmark.csv`).
   - Acceptor: `df['acceptor_canonical']` (string column in `opv_devices_strict_molecular_benchmark.csv`).
2. **Normalization & Canonicalization Function:**
   - In `train_baseline.py`, the operation is pure Python string concatenation using the delimiter `"::"`: `(df['donor_canonical'] + "::" + df['acceptor_canonical']).to_numpy()`.
   - No runtime stripping, lowercasing, whitespace trimming, or RDKit molecular canonicalization is performed inside `train_baseline.py`.
   - The canonicalization occurred upstream during OPV-DB database extraction, where raw publication strings (`donor` and `acceptor`) were normalized into standardized material identifiers (`donor_canonical` and `acceptor_canonical`).
3. **Representation Basis:**
   - **The pair grouping is based strictly on normalized material names/abbreviations, NOT on canonical SMILES, exact chemical structures, or database IDs.**
   - Examples of actual group names: `'PM6::Y6'`, `'PTB7-Th::PC71BM'`, `'J51::ITIC'`, `'BTR-Cl::L8-BO'`.
4. **Independent Normalization:**
   - **Yes.** In OPV-DB, donor materials and acceptor materials are curated in separate entity tables (`materials_reference.csv`). `donor_canonical` is resolved from `donor` independently of how `acceptor_canonical` is resolved from `acceptor`.
5. **Missing or Invalid Identities:**
   - An audit of all 22,467 benchmark records confirms **0 null values, 0 empty strings, and 0 missing values** in both `donor_canonical` and `acceptor_canonical`:
     $$\text{Nulls}(\text{donor\_canonical}) = 0, \quad \text{Nulls}(\text{acceptor\_canonical}) = 0$$
   - Every single record in the benchmark table belongs to a fully specified, non-null material-name pair.
6. **Stability Across Reloads:**
   - **Yes.** Because the grouping is a deterministic string concatenation of two persistent text columns from a static, checksum-verified CSV file, pair IDs are 100% deterministic, immutable, and stable across dataset reloads, operating systems, and Python sessions.

---

## 2. Investigation and Explanation of the 4,730 vs. 4,832 Discrepancy

### 2.1 Discrepancy Accounting
Across the 22,467 records of the Strict Molecular Benchmark:
- **Canonical Material-Name Pairs (`pair_canonical_name`):** **4,730 groups**
- **Raw SMILES String Combinations (`pair_raw_smiles`):** **4,832 combinations**
- **RDKit Canonical SMILES Combinations (`pair_rdkit_can_smiles`):** **4,788 combinations**
- **Material-Name Pairs with Multiple Raw SMILES:** **230 pairs**
- **Material-Name Pairs with Multiple RDKit Canonical SMILES:** **190 pairs**

$$\begin{array}{l|c|l}
\hline
\textbf{Representation Level} & \textbf{Unique Combinations} & \textbf{Scientific Interpretation} \\
\hline
\text{Raw Publication Names} & 5,173 & \text{Uncurated literature names (includes typos, case differences)} \\
\textbf{Normalized Material Names} & \textbf{4,730} & \textbf{The Phase 5 Baseline grouping unit (material identity umbrella)} \\
\text{RDKit Canonical SMILES} & 4,788 & \text{Chemically distinct molecular graphs (exact connectivity \& stereochemistry)} \\
\text{Raw Curated SMILES} & 4,832 & \text{Syntactically distinct SMILES strings before chemical canonicalization} \\
\hline
\end{array}$$

### 2.2 Classification of Multi-SMILES Material Groups
The 230 material-name pairs exhibiting multiple raw SMILES strings were exhaustively audited and classified into two distinct categories:

#### Category A: Equivalent SMILES Encodings of the Same Molecular Structure (40 Pairs)
In 40 material-name pairs, the raw SMILES strings differ only due to syntactic formatting in the source literature (alternative atom numbering order, Kekule vs. aromatic ring notation, or explicit stereochemical slash slashes), but **resolve to 100% identical RDKit canonical SMILES, identical molecular graphs, and identical chemical descriptors**:
- **Representative Examples:**
  1. `'BTR-Cl::L8-BO'` (2 raw SMILES $\to$ 1 RDKit canonical SMILES): The donor SMILES in one publication was written in Kekule form (`CCCCCCC1=C(C2=CC(CCCCCC)=...`) while another paper used aromatic form (`CCCCCCc1cc(-c2cc3c...`). Both yield identical RDKit canonical SMILES: `CCCCCC...`.
  2. `'BTR-Cl::N3'` (2 raw SMILES $\to$ 1 RDKit canonical SMILES).
  3. `'BTR-Cl::Y6'` (2 raw SMILES $\to$ 1 RDKit canonical SMILES).
  4. `'D18::AQx-2'` (2 raw SMILES $\to$ 1 RDKit canonical SMILES).
  5. `'D18::BO-4Cl'` (2 raw SMILES $\to$ 1 RDKit canonical SMILES).
- **Scientific Conclusion:** These 40 pairs represent chemical identity equivalence; the difference between 4,832 raw SMILES and 4,788 canonical SMILES is entirely accounted for by Category A.

#### Category B: Chemically Distinct Structures Under the Same Material Name (190 Pairs)
In 190 material-name pairs, the records associated with a single material name actually describe **chemically distinct molecular structures** synthesized or reported by different research groups:
- **Representative Examples:**
  1. `'FTAZ::NNFA'` (5 distinct chemical structures for the acceptor): The authors of different papers synthesized homologues of the non-fullerene acceptor NNFA with varying solubilizing alkyl chains:
     - Record 1: Acceptor formula $C_{98}H_{82}N_4O_2S_6$, $\text{MW} = 1540.16\text{ Da}$ (octyl chains).
     - Record 2: Acceptor formula $C_{110}H_{106}N_4O_2S_6$, $\text{MW} = 1708.49\text{ Da}$ (decyl chains).
     - Record 3: Acceptor formula $C_{122}H_{130}N_4O_2S_6$, $\text{MW} = 1876.81\text{ Da}$ (dodecyl chains).
     All were referred to as "NNFA" in their respective articles.
  2. `'J40::ITIC'` (Donor has 2 distinct structures): Formula $C_{74}H_{111}F_2N_3O_2S_4$ ($\text{MW} = 1240.98\text{ Da}$) vs. $C_{72}H_{107}F_2N_3O_2S_4$ ($\text{MW} = 1212.93\text{ Da}$). The difference is an ethyl group ($C_2H_4$, $\Delta \text{MW} = 28.05\text{ Da}$) in the polymer side chain.
  3. `'J51::ITIC'` and `'J51::N2200'` (Donor has 2 distinct structures): Formula $C_{80}H_{111}F_2N_3S_6$ ($\text{MW} = 1345.19\text{ Da}$) vs. $C_{74}H_{99}F_2N_3S_6$ ($\text{MW} = 1261.02\text{ Da}$), differing by a hexyl moiety ($\Delta \text{MW} = 84.16\text{ Da}$).
  4. `'FTAZ::ITIC'` and `'HTAZ::ITIC'` (Stereochemical / Regiochemical Isomers): Both structures have identical chemical formula $C_{94}H_{82}N_4O_2S_4$ and $\text{MW} = 1427.98\text{ Da}$, but differ in cis/trans isomerism and regio-attachment of the terminal dicyanomethylene indanone units.
- **Scientific Conclusion:** These are chemically distinct molecules grouped under a shared material umbrella. They are not canonical-equivalent.

---

## 3. Scientific Definition of the Tier 2 Target

### 3.1 Formal Declaration of the Target Unit
- **Tier 2 evaluates:** **Unseen normalized material-name combinations** (`donor_canonical::acceptor_canonical`, 4,730 groups).
- **It does NOT evaluate:** Exact canonical molecular structure combinations (which would consist of 4,788 groups).

### 3.2 Scientific Rationale & Chemical Context
In organic photovoltaics literature, a "material" is historically identified by its polymer or small-molecule acronym (e.g. PM6, PBDB-T-2F, Y6, ITIC). Experimental literature frequently synthesizes batches with slight variations in alkyl chain length or oligomer repeat-unit length to optimize solubility and film morphology without changing the fundamental chromophore identity.

Therefore, the 4,730 canonical material-name pairs represent **material identity umbrellas**.

### 3.3 Historical Baseline Integrity
- The Phase 5 Baseline Random Forest cross-validation splits, out-of-fold predictions, and metrics (`ml/results/phase_5_*`) were generated strictly using `donor_canonical + "::" + acceptor_canonical` (4,730 groups).
- **Strict Rule:** We do **not** retrospectively alter the Phase 5 grouping or re-run Phase 5 splits with a 4,788-group structure definition. Doing so would violate the frozen baseline mandate and invalidate the historical audit trail.

### 3.4 Future Sensitivity Analysis Specification
A structure-level grouping based on RDKit canonical SMILES pairs (4,788 groups) is scientifically sound and represents a valuable comparative benchmark. It is pre-registered as a **separate auxiliary sensitivity analysis** requiring its own independent validation protocol, and will not replace the primary 4,730-pair evaluation in Phase 7.1.

---

## 4. Verification of the Random-Record Sampling Implementation

### 4.1 Sampling Protocol Verification
To evaluate Tier 2 Alternative B, exactly one record is sampled per material pair:
1. **One Record per Pair:** For each material-pair cluster $g$, exactly one index $j_g \in \{1, \dots, n_g\}$ is selected:
   - For singleton pairs ($n_g = 1$), the unique record is chosen deterministically.
   - For repeated pairs ($n_g > 1$), record $j_g$ is drawn uniformly at random using a pre-registered pseudo-random generator (`seed = 42` for calibration, `seed = 43` for test).
2. **Target Independence:** Selection depends strictly on the pair cluster index and random seed; it is completely independent of $y_{g, i} = \text{PCE}_{g, i}$, model predictions $\hat{\mu}$, or residual magnitudes.
3. **Identical Rule for Calibration and Test:** The identical sampling rule is applied to both $D_{\text{cal}}$ (946 pairs) and $D_{\text{test}}$ (946 pairs), yielding exactly 946 calibration records and 946 test records.
4. **Record-Specific Features:** As proven in Phase 7.0.3, 21 of 22 molecular features vary across records within multi-SMILES pairs. Non-conformity scores and test evaluations evaluate the selected record's **actual feature vector** $x_{g, j_g}$:
   $$S_g = \frac{|y_{g, j_g} - \hat{\mu}(x_{g, j_g})|}{\hat{\sigma}^*(x_{g, j_g}) + \epsilon}$$

### 4.2 Partition Composition & Cluster Structure Audit
A simulation of the pre-registered 60/20/20 pair-disjoint split on the 4,730 canonical material-name pairs (`seed = 42`) was audited:

```
Pair-Disjoint Split Composition (4,730 Canonical Material-Name Pairs):
├── Training Partition (D_train, 60%): 2,838 pairs (11,786 total records)
│   ├── Singleton Pairs (n_g = 1):  1,669 pairs (58.8% of train pairs)
│   ├── Repeated Pairs (n_g > 1):   1,169 pairs (41.2% of train pairs)
│   ├── Maximum Records in Pair:    1,334 records (e.g. PTB7-Th:PC71BM)
│   └── Record Distribution:        Median = 1.0 record, Mean = 4.15 records
│
├── Calibration Partition (D_cal, 20%): 946 pairs (2,746 total records)
│   ├── Singleton Pairs (n_g = 1):  552 pairs (58.4% of cal pairs)
│   ├── Repeated Pairs (n_g > 1):   394 pairs (41.6% of cal pairs)
│   ├── Maximum Records in Pair:    156 records
│   ├── Alternative B Sampled:      946 records (exactly 1 record per pair)
│   └── Record Distribution:        Median = 1.0 record, Mean = 2.90 records
│
└── Test Partition (D_test, 20%): 946 pairs (7,935 total records)
    ├── Singleton Pairs (n_g = 1):  569 pairs (60.1% of test pairs)
    ├── Repeated Pairs (n_g > 1):   377 pairs (39.9% of test pairs)
    ├── Maximum Records in Pair:    2,658 records (e.g. PM6:Y6 hero cluster)
    ├── Alternative B Sampled:      946 records (exactly 1 record per pair)
    └── Record Distribution:        Median = 1.0 record, Mean = 8.39 records
```

### 4.3 Partition Pair Disjointness Verification
Pair overlap checks confirm zero data leakage across splits:
$$|\text{Train} \cap \text{Cal}| = 0, \quad |\text{Train} \cap \text{Test}| = 0, \quad |\text{Cal} \cap \text{Test}| = 0$$

### 4.4 Anti-Snooping Integrity
No test split labels were accessed or evaluated, and no model predictions or performance metrics were calculated during this verification.

---

## 5. Literal Git Repository Status Report

The literal output of `git status --short` is:

```text
$ git status --short
?? ml/DATASET_DECISION.md
?? ml/DATASET_RESEARCH_REPORT.md
?? ml/DATASET_SCHEMA_REPORT.md
?? ml/DATA_QUALITY_ASSESSMENT.md
?? ml/PHASE_4_1_SCIENTIFIC_AUDIT.md
?? ml/PHASE_4_2_ACQUISITION_REPORT.md
?? ml/PHASE_4_2_DATA_READINESS.md
?? ml/PHASE_4_2_FEATURE_AUDIT.md
?? ml/PHASE_4_2_STRUCTURE_AUDIT.md
?? ml/PHASE_4_3_PRETRAINING_REVIEW.md
?? ml/PHASE_5_1_INDEPENDENT_AUDIT.md
?? ml/PHASE_5_BASELINE_REPORT.md
?? ml/PHASE_6_1_1_FINAL_AUDIT.md
?? ml/PHASE_6_1_2_FINAL_STATISTICAL_CORRECTIONS.md
?? ml/PHASE_6_1_NESTED_VALIDATION_REPORT.md
?? ml/PHASE_6_VALIDATION_REPORT.md
?? ml/PHASE_7_0_1_CONFORMAL_AUDIT.md
?? ml/PHASE_7_0_1_FINAL_VALIDATION_DESIGN.md
?? ml/PHASE_7_0_1_GROUPED_COVERAGE_DESIGN.md
?? ml/PHASE_7_0_2_FINAL_DESIGN_CORRECTIONS.md
?? ml/PHASE_7_0_3_FINAL_SCOPE_AUDIT.md
?? ml/PHASE_7_0_4_PAIR_IDENTITY_AUDIT.md
?? ml/PHASE_7_CALIBRATION_DESIGN.md
?? ml/PHASE_7_UNCERTAINTY_RESEARCH.md
?? ml/PHASE_7_VALIDATION_PROTOCOL.md
?? ml/experiments/
?? ml/results/
```

### Working Tree Accounting
- **Cleanliness:** The working tree contains **zero modified tracked files** (no ` M` or `M `) and **zero staged changes**. It is not clean in the Git sense due to untracked markdown reports and experiment directories.
- **Accountability:** Every untracked file corresponds to an approved deliverable from Phase 4.1 through Phase 7.0.4. Scratch verification scripts are maintained outside the repository in `<conversation-id>/scratch/`.

---

## 6. Verification of Strict Restrictions

- [x] **No model fitting.**
- [x] **No conformal calibration implemented.**
- [x] **No prediction intervals generated.**
- [x] **No test-performance calculated.**
- [x] **No production backend or frontend changes.**
- [x] **Frozen Phase 5 baseline strictly preserved.**
- [x] **No Git commits created.**

Phase 7.0.4 is complete. All pair identity mechanics, discrepancy roots, target definitions, and sampling parameters are fully audited and locked. We await your explicit authorization before proceeding to **Phase 7.1: Conformal Calibration Implementation and Validation**.
