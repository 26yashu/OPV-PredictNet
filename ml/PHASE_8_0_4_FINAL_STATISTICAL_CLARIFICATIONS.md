# OPV-PredictNet — Phase 8.0.4: Final Statistical Clarifications

**Date**: 2026-11-24  
**Scope**: Final statistical clarifications resolving partial-input pattern counts and overlaps, correcting validation terminology and evidence gates, and formalizing the scientific interpretation of the Molecular-Descriptor-Only PCE Baseline (Model M0).  
**Constraint Compliance**: Strictly documentation and data verification only. Zero models trained, zero calibration quantiles calculated, zero test predictions generated, zero production code modified, zero Git commits created.

---

## 1. Reconciliation of Partial-Input Counts

### 1.1 Explanation of the 13,731 vs. 14,704 Discrepancy
In Phase 8.0.3, a preliminary table of seven input patterns reported counts summing to 13,731 records, compared to the 14,704 total records in the training partition ($D_{\text{train}}$).

Audit of the underlying selection queries reveals that the seven illustrative queries were **neither mutually exclusive nor exhaustive**:
1. **Extensive Pairwise Overlaps**:
   - Query P2 (*Missing thickness*: 2,260 records) and Query P3 (*Missing additive concentration*: 5,068 records) overlap by **1,428 records** where both thickness and additive ratio were omitted.
   - Query P2 and Query P6 (*Multiple missing*: 1,629 records) overlap by **1,212 records**.
   - Query P3 and Query P5 (*Missing architecture*: 1,544 records) overlap by **882 records**.
   - Query P3 and Query P6 overlap by **1,072 records**.
   - Query P5 and Query P6 overlap by **417 records**.
2. **Unclassified Records (Incomplete Coverage)**:
   - Because the queries were illustrative rather than a partition, their mathematical union spanned only **9,792 unique records**.
   - Exactly **4,912 records in $D_{\text{train}}$** (33.40%) were left unclassified by those ad-hoc masks (e.g., records reporting thickness and thermal annealing but omitting blend ratio and host solvent).
   - The apparent sum of 13,731 was an artifact of adding overlapping subsets while simultaneously omitting unclassified combinations.

### 1.2 Mutually Exclusive, Exhaustive Partition of $D_{\text{train}}$ ($N = 14,704$)
To represent all 14,704 records in $D_{\text{train}}$ rigorously without overlaps or unclassified gaps, every record is classified by the exact count $k \in \{0, 1, \dots, 7\}$ of primary processing parameters reported (`d_a_ratio`, `solvent_canonical`, `active_layer_thickness`, `annealing_temp`, `device_type`, `additive_canonical`, and `interlayers`):

```
+---------------------------------------------------------------------------------------------------------+
| Mutually Exclusive and Exhaustive Field-Count Partition (D_train, N = 14,704)                           |
+----+--------------------------------+---------------+-----------+---------------------------------------+
| k  | Reported Field Count           | Record Count  | % of Data | Input Status                          |
+----+--------------------------------+---------------+-----------+---------------------------------------+
| 0  | Zero Fields Reported           | **1,548**     | 10.53%    | Molecular-Descriptor-Only Input       |
| 1  | Exactly 1 Field Reported       | **1,633**     | 11.11%    | Minimal Processing Input              |
| 2  | Exactly 2 Fields Reported      | **2,282**     | 15.52%    | Low Partial Processing Input          |
| 3  | Exactly 3 Fields Reported      | **2,230**     | 15.17%    | Low Partial Processing Input          |
| 4  | Exactly 4 Fields Reported      | **2,294**     | 15.60%    | Moderate Partial Processing Input     |
| 5  | Exactly 5 Fields Reported      | **2,510**     | 17.07%    | Moderate Partial Processing Input     |
| 6  | Exactly 6 Fields Reported      | **1,750**     | 11.90%    | Rich Processing Input                 |
| 7  | All 7 Primary Fields Reported  | **457**       | 3.11%     | Fully Specified Processing Recipe     |
+----+--------------------------------+---------------+-----------+---------------------------------------+
|    | **Total (Exhaustive Sum)**     | **14,704**    | **100.0%**| Exact bit-for-bit reconciliation      |
+----+--------------------------------+---------------+-----------+---------------------------------------+
```

```
+---------------------------------------------------------------------------------------------------------+
| Mutually Exclusive Completeness Tier Partition (D_train, N = 14,704)                                    |
+--------------------------+---------------------+----------------+---------------+-----------------------+
| Completeness Tier        | Primary Fields (k)  | Record Count   | % of D_train  | Model Routing         |
+--------------------------+---------------------+----------------+---------------+-----------------------+
| **Molecular-Only Tier**  | k = 0 fields        | 1,548          | 10.53%        | Model M0              |
| **Sparse Tier**          | k = 1 field         | 1,633          | 11.11%        | Model M2 / M3         |
| **Low-Partial Tier**     | k = 2 to 3 fields   | 4,512          | 30.69%        | Model M2 / M3         |
| **High-Partial Tier**    | k = 4 to 5 fields   | 4,804          | 32.67%        | Model M2 / M3         |
| **Rich Tier**            | k = 6 to 7 fields   | 2,207          | 15.01%        | Model M2 / M3         |
+--------------------------+---------------------+----------------+---------------+-----------------------+
| **Total**                | **k = 0 to 7**      | **14,704**     | **100.00%**   | Zero unclassified     |
+--------------------------+---------------------+----------------+---------------+-----------------------+
```

---

## 2. Correction of Validation Terminology and Evidence Gates

### 2.1 Terminology Corrections
1. **Replacement of "Validated Pattern"**:
   - The phrase "validated pattern" has been removed from all documentation.
   - Replaced with: **"supported input pattern observed in training data."**
   - Merely observing a missingness pattern in $D_{\text{train}}$ establishes representation, not predictive accuracy or statistical validity.
2. **Replacement of "High Confidence"**:
   - Unsupported qualitative claims such as "high confidence" have been removed.
   - Replaced with neutral, factual descriptions: **"fully specified processing recipe"** or **"unconditional baseline prediction."**

### 2.2 Pre-Registered Evidence Gates for Pattern Acceptance
Before any partial-input pattern or model configuration can be claimed as evaluated or statistically acceptable, it must satisfy four pre-registered evidence gates during Phase 8.1 final-test evaluation:
1. **Calibration Representation Gate**:
   - The pattern must be represented in $D_{\text{cal}}$ with sufficient sample size ($N_{\text{cal, slice}} \ge 30$) to support stable empirical quantile computation without extreme rank variance.
2. **Held-Out Out-of-Pair Generalization Gate**:
   - The pattern must be evaluated on held-out test records ($D_{\text{test}}$) from **unseen material-name pairs** drawn via Alternative B.
3. **Subgroup Slice Coverage Gate**:
   - For confidence level $1 - \alpha = 0.90$, empirical interval coverage on that pattern slice must fall within the nominal acceptance band:
     $$85.0\% \le \text{Coverage}_{90} \le 95.0\%$$
   - The slice must not achieve marginal coverage merely because undercoverage in one slice is masked by overcoverage in another.
4. **Subgroup Bias and Error Gate**:
   - Sliced Mean Signed Bias must be bounded: $|\text{Bias}| \le 0.50\text{ PCE pp}$.
   - Sliced MAE must demonstrate improvement over Model M0 on the same test slice.

*Protocol Rule*: Zero claims of performance validation are asserted prior to Phase 8.1 frozen test execution.

---

## 3. Scientific Interpretation of Model M0

### 3.1 Formal Model Re-Naming
Model M0 is formally designated across all specifications and reports as:
$$\textbf{Model M0: Molecular-Descriptor-Only PCE Baseline}$$

### 3.2 Scientific Scope and Physical Clarification
1. **What Model M0 Actually Predicts**:
   - Model M0 predicts measured power conversion efficiency (PCE) using **22 intrinsic molecular descriptors** computed from donor and acceptor chemical structures, **without explicit processing inputs**.
2. **What Model M0 Reflects**:
   - Predictions from Model M0 reflect the **empirical distribution of processing conditions, device architectures, and laboratory optimization practices present in the literature training dataset**.
   - For example, if a material pair was historically tested primarily in inverted architectures with DIO additives, Model M0 implicitly reflects the average efficiency achieved under those historical laboratory conventions.
3. **What Model M0 Does NOT Predict**:
   - Model M0 does **not** isolate or predict the "intrinsic molecular efficiency," "theoretical potential," or "thermodynamic ceiling" of the chemical structures.
   - It cannot decouple the chemical donor-acceptor electronic coupling from the morphological, solvent-dependent, and thermal processing variations present in the training publications.
   - All claims describing Model M0 as evaluating "intrinsic potential" or "intrinsic chemical efficiency" are eliminated.

---

## 4. Verification Checklist

- [x] **Partial-Input Reconciliation**: Explained the 13,731 count (overlapping queries double-counting records while leaving 4,912 records unclassified). Provided a clean, mutually exclusive, exhaustive field-count and tier partition summing to exactly 14,704 records.
- [x] **Validation Terminology Corrected**: Replaced "validated pattern" with "observed in training data"; removed "high confidence"; defined four rigorous pre-registered evidence gates required for acceptance.
- [x] **Model M0 Renamed and Clarified**: Formally renamed to "Molecular-Descriptor-Only PCE Baseline"; clarified that it predicts measured PCE under the empirical processing distribution of the training data, not intrinsic thermodynamic potential.
- [x] **Documentation Synchronized**: Updated Phase 8 specifications and contracts to align with these clarifications.
- [x] **Scope Boundaries Preserved**: Zero models trained, zero calibration quantiles calculated, zero final-test predictions generated, zero production code modified, zero Git commits created.

---

**Clarifications complete. All pre-experiment audits and corrections are finalized.**
