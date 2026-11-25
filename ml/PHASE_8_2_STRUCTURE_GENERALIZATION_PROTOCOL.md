# OPV-PredictNet — Phase 8.2: Structure-Level Generalization Experimental Protocol

**Date**: 2026-11-25  
**Scope**: Pre-registered, locked experimental protocol for evaluating model generalization to unseen molecular structures and Bemis–Murcko scaffolds.  
**Benchmark Candidates**: Models M0 (22 feats), M1 (50 feats), M2A (37 feats), and M3 (44 feats).  
**Benchmark Strategies**: Strategy A (Donor Scaffold Holdout), Strategy B (Acceptor Scaffold Holdout), and Strategy C (Dual-Scaffold Holdouts).

---

## 1. Executive Summary & Protocol Principles

Prior project phases (Phase 5 through 8.1.2) evaluated models on held-out **material-name pairs**. Phase 8.2 directly tests the central scientific question of organic photovoltaic machine learning:
> *Can models trained on known chemical structures predict power conversion efficiency for donor and acceptor molecules with chemical scaffolds completely absent from the training set?*

### Non-Negotiable Protocol Rules
1. **Zero Chemical Scaffold Leakage**: In Strategy A, no donor scaffold present in the test set may appear in training or calibration. In Strategy B, no acceptor scaffold present in the test set may appear in training or calibration.
2. **Deterministic Partition Locking**: Partitions are locked using fixed random seeds (`7701`, `7702`, `7703`), with scaffolds sorted deterministically before shuffling.
3. **Strict Preprocessing Non-Leakage**: All feature extraction, one-hot encoders, and continuous medians are fitted strictly on $D_{\text{train}}$. No calibration or test structures may influence training-time transformations.
4. **Independent Split-Conformal Calibration**: Conformal calibration quantiles must be calibrated independently on disjoint calibration partitions ($D_{\text{cal}}$) under each respective scaffold split. Reusing Phase 8.1 quantiles is prohibited.
5. **Single-Threaded Determinism**: Single-threaded execution (`n_jobs=1`) is enforced across all estimators and evaluations to guarantee bit-for-bit reproducibility.

---

## 2. Benchmark Partitioning Strategies

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Phase 8.2 Scaffold-Generalization Benchmark Strategy Specifications                                                           |
+----------+-----------------------------+---------------+-----------------------+----------------------+-----------------------+
| Strategy | Generalization Dimension    | Group Key     | Locked Partition Seed | Evaluation Population| Target Generalization |
+----------+-----------------------------+---------------+-----------------------+----------------------+-----------------------+
| **A**    | **Donor Scaffold Holdout**  | `donor_scaff` | `seed = 7701`         | Held-out donor scaff.| Novel donor polymers  |
|          | (Unseen donor backbones)    | (1,073 total) | (60% Tr / 20% Cl/Te)  | (Identical test set) | with known acceptors  |
+----------+-----------------------------+---------------+-----------------------+----------------------+-----------------------+
| **B**    | **Acceptor Scaffold Holdout**| `acceptor_sc`| `seed = 7702`         | Held-out acceptor sc.| Novel NFA/Fullerenes  |
|          | (Unseen acceptor backbones) | (812 total)   | (60% Tr / 20% Cl/Te)  | (Identical test set) | with known donors     |
+----------+-----------------------------+---------------+-----------------------+----------------------+-----------------------+
| **C1**   | **Strict Dual-Scaffold**    | Dual global   | `seeds = 7701+7702`   | Mutually disjoint    | Simultaneously novel  |
|          | (Unseen D and A scaffolds)  | disjointness  | (Cross-edges deleted) | test (N=77, underpow)| donor & acceptor cores|
+----------+-----------------------------+---------------+-----------------------+----------------------+-----------------------+
| **C2**   | **Scaffold-Pair Holdout**   | `scaff_pair`  | `seed = 7703`         | Unseen combination   | Novel combination of  |
|          | (Unseen scaffold pairs)     | (2,819 total) | (60% Tr / 20% Cl/Te)  | (39-44% single overlap)| diverse scaffolds     |
+----------+-----------------------------+---------------+-----------------------+----------------------+-----------------------+
```

---

## 3. Partition Construction Specifications

### 3.1 Strategy A: Donor Scaffold Holdout (Seed = 7701)
1. **Scaffold Identification**: Every record is mapped to its RDKit Bemis–Murcko donor scaffold. Empty scaffolds are assigned `"[NO_RING]:" + smiles`.
2. **Anchor Group Assignment**: The dominant donor scaffold (BDT-BDD core, 9,251 records, 41.18% of data) is assigned deterministically to $D_{\text{train}}$ to preserve training set representativeness.
3. **Partitioning**: The remaining 1,072 donor scaffolds are sorted lexicographically, shuffled using `np.random.RandomState(7701)`, and partitioned:
   - Training ($D_{\text{train}}$): Anchor scaffold + 642 scaffolds (total 643 scaffolds, $\approx 17,800$ records).
   - Calibration ($D_{\text{cal}}$): 214 donor scaffolds ($\approx 3,600$ records).
   - Final Test ($D_{\text{test}}$): 216 donor scaffolds ($\approx 900-1,000$ records).
4. **Evaluation Sampling**: Alternative B sampling (1 record per test donor scaffold) is evaluated alongside the complete test population to assess both per-scaffold and per-device generalization.
5. **Disjointness Assertion**:
   $$S_D^{\text{train}} \cap S_D^{\text{cal}} = \emptyset, \quad S_D^{\text{train}} \cap S_D^{\text{test}} = \emptyset, \quad S_D^{\text{cal}} \cap S_D^{\text{test}} = \emptyset$$

### 3.2 Strategy B: Acceptor Scaffold Holdout (Seed = 7702)
1. **Scaffold Identification**: Every record is mapped to its RDKit Bemis–Murcko acceptor scaffold. Empty scaffolds assigned `"[NO_RING]:" + smiles`.
2. **Anchor Group Assignment**: The dominant acceptor scaffold (Y6 core, 7,902 records, 35.17% of data) is assigned deterministically to $D_{\text{train}}$.
3. **Partitioning**: The remaining 811 acceptor scaffolds are sorted, shuffled with `seed = 7702`, and partitioned:
   - Training ($D_{\text{train}}$): Anchor scaffold + 486 scaffolds (total 487 scaffolds, $\approx 19,900$ records).
   - Calibration ($D_{\text{cal}}$): 162 acceptor scaffolds ($\approx 800$ records).
   - Final Test ($D_{\text{test}}$): 163 acceptor scaffolds ($\approx 1,700$ records).
4. **Disjointness Assertion**:
   $$S_A^{\text{train}} \cap S_A^{\text{cal}} = \emptyset, \quad S_A^{\text{train}} \cap S_A^{\text{test}} = \emptyset, \quad S_A^{\text{cal}} \cap S_A^{\text{test}} = \emptyset$$

### 3.3 Strategy C: Dual-Scaffold Holdout Specifications
- **Strategy C1 (Strict Global Dual Holdout)**:
  - Formed by taking test donor scaffolds from Strategy A ($D_{\text{test}}$) and test acceptor scaffolds from Strategy B ($A_{\text{test}}$).
  - Strict Test: $\text{donor\_scaffold} \in D_{\text{test}} \text{ AND } \text{acceptor\_scaffold} \in A_{\text{test}}$ ($N = 77$ records).
  - Strict Train: $\text{donor\_scaffold} \notin D_{\text{test}} \text{ AND } \text{acceptor\_scaffold} \notin A_{\text{test}}$ ($N = 19,887$ records).
  - All cross-edges ($N = 2,503$ records) deleted.
  - Documented as an underpowered, exploratory boundary ($N=77 < 100$).
- **Strategy C2 (Scaffold-Pair Holdout, Seed = 7703)**:
  - 2,819 scaffold pairs partitioned 60 / 20 / 20.
  - Unseen combinations of scaffolds; explicitly reports single-scaffold overlap.

---

## 4. Models and Evaluation Standards

### 4.1 Models Evaluated
```
+-------+-------------------------+----------+---------------------------------+--------------------------------+
| Model | Identifier              | Features | Architecture                    | Parameters                     |
+-------+-------------------------+----------+---------------------------------+--------------------------------+
| **M0**| Molecular Baseline      | 22       | Random Forest Regressor         | n_est=100, seed=8888, n_jobs=1 |
| **M1**| Naive OHE Processing    | 50       | Random Forest Regressor         | n_est=100, seed=8888, n_jobs=1 |
|**M2A**| Continuous Regimes (No F)| 37      | Random Forest Regressor         | n_est=100, seed=8888, n_jobs=1 |
| **M3**| Native-NaN GBDT         | 44       | HistGradientBoostingRegressor   | max_iter=100, seed=8888        |
+-------+-------------------------+----------+---------------------------------+--------------------------------+
```

### 4.2 Split-Conformal Calibration
For each strategy independently:
1. Compute absolute residuals on $D_{\text{cal}}$: $S_i = |y_i - \hat{y}_i|$.
2. Calculate cutoffs $q_\alpha$ at finite-sample order statistic rank:
   $$k = \left\lceil (n_{\text{cal}} + 1)(1 - \alpha) \right\rceil$$
   for $\alpha \in \{0.20, 0.10, 0.05\}$ (80%, 90%, 95% levels).
3. Evaluate empirical test coverage and median interval width on $D_{\text{test}}$.

---

## 5. Automated Verification and Leakage Assertions

Automated test suite (`tests/ml/test_phase_8_2_scaffold_benchmark.py`) must enforce:
1. `test_donor_scaffold_disjointness_strategy_a()`: 0 donor scaffolds shared between train, cal, and test.
2. `test_acceptor_scaffold_disjointness_strategy_b()`: 0 acceptor scaffolds shared between train, cal, and test.
3. `test_strict_dual_scaffold_disjointness_strategy_c1()`: 0 donor AND 0 acceptor scaffolds shared between train and test.
4. `test_scaffold_pair_disjointness_strategy_c2()`: 0 scaffold pairs shared between train, cal, and test.
5. `test_zero_empty_scaffold_leakage()`: Pseudo-scaffolds `[NO_RING]:*` cleanly partitioned with zero cross-contamination.
6. `test_preprocessing_leakage_freedom()`: Imputers, scalers, and encoders fitted strictly on $D_{\text{train}}$.

This protocol is approved and locked for execution.
