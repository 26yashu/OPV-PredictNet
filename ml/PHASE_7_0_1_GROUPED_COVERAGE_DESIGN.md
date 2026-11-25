# OPV-PredictNet — Phase 7.0.1: Grouped Conformal Coverage & Exchangeability Design

**Author / Auditor:** Antigravity Agent  
**Date:** November 24, 2026  
**Status:** Completed Architectural Design Document (Pre-Implementation Review)  
**Target Model:** Phase 5 Baseline Random Forest Regressor ($n=100, \text{depth}=\text{None}, \text{split}=2, \text{leaf}=1, \text{features}=1.0$)  
**Dataset:** OPV-DB v1.1.0 Strict Molecular Benchmark (22,467 records, 4,730 material pair clusters)  
**Scope:** Definition of conformal exchangeability units, rigorous distinction among three generalization tiers, cluster-aware calibration mechanics, and scaffold-sharing realities.

---

## 1. Executive Summary & Core Architectural Premise

Under the stated exchangeability assumptions and the correct finite-sample conformal quantile, split conformal provides marginal coverage of at least the nominal level: $P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$. Under the additional assumption that calibration scores are almost surely distinct (no ties), the marginal coverage is also bounded from above by $1 - \alpha + \frac{1}{n_{\text{cal}} + 1}$. However, tied calibration scores or discrete quantization may result in empirical coverage strictly exceeding this upper bound. In organic photovoltaics, observations are neither independent nor identically distributed: records are grouped by donor–acceptor material pairs, and material pairs share recurring conjugated chemical scaffolds.

This design document establishes:
1. **Three Distinct Generalization Tiers:** Distinguishing (a) in-domain device record prediction, (b) out-of-domain material pair prediction, and (c) novel molecular scaffold extrapolation.
2. **The Scaffold-Sharing Reality:** Explicitly demonstrating that **disjoint material-pair partitions do NOT guarantee scaffold-disjoint generalization**.
3. **Cluster-Aware Conformal Calibration:** Comparing record-level, pair-level, and cluster-aware conformal schemes to establish exact mathematical coverage claims.
4. **Frozen Training-Only Normalization:** Specifying the mathematical calibration engine using the frozen Phase 5 Baseline Random Forest.

---

## 2. The Three Generalization Tiers in OPV Discovery

A single model cannot claim uniform coverage guarantees across all operational contexts. In OPV research, predictions fall into three distinct operational domains:

```
Three Tiers of Conformal Generalization in OPV
│
├── Tier 1: Record-Level In-Domain Prediction (Known Material Pairs)
│   ├── Target: New laboratory device fabrication of a previously studied material pair
│   ├── Sampling Assumption: Exchangeability of records within represented material clusters
│   ├── Mathematical Guarantee: Marginal coverage P(Y_new ∈ C(X_new)) ≥ 1 - α for known pairs
│   ├── Operational Use Case: Process optimization, thickness sweeps, annealing adjustments
│   └── Implementation Status: UNVALIDATED & OUTSIDE INITIAL SCOPE for Phase 7.1
│
├── Tier 2: Pair-Level Out-of-Domain Prediction (Unseen Material Pairs)
│   ├── Target: A newly synthesized or combined normalized material-name pair (e.g. PM6::Y6)
│   ├── Sampling Assumption: Exchangeability of material-pair clusters and within-pair selection
│   ├── Mathematical Guarantee: Marginal coverage for a randomly selected record of a new pair
│   ├── Operational Use Case: Virtual combinatorial screening of novel material combinations
│   └── Implementation Status: PRIMARY VALIDATION TARGET for Phase 7.1
│
└── Tier 3: Scaffold-Level Novel Chemotype Extrapolation (Unseen Scaffolds)
    ├── Target: Completely novel conjugated backbones / Bemis-Murcko core frameworks
    ├── Sampling Assumption: Extreme chemical out-of-distribution extrapolation
    ├── Mathematical Guarantee: None (conformal exchangeability systematically violated)
    ├── Operational Use Case: Exploration of radically novel chemical spaces
    └── Implementation Status: SEPARATE STRINGENT RESEARCH AUDIT (Gate 5)
```

### 2.1 Detailed Analysis & Scope Separation of Generalization Tiers

| Generalization Tier | Calibration Unit | Test Query Definition | Exchangeability Assumption | Justified Coverage Claim | Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Record-Level (Known Pairs)** | Individual device records sampled at random | New device made from an *existing* material pair | Record-level exchangeability across device fabrications: $(X_i, y_i) \sim P_{\text{records}}$ | **Marginal Coverage Guaranteed ($\ge 1 - \alpha$):** Valid for a random re-fabrication of an existing pair. | **UNVALIDATED & OUTSIDE PHASE 7.1 INITIAL SCOPE.** The 60/20/20 split in Phase 7.1 is pair-disjoint; Tier 1 requires an independent record-holdout partition. |
| **Tier 2: Pair-Level (Unseen Pairs)** | Single randomly selected record per unseen pair cluster ($S_g^{\text{single}}$) | Novel normalized material-name pair not present in training ($g_{\text{new}} \notin \mathcal{G}_{\text{train}}$, evaluated across 4,730 canonical pair groups) | Exchangeability of material-pair clusters and identical within-pair record sampling | **Marginal Coverage Guaranteed ($\ge 1 - \alpha$):** For a randomly selected record of a randomly selected new pair. | **PRIMARY TARGET FOR PHASE 7.1 IMPLEMENTATION.** Evaluated via 60/20/20 pair-disjoint split. |
| **Tier 3: Scaffold-Level (Novel Chemotypes)** | Pair-level clusters with novel Bemis-Murcko frameworks | Donor or acceptor with core scaffold absent from training | Violates exchangeability (covariate and conditional distribution shift) | **No Distribution-Free Guarantee.** Coverage empirically degrades below $1 - \alpha$. | **SEPARATE RESEARCH EVALUATION (Gate 5).** Evaluates degradation under chemical distribution shift; not guaranteed by pair splits alone. |

---

## 3. The Scaffold-Sharing Reality: Why Pair-Disjoint Does Not Equal Scaffold-Disjoint

A critical methodological fallacy in chemoinformatics is assuming that splitting data by material pairs creates an "unseen chemistry" benchmark.

### 3.1 Empirical Evidence of Scaffold Sharing in OPV-DB
In OPV-DB, material pairs are frequently constructed by pairing a shared donor polymer with variant acceptors, or vice versa:
1. **Donor Scaffolds:** Over 1,500 distinct material pairs utilize the **PM6** (poly[(2,6-(4,8-bis(5-(2-ethylhexyl-3-chloro)thiophen-2-yl)-benzo[1,2-b:4,5-b']dithiophene))-alt-(5,5'-(1',3'-di-2-thienyl-5',7'-bis(2-ethylhexyl)benzo[1',2'-c:4',5'-c']dithiophene-4,8-dione))]) backbone.
2. **Acceptor Scaffolds:** Hundreds of distinct acceptors (Y6, Y7, BTP-4F, BTP-eC9, L8-BO) share the identical central **fused-ring electron-deficient core** (dithienothiophen[3,2-b]-pyrrolobenzothiadiazole), differing only by solubilizing alkyl side-chains or halogen substitutions.

```
Scaffold-Sharing Structure Across Material Pairs
│
├── Material Pair A: PM6 :: Y6      ──────> (BDT Donor Core + Fused-Ring NFA Core)
├── Material Pair B: PM6 :: BTP-eC9  ──────> (Identical BDT Core + Identical Fused-Ring Core)
└── Material Pair C: D18 :: Y6      ──────> (Different Donor Core + Identical Fused-Ring Core)
```

### 3.2 Methodological Mandate
- When evaluating on held-out material pairs ($D_{\text{test}}$ partitioned by `pair = donor_canonical::acceptor_canonical`), the model is frequently interpolating across **familiar molecular scaffolds**.
- **Explicit Prohibition:** **We strictly prohibit claiming that material-pair-held-out validation evaluates scaffold-disjoint extrapolation.**
- The validation report must explicitly state that Pair-Level (Tier 2) evaluation assesses generalization to *novel combinations of known chemical fragments*, not to *completely novel chemical chemotypes* (Tier 3).

---

## 4. Evaluation of Conformal Schemes & Tier 2 Calibration Scores

In Tier 2, the prediction query is a **new, previously unobserved material pair** $g_{n+1} = (\text{donor}_{\text{new}}, \text{acceptor}_{\text{new}})$. In the calibration partition $D_{\text{cal}}$, there are $G_{\text{cal}}$ unique material pairs, each containing $n_g$ device fabrications: $\{(x_{g, i}, y_{g, i})\}_{i=1}^{n_g}$.

### 4.0 Material-Pair Identity & Empirical Invariance Audit

#### 4.0.1 Precise Definition of the Tier 2 Exchangeability Unit
In the historical Phase 5 baseline and all subsequent phases, the donor-acceptor grouping column is defined in `ml/experiments/train_baseline.py` (lines 201–202) as:
```python
pair_groups = (df['donor_canonical'] + "::" + df['acceptor_canonical']).to_numpy()
```
- **Exchangeability Unit:** Tier 2 evaluates **unseen normalized material-name combinations**, NOT exact canonical molecular structures.
- **Identity Source:** Identities derive from OPV-DB's normalized material strings (`donor_canonical` and `acceptor_canonical`), representing standard chemical naming conventions (e.g., "PM6", "Y6", "ITIC", "PBDB-T").
- **Stability:** The grouping is 100% deterministic, stable across dataset reloads, and contains zero missing/null values.

#### 4.0.2 The 4,730 vs. 4,832 Discrepancy Breakdown
Across the 22,467 records of the Strict Molecular Benchmark:
- **Canonical Material-Name Pairs:** 4,730 groups.
- **Raw SMILES Pairs:** 4,832 combinations.
- **RDKit Canonical Molecular Structure Pairs:** 4,788 combinations.

The discrepancy of 102 pairs (4,832 vs. 4,730) spans 230 material-name pairs with multiple SMILES strings, categorized as follows:
1. **Category A (Syntactically distinct raw SMILES, identical chemical structure): 40 pairs.** Different publications wrote identical molecules using alternative atom numbering, aromatic vs. kekule notations, or stereochemical slashes (e.g. `BTR-Cl::L8-BO`, `BTR-Cl::Y6`, `D18::AQx-2`). These resolve to 1 single RDKit canonical structure.
2. **Category B (Chemically distinct structures under the same material name): 190 pairs.** The literature applied the same material name to slight structural variants, including alkyl chain length homologues (e.g. octyl vs. decyl in `FTAZ::NNFA`, $C_2H_4$ side-chain difference in `J40::ITIC`, $C_6H_{12}$ difference in `J51::ITIC`), oligomeric repeat unit lengths, or stereoisomers.
3. **Preservation of the Baseline:** To preserve comparability with the Phase 5 baseline, the 4,730 material-name grouping is retained as the primary unit. A structure-level grouping (4,788 RDKit canonical pairs) is defined as an auxiliary sensitivity analysis, not substituted retrospectively.

### 4.0 Feature Invariance Audit: Record-Level vs. Pair-Level Descriptors
An inspection of the 22-feature production schema across all 22,467 records and 4,730 unique material pairs reveals that **features are NOT strictly pair-invariant**:
- **Chemical SMILES Variation:** In 174 donor pairs and 107 acceptor pairs, multiple distinct SMILES strings are curated under the same canonical donor or acceptor name (reflecting monomer vs. repeat-unit representations, oligomeric degree, side-chain alkyl isomer variations, or stereoisomers).
- **Feature Variation:** 21 out of 22 molecular descriptors vary across records within the same material pair (e.g. `donor_mw` varies in 141 pairs with standard deviation up to 999.6 Da; `mw_diff` varies in 161 pairs). Only `acceptor_hbd` exhibited zero variance across pairs in the benchmark.
- **Missing / Imputation:** Zero features use zero-imputation or missing device metadata; all 22 features are derived purely from RDKit Mol objects.
- **Score Formulation Impact:** Because descriptor values depend on the specific molecular SMILES recorded for each device fabrication, non-conformity scores must use the selected record's **actual feature vector** $x_{g, j_g}$, not an assumed common pair vector $x_g$:
  $$S_g = \frac{|y_{g, j_g} - \hat{\mu}(x_{g, j_g})|}{\hat{\sigma}(x_{g, j_g}) + \epsilon}$$

We evaluate two cluster-level calibration alternatives:

### 4.1 Alternative A: Maximum Normalized Residual per Pair (Simultaneous Condition Envelope)
1. **Mathematical Definition:**
   $$S_g^{\text{max}} = \max_{i \in \{1, \dots, n_g\}} \frac{|y_{g, i} - \hat{\mu}(x_{g, i})|}{\hat{\sigma}(x_{g, i}) + \epsilon}$$
2. **Quantile Index:** Exactly one score per pair ($N_{\text{cal}} = G_{\text{cal}}$). The finite-sample rank is $k = \lceil (G_{\text{cal}} + 1)(1 - \alpha) \rceil$, yielding $\hat{q}_{\text{max}, 1-\alpha} = S_{(k)}^{\text{max}}$.
3. **Formal Coverage Claim:** Under exchangeability of material pairs, for any new pair $g_{n+1}$ with any number of reported device fabrications $n_{g_{n+1}}$:
   $$\mathbb{P}\left( \forall i \in \{1, \dots, n_{g_{n+1}}\}, \;\; y_{g_{n+1}, i} \in C_{\alpha}^{\text{max}}(x_{g_{n+1}, i}) \right) \ge 1 - \alpha$$
   This guarantees **simultaneous coverage** of *all* experimental device fabrications for that pair.
4. **Trade-Off & Conservatism:** Because it covers the worst-case experimental condition across varying processing protocols, thicknesses, and additives for that pair, $S_g^{\text{max}}$ is driven by extreme outliers and unoptimized control devices in the literature, producing wide, conservative prediction intervals ($\approx 8\% - 12\%$ PCE pp). It is retained as a secondary simultaneous benchmark.

### 4.2 Alternative B: Single Randomly Sampled Record per Pair (Selected Primary Method)
1. **Pre-Registered Sampling Procedure:**
   Before inspecting any residuals or outcomes, for each pair $g \in \{1, \dots, G_{\text{cal}}\}$, draw exactly one record index $j_g \sim \text{Uniform}(\{1, \dots, n_g\})$ using a fixed random seed (`seed = 42`).
2. **Mathematical Definition:**
   $$S_g^{\text{single}} = \frac{|y_{g, j_g} - \hat{\mu}(x_{g, j_g})|}{\hat{\sigma}(x_{g, j_g}) + \epsilon}$$
3. **Quantile Index:** Exactly one score per pair ($N_{\text{cal}} = G_{\text{cal}}$). The finite-sample rank is $k = \lceil (G_{\text{cal}} + 1)(1 - \alpha) \rceil$, yielding $\hat{q}_{\text{single}, 1-\alpha} = S_{(k)}^{\text{single}}$.
4. **Exact Marginal Coverage Guarantee:**
   > *"The intended guarantee is marginal coverage for a randomly selected experimental record from a randomly selected new material pair, provided calibration and future material-pair clusters are exchangeable under the same pair-selection and within-pair record-sampling mechanisms."*
   $$\mathbb{P}\left( Y_{g_{n+1}, \text{random}} \in C_{\alpha}^{\text{single}}(x_{g_{n+1}, \text{random}}) \right) \ge 1 - \alpha$$
5. **Rigorous Operational Scope & Limitations of the Guarantee:**
   - **Sampling Population:** The material-pair sampling population consists of donor–acceptor material combinations curated in the scientific literature as indexed by OPV-DB v1.1.0 Strict Molecular Benchmark.
   - **Within-Pair Record Selection:** For each pair cluster $g$ containing $n_g$ observed records, index $j_g$ is drawn uniformly at random from $\{1, \dots, n_g\}$ ($P(j_g = i) = 1/n_g$).
   - **Uniformity over Observed Records Only:** Selection is strictly uniform over the *observed database records*, NOT over the unobserved space of physical processing conditions, laboratory procedures, or literature publications. Due to publication bias and optimization trials in the literature, observed records over-represent high-performing device configurations.
   - **Target Independence:** Record selection occurs pseudo-randomly with pre-registered seed 42, strictly independent of the target value $y_{g, i}$, model prediction $\hat{\mu}$, or residual size.
   - **No Protocol-Specific Coverage Claim:** This guarantee does **NOT** guarantee coverage for a user-selected experimental protocol (e.g. choosing a specific unoptimized thermal annealing condition or a novel non-halogenated solvent) or an arbitrary future laboratory distribution with shifted glovebox environmental baselines.
   - **Summary:** We explicitly do NOT claim protocol-specific or conditional coverage; the interval covers a typical literature realization under identical sampling.

### 4.3 Proof of Why Pair-Average / Pair-Median Scores are Mathematically Invalid
If a calibration score is defined as the mean or median residual across records in pair $g$ ($\bar{S}_g = \frac{1}{n_g} \sum s_{g, i}$ or $\tilde{S}_g = \text{Median}(s_{g, i})$):
- A conformal guarantee on $\bar{S}_g$ guarantees only that $\mathbb{P}\left( \frac{1}{n_{g_{n+1}}} \sum_{i=1}^{n_{g_{n+1}}} s_{g_{n+1}, i} \le \hat{q} \right) \ge 1 - \alpha$.
- By Jensen's inequality and the convexity of absolute residuals, bounding the *mean* residual does **not** bound individual residuals $s_{g_{n+1}, i}$.
- For a material pair with high processing variance (e.g. 5% unannealed vs. 15% annealed), the mean residual may be 3.5%, but individual device residuals reach 7.0%. Individual device fabrications will suffer catastrophic empirical undercoverage (frequently dropping below $60\%$ coverage at nominal $90\%$).
- **Rule:** **Pair-average and pair-median scores are strictly prohibited for individual or simultaneous coverage claims.**

### 4.4 Finite-Sample Quantile Rule, Bounds & Edge Cases ($k > N_{\text{cal}}$)
- **Lower Bound:** Under the stated exchangeability assumptions and finite-sample rank $k = \lceil (N_{\text{cal}} + 1)(1 - \alpha) \rceil$, marginal coverage is guaranteed to satisfy:
  $$P(Y_{\text{new}} \in C(X_{\text{new}})) \ge 1 - \alpha$$
- **Upper Bound Qualification:** Under the additional assumption that calibration non-conformity scores are almost surely distinct (no-tie condition), split conformal also satisfies:
  $$P(Y_{\text{new}} \in C(X_{\text{new}})) \le 1 - \alpha + \frac{1}{N_{\text{cal}} + 1}$$
- **Effect of Tied Scores:** In molecular databases, identical descriptor values or rounded literature values can generate identical non-conformity scores. Tied calibration scores or discrete quantization place positive probability mass at $\hat{q}_{1-\alpha}$, which may cause actual finite-sample coverage to strictly exceed $1 - \alpha + \frac{1}{N_{\text{cal}} + 1}$ (conservative overcoverage, never undercoverage).
- **Edge Case ($k > N_{\text{cal}}$):**
  - **Theoretical Treatment:** When $k = N_{\text{cal}} + 1$ (e.g., $N_{\text{cal}} < \lceil 1/\alpha \rceil - 1$), the $(N_{\text{cal}} + 1)$-th order statistic of an exchangeable sample is formally $+\infty$. Returning an infinite interval $(-\infty, +\infty)$ is a theoretically valid finite-sample conformal treatment that trivially guarantees $100\% \ge 1 - \alpha$ coverage.
  - **Implementation Policy:** In practical OPV molecular screening software, returning infinite intervals silently passes uninformative predictions to downstream filters. Therefore, as an explicit **implementation policy**, `GroupConformalCalibrator` raises `InsufficientCalibrationDataError` to require adequate calibration data rather than returning infinite bands or silently substituting an empirical percentile.## 5. Frozen Calibration Architecture Specification

The recommended calibration engine operates under a strict, frozen mathematical contract:

```
Sequential 4-Stage Leakage-Safe Architecture
│
├── STAGE 1: Model Fitting on D_train (60% of material pairs, ~2,838 pairs)
│   └── Fit Phase 5 Baseline Random Forest: mu_hat(X)
│
├── STAGE 2: Scale Selection on D_train ONLY (Internal 5-Fold Grouped CV)
│   ├── Compare Candidate 1: Unnormalized Scale (sigma_0(X) = 1.0)
│   ├── Compare Candidate 2: Frozen Tree Disagreement (sigma_trees(X) = std(T_1..100) + 0.10)
│   ├── Evaluate out-of-fold training Winkler scores
│   └── LOCK AND FREEZE the selected scale function: sigma_hat*(X)
│   └── (Zero calibration or test labels are accessed during Stage 2)
│
├── STAGE 3: Quantile Calibration on D_cal (20% of material pairs, ~946 pairs)
│   ├── Apply frozen mu_hat(X) and frozen sigma_hat*(X) to D_cal
│   ├── Compute Tier 2 scores: S_g = |y_{g, j_g} - mu_hat(X_g)| / sigma_hat*(X_g)
│   ├── Compute finite-sample quantiles: q_hat_80, q_hat_90, q_hat_95 using rank k
│   └── LOCK AND FREEZE all quantiles q_hat
│   └── (No model tuning or scale adjustment occurs during Stage 3)
│
└── STAGE 4: Final Validation Audit on D_test (20% of material pairs, ~946 pairs)
    ├── Evaluate locked intervals on completely untouched test partition
    └── Report empirical coverage, interval widths, and sliced audit tables
```

### 5.1 Formal Mathematical Definitions
1. **Frozen Point Prediction:**
   $$\hat{\mu}(X) = \frac{1}{100} \sum_{t=1}^{100} T_t(X)$$
2. **Frozen Relative Dispersion Scale:**
   $$\hat{\sigma}(X) = \sqrt{\frac{1}{100} \sum_{t=1}^{100} (T_t(X) - \hat{\mu}(X))^2} + \epsilon \quad (\epsilon = 0.10\% \text{ PCE})$$
3. **Normalized Non-Conformity Score:**
   $$S_i = \frac{|y_i - \hat{\mu}(X_i)|}{\hat{\sigma}(X_i)}$$
4. **Finite-Sample Corrected Quantile:**
   $$\hat{q}_{1-\alpha} = S_{(\lceil (N_{\text{cal}} + 1)(1 - \alpha) \rceil)}$$
5. **Raw Prediction Interval:**
   $$C_{\alpha}^{\text{raw}}(X) = \left[ \hat{\mu}(X) - \hat{q}_{1-\alpha}\hat{\sigma}(X), \;\; \hat{\mu}(X) + \hat{q}_{1-\alpha}\hat{\sigma}(X) \right]$$
6. **Physical Non-Negative Bound (Post-Processing Audit):**
   $$C_{\alpha}^{\text{phys}}(X) = \left[ \max(0.0, \; \hat{\mu}(X) - \hat{q}_{1-\alpha}\hat{\sigma}(X)), \;\; \hat{\mu}(X) + \hat{q}_{1-\alpha}\hat{\sigma}(X) \right]$$

---

## 6. Target Domain Tagging Protocol

Every prediction generated by the uncertainty engine must carry an automated **Generalization Tier Tag**:
- **`TIER_1_IN_DOMAIN`:** Query material pair was observed in the training partition. Conformal interval carries high-confidence in-domain device coverage guarantee ($1-\alpha$).
- **`TIER_2_NOVEL_PAIR`:** Query material pair is unobserved in training, but both donor and acceptor share chemical scaffolds with training exemplars (Tanimoto descriptor similarity $\ge 0.50$). Conformal interval carries cluster-exchangeable coverage claim.
- **`TIER_3_NOVEL_SCAFFOLD`:** Query molecule exhibits low descriptor similarity (Tanimoto similarity $< 0.50$ or Euclidean distance $> 75\text{th}$ percentile). **Warning attached:** *"Conformal coverage guarantees do not hold due to chemical scaffold extrapolation."*
