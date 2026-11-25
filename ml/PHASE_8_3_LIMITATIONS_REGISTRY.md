# OPV-PredictNet — Phase 8.3: Final Scientific Limitations Registry

**Date:** 2026-11-25  
**Scope:** Canonical registry of unresolved scientific, data, algorithmic, and statistical limitations across OPV-PredictNet models **M0, M1, M2A, and M3**.  
**Rule:** None of the limitations below have been resolved by Phase 8.1–8.2.1 experimentation; they remain hard constraints on model interpretation and integration.

---

## Summary Registry Table

| ID | Limitation Category | Severity | Affected Models | Empirical Evidence Summary | Operational Consequence |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`LIM-01`** | **Scaffold Extrapolation Failure** | **CRITICAL** | M0, M1, M2A, M3 | Dual holdout (C1): $R^2 = 0.0786\text{–}0.2440$; Unseen acceptors (B): $R^2 = 0.4396\text{–}0.5375$. | Block/reject zero-shot dual-scaffold claims; warn on unseen single-axis scaffolds. |
| **`LIM-02`** | **High-PCE ($\ge 15\%$) Underprediction** | **CRITICAL** | M0, M1, M2A, M3 | Bias `-2.24` to `-6.33 pp`; Strategy B 90% cov `45.07%–59.15%`; Phase 8.1 cov `81.3%–84.0%`. | Prohibit champion-cell discovery or hard efficiency thresholding. |
| **`LIM-03`** | **Conformal Breakdown Under Distribution Shift** | **HIGH** | M0, M1, M2A, M3 | C1 90% cov `98.6%–99.7%` (width `15.7–18.6 pp`); C2 80% cov `75.7%–78.5%`. | Strip distribution-free coverage guarantees outside exchangeable pairs. |
| **`LIM-04`** | **Processing Metadata Missingness & Confounding** | **HIGH** | M1, M2A, M3 | Only `3.11%` (457/14,704) of $D_{\text{train}}$ has all 7 fields; $k=7$ test $N=22$. | Label partial-input modes exploratory; avoid causal "regularization" claims. |
| **`LIM-05`** | **Polymer Repeat-Unit & 2D Descriptor Blindness** | **HIGH** | M0, M1, M2A, M3 | Monomer SMILES omit $M_n$, PDI, regioregularity, conjugation length, morphology, HOMO/LUMO. | Cannot distinguish polymer batch variation or electronic alignment shifts. |
| **`LIM-06`** | **Publication, Era, & Laboratory Bias** | **HIGH** | M0, M1, M2A, M3 | Top 5 donors = `66.7%`, top 5 acceptors = `76.4%`, `PC61BM` = `61.2%` of B; `18–42%` DOI overlap. | Sample metrics reflect literature reporting skew; lab variance floor $\approx 8.4\text{–}9.0\text{ pp}$. |
| **`LIM-07`** | **Material-Name vs. Structure Ambiguity** | **MEDIUM** | Data / Pipeline | 79 donor & 68 acceptor names map to $>1$ SMILES; 59 SMILES map to $>1$ name; 12 ringless rows. | Require canonical SMILES + `[NO_RING]:` hashing for OOD detection, never names alone. |
| **`LIM-08`** | **Lack of Prospective Experimental Validation** | **CRITICAL** | M0, M1, M2A, M3 | All metrics derived retrospectively from OPV-DB v1.1.0; zero blind lab synthesis tests. | Restrict all outputs to "experimental research estimates" only. |

---

## Detailed Limitation Records

### `LIM-01`: Chemical Scaffold Extrapolation Failure
- **Description:** Models trained on 22 RDKit 2D physicochemical descriptors (and optional processing features) fail to generalize when evaluated on chemical scaffolds absent from training.
- **Empirical Evidence:**
  - **Strict Dual-Scaffold Holdout (Strategy C1, $N=284$):** $R^2$ collapses to **0.0786 (M0)**, **0.1672 (M1)**, **0.2440 (M2A)**, and **0.2385 (M3)**; MAE rises to `2.30–2.57 pp` and RMSE to `2.86–3.16 pp`.
  - **Acceptor Scaffold Holdout (Strategy B, $N=3,797$):** $R^2$ drops to **0.4396–0.5375**, global signed bias rises to `+0.43` to `+0.80 pp`, and cluster bootstrap 95% CIs show no significant improvement of M1/M2A/M3 over M0.
- **Root Cause:** Bipartite graph connectivity in OPV-DB v1.1.0 places **99.95% (22,455/22,467)** of records in a single giant component dominated by standard hubs (`BDT-BDD`, `Y6`, `PC71BM`). Tree ensembles partition descriptor space around these training clusters and cannot extrapolate structure–efficiency relationships across unseen conjugated ring topologies.
- **Mandatory Control:** Runtime inference must detect unseen donor/acceptor Bemis–Murcko scaffolds relative to the training scaffold set and reject or hard-flag dual-unseen queries.

---

### `LIM-02`: Systematic High-PCE Underprediction ($\text{PCE} \ge 15.0\%$)
- **Description:** Every evaluated ensemble model systematically underpredicts high-efficiency solar cells and loses conformal coverage in the right tail of PCE.
- **Empirical Evidence:**
  - **Phase 8.1 ($N=75$):** Signed bias `-2.57` to `-2.84 pp`, MAE `2.67–3.05 pp`, 90% coverage `81.33%–84.00%`.
  - **Phase 7.1 ($N=63$):** Unnormalized M0 90% coverage `77.78%` (though locally normalized M0 reached `92.06%`, it degraded on repeated pairs `84.35%`).
  - **Phase 8.2 Strategy A ($N=115$):** Bias `-2.24` to `-3.55 pp`; Strategy C2 ($N=205$): bias `-2.67` to `-3.22 pp`, 90% coverage `73.66%–82.93%`.
  - **Phase 8.2 Strategy B ($N=71$):** Bias **`-5.25` to `-6.33 pp`**, MAE `5.26–6.33 pp`, 90% coverage collapses to **`45.07%` (M0) – `59.15%` (M3)**.
- **Root Cause:** Tree-based regressors (`RandomForestRegressor`, `HistGradientBoostingRegressor`) output weighted averages of training leaf targets, causing regression-to-the-mean shrinkage bounded by training regimes, especially when novel acceptors introduce non-covalent locks or end-group halogenations absent from training leaves.
- **Mandatory Control:** High-PCE discovery and threshold-based candidate filtering are prohibited; outputs $\ge 12.0\%$ PCE must emit an explicit `HIGH_PCE_UNDERPREDICTION_WARNING`.

---

### `LIM-03`: Conformal Coverage Breakdown Under Distribution Shift
- **Description:** Split-conformal prediction intervals fail to maintain nominal finite-sample coverage guarantees when calibration and test sets are not exchangeable across scaffold partitions.
- **Empirical Evidence:**
  - **Strategy C1 Excess Conservatism:** $D_{\text{cal}}$ (`N=260`, `std=6.53%`, `21.2%` $\ge 15\%$ PCE) has far heavier residuals than $D_{\text{test}}$ (`N=284`, `std=3.29%`, `2.1%` $\ge 15\%$ PCE), inflating 90% widths to **15.69–18.62 pp** and empirical coverage to **98.59%–99.65%**.
  - **Strategy C2 Undercoverage:** Reverse target variance shift (`Cal std=4.23%` vs `Test std=4.94%`) causes observed coverage to miss nominal targets across every model at 80% (`75.71%–78.47%`) and 90% (`88.05%–89.89%`).
  - **Unnormalized Constant Widths:** Because `mean_width == median_width == 2 * q_alpha`, interval widths do not adapt to local chemical novelty or prediction difficulty in Phase 8.1/8.2 models.
- **Mandatory Control:** Uncertainty intervals must never be presented as guaranteed physical error bounds for out-of-distribution molecules or uncalibrated lab protocols.

---

### `LIM-04`: Processing Metadata Missingness & Historical Confounding
- **Description:** Experimental processing fields (`d_a_ratio`, `solvent`, `active_layer_thickness`, `annealing_temp`, `device_type`, `additive`, `etl`/`htl`) are overwhelmingly incomplete in literature data and cannot be assumed missing-at-random.
- **Empirical Evidence:**
  - In Phase 8.1 $D_{\text{train}}$ ($N=14,704$), only **457 records (3.11%)** report all 7 processing parameters (`P1`), `1,548 (10.53%)` report zero processing fields (`P6`), and `86.37%` are partially missing (`P2–P5`).
  - In Phase 8.1 $D_{\text{test}}$ ($N=946$), complete `k=7` recipes comprise only **22 records** (underpowered/descriptive only).
  - Unobserved processing fields are **unrecorded**, not experimentally absent (e.g., `annealing_temp = NaN` in 75.42% of the full dataset).
  - Observed MAE reductions from processing features (M1/M2A/M3 vs M0) remain consistent with **shortcut learning** (publication era and reporting habits correlating with efficiency regimes) as well as genuine physical conditioning.
- **Mandatory Control:** Never silently impute missing user parameters as active laboratory defaults; always declare exact missing fields and avoid causal "physical regularization" claims.

---

### `LIM-05`: Polymer Repeat-Unit & 2D RDKit Descriptor Blindness
- **Description:** Conjugated donor polymers and macromolecular acceptors are encoded solely as monomeric repeat-unit SMILES strings via 22 scalar 2D RDKit descriptors (`MolWt`, `MolLogP`, `TPSA`, H-bond counts, ring counts, `FractionCSP3`, `HeavyAtomCount`).
- **Omitted Physics:**
  - Number-average molecular weight ($M_n$), weight-average molecular weight ($M_w$), and polydispersity index (PDI), which govern domain crystallinity and miscibility.
  - Polymer regioregularity, end-group capping, chain conformation, and 3D $\pi$–$\pi$ stacking geometry.
  - Electronic energy levels ($\text{HOMO}$, $\text{LUMO}$, optical bandgap $E_g$) — omitted from the 22-feature schema due to high missingness and DFT/UPS/CV protocol inconsistency in OPV-DB.
- **Mandatory Control:** Document that identical repeat-unit SMILES with different batch $M_n$ or PDI collapse to identical molecular feature vectors.

---

### `LIM-06`: Dataset Publication, Era, and Laboratory Bias
- **Description:** OPV-DB v1.1.0 is an observational literature compilation ($N=22,467$) subject to strong positive-publication bias, material concentration, and unmodeled inter-laboratory variance.
- **Empirical Evidence:**
  - Top 5 donor scaffolds cover **66.74% (14,994/22,467)** of all records (`BDT-BDD` alone = **41.18%**). Top 5 acceptor scaffolds cover **76.44% (17,174/22,467)** (`Y6` = **35.17%**, `PC71BM` = **16.20%**, `PC61BM` = **10.34%**).
  - In Phase 8.2, **18.1%–41.9%** of test DOIs overlap with training DOIs across scaffold splits.
  - Even on held-out pairs with complete or high-partial recipes (Phase 8.1), 90% conformal intervals remain **8.43–8.66 pp wide**, reflecting unmeasured variables (glovebox $O_2/H_2O$, spin speed, solvent drying kinetics, active-area masking, solar simulator mismatch).
  - Furthermore, **tandem/multi-junction devices are unidentifiable** in raw metadata (Phase 6.1 audit).

---

### `LIM-07`: Material-Name vs. Canonical Structure Identity Ambiguities
- **Description:** Material names in literature are non-unique and inconsistent with molecular structure graphs.
- **Empirical Evidence (Phase 8.2 Mapping Audit):**
  - **79 donor canonical names** and **68 acceptor canonical names** map to multiple distinct RDKit canonical SMILES (e.g., `D18` variant end-groups, `AQx` regioisomers).
  - **30 donor SMILES** and **29 acceptor SMILES** appear under multiple synonymous names.
  - Strategy A/B/C1 had `3–8` material-name homonym collisions despite `0` canonical SMILES leakage.
  - **5 non-ring structures** (`12` records: e.g., inorganic `CuSCN` [7], `TiO2` [1] mislabeled as active layers, plus acyclic `PR1`, `GT-s`, `ITN`) yield empty Bemis–Murcko strings and require `[NO_RING]:<smiles>` hashing.
- **Mandatory Control:** All feature computation, scaffold OOD checks, and deduplication must key strictly on **RDKit canonical SMILES** and Bemis–Murcko hashes—never raw material names.

---

### `LIM-08`: Absence of Independent Prospective Experimental Validation
- **Description:** All metrics across Phases 5–8.2.1 are retrospective holdout estimates on OPV-DB v1.1.0 literature records collected prior to dataset freeze.
- **Empirical Gap:** Zero prospective blind laboratory synthesis, device fabrication, or out-of-database multi-lab validation campaigns have been conducted on predictions generated by M0, M1, M2A, or M3.
- **Mandatory Control:** Every API response, UI badge, and exported report must designate predictions as **unvalidated prospective / retrospective research estimates only**.
