# OPV-PredictNet — Phase 8.2.1: Final Model Readiness and Regime-Specific Assessment

**Date:** 2026-11-25  
**Scope:** Reassessment of candidate models **M0, M1, M2A, and M3** across five distinct chemical/device regimes following Phase 8.2.1 audit corrections.  
**Policy Status:** Research-use only. No universal model winner is selected. Zero production deployment authorized.

---

## 1. Executive Reassessment Principles

Following the Phase 8.2.1 statistical and scientific audit:
1. **No Universal Winner:** Model rankings depend on the chemical split regime and evaluation metric (MAE vs. RMSE vs. $R^2$ vs. bias). No single model dominates across all conditions:
   - In Phase 8.1 (held-out material-name pairs), **M1** achieved the lowest MAE (`2.0321 pp`), whereas **M3** achieved the highest $R^2$ (`0.6502`) and lowest RMSE (`2.7525 pp`).
   - In Strategy A (unseen donor scaffolds), **M3** achieved the lowest MAE (`2.0025 pp`), whereas **M2A** achieved near-zero signed bias (`-0.0040 pp`).
   - In Strategy C2 (unseen scaffold pairs), **M1** achieved the lowest MAE (`2.1393 pp`), whereas **M3** achieved the lowest RMSE (`2.8526 pp`) and highest $R^2$ (`0.6667`).
2. **Causal Mechanism of Processing Gain Unproven:** Adding fabrication parameters reduces sample MAE relative to M0 in every benchmark split, but this experiment does not prove "physical regularization" over dataset shortcut correlations (e.g., laboratory reporting practices or historical material-era associations). This interpretation remains a plausible hypothesis, not an established mechanism.
3. **Conformal Uncertainty Does Not Survive Scaffold Distribution Shift:** Across scaffold splits, empirical coverage fluctuates from `75.7%` (Strategy C2 at 80% nominal) to `99.7%` (Strategy C1 at 90% nominal) and collapses to `45.1%–59.2%` on high-PCE unseen acceptors.

---

## 2. Regime-by-Regime Model Assessment

### 2.1 Regime 1: Known-Scaffold Combinatorial Research (Phase 8.1 Held-Out Pairs & Strategy C2)
*Scope:* Novel combinations of previously observed donor/acceptor chemical scaffolds (in C2, 73.25% of test records share a training donor scaffold, 84.86% share a training acceptor scaffold, and 58.35% share *both* individual scaffolds).
- **Evidence Summary:**
  - Phase 8.1 ($N=946$, 1 rec/pair): M1 MAE `2.0321` ($R^2=0.6422$), M3 MAE `2.0540` ($R^2=0.6502$), M2A MAE `2.0689` ($R^2=0.6332$), M0 MAE `2.1727` ($R^2=0.6110$).
  - Strategy C2 ($N=2,067$, unweighted): M1 MAE `2.1393` ($R^2=0.6578$), M2A MAE `2.1436` ($R^2=0.6582$), M3 MAE `2.1536` ($R^2=0.6667$), M0 MAE `2.3163` ($R^2=0.6172$). All five one-way cluster bootstraps show M1, M2A, and M3 below zero vs. M0.
- **Model Readiness:**
  - **M1, M2A, M3:** **Supported for exploratory research** on known scaffold combinations (M1 for lowest sample MAE with standard laboratory vocabularies; M2A for continuous parameter sweeps without categorical zero-vector barriers; M3 for lowest RMSE/$R^2$ and native missingness handling).
  - **M0:** Baseline molecular comparator only.
  - **Limitation:** Even with known scaffolds, 90% prediction intervals remain `8.28–9.30 pp` wide and nominal 80% intervals under-cover in C2 (`75.7%–78.5%`).

### 2.2 Regime 2: Unseen Donor Scaffolds with Known Acceptors (Strategy A)
*Scope:* Novel donor core scaffolds paired largely with known acceptor scaffolds (74.6% of Strategy A unique test acceptors overlap with training; `1,481` test devices across `216` donor scaffolds).
- **Evidence Summary:**
  - M3 (`MAE=2.0025`, `R^2=0.6428`, `90% width=9.26 pp`) and M1 (`MAE=2.0916`, `R^2=0.6318`, `90% width=10.25 pp`) improve over M0 (`MAE=2.2302`, `R^2=0.5839`) across all five one-way cluster bootstrap sensitivity margins (`95% CI < 0`).
  - M2A (`MAE=2.1470`, `R^2=0.6175`) crosses zero on all five cluster bootstrap margins vs. M0 (`[-0.2387, +0.0456]`).
  - Note: `215/513 (41.9%)` test DOIs also appear in training, so literature confounding cannot be ruled out.
- **Model Readiness:**
  - **M1 & M3:** **Supported as exploratory research candidates** for donor screening when paired with well-characterized acceptors (subject to wide intervals $\approx 9.3–10.3\text{ pp}$ and DOI overlap).
  - **M2A & M0:** Weak / non-preferred relative to M3 and M1 in this regime (M2A improvement over M0 is not statistically distinguishable under donor clustering).

### 2.3 Regime 3: Unseen Acceptor Scaffolds with Known Donors (Strategy B)
*Scope:* Novel acceptor scaffolds paired largely with known donors (60.7% of unique test donor scaffolds overlap with training; `3,797` test devices across `163` acceptor scaffolds).
- **Evidence Summary:**
  - Sample point metrics: M3 MAE `1.9463` ($R^2=0.5375$), M2A MAE `2.1158` ($R^2=0.4795$), M1 MAE `2.1820` ($R^2=0.4415$), M0 MAE `2.2243` ($R^2=0.4396$).
  - **Inferential Breakdown:** A single cluster (`PC61BM`) constitutes **61.18% (2,323/3,797)** of the test set. Under acceptor-scaffold cluster bootstrapping, **none of M1, M2A, or M3 achieve statistical significance over M0** (95% CIs cross zero: M3 `[-0.3781, +0.0386]`, M2A `[-0.1471, +0.0168]`, M1 `[-0.1191, +0.0443]`). M3 also underperforms M0 on two of the largest non-fullerene acceptor hubs (`+0.1837 pp`, `+0.3136 pp`).
- **Model Readiness:**
  - **M0, M1, M2A, M3:** **Weak / Unreliable for unseen acceptor generalization.** No processing model demonstrates statistically robust superiority across independent unseen acceptor scaffolds, and all models exhibit substantial positive global bias (`+0.43 to +0.80 pp`) alongside severe high-PCE failure.

### 2.4 Regime 4: Strict Dual-Scaffold Extrapolation (Strategy C1)
*Scope:* Simultaneously unseen donor AND unseen acceptor scaffolds (`N_train=13,324`, `N_cal=260`, `N_test=284`; `8,599` cross-edges dropped).
- **Evidence Summary:**
  - Variance explained collapses across every model: M0 $R^2 = 0.0786$ (`MAE=2.5740`), M1 $R^2 = 0.1672$ (`MAE=2.3999`), M3 $R^2 = 0.2385$ (`MAE=2.3013`), M2A $R^2 = 0.2440$ (`MAE=2.3314`).
  - Paired comparisons vs. M0 cross zero when clustered by `donor_scaffold` or `acceptor_scaffold`.
  - Conformal 90% intervals explode to **15.69–18.62 pp** width with excessive empirical overcoverage (**98.59%–99.65%**) caused by severe calibration-vs-test variance mismatch ($D_{\text{cal}} \text{ std}=6.53\%$ vs. $D_{\text{test}} \text{ std}=3.29\%$).
- **Model Readiness:**
  - **ALL MODELS (M0, M1, M2A, M3):** **UNSUPPORTED & PROHIBITED** for zero-shot novel dual-scaffold prediction, virtual screening of de novo donor–acceptor pairs, or candidate ranking.

### 2.5 Regime 5: High-PCE Discovery ($\text{PCE} \ge 15.0\%$)
*Scope:* Discovering or filtering champion-efficiency solar cells across unseen scaffolds or unseen pairs.
- **Evidence Summary:**
  - Strategy B ($\ge 15\%$, $N=71$): MAE `5.26–6.33 pp`, signed bias **`-5.25 to -6.33 pp`**, 90% coverage **`45.07%` (M0) to `59.15%` (M3)**.
  - Strategy C2 ($\ge 15\%$, $N=205$): MAE `2.76–3.32 pp`, signed bias **`-2.67 to -3.22 pp`**, 90% coverage **`73.66%` to `82.93%`**.
  - Strategy A ($\ge 15\%$, $N=115$): MAE `2.35–3.55 pp`, signed bias **`-2.24 to -3.55 pp`**, 90% coverage `86.96%–96.52%`.
  - Strategy C1 ($\ge 15\%$, $N=6$): Underpowered descriptive (`MAE=4.34–5.34 pp`, bias `-4.34 to -5.34 pp`).
- **Model Readiness:**
  - **ALL MODELS (M0, M1, M2A, M3):** **UNSUPPORTED & PROHIBITED** for autonomous high-PCE candidate discovery or threshold-based filtering. Tree ensembles compress champion efficiencies downward by 2.2–6.3 percentage points and lose conformal coverage reliability.

---

## 3. Consolidated Readiness Matrix & Phase 8.3 Preconditions

| Model | Known Scaffolds / Pair Combos (P8.1 & C2) | Unseen Donor Scaffolds (Strategy A) | Unseen Acceptor Scaffolds (Strategy B) | Dual-Scaffold Extrapolation (Strategy C1) | High-PCE Discovery ($\ge 15\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M0** (22 feats) | Baseline comparator | Baseline comparator | Weak / Unsupported | **Unsupported** ($R^2=0.079$) | **Unsupported** (Bias up to -6.3 pp) |
| **M1** (50 feats) | Exploratory Research (lowest sample MAE) | Exploratory Research | Weak / Inconclusive vs. M0 | **Unsupported** ($R^2=0.167$) | **Unsupported** (45–89% cov) |
| **M2A** (37 feats) | Exploratory Research (continuous sweeps) | Inconclusive vs. M0 | Weak / Inconclusive vs. M0 | **Unsupported** ($R^2=0.244$) | **Unsupported** (54–89% cov) |
| **M3** (44 feats) | Exploratory Research (highest $R^2$/RMSE) | Exploratory Research | Weak / Inconclusive vs. M0 | **Unsupported** ($R^2=0.239$) | **Unsupported** (59–87% cov) |

### Required Preconditions Before Authorizing Phase 8.3
1. Any Phase 8.3 modeling work aimed at improving scaffold generalization must address **2D descriptor saturation and tree extrapolation bounds** (e.g., testing domain-specific electronic representations, frontier orbital / graph representations, or similarity-weighted / out-of-distribution applicability domains) rather than tuning hyperparameters on existing Random Forests / GBDTs.
2. Evaluation designs in Phase 8.3 must explicitly separate **per-scaffold** (equal cluster weight / 1 record per scaffold) from **per-device** (cluster-size-weighted) metrics so giant hubs like `PC61BM` (61.2% of Strategy B) do not mask behavior on diverse chemical scaffolds.
3. Conformal calibration under scaffold shift must explicitly measure calibration-vs-test residual shift rather than claiming distribution-free guarantees across non-exchangeable scaffold splits.
