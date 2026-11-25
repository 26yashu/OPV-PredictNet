# OPV-PredictNet — Phase 8.3: Scientific Model Governance Policy

**Date:** 2026-11-25  
**Scope:** Binding scientific, operational, and UI/API governance controls for OPV-PredictNet models **M0, M1, M2A, and M3**.  
**Authority:** Overrides any permissive inference or marketing claims in prior research drafts.

---

## 1. Six Non-Negotiable Application Restrictions

All future backend endpoints, frontend interfaces, CLI scripts, and exported research reports exposing M0, M1, M2A, or M3 predictions **must enforce all six restrictions below**:

### Restriction 1: No Autonomous Discovery Claims
- Models must **never** be marketed, documented, or configured for autonomous "closed-loop" material discovery, automated synthesis ranking without human chemist review, or production decision-making.
- Predictions reflect retrospective statistical associations learned from OPV-DB v1.1.0 (`22,467` literature rows), not verified physical simulations (`LIM-05`, `LIM-06`, `LIM-08`).

### Restriction 2: No Reliable High-PCE Discovery Claims
- No model in OPV-PredictNet reliably predicts champion-efficiency devices ($\text{PCE} \ge 15.0\%$).
- Across Phase 8.1 and Phase 8.2, high-PCE cells suffer systematic negative bias (`-2.24` to `-6.33 pp`) and conformal coverage failure (`45.07%–59.15%` in Strategy B; `73.66%–82.93%` in C2; `81.33%–84.00%` in Phase 8.1) (`LIM-02`).
- **Rule:** Hard threshold filtering (e.g., `"return candidates > 15% PCE"`) and claims of high-efficiency screening are **strictly prohibited**. Any prediction $\ge 12.0\%$ PCE must emit `HIGH_PCE_UNDERPREDICTION_WARNING`.

### Restriction 3: No Zero-Shot Dual-Scaffold Prediction Claims
- Under strict dual-scaffold holdout (Strategy C1, $N=284$), $R^2$ collapses to **`0.0786` (M0), `0.1672` (M1), `0.2440` (M2A), and `0.2385` (M3)** (`LIM-01`).
- **Rule:** Queries where *both* donor and acceptor Bemis–Murcko scaffolds are absent from training (`DUAL_UNSEEN_SCAFFOLDS`) must be **hard-rejected by default** (`OutOfDistributionScaffoldError`), and zero-shot novel scaffold screening claims are **strictly prohibited**.

### Restriction 4: Mandatory "Experimental Research Estimate" Labeling
- Every prediction payload, UI card, CSV export, and report must explicitly contain:
  - `research_estimate_only: true`
  - `validated_for_production: false`
  - Human-readable disclaimer: *"Experimental research estimate from retrospective OPV-DB literature data; not prospectively validated for synthesis or commercial decisions."*

### Restriction 5: Explicit Uncertainty Interval Limitations
- Conformal prediction intervals must never be displayed as unconditional error bounds or distribution-free guarantees (`LIM-03`).
- Every displayed interval must explicitly declare `available`, `exchangeability_valid`, and non-guarantee flags proving nominal 90% coverage is **not guaranteed** for:
  1. **Unseen scaffolds** (donor, acceptor, or dual unseen).
  2. **High-PCE predictions** ($\ge 12\% / 15\%$, collapsing to `45.1%–59.2%` in Strategy B).
  3. **Arbitrary individual records** (split-conformal coverage is marginal across exchangeable draws, never conditional on a specific single device).
  4. **Distribution-shifted inputs** (coverage fluctuates from `75.7%` in C2 80% to `99.7%` in C1 90%).
- Calibration source (`Phase 8.1 Pair Holdout, N_cal=946, rank k=853`) and widths (`8.43–8.97 pp` on familiar pairs) must be explicitly separated from `point_prediction`, `ood_status`, and `scientific_support_status`.
- Intervals must be suppressed (`available: false`) whenever `scaffold_ood_class` is `UNSEEN_ACCEPTOR_KNOWN_DONOR`, `DUAL_UNSEEN_SCAFFOLDS`, or an unsupported polymer/mixture approximation.

### Restriction 6: Explicit Warnings or Hard Rejection for Unsupported Inputs
Unsupported molecular structures, missing/ambiguous identities, invalid categories, unphysical fabrication parameters, and high-risk OOD scaffolds must trigger deterministic warnings or hard rejection—**never silent defaults, clipping, or in-domain assumptions**.

---

## 2. Input Enforcement & Routing Governance Matrix

| Input / Scaffold State | Permitted Model(s) | System Action | Mandatory Warnings / Error Codes |
| :--- | :---: | :--- | :--- |
| Missing Donor or Acceptor Identity | None | **HARD REJECT** | `MissingMolecularIdentityError` |
| Material-Name Query Without SMILES (or Ambiguous Collision) | None | **HARD REJECT** | `AmbiguousMaterialIdentityError` (if SMILES supplied & conflicts: `MATERIAL_NAME_STRUCTURE_CONFLICT_WARNING`) |
| Unparseable / Empty SMILES or BigSMILES | None | **HARD REJECT** | `InvalidMolecularInputError` |
| Unseen Wildcard `*` Polymer or `.` Salt/Mixture | None | **HARD REJECT** | `UnsupportedPolymerOrMixtureError` (if matched in training/override: `UNSUPPORTED_POLYMER_OR_MIXTURE_WARNING`, `interval.available=false`) |
| Empty Bemis–Murcko Scaffold (`[NO_RING]:<smiles>`) | Per OOD lookup | **OOD LOOKUP + WARN** | `NON_RING_PSEUDO_SCAFFOLD_WARNING`; hard reject if dual unseen |
| Invalid Categorical Type / Empty String (`M1`) | None | **HARD REJECT** | `PhysicalInputValidationError` (reject prior to `map_*_nominal`) |
| Unphysical Processing ($T \le 0$ nm, $T_{\text{ann}} > 350^\circ\text{C}$, $D:A \le 0$) | None | **HARD REJECT** | `PhysicalInputValidationError` |
| `DUAL_UNSEEN_SCAFFOLDS` (both D & A unseen) | None (unless diagnostic override) | **HARD REJECT** | `OutOfDistributionScaffoldError` (`UNSUPPORTED_EXTRAPOLATION` + `interval.available=false` if debug override) |
| `UNSEEN_ACCEPTOR_KNOWN_DONOR` | Exploratory diagnostic only | **ALLOW WITH CRITICAL WARN** | `CRITICAL_UNSEEN_ACCEPTOR_OOD_WARNING`, `INCONCLUSIVE_EXPLORATORY_ONLY`, `interval.available=false` |
| `UNSEEN_DONOR_KNOWN_ACCEPTOR` | M3, M1 (exploratory) | **ALLOW WITH OOD WARN** | `UNSEEN_DONOR_SCAFFOLD_OOD_WARNING`, `exchangeability_valid=false` |
| `UNSEEN_PAIR_KNOWN_SCAFFOLDS` | M1, M2A, M3, M0 | **ALLOW WITH PAIR WARN** | `UNSEEN_SCAFFOLD_PAIR_WARNING`, `exchangeability_valid=false` |
| `FAMILIAR_SCAFFOLD_PAIR` ($k = 0$) | M0 | **ALLOW (RESEARCH ESTIMATE)** | `RESEARCH_ESTIMATE_ONLY`, `MOLECULAR_ONLY_BASELINE` |
| `FAMILIAR_SCAFFOLD_PAIR` ($k \ge 1$) | M3, M1, M2A | **ALLOW (RESEARCH ESTIMATE)** | `RESEARCH_ESTIMATE_ONLY`; list `missing_fields` explicitly; warn if out-of-range or `'other'` mapped |
| Predicted $\text{PCE} < 0\%$ or $> 34\%$ | None | **HARD REJECT** | `UnphysicalPredictionError` (no silent clipping) |

---

## 3. Prohibited Marketing, UI, and Reporting Language

The following phrases are **prohibited** in backend responses, frontend tooltips, documentation, and presentations unless superseded by new prospective evidence passing Section 4 promotion gates:
- *"Production-ready predictor"* or *"scientifically certified for general OPV screening"*
- *"Accurately discovers novel donor–acceptor scaffolds"*
- *"Guaranteed 90% confidence intervals for new molecules"*
- *"Physical regularization proven by processing models"*
- *"Validated for complete devices"* (when referencing Phase 8.1 $k=6\text{--}7$ or $k=7$ with $N=22$)

---

## 4. Promotion Gates Required Before Any Future Production Authorization

Before any OPV-PredictNet model may advance beyond `research_estimate_only`, all five gates must be satisfied and independently audited:
1. **Gate 1 — Prospective Multi-Lab Validation (`LIM-08`):** Blind prospective prediction and subsequent device fabrication across at least 3 independent laboratories covering novel structures outside OPV-DB v1.1.0.
2. **Gate 2 — Scaffold-Level Extrapolation Recovery (`LIM-01`):** Dual-scaffold holdout (Strategy C1 or equivalent) must achieve $R^2 \ge 0.50$ under balanced per-scaffold evaluation without relying on hub leakage or DOI overlap.
3. **Gate 3 — High-PCE Tail Calibration (`LIM-02`):** Conditional high-PCE ($\ge 15\%$) bias must be reduced within $\pm 1.0\text{ pp}$ and empirical 90% coverage restored within $85\%\text{–}95\%$ across unseen acceptors.
4. **Gate 4 — Polymer & Electronic Representation (`LIM-05`):** Incorporate macromolecular batch features ($M_n$, PDI) and calibrated electronic descriptors beyond 2D monomer RDKit averages.
5. **Gate 5 — Production Software Sign-Off:** Complete backend integration tests covering `PHASE_8_3_INTEGRATION_CONTRACT.md` with zero fallback paths.
