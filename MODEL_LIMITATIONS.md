# OPV-PredictNet — Scientific Model Limitations and Governance Specification

**Document Version:** 1.0 (Phase 12 Release)  
**Governance Status:** Mandatory System Governance  
**Target Audience:** Computational Chemists, Materials Scientists, Machine Learning Engineers  

---

## 1. Scientific Governance and Status

Every model artifact, prediction endpoint, and serialized data structure within OPV-PredictNet is strictly bounded by non-negotiable governance constants:

```json
{
  "research_estimate_only": true,
  "validated_for_production": false
}
```

### Core Disclaimers
1. **Statistical Estimates, Not Physical Certifications:** Predicted figures of merit—including Power Conversion Efficiency ($PCE$), Open-Circuit Voltage ($V_{OC}$), Short-Circuit Current Density ($J_{SC}$), and Fill Factor ($FF$)—are statistical estimates derived from empirical literature datasets. They do not constitute experimentally verified physical measurements or performance guarantees for device fabrication.
2. **Device-Level Emergence:** In organic photovoltaics, solar cell efficiency is an emergent property of the active layer blend nanomorphology, donor:acceptor interpenetrating network, interfacial contact layers, substrate topography, and degradation kinetics. Two-dimensional chemical structures and nominal processing descriptors cannot fully capture three-dimensional domain crystallite orientation, molecular packing, or batch-to-batch molecular weight polydispersity ($M_w, M_n, \text{PDI}$).

---

## 2. Approved Model Roles and Architectural Scopes

Based on the Phase 8.3 Scientific Governance Contract and Phase 8.5 Benchmark Reproduction, four research models are maintained:

| Model ID | Input Features | Architecture | Authoritative Governance Role | Recommended Research Scope |
| :---: | :---: | :---: | :--- | :--- |
| **M0** | 22 molecular | Random Forest (100 trees) | Pure Molecular Baseline | Rapid virtual screening across known chemical families when processing metadata is absent. |
| **M1** | 50 (22 mol + 28 one-hot) | Random Forest (100 trees) | One-Hot Categorical Processing Model | Benchmark comparison on familiar categorical device architectures; strictly bounded to known training categories. |
| **M2A** | 37 (22 mol + 15 continuous) | Random Forest (100 trees) | Continuous Processing Model | Fabrication optimization when numeric film thicknesses, temperatures, and concentration values are available. |
| **M3** | 44 (22 mol + 22 native missing) | HistGradientBoosting | Native-Missingness Champion Model | **Default primary model**; handles partial or missing fabrication records natively without synthetic imputation. |

---

## 3. Chemical Scaffold Generalization and OOD Policy

Molecules are assigned to structural clusters using **Bemis-Murcko generic scaffolds** (`MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))`). The training scaffold registry comprises 430 donor scaffolds and 172 acceptor scaffolds from OPVdb.

```
                  ┌───────────────────────────────┐
                  │    Donor Scaffold Status?     │
                  └───────┬───────────────┬───────┘
                          │               │
                  Known   │               │ Unseen (OOD)
                          ▼               ▼
           ┌──────────────────────┐   ┌──────────────────────┐
           │ Acceptor Scaffold?   │   │ Acceptor Scaffold?   │
           └──────┬────────┬──────┘   └──────┬────────┬──────┘
                  │        │                 │        │
           Known  │        │ Unseen   Known  │        │ Unseen
                  ▼        ▼                 ▼        ▼
             [In-Domain] [Strategy B]   [Strategy A] [Dual OOD]
             Severity:   Severity:      Severity:    Severity:
             NORMAL      WARNING        WARNING      CRITICAL
             (HTTP 200)  (HTTP 200)     (HTTP 200)   (HTTP 422 Rejection)
```

### 3.1 Strategy A: Unseen Donor Scaffold (Donor OOD)
- **Observed Behavior:** Moderately degraded accuracy compared to random splits.
- **Support Level:** `exploratory_unseen_donor_candidate`.
- **API Status:** HTTP 200 with `ood_donor_scaffold=True` and `severity="warning"`.
- **Scientific Caveat:** Prediction intervals retain valid nominal coverage under single-scaffold shift if the acceptor partner is well-represented in the training distribution.

### 3.2 Strategy B: Unseen Acceptor Scaffold (Acceptor OOD)
- **Observed Behavior:** Marked coverage degradation in high-PCE regimes.
- **Support Level:** `exploratory_unseen_acceptor_candidate`.
- **API Status:** HTTP 200 with `ood_acceptor_scaffold=True` and `severity="warning"`.
- **Critical Coverage Finding:** Under Strategy B scaffold shifts, empirical coverage for high-performing acceptor blends drops to **~45% at nominal 90% confidence**. Conformal intervals under-represent uncertainty for novel non-fullerene acceptor classes (e.g., novel fused-ring electron acceptors).

### 3.3 Strategy D: Strict Dual-Scaffold Extrapolation (Dual OOD)
- **Observed Behavior:** **Predictive failure.** Across all models (M0, M1, M2A, M3), test performance under joint donor and acceptor scaffold shift drops to $R^2 \approx 0.08$ with root-mean-square errors exceeding $2.4\%$.
- **System Policy:** **Strict HTTP 422 Rejection by default** (`DUAL_SCAFFOLD_OOD_REJECTED`).
- **Explicit Override (`allow_dual_scaffold_ood=True`):** Returns HTTP 200 with `severity="critical"`, emitting explicit alerts stating that the prediction has no verified predictive power and is mathematically equivalent to uncalibrated prior guessing.

---

## 4. Conformal Prediction and Uncertainty Calibration

### 4.1 Nominal vs. Observed Empirical Coverage
Conformal prediction intervals are constructed via locally scaled absolute residual nonconformity scores using the Phase 8.5 Strategy C2 calibration partition ($N_{\text{cal}} = 3,863$):

$$\hat{C}(x) = \left[ \hat{\mu}(x) - q_{1-\alpha} \cdot \hat{\sigma}(x), \; \hat{\mu}(x) + q_{1-\alpha} \cdot \hat{\sigma}(x) \right]$$

| Nominal Confidence ($1 - \alpha$) | Conformal Quantile ($q$) | In-Distribution Coverage (Observed) | Unseen Acceptor Coverage (Strategy B) |
| :---: | :---: | :---: | :---: |
| **80%** | 1.341 | 80.2% | 58.4% |
| **90%** | 1.832 | 90.1% | 67.2% (Drops to ~45% for PCE > 12%) |
| **95%** | 2.215 | 95.0% | 76.8% |

### 4.2 Non-Exchangeability Caveat
Conformal calibration guarantees marginal validity strictly under the assumption of **exchangeability** between the calibration and test sets. Whenever a query molecule introduces an unseen scaffold or extreme fabrication conditions, exchangeability is formally violated. In these scenarios, nominal $90\%$ confidence intervals must not be interpreted as guaranteeing $90\%$ physical containment.

---

## 5. Chemoinformatics and Structural Representation Nuances

### 5.1 Finite Repeat-Unit Representation for Polymers
- In bulk-heterojunction solar cells, conjugated donors are predominantly high-molecular-weight polymers (e.g., PM6, PTB7-Th, P3HT).
- RDKit operates on discrete, stoichiometric molecular graphs. Consequently, polymers are represented as **finite repeat units (monomers or end-capped oligomer subunits)**.
- **Consequence:** Molecular weight ($MW$), total polar surface area ($TPSA$), and heavy atom counts represent the repeating unit, not the macroscale polymer chain. Extended conformational properties (e.g., chain entanglement, persistence length, aggregation state) are not explicitly captured in 2D topological descriptors.

### 5.2 Polymer Wildcards and Mixture Rejection
- Polymer SMILES containing BigSMILES notation (`{}`), bracket wildcards (`*`), or multi-component mixture dot notation (`.`) are strictly rejected with HTTP 422.
- Users must supply the resolved representative monomer or discrete chemical subunit without open bonding wildcards.

### 5.3 Reporting Bias and Missing Fabrication Metadata in OPVdb
- Literature experimental data compiled in OPVdb (Zenodo DOI: 10.5281/zenodo.4704870) originates from diverse peer-reviewed laboratories with varying fabrication protocols, spin-coating environments (glovebox $O_2/\text{H}_2O$ levels), and illumination mask geometries.
- **Reporting Bias:** Published literature heavily over-represents successful, high-efficiency devices ("positive reporting bias"). Low-efficiency failures ($PCE < 2\%$) are under-reported in scientific journals, skewing the training distribution towards moderately successful blend systems.
- Processing conditions (e.g., thermal annealing duration, solvent additive purity) are frequently omitted or reported with ambiguous phrasing in original research articles. Model M3 is designed to accommodate missing fabrication metadata without inventing synthetic default values.
