# Phase 4.2 — 22-Feature Pipeline Compatibility & Distribution Audit

**Project:** OPV-PredictNet  
**Phase:** 4.2 — Approved Dataset Acquisition and Empirical Audit  
**Date:** November 2026  
**Auditor:** Quality & Machine Learning Engineering Lead  
**Feature Source File:** `ml/src/features/descriptors.py` (Unmodified)  
**Target Table:** `opv_devices_strict_molecular_benchmark.csv` (22,467 records)  
**Status:** **AUDITED — 100% EXTRACTION SUCCESS (ZERO FAILURES, ZERO NON-FINITE VALUES)**

---

## 1. Executive Summary

The production 22-feature molecular descriptor extraction pipeline (`ml/src/features/descriptors.py`) was executed across all **22,467 records** of the strict molecular benchmark table.

### Primary Audit Findings:
1. **Zero Modifications Required:** The pipeline was executed strictly as written during Phase 3, without schema modifications, exception overrides, or fallback hacks.
2. **100% Extraction Success:** Exactly **22,467 out of 22,467 feature vectors (100.0000%)** were successfully computed.
3. **Zero Missing or Non-Finite Values:**
   - `NaN` count across all 22 features: **0**.
   - `Inf` count across all 22 features: **0**.
4. **Vector Conformity:** Every computed vector strictly adheres to the canonical 22-dimensional floating-point schema (`np.float64`).

---

## 2. Feature Schema & Extraction Architecture

The feature schema is defined in `ml/src/features/descriptors.py` and consists of 9 donor descriptors, 9 acceptor descriptors, and 4 interaction terms:

```python
FEATURE_NAMES = [
    # 9 Donor Descriptors
    "donor_mw", "donor_logp", "donor_tpsa", "donor_hbd", "donor_hba",
    "donor_rotatable_bonds", "donor_aromatic_rings", "donor_fraction_csp3", "donor_heavy_atoms",
    # 9 Acceptor Descriptors
    "acceptor_mw", "acceptor_logp", "acceptor_tpsa", "acceptor_hbd", "acceptor_hba",
    "acceptor_rotatable_bonds", "acceptor_aromatic_rings", "acceptor_fraction_csp3", "acceptor_heavy_atoms",
    # 4 Pairwise Interaction Terms
    "mw_diff", "logp_diff", "tpsa_total", "aromatic_rings_total"
]
```

---

## 3. Empirical Feature Distributions Across the Strict Benchmark (N = 22,467)

The table below reports the empirical statistics (mean, standard deviation, minimum, median, and maximum) for every feature extracted across the 22,467 benchmark records.

| Feature Name | Category | Mean | Std Dev | Min | Median (50%) | Max | Physical Units / Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `donor_mw` | Donor Physical | 1,094.32 | 406.30 | 121.63 | 1,221.90 | 3,617.15 | g/mol (Molecular weight) |
| `donor_logp` | Donor Hydrophobic | 22.46 | 8.32 | -3.28 | 24.91 | 66.02 | Wildman-Crippen LogP |
| `donor_tpsa` | Donor Polarity | 35.23 | 22.42 | 0.00 | 34.14 | 439.83 | Å² (Topological polar surface) |
| `donor_hbd` | Donor H-Bonding | 0.02 | 0.32 | 0.00 | 0.00 | 14.00 | Hydrogen bond donors (Lipinski) |
| `donor_hba` | Donor H-Bonding | 8.91 | 3.68 | 0.00 | 10.00 | 44.00 | Hydrogen bond acceptors (Lipinski) |
| `donor_rotatable_bonds` | Donor Flexibility | 28.56 | 12.27 | 0.00 | 29.00 | 113.00 | Rotatable single bonds |
| `donor_aromatic_rings` | Donor Conjugation | 8.01 | 3.32 | 0.00 | 9.00 | 36.00 | Aromatic ring count |
| `donor_fraction_csp3` | Donor Saturation | 0.5013 | 0.0855 | 0.0000 | 0.4754 | 0.7692 | Fraction of sp3 carbons |
| `donor_heavy_atoms` | Donor Size | 72.21 | 26.87 | 4.00 | 80.00 | 236.00 | Non-hydrogen atom count |
| `acceptor_mw` | Acceptor Physical | 1,276.99 | 410.45 | 79.87 | 1,451.97 | 5,828.81 | g/mol (Molecular weight) |
| `acceptor_logp` | Acceptor Hydrophobic | 23.25 | 7.57 | -3.25 | 25.76 | 117.61 | Wildman-Crippen LogP |
| `acceptor_tpsa` | Acceptor Polarity | 110.63 | 64.46 | 0.00 | 129.30 | 508.17 | Å² (Topological polar surface) |
| `acceptor_hbd` | Acceptor H-Bonding | 0.0019 | 0.0709 | 0.00 | 0.00 | 6.00 | Hydrogen bond donors (Lipinski) |
| `acceptor_hba` | Acceptor H-Bonding | 8.79 | 5.18 | 0.00 | 10.00 | 32.00 | Hydrogen bond acceptors (Lipinski) |
| `acceptor_rotatable_bonds`| Acceptor Flexibility | 25.37 | 17.09 | 0.00 | 29.00 | 184.00 | Rotatable single bonds |
| `acceptor_aromatic_rings` | Acceptor Conjugation | 12.83 | 6.65 | 0.00 | 10.00 | 54.00 | Aromatic ring count |
| `acceptor_fraction_csp3` | Acceptor Saturation | 0.3345 | 0.1670 | 0.0000 | 0.3684 | 1.0000 | Fraction of sp3 carbons |
| `acceptor_heavy_atoms` | Acceptor Size | 91.99 | 26.18 | 3.00 | 101.00 | 434.00 | Non-hydrogen atom count |
| `mw_diff` | Pair Interaction | 397.67 | 339.92 | 0.00 | 266.05 | 4,466.53 | \|Donor MW - Acceptor MW\| (g/mol) |
| `logp_diff` | Pair Interaction | 6.83 | 7.07 | 0.00 | 4.47 | 88.76 | \|Donor LogP - Acceptor LogP\| |
| `tpsa_total` | Pair Interaction | 145.85 | 68.20 | 0.00 | 164.94 | 612.33 | Donor TPSA + Acceptor TPSA (Å²) |
| `aromatic_rings_total` | Pair Interaction | 20.84 | 6.11 | 1.00 | 20.00 | 64.00 | Donor + Acceptor Aromatic Rings |

---

## 4. Chemical & Physical Interpretation of Descriptors

The computed statistics provide a clear, scientifically coherent signature of modern organic photovoltaic active layer materials:

### 1. Molecular Weight and Size
- **Donors:** Mean molecular weight is $1,094 \pm 406$ g/mol with a median of $1,222$ g/mol, reflecting the oligomeric/monomeric repeat units of high-performance donor polymers (e.g. PM6 at 1,221.9 g/mol, PTB7-Th at ~750 g/mol, D18 at ~1,400 g/mol).
- **Acceptors:** Mean molecular weight is $1,277 \pm 410$ g/mol with a median of $1,452$ g/mol, reflecting dominant small-molecule non-fullerene acceptors (e.g. Y6 at 1,452.0 g/mol, L8-BO at 1,452.0 g/mol, BTP-eC9 at 1,520.8 g/mol) and fullerene derivatives (PCBM at 910.9 g/mol).

### 2. High Hydrophobicity and Alkyl Solubilization
- **LogP Values:** Extreme mean LogP values ($22.46$ for donors, $23.25$ for acceptors) arise from long solubilizing alkyl side chains (e.g., 2-ethylhexyl, butyloctyl, 4-hexylphenyl).
- **Rotatable Bonds:** A mean of $28.56$ rotatable bonds in donors and $25.37$ in acceptors directly reflects these flexible alkyl chains, which are essential for solution processability in halogenated or non-halogenated aromatic solvents (chlorobenzene, chloroform, o-xylene).
- **Fraction CSP3:** Donors exhibit a higher fraction of $sp^3$ carbons (mean $0.501 \pm 0.085$) compared to acceptors (mean $0.334 \pm 0.167$), reflecting the heavier alkyl branching required to solubilize conjugated polymer backbones.

### 3. Extended Conjugation and Aromatic Topology
- **Aromatic Ring Counts:** Device pairs contain an average of $20.84 \pm 6.11$ aromatic rings (ranging up to 64 rings). Acceptors feature more aromatic rings on average ($12.83$ rings) than donors ($8.01$ rings), reflecting the fused polycyclic central cores of A-D-A' or A-DA'D-A type NFAs (e.g., benzothiadiazole, thieno[3,2-b]thiophene, dithienopyran cores).

### 4. Absence of Hydrogen Bond Donors
- Both donors and acceptors exhibit near-zero hydrogen bond donors (mean HBD $0.02$ for donors, $0.002$ for acceptors). Median HBD is exactly $0.00$. This is physically necessary: strong hydrogen bond donor motifs would cause uncontrolled aggregation or precipitation in non-polar casting solvents.

---

## 5. Non-Finite Value Audit

A rigorous vectorized check was performed across the entire $22,467 \times 22$ feature matrix:

```python
nan_mask = np.isnan(feature_matrix)  # Total NaNs: 0
inf_mask = np.isinf(feature_matrix)  # Total Infs: 0
```

- **NaN vectors:** 0
- **Inf vectors:** 0
- **Descriptor function exceptions:** 0
- **Feature alignment:** The ordering in `FEATURE_NAMES` matches the vector returned by `extract_features_from_mols` with zero discrepancy.

---

## 6. Compatibility Conclusion

The OPV-PredictNet 22-feature molecular descriptor schema is **100% compatible** with the official OPV-DB v1.1.0 strict molecular benchmark. All 22,467 records yield clean, physically interpretable numeric vectors suitable for immediate input into Scikit-learn regressors without requiring synthetic imputation or schema restructuring.
