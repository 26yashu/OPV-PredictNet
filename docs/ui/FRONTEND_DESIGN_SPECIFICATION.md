# OPV-PredictNet — Frontend Design Specification
**Document Version:** 1.0.0  
**Phase:** 3.1 Architecture Preservation  
**Classification:** Scientific UI / UX Design Baseline  

---

## 1. Executive Vision

OPV-PredictNet combines rigorous computational materials science with a modern, high-performance web interface. The frontend is conceived as a **hybrid cinematic/scientific dashboard**—merging the visual immersion of deep dark-mode aesthetic spaces with the uncompromising precision required by quantum chemists, device physicists, and photovoltaic engineers.

The UI avoids generic enterprise SaaS styling in favor of an interface optimized for molecular structures, multidimensional device parameters, energy level alignments, and real-time ML inference.

---

## 2. Core Aesthetic & Theme System

### 2.1 Color Palette

| Token Name | Hex Code | Purpose / Application |
| :--- | :--- | :--- |
| **Space Void (Base)** | `#0a0e17` | Root background for 3D molecular canvases and application viewport |
| **Deep Slate (Surface)** | `#0f172a` | Primary card, container, and sidebar background |
| **Elevated Obsidian** | `#1e293b` | Secondary elevated cards, dialogs, dropdowns, and data tables |
| **Glass Border** | `rgba(148, 163, 184, 0.15)` | Subtle 1px borders providing glassmorphism edge definition |
| **Electric Cyan** | `#00f0ff` | Primary interactive highlights, active state indicators, HOMO/LUMO levels |
| **Scientific Turquoise** | `#10b981` | Positive performance indicators, valid chemistry badges, donor metrics |
| **Photovoltaic Amber** | `#f59e0b` | Warnings, intermediate confidence bounds, solar spectrum overlays |
| **Quantum Violet** | `#8b5cf6` | Acceptor metrics, non-fullerene core features, ML ensemble highlights |
| **Danger Coral** | `#ef4444` | Chemical valence errors, unphysical output warnings, validation failures |
| **Text Primary** | `#f8fafc` | Headings, primary figures of merit, active data values |
| **Text Muted** | `#94a3b8` | Parameter labels, scientific units, provenance citations, disclaimers |

### 2.2 Glassmorphism & Atmospheric Depth
- **Panel Construction**: Translucent dark slate surfaces (`rgba(15, 23, 42, 0.75)`) layered over subtle radial gradient glows.
- **Backdrop Blur**: `backdrop-filter: blur(16px)` applied to floating panels and modal overlays.
- **Illumination**: Soft edge glows (`box-shadow: 0 0 25px rgba(0, 240, 255, 0.08)`) on focused input groups and active molecular viewers.

### 2.3 Typography & Scientific Notation
- **Body & Interface**: Inter / System sans-serif (`font-feature-settings: "cv02", "cv03", "cv04", "cv11"`).
- **Code & Chemistry**: JetBrains Mono or Fira Code for SMILES strings, canonical formulas, PDB/SDF coordinate tables, and tensor shapes.
- **Scientific Units**: Strict typographic convention for standard OPV units:
  - Power Conversion Efficiency: `%`
  - Open-Circuit Voltage: `V` ($V_{OC}$)
  - Short-Circuit Current Density: `mA/cm²` ($J_{SC}$)
  - Fill Factor: `%` ($FF$)
  - Molecular Weight: `g/mol`
  - Topological Polar Surface Area: `Å²`
  - Active Layer Thickness: `nm`

---

## 3. Structural Navigation & Workspace Layout

The application architecture supports six dedicated workspaces accessible via a persistent, collapsible scientific sidebar:

```
+---------------------------------------------------------------------------------------+
|  OPV-PredictNet Header: Model Registry Status [Validated/Pending] | User Session | GPU |
+----+----------------------------------------------------------------------------------+
|    |                                                                                  |
| S  |  1. Dashboard & Key Metrics Overview                                             |
| I  |     - Quick screening summary, recent runs, active model version, sanity guards   |
| D  |                                                                                  |
| E  |  2. Molecular Designer & Structure Viewer                                        |
| B  |     - Interactive 3Dmol.js / Three.js viewport for donor and acceptor structures |
| A  |     - Real-time RDKit 2D depiction and valence error detection                    |
| R  |                                                                                  |
|    |  3. Prediction Studio                                                            |
|    |     - Donor/Acceptor formulation inputs (Monomer Unit / Core Fragment / Molecule)   |
|    |     - Collapsible experimental device parameter controls (Pillar 3)              |
|    |     - Execution pipeline with physical sanity validation                         |
|    |                                                                                  |
|    |  4. Dataset Explorer                                                             |
|    |     - Interactive exploration of curated experimental OPV-DB records             |
|    |     - Chemical space distributions, donor-acceptor frequency matrices            |
|    |                                                                                  |
|    |  5. Model Analytics & Validation Suite                                            |
|    |     - Feature importance rankings (22-feature schema)                            |
|    |     - Parity plots (experimental vs. predicted PCE, Voc, Jsc, FF)                |
|    |     - Residual distributions and domain of applicability boundaries              |
|    |                                                                                  |
|    |  6. Knowledge Assistant & System Settings                                        |
|    |     - Local Python knowledge chatbot for OPV literature and mechanism queries    |
|    |     - Database connection status, cache controls, and export tools               |
+----+----------------------------------------------------------------------------------+
```

---

## 4. Interaction Patterns & Scientific Safeguards

### 4.1 Chemical Input Clarification
To prevent scientific ambiguity between polymer macromolecules and computational representation:
- **Explicit Scoping Labels**: Chemical input fields must explicitly identify whether the input represents a *Monomer Repeat Unit*, a *Conjugated Core Fragment*, or a *Complete Small Molecule*.
- **Live Valence Feedback**: As the user edits SMILES strings, client-side debounce queries the backend RDKit endpoint (`/api/predictions/analyze`) to canonicalize the structure and display computed properties immediately.

### 4.2 Device Parameter Attribution
- **Provenance vs. Feature Vector**: The UI maintains a distinct visual distinction between features currently utilized by the 22-descriptor ML model and device processing parameters (film thickness, solvent additive, annealing) collected for provenance and downstream multi-modal architectures.
- **Scope Banner**: A persistent notice informs the researcher: *"Experimental processing parameters are recorded for provenance but are not currently included in the 22-feature molecular descriptor model."*

### 4.3 Output Sanity & Integrity Guardrails
- **No Unphysical Displays**: Predictions exceeding single-junction thermodynamic bounds (PCE > 34%, Voc > 2.0 V, Jsc > 45 mA/cm², FF > 100%) are rejected explicitly with high-visibility diagnostic banners rather than truncated or silently clipped.
- **Model Validation Pill**: The prediction status badge reflects real artifact health:
  - `Model Pending (No Fabrication)` when no trained artifact exists.
  - `Rejected: Missing Scientific Validation` when an unverified binary is detected.
  - `Validated Model Active` only when verified experimental provenance and performance metrics are present.

---

## 5. Technology Alignment & Phased Rollout

- **Phase 3 / 3.1 Status**: Core RDKit analysis, authentication, preset terminology correction, and physical output sanity checks implemented.
- **Phase 4 (Future)**: Interactive 3Dmol.js / React Three Fiber molecular canvas integration, Recharts parity plots, and full navigation sidebar layout.
- **Constraints**: 100% client-side compatible with Windows 10/11 and 8 GB RAM hardware profile; zero mandatory paid APIs.
