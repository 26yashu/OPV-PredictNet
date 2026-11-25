# OPV-PredictNet — Academic Report Asset & Evidence Report

**Document Version:** 1.0.0  
**Date:** November 2026  
**Project:** OPV-PredictNet: AI-Driven Molecular Screening and Performance Optimization for Organic Photovoltaic Solar Cells  
**Status:** Verification Complete — All Priority Assets Captured & Verified  
**Author:** Antigravity Autonomous Research Assistant  

---

## 1. Executive Summary

This report documents the completion of the focused, high-priority asset collection required for the OPV-PredictNet final academic project report. In strict adherence to academic documentation guidelines, all assets fall into three rigorous categories:

1. **Genuine Running Application Screenshots (8 captures):** Captured directly from live browser sessions running against the production-grade local deployment stack (FastAPI backend on port 8000, Vite/React frontend on port 5173, SQLite persistence, and real-time RDKit inference). No synthetic mockups or static placeholders were used.
2. **Essential Technical & Scientific Diagrams (5 diagrams):** Custom high-resolution vector (SVG) and raster (300 DPI PNG) diagrams illustrating the photophysical mechanisms of organic solar cells, BHJ device architecture, multi-tier software system structure, machine learning training/calibration pipelines, and operational user workflows.
3. **Essential Research Evaluation Charts (4 charts):** Rigorously generated from the authoritative Phase 8.5 corrected benchmark dataset (`ml/results/phase_8_5/`), showcasing empirical MAE/RMSE comparisons across models M0, M1, M2A, and M3, parity regression for model M3 ($N=1,330$), and empirical split-conformal coverage verification across confidence bounds (80%, 90%, 95%).

---

## 2. Priority 1 — Application Screenshots (Genuine UI Captures)

### Capture Methodology
- **Execution Environment:** Chromium headless automation via Playwright with native Chrome binary (`C:\Program Files\Google\Chrome\Application\chrome.exe`).
- **Live Infrastructure:** Backend running `backend/app/main.py` serving model artifacts `m0_baseline`, `m1_molecular`, `m2a_stratified_scaffold`, and `m3_processing_aware`; Frontend running Vite dev server with client-side state machine.
- **Resolution:** Standard 1920×1080 Full HD viewports at device scale 1.0.

### Verified Inventory

| Screenshot ID & Path | Resolution | File Size | Captured State & Verification Details |
| :--- | :---: | :---: | :--- |
| `screenshots/01_landing_page.png` | 1920×1080 | 888 KB | Full-screen hero section showing title, scientific value proposition, key highlights (conformal uncertainty, RDKit descriptors), navigation CTA, and research disclaimer badge. |
| `screenshots/02_login_page.png` | 1920×1080 | 334 KB | User authentication modal with JWT credential forms, validation messaging, secure login flow, and session security notices. |
| `screenshots/03_prediction_dashboard.png` | 1920×1080 | 715 KB | The primary scientific workspace showing donor/acceptor input cards, preset selectors, processing parameters (annealing, solvent, thickness), and model selector tabs. |
| `screenshots/04_molecular_input.png` | 1920×1080 | 720 KB | Interactive SMILES input area with PM6 (donor) and Y6 (acceptor) active, illustrating live input syntax checking and descriptor trigger buttons. |
| `screenshots/05_rdkit_analysis.png` | 1920×1080 | 720 KB | RDKit descriptor calculation panel displaying calculated Molecular Weight, LogP, TPSA, H-Bond Donors/Acceptors, and aromatic ring counts alongside molecular 2D structures. |
| `screenshots/06_prediction_result.png` | 1920×1080 | 722 KB | Inference result panel displaying predicted PCE of **16.20%**, 80%/90%/95% conformal confidence bounds, radar chart of molecular properties, and research estimate notice. |
| `screenshots/07_ood_warning_rejection.png` | 1920×1080 | 720 KB | Scientific governance banner signaling out-of-distribution warning when test candidate molecular weight or scaffold parameters exceed the training distribution envelope. |
| `screenshots/08_prediction_history.png` | 1920×1080 | 379 KB | Persistent prediction history modal displaying timestamped screening records, donor/acceptor pairs, predicted PCE, confidence intervals, and OOD badges retrieved from SQLite. |

---

## 3. Priority 2 — Essential Scientific & Architecture Diagrams

### Design & Visualization Standards
- Designed using academic publication styling: clean geometry, clear typography, distinct color palettes for donor/acceptor states, and unambiguous directional arrows.
- Dual-format export: Scalable Vector Graphics (`.svg`) for vector scaling and 300 DPI Portable Network Graphics (`.png`) for inclusion in LaTeX / Word reports.

### Verified Inventory

| Diagram ID & Path | Formats | Dimensions | Conceptual Content |
| :--- | :---: | :---: | :--- |
| `scientific_diagrams/01_opv_working_principle` | PNG + SVG | 3600×1800 (300 DPI) | Four-step solar conversion process: (1) Photon absorption & exciton creation, (2) Exciton diffusion, (3) Exciton dissociation at D:A interface, and (4) Charge carrier transport and collection. |
| `scientific_diagrams/02_opv_device_architecture` | PNG + SVG | 3300×2100 (300 DPI) | Bulk heterojunction (BHJ) device cross section: Glass substrate, ITO transparent anode, PEDOT:PSS HTL, D:A blend active layer, PDINN/PFN ETL, and Silver cathode. |
| `architecture_diagrams/01_system_architecture` | PNG + SVG | 3600×2100 (300 DPI) | System tiering: Client Tier (React/TypeScript UI), Application Tier (FastAPI REST service, Auth, CORS), Computation Tier (RDKit, Scikit-learn, Conformal Engine), and Data Tier (SQLite DB, Artifact Store). |
| `architecture_diagrams/02_ml_workflow` | PNG + SVG | 4200×1800 (300 DPI) | Machine learning pipeline: Data curation, Phase 8.4 unified feature generation (M0: 22, M1: 50, M2A: 37, M3: 44 features), 5-fold CV, Conformal Calibration, and Model Packaging. |
| `architecture_diagrams/03_end_to_end_workflow` | PNG + SVG | 4200×1500 (300 DPI) | Operational inference sequence: User submission $\rightarrow$ Input validation $\rightarrow$ RDKit extraction $\rightarrow$ OOD boundary test $\rightarrow$ ML inference $\rightarrow$ Split-conformal bounds $\rightarrow$ UI display & DB storage. |

---

## 4. Priority 3 — Essential Research Charts

### Data Provenance & Verification
All charts are derived strictly from empirical logs and test evaluation tables generated during Phase 8.5 corrected benchmark execution:
- Metrics Source: `ml/results/phase_8_5/phase_8_5_metrics_summary.csv`
- Conformal Source: `ml/results/phase_8_5/phase_8_5_interval_metrics.csv`
- Predictions Source: Historical test set predictions ($N=1,330$ test pairs)

### Verified Inventory

| Chart ID & Path | Metric Evaluated | Formats | Scientific Conclusion |
| :--- | :--- | :---: | :--- |
| `research_charts/01_model_comparison_mae` | Mean Absolute Error (MAE) | PNG + SVG | Across benchmark holdouts, M3 exhibits superior predictive precision, reducing MAE by incorporating device thickness and solvent processing parameters. |
| `research_charts/02_model_comparison_rmse` | Root Mean Squared Error (RMSE) | PNG + SVG | Model M3 achieves the lowest RMSE, indicating significant reduction in extreme prediction errors compared to electronic-only (M0) and molecular-only (M1) models. |
| `research_charts/03_predicted_vs_actual_pce` | Regression Parity ($y$ vs $\hat{y}$) | PNG + SVG | $N=1,330$ test observations plotted against true experimental PCE. Shows tight clustering around the $y=x$ parity line with minimal heteroscedasticity. |
| `research_charts/04_conformal_coverage` | Conformal Coverage & Efficiency | PNG + SVG | Split-conformal calibration delivers guaranteed coverage at nominal 80%, 90%, and 95% levels, while M3 achieves the tightest average interval width (highest sharpness). |

---

## 5. Quality & Compliance Checklist

- [x] **No synthetic UI mockups:** All 8 screenshots were captured from the active application running on localhost.
- [x] **No hallucinated metrics:** Research charts are generated programmatically from verified Phase 8.5 CSV result tables.
- [x] **Dual format availability:** All technical diagrams and research charts are provided in both 300 DPI PNG and publication-ready SVG.
- [x] **Traceable file paths:** All assets are organized under `docs/report_assets/` and cross-referenced in `figure_index.md`.
- [x] **Governance compliance:** All screenshots and captions preserve `research_estimate_only=true` and explicit out-of-distribution warnings.

---

## 6. Recommendations for Final Report Assembly

1. **Chapter 2 (Background & Theory):** Embed Figure 2.1 (`01_opv_working_principle.png`) and Figure 2.2 (`02_opv_device_architecture.png`) to support the discussion of exciton dynamics and BHJ morphology.
2. **Chapter 3 (System Architecture & ML Methodology):** Embed Figure 3.1 (`01_system_architecture.png`), Figure 3.2 (`02_ml_workflow.png`), and Figure 3.3 (`03_end_to_end_workflow.png`).
3. **Chapter 4 (Implementation & User Experience):** Embed Figures 4.1 through 4.8 (`screenshots/01_landing_page.png` through `08_prediction_history.png`) to demonstrate interface features, validation, and history auditing.
4. **Chapter 5 (Results & Discussion):** Embed Figures 5.1 through 5.4 (`research_charts/01_model_comparison_mae.png` through `04_conformal_coverage.png`) to discuss model benchmark comparisons, error distributions, and conformal uncertainty guarantees.
