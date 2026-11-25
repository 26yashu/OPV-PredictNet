# OPV-PredictNet — Academic Report Figure Index

**Project:** OPV-PredictNet — AI-Driven Molecular Screening and Performance Optimization for Organic Photovoltaic Solar Cells  
**Asset Repository:** `docs/report_assets/`  
**Generated Date:** November 2026  
**Scope:** Essential Academic Report Asset Collection (8 Application Screenshots, 5 System/Scientific Diagrams, 4 Research Evaluation Charts)

---

## 1. Summary of Asset Catalog

| Asset Category | Target Count | Actual Count | Formats Available | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Application Screenshots** | 8 | 8 | PNG (1920×1080) | Verified Genuine |
| **Essential Diagrams** | 5 | 5 | PNG (300 DPI) + SVG | High-Res Vector/Raster |
| **Research Charts** | 4 | 4 | PNG (300 DPI) + SVG | Verified Phase 8.5 Outputs |
| **Total Core Assets** | **17** | **17** | **PNG + SVG** | **Complete** |

---

## 2. Priority 1 — Application Screenshots (Genuine UI Captures)

All screenshots are genuine captures taken from the running OPV-PredictNet application stack (FastAPI backend + Vite/React frontend connected to an SQLite database with real model inference and RDKit descriptors).

| Figure ID | Filename | Title | Target Report Chapter | Asset Type | Description & Academic Caption |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 4.1** | `screenshots/01_landing_page.png` | Landing Page Hero & Value Proposition | Chapter 4: System Implementation & UI | Running Screenshot | **Hero section of OPV-PredictNet landing page.** Displays the core scientific proposition: AI-driven molecular screening and PCE prediction with conformal uncertainty quantification, navigation to the prediction workspace, and research-grade notice. |
| **Figure 4.2** | `screenshots/02_login_page.png` | User Authentication & Secure Access | Chapter 4: System Implementation & Security | Running Screenshot | **User authentication portal.** Highlights JWT-based access control, secure credential handling, and modal transitions protecting user history and proprietary molecular candidate records. |
| **Figure 4.3** | `screenshots/03_prediction_dashboard.png` | Main Prediction Dashboard Workspace | Chapter 4: User Interface & Workflow | Running Screenshot | **Overview of the primary prediction dashboard.** Displays the two-column scientific workspace with donor/acceptor SMILES inputs, preset loading, processing parameters, and model selector (M0, M1, M2A, M3). |
| **Figure 4.4** | `screenshots/04_molecular_input.png` | Molecular Structure Input Panel | Chapter 4: Input Validation & UI | Running Screenshot | **Donor and Acceptor SMILES input interface.** Illustrates interactive molecular entry with pre-populated benchmark systems (PM6 and Y6), showing real-time SMILES validation and structural feedback. |
| **Figure 4.5** | `screenshots/05_rdkit_analysis.png` | Real-Time RDKit Chemical Analysis | Chapter 4: Descriptor Extraction Engine | Running Screenshot | **RDKit-driven molecular property inspection.** Displays real-time computed physicochemical properties (Molecular Weight, LogP, TPSA, HBD, HBA, aromatic rings) and active structure visualization for entered donor and acceptor pairs. |
| **Figure 4.6** | `screenshots/06_prediction_result.png` | PCE Prediction & Uncertainty Intervals | Chapter 5: Prediction Results & Evaluation | Running Screenshot | **Model prediction display.** Shows predicted Power Conversion Efficiency (PCE = 16.20%), split conformal prediction intervals (80%, 90%, 95%), feature contribution radar/breakdown, and clear `research_estimate_only=true` badges. |
| **Figure 4.7** | `screenshots/07_ood_warning_rejection.png` | Out-of-Distribution (OOD) Warning & Scope Validation | Chapter 4: Risk Mitigation & Governance | Running Screenshot | **Rigorous OOD detection interface.** Displays scientific domain alerts when candidates exceed molecular weight bounds or fall into unsupported donor/acceptor scaffold clusters, preventing ungrounded extrapolations. |
| **Figure 4.8** | `screenshots/08_prediction_history.png` | Prediction History & Audit Trail | Chapter 4: Data Management & Auditing | Running Screenshot | **Auditable prediction history table.** Demonstrates persisted screening records stored in SQLite, showing timestamp, model version, donor/acceptor identifiers, predicted PCE, confidence intervals, and OOD status flags. |

---

## 3. Priority 2 — Essential Scientific & Architecture Diagrams

Diagrams provide formal representations of the physical OPV mechanism, multi-layer device architecture, end-to-end software stack, ML training/calibration pipeline, and runtime inference lifecycle.

| Figure ID | Filename | Title | Target Report Chapter | Asset Type | Description & Academic Caption |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 2.1** | `scientific_diagrams/01_opv_working_principle.png` (and `.svg`) | Working Principle of Organic Photovoltaic Cells | Chapter 2: Theoretical Background | Scientific Diagram (SVG/PNG) | **Four-stage photovoltaic conversion mechanism.** Illustrates: (1) Photon absorption and exciton generation in donor, (2) Exciton diffusion toward the donor-acceptor interface, (3) Charge separation driven by LUMO/HOMO offset, and (4) Charge collection at ITO anode and metal cathode. |
| **Figure 2.2** | `scientific_diagrams/02_opv_device_architecture.png` (and `.svg`) | Standard Bulk Heterojunction (BHJ) Architecture | Chapter 2: Device Physics & Fabrication | Scientific Diagram (SVG/PNG) | **Layered cross-section of a BHJ solar cell.** Visualizes Glass substrate, transparent ITO anode, PEDOT:PSS hole transport layer (HTL), interpenetrating Donor:Acceptor active layer, PDINN/PFN electron transport layer (ETL), and Silver/Aluminum top cathode. |
| **Figure 3.1** | `architecture_diagrams/01_system_architecture.png` (and `.svg`) | OPV-PredictNet Full System Architecture | Chapter 3: Software Architecture | Architecture Diagram (SVG/PNG) | **Multi-tier software architecture diagram.** Depicts the React + TypeScript frontend, FastAPI backend application tier, RDKit physicochemical descriptor engine, scikit-learn ML inference engine, and SQLite database storage. |
| **Figure 3.2** | `architecture_diagrams/02_ml_workflow.png` (and `.svg`) | Unified Machine Learning Training & Inference Workflow | Chapter 3: Methodology & ML Pipeline | Pipeline Diagram (SVG/PNG) | **End-to-end ML lifecycle.** Details Phase 8.4 unified feature extraction, multi-level feature tiering (M0, M1, M2A, M3), cross-validation, hyperparameter tuning, non-conformity calibration, and model artifact serialization. |
| **Figure 3.3** | `architecture_diagrams/03_end_to_end_workflow.png` (and `.svg`) | End-to-End Application Workflow | Chapter 3: Operational Workflow | Flow Diagram (SVG/PNG) | **User-to-system sequential workflow.** Maps user SMILES submission, real-time RDKit property verification, model selection, ML inference, conformal interval derivation, OOD guardrails, and persistent database logging. |

---

## 4. Priority 3 — Essential Research Charts (Verified Phase 8.5 Outputs)

Charts are generated strictly from the authoritative Phase 8.5 experimental results (`ml/results/phase_8_5/`), without interpolation or synthetic values.

| Figure ID | Filename | Title | Target Report Chapter | Asset Type | Description & Academic Caption |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 5.1** | `research_charts/01_model_comparison_mae.png` (and `.svg`) | MAE Comparison Across M0, M1, M2A, and M3 | Chapter 5: Experimental Evaluation | Research Chart (SVG/PNG) | **Mean Absolute Error (MAE) performance comparison across model tiers.** Demonstrates performance improvement from M0 baseline (electronic descriptors only) to M3 (processing-aware descriptors), showing holdout evaluation across scaffold generalization strategies. |
| **Figure 5.2** | `research_charts/02_model_comparison_rmse.png` (and `.svg`) | RMSE Comparison Across M0, M1, M2A, and M3 | Chapter 5: Experimental Evaluation | Research Chart (SVG/PNG) | **Root Mean Squared Error (RMSE) performance comparison across model tiers.** Shows penalization of large residual errors across models, verifying M3's lower variance and enhanced robustness when processing conditions are accounted for. |
| **Figure 5.3** | `research_charts/03_predicted_vs_actual_pce.png` (and `.svg`) | Predicted vs. Actual PCE for M3 (Full Holdout Set) | Chapter 5: Experimental Evaluation | Research Chart (SVG/PNG) | **Parity scatter plot for Model M3.** Evaluates $N=1,330$ experimental observations. Features identity line ($y=x$), $\pm 10\%$ error margins, kernel density marginals, and statistical regression metrics ($R^2$, RMSE, MAE). |
| **Figure 5.4** | `research_charts/04_conformal_coverage.png` (and `.svg`) | Conformal Prediction Empirical Coverage & Efficiency | Chapter 5: Uncertainty Quantification | Research Chart (SVG/PNG) | **Split conformal prediction calibration verification.** Shows empirical vs. nominal coverage rates for 80%, 90%, and 95% confidence levels across M0, M1, M2A, and M3 on Strategy A (Donor Scaffold Holdout), alongside mean prediction interval widths. |

---

## 5. File System Location Reference

```
docs/report_assets/
├── figure_index.md
├── SCREENSHOT_AND_FIGURE_REPORT.md
├── screenshots/
│   ├── 01_landing_page.png
│   ├── 02_login_page.png
│   ├── 03_prediction_dashboard.png
│   ├── 04_molecular_input.png
│   ├── 05_rdkit_analysis.png
│   ├── 06_prediction_result.png
│   ├── 07_ood_warning_rejection.png
│   └── 08_prediction_history.png
├── scientific_diagrams/
│   ├── 01_opv_working_principle.png
│   ├── 01_opv_working_principle.svg
│   ├── 02_opv_device_architecture.png
│   └── 02_opv_device_architecture.svg
├── architecture_diagrams/
│   ├── 01_system_architecture.png
│   ├── 01_system_architecture.svg
│   ├── 02_ml_workflow.png
│   ├── 02_ml_workflow.svg
│   ├── 03_end_to_end_workflow.png
│   └── 03_end_to_end_workflow.svg
└── research_charts/
    ├── 01_model_comparison_mae.png
    ├── 01_model_comparison_mae.svg
    ├── 02_model_comparison_rmse.png
    ├── 02_model_comparison_rmse.svg
    ├── 03_predicted_vs_actual_pce.png
    ├── 03_predicted_vs_actual_pce.svg
    ├── 04_conformal_coverage.png
    └── 04_conformal_coverage.svg
```
