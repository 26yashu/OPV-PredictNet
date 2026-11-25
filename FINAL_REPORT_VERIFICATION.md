# OPV-PredictNet — Final Academic Project Report Verification

**Document Version:** 1.0.0  
**Generated Date:** November 25, 2026  
**Target Submission:** Bachelor of Technology in Computer Science & Engineering (AIML), VII Semester  
**Institution:** JAIN (Deemed-to-be University), Faculty of Engineering & Technology, Bengaluru  
**Generated Artifacts:**
- Microsoft Word Document: [`OPV_PredictNet_Final_Report.docx`](file:///c:/Users/DELL/OPV-PredictNet/OPV_PredictNet_Final_Report.docx) (5.82 MB)
- Adobe Portable Document Format: [`OPV_PredictNet_Final_Report.pdf`](file:///c:/Users/DELL/OPV-PredictNet/OPV_PredictNet_Final_Report.pdf) (1.76 MB)

---

## 1. Sections Completed

The final report preserves all mandatory institutional sections from `Report_Template[1].docx` and includes complete technical coverage of all 12 project phases:

- [x] **Front Matter (Jain University Specifications):**
  - **Title Page:** Department of Computer Science & Engineering (AIML), Jain Global Campus, Ramanagara; full project title; academic year 2025–2026 (VII Semester); student authors and guide.
  - **Certificate Page:** Formal departmental certificate with faculty guide, coordinator, head of department, and blank examiner fill lines (no forged or simulated signatures).
  - **Declaration Page:** Authentic student declaration with candidate names and USN codes.
  - **Acknowledgement Page:** Comprehensive academic acknowledgement of faculty guide Dr. Kushal Chakraborthy, department leadership, open-source communities (RDKit, OPV-DB, FastAPI), and families.
  - **Abstract:** Detailed 3-paragraph executive summary covering the OPV chemical design challenge, 12-phase engineering methodology, feature tiers (M0–M3), scaffold generalization, conformal coverage guarantees, and governance disclaimers.
  - **Table of Contents:** Complete pagination mapping covering all front matter, 6 main chapters, 28 subsections, references, and 4 appendices.
  - **List of Figures:** Itemized table of all 17 embedded figures with numbers, titles, and page references.
  - **List of Tables:** Itemized table of all 7 embedded tables with numbers, titles, and page references.

- [x] **Core Academic Chapters (All 12 Phases Covered):**
  - **Chapter 1: Introduction:** Global renewable energy context, OPV physical advantages, 4-stage exciton conversion mechanism, bulk heterojunction (BHJ) device cross-section, problem statement, objectives, and scope.
  - **Chapter 2: Literature Survey and Theoretical Methodology:** Critical review of high-throughput DFT (Harvard CEP), literature databases (HOPV15, OPV-DB), identification of research gaps, end-to-end ML workflow, dataset acquisition and provenance, RDKit descriptor formulation, multi-tier feature engineering (M0: 22f, M1: 50f, M2A: 37f, M3: 44f), and split conformal prediction theory.
  - **Chapter 3: System Design and Architecture:** High-level specifications, 4-tier microservice architecture (React Client, FastAPI Gateway, RDKit/ML Engine, SQLite Persistence), operational screening workflow, REST API contract, and relational schema.
  - **Chapter 4: System Implementation and User Interface:** Development phase timeline (Phases 1–12), user authentication, cinematic landing page, two-column prediction dashboard, interactive SMILES input with PM6:Y6 benchmarks, real-time RDKit property inspection, inference execution with conformal bounds, out-of-distribution (OOD) guardrails, and persistent history auditing.
  - **Chapter 5: Experimental Results and Discussion:** Experimental protocol, comparative MAE/RMSE across M0–M3, parity scatter plot for M3, split conformal empirical coverage and efficiency, scaffold generalization benchmark across Strategies A, B, C1, and C2, feature attribution decomposition, and honest scientific limitations.
  - **Chapter 6: System Verification, Deployment and Conclusion:** Phase 11 full-system verification (100% pass across 92 tests), Phase 12 deployment readiness and containerization (Docker, production CORS, secret hygiene), summary of engineering contributions, and future research directions.

- [x] **Back Matter:**
  - **References:** 21 formally formatted, numbered academic citations spanning photovoltaic physics, literature databases, conformal prediction, cheminformatics, and software frameworks.
  - **Appendix A:** Complete Feature Dictionary for Models M0, M1, M2A, and M3.
  - **Appendix B:** REST API Contract and JSON Request/Response Schemas.
  - **Appendix C:** Complete Phase 8.5 Numerical Benchmark Evaluation Table.
  - **Appendix D:** Student and Faculty Institutional Details.

---

## 2. Figures Inserted & Verified

All 17 genuine visual assets from `docs/report_assets/` were embedded at high resolution without stretching or distortion:

| Figure ID | Asset Filename | Visual Content & Provenance | Placement |
| :--- | :--- | :--- | :--- |
| **Figure 1.1** | `scientific_diagrams/01_opv_working_principle.png` | 4-step exciton photophysical conversion mechanism | Section 1.2 |
| **Figure 1.2** | `scientific_diagrams/02_opv_device_architecture.png` | Bulk heterojunction (BHJ) layered device cross-section | Section 1.3 |
| **Figure 2.1** | `architecture_diagrams/02_ml_workflow.png` | End-to-end ML training, calibration, and validation lifecycle | Section 2.3 |
| **Figure 3.1** | `architecture_diagrams/01_system_architecture.png` | Multi-tier software architecture (Client, API, ML, SQLite) | Section 3.2 |
| **Figure 3.2** | `architecture_diagrams/03_end_to_end_workflow.png` | Operational workflow from user SMILES to conformal interval | Section 3.3 |
| **Figure 4.1** | `screenshots/02_login_page.png` | Running UI screenshot: User authentication & JWT portal | Section 4.2 |
| **Figure 4.2** | `screenshots/01_landing_page.png` | Running UI screenshot: Cinematic landing page & hero section | Section 4.3 |
| **Figure 4.3** | `screenshots/03_prediction_dashboard.png` | Running UI screenshot: Primary prediction dashboard | Section 4.3 |
| **Figure 4.4** | `screenshots/04_molecular_input.png` | Running UI screenshot: Donor & acceptor SMILES input panel | Section 4.4 |
| **Figure 4.5** | `screenshots/05_rdkit_analysis.png` | Running UI screenshot: Real-time RDKit chemical property view | Section 4.4 |
| **Figure 4.6** | `screenshots/06_prediction_result.png` | Running UI screenshot: Point estimate & conformal bounds (16.20%) | Section 4.5 |
| **Figure 4.7** | `screenshots/07_ood_warning_rejection.png` | Running UI screenshot: Out-of-distribution (OOD) domain warning | Section 4.6 |
| **Figure 4.8** | `screenshots/08_prediction_history.png` | Running UI screenshot: SQLite-backed screening audit history | Section 4.7 |
| **Figure 5.1** | `research_charts/01_model_comparison_mae.png` | Verified chart: MAE comparison across M0–M3 and 4 strategies | Section 5.2 |
| **Figure 5.2** | `research_charts/02_model_comparison_rmse.png` | Verified chart: RMSE comparison across M0–M3 and 4 strategies | Section 5.2 |
| **Figure 5.3** | `research_charts/03_predicted_vs_actual_pce.png` | Verified chart: Predicted vs. Actual PCE for M3 (N=2,067) | Section 5.3 |
| **Figure 5.4** | `research_charts/04_conformal_coverage.png` | Verified chart: Conformal empirical coverage and interval sharpness | Section 5.4 |

---

## 3. References Checked & DOI Discrepancy Resolved

- **DOI Discrepancy Resolution:**
  - *Earlier Citing:* Some preliminary documents cited `10.5281/zenodo.4704870`, while others cited `10.5281/zenodo.22177960`.
  - *Verified Origin:*
    - `10.5281/zenodo.4704870` is the original Zenodo deposit for the initial 2021 release of OPV-DB by Jiangjie Qiu et al.
    - `10.5281/zenodo.22177960` is the immutable Zenodo Version DOI for the frozen OPV-DB v1.1.0 release snapshot (2026 Q3 release, archive `opvdb_2026_q3.zip`, 6.62 MB), registered under the Zenodo Concept DOI `10.5281/zenodo.20841542`.
  - *Report Alignment:* Both records are explicitly cited and explained in Section 2.4 and in References [1] and [2], crediting Jiangjie Qiu and Xuan Yang (Tsinghua University) under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## 4. Numerical Claims & Benchmark Metrics Verified

All numerical values in the report correspond strictly to verified Phase 8.5 experimental result tables (`ml/results/phase_8_5/`):

- **Dataset Counts:**
  - Full OPV-DB archive: 39,998 records (8,073 literature DOIs).
  - Strict Performance Benchmark: 32,457 records (7,379 literature DOIs).
  - Strict Molecular Benchmark (Training Base): 22,467 records (5,615 literature DOIs).
- **Evaluation Split Sample Counts:**
  - Strategy A (Donor Scaffold Holdout): Train = 15,750, Cal = 5,236, Test = 1,481.
  - Strategy B (Acceptor Scaffold Holdout): Train = 17,658, Cal = 1,012, Test = 3,797.
  - Strategy C1 (Strict Dual Holdout): Train = 13,324, Cal = 260, Test = 284.
  - Strategy C2 (Scaffold-Pair Holdout): Train = 16,537, Cal = 3,863, Test = 2,067.
- **Verification of N=1,330 vs. N=2,067:**
  - An earlier draft script had hardcoded `N=1,330` in the title string of the predicted-versus-actual plot, even though the internal statistical legend and actual Strategy C2 test partition evaluated $N=2,067$ samples.
  - The parity plot (`03_predicted_vs_actual_pce.png`) was regenerated to accurately state $N=2,067$, matching the verified metrics: MAE = 2.16% PCE, RMSE = 2.86% PCE, $R^2 = 0.664$.
- **Conformal Coverage Verification:**
  - Evaluated on Strategy A for Model M3:
    - 80% Nominal Target $\rightarrow$ 86.29% Empirical Coverage (Cutoff $q=3.73\%$, mean width $7.47\%$).
    - 90% Nominal Target $\rightarrow$ 92.37% Empirical Coverage (Cutoff $q=4.71\%$, mean width $9.43\%$).
    - 95% Nominal Target $\rightarrow$ 95.68% Empirical Coverage (Cutoff $q=5.83\%$, mean width $11.65\%$).
  - In all instances, empirical coverage satisfies the theoretical condition $P(y \in C) \ge 1 - \alpha$.
- **Scientific Governance:**
  - Point predictions and intervals are explicitly identified as computational research estimates (`research_estimate_only = True`), not certified physical measurements.

---

## 5. Missing Information Requiring Student Confirmation Before Printing

1. **Date of Submission:** The report displays the current date (`25-11-2026`). Students should update this to the exact date of final print submission.
2. **University Examiner Names:** In accordance with instructions, blank lines are provided for external and university examiners (`Name: ____________`, `Signature: ____________`). Do not fill these in prior to official examination.
3. **Institutional Email Addresses:** Optional contact emails for student authors Gandlaparthi Yaswanthi and Gooty Kumara Snigdha can be appended in Appendix D if requested by the department.

---

## 6. Template Requirements Compliance

- **Layout & Margins:** Standard A4 page geometry with 1.0-inch (72 pt) margins on all sides.
- **Typography:** Consistent Times New Roman body text (11 pt, 1.15 line spacing, 6 pt paragraph spacing) and bold numbered headings (Heading 1: 16 pt, Heading 2: 13 pt, Heading 3: 11.5 pt).
- **Front Matter Integrity:** The Jain University header, faculty title, degree nomenclature (B.Tech in CSE - AIML), guide designation, certificate layout, declaration, and acknowledgement have been preserved without distortion.
- **Dual Export:** Generated in native Microsoft Word `.docx` format (`OPV_PredictNet_Final_Report.docx`) and compiled to PDF (`OPV_PredictNet_Final_Report.pdf`) via Word 16.0 COM automation.
