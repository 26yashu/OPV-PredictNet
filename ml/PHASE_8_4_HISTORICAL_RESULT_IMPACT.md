# OPV-PredictNet — Phase 8.4: Historical Result Impact & Governance

**Date:** 2026-11-25  
**Scope:** Impact assessment of the Phase 8.2 `extract_m3` thermal-annealing defect and extraction drift on all historical benchmark results (Phases 5 through 8.2.1).  
**Preservation Policy:** Original result CSV and JSON files in `ml/results/` are **strictly preserved unmodified**.

---

## 1. Defect Summary: What Happened in Phase 8.2 Code

During Phase 8.3.1 inspection of `ml/experiments/phase_8_2_scaffold_benchmark.py`, the following defects were identified relative to the locked Phase 8.0.3 / Phase 8.1 pipeline:

1. **Silent `NameError` in `extract_m3`'s `reg_ann` (Lines 614–617):**
   ```python
   def reg_ann(val):
       if pd.isna(val): return 0
       try: return 1 if 40.0 <= v <= 250.0 else 0
       except: return 0
   ```
   Because `v = float(val)` was omitted, `40.0 <= v <= 250.0` raised `NameError` on every non-null input, silently swallowed by `except: return 0`. Consequently, **`regime_thermal_annealed` evaluated to `0` across 100% of Train, Cal, and Test rows for Model `M3` in Phase 8.2** (note: `annealing_temp_norm`, `missing_annealing`, and `regime_as_cast_room_temp` were unaffected, and `extract_m2a` properly converted `v = float(val)`).
2. **`map_etl_nominal` Substring Collision (`'ca' in s`):**
   Phase 8.2 matched `'ca' in s` rather than `s == 'ca' or 'calcium' in s`, misclassifying **46 distinct ETL strings** (e.g., `ZrAcac`, `Zr(acac)3`, `PDIN:DCA`, `PDIN:TCA`, `TNP CA`) as `'ca'` (calcium) instead of `'other'` in Phase 8.2 `M1`, and missing `PFN`/`PFNDI-Br` as `'pfn_br'`.
3. **Physical Constant Table Drift (`SOLV_BP_MAP`, `ETL_WF_MAP`, `HTL_WF_MAP`):**
   Phase 8.2 modified lookup tables (`'o-xylene': 144.4` $\to$ `'xylene': 138.5`, `'pdinn': 4.00` $\to$ `4.15`, `'pfn-br': 4.20` $\to$ `4.10`, dropped `'2pacz': 5.20`, added `'tio2'`, `'vox'`, `'ni'`), altering `34` solvent, `118` ETL, and `80` HTL unique string mappings in Phase 8.2 `M2A` and `M3`.

---

## 2. Exact Impact Classification Across Historical Results

| Historical Phase & Artifacts | Affected? | Technical Verification |
| :--- | :---: | :--- |
| **Phase 5** (`phase_5_summary.json`, `phase_5_fold_metrics.csv`) | **NO (Unchanged)** | Evaluates 22 RDKit molecular descriptors (`descriptors.py`) only. Zero processing features used. |
| **Phase 6.1** (`phase_6_1_summary.json`, `phase_6_1_nested_fold_metrics.csv`) | **NO (Unchanged)** | Evaluates 22 RDKit molecular descriptors (`descriptors.py`) only. |
| **Phase 7.1** (`phase_7_1_final_test_metrics.json`, `phase_7_1_*.csv/json`) | **NO (Unchanged)** | Evaluates M0 (22 molecular descriptors) conformal calibration only. |
| **Phase 8.1** (`phase_8_1_model_metrics.csv`, `phase_8_1_interval_metrics.csv`, `phase_8_1_paired_comparisons.csv`, `phase_8_1_subgroup_metrics.csv`, `phase_8_1_test_predictions.csv`) | **NO (Unchanged & Clean)** | Inspected `ml/experiments/phase_8_1_processing_aware_models.py`: `reg_ann` properly assigned `v = float(val)` (L460), `map_etl_nominal` used `s == 'ca'` (L134), and lookup dictionaries matched locked Phase 8.0.3 verbatim. |
| **Phase 8.2 `M0`** (`phase_8_2_scaffold_metrics.csv` rows for M0) | **NO (Unchanged)** | Uses 22 molecular descriptors only. |
| **Phase 8.2 `M1`, `M2A`, `M3`** (`phase_8_2_scaffold_metrics.csv`, `phase_8_2_interval_metrics.csv`, `phase_8_2_paired_comparisons.csv`, `phase_8_2_subgroup_metrics.csv`) | **YES (Potentially Affected)** | `M1` used drifted `map_etl_nominal`/`map_htl_nominal`; `M2A` used drifted `SOLV_BP_MAP`/`ETL_WF_MAP`/`HTL_WF_MAP`; **`M3` ran with `regime_thermal_annealed = 0` on all rows AND drifted lookup tables.** |

---

## 3. Governance Rules for Historical Reporting

1. **No Silent Rewriting of Phase 8.2 Results:** Original `ml/results/phase_8_2_*.csv` and `phase_8_2_config.json` files remain permanently frozen as the exact numerical record of what `phase_8_2_scaffold_benchmark.py` executed.
2. **Prohibition on Presenting Old Metrics as Corrected-Pipeline Metrics:** Phase 8.2 `M1`, `M2A`, and `M3` metrics **must not** be presented as performance metrics of the corrected `UnifiedFeaturePipeline`. Whenever cited, Phase 8.2 `M3` metrics must note that `regime_thermal_annealed` was zeroed due to the historical extraction defect.
3. **Why Phase 8.2 Qualitative Conclusions Remain Valid Boundaries (Until Rerun):** The overarching structural failure demonstrated in Phase 8.2—that `M0` collapses to $R^2=0.0786$ under dual-scaffold holdout (`Strategy C1`), that conformal exchangeability fails across scaffold splits, and that high-PCE acceptors suffer severe underprediction (`Strategy B`)—does not depend on `M3` `regime_thermal_annealed` alone (`M0` and `M2A` both show the same structural collapse). Nevertheless, **zero claims of improved `M3` scaffold accuracy may be made from the unified pipeline until a controlled rerun is authorized.**

---

## 4. Required Evaluations Before Any New `M3` / `M2A` / `M1` Scaffold Claim

Before claiming updated scaffold generalization performance for the corrected `UnifiedFeaturePipeline`:
1. Execute a separately authorized **Phase 8.4.1 / Phase 9 Verification Rerun** running `UnifiedFeaturePipeline` across the exact locked Phase 8.2 partition seeds (`7701`, `7702`, `7703`, `8888`) and scaffold cache.
2. Compare regenerated `M1`, `M2A`, and `M3` metrics directly against `M0` under identical one-way cluster-bootstrap and balanced per-scaffold checks.
3. Archive new results under distinct versioned filenames (never overwriting `phase_8_2_*.csv`).
