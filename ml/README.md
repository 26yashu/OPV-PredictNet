# OPV-PredictNet — Machine Learning Subsystem

## Overview

The machine learning subsystem in OPV-PredictNet is responsible for translating molecular structures (donors and acceptors) and optional processing parameters into numerical representations, and generating statistical predictions for organic photovoltaic (OPV) device performance metrics.

## Scientific Principles

1. **Molecular Screening vs. Device Prediction**:
   - An isolated molecule possesses intrinsic electronic characteristics (HOMO, LUMO, optical bandgap, molecular weight, dipole moment).
   - In organic solar cells, light absorption and charge separation occur in an active layer comprising a **donor–acceptor blend** (bulk heterojunction).
   - Real-world device efficiency (Power Conversion Efficiency / PCE) is an emergent property of the entire fabricated device stack (ITO / PEDOT:PSS / Active Layer / Interfacial Layers / Metal Electrode) and processing history (solvents, additives like DIO, thermal annealing, film morphology).
   - Machine learning models in OPV-PredictNet predict expected device-level figures of merit under standard test conditions (AM1.5G, 100 mW/cm²) based on experimental training distributions, rather than claiming an isolated molecule determines efficiency in isolation.

2. **Phase 1 Status**:
   - Directory structure and feature pipeline interfaces defined.
   - **No model training has been conducted in Phase 1.**
   - **No synthetic training data or fabricated performance metrics are present.**

## Directory Layout

- `data/raw/`: Original literature datasets (e.g., HOPV, Harvard Clean Energy Project, peer-reviewed solar cell tables).
- `data/processed/`: Cleaned, canonicalized, deduplicated tabular datasets with computed features.
- `models/saved_models/`: Serialized Joblib model artifacts (`.joblib`) accompanied by JSON metadata.
- `notebooks/`: Exploratory data analysis, feature correlation, and model validation notebooks.
- `src/features/`: RDKit-based molecular descriptor calculation and blend feature composition.
- `src/models/`: Random Forest Regressor wrappers and pipeline definitions.
- `src/evaluation/`: Cross-validation (scaffold splitting), MAE, RMSE, and R² evaluation metrics.
- `src/utils/`: Serialization and data cleaning helpers.
