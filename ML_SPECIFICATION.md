# OPV-PredictNet — Machine Learning Technical Specification

**Algorithm**: Random Forest Regression (`sklearn.ensemble.RandomForestRegressor`)  
**Feature Generation**: RDKit (2D physicochemical descriptors + Morgan circular fingerprints)  
**Target Metrics**: PCE (Power Conversion Efficiency), Voc, Jsc, FF  
**Status**: Phase 1 Specification (No training executed; zero synthetic data fabricated)  

---

## 1. Rationale for Random Forest as Primary Baseline

In organic solar cell informatics, dataset sizes are constrained by physical laboratory synthesis and device fabrication costs (typically ranging from a few hundred to several thousand published records).

Random Forest Regression is selected as the initial primary architecture due to:
1. **Low Sample Complexity & Overfitting Resistance**: Ensembling decorrelated decision trees reduces model variance, avoiding the catastrophic overfitting often observed with deep neural networks on small chemical datasets ($N < 5000$).
2. **Tabular Descriptor Compatibility**: Handles heterogeneous molecular descriptors (continuous physicochemical descriptors, discrete ring counts, binary/count Morgan fingerprint bits) without requiring extensive feature normalization.
3. **Execution Efficiency on Constrained Hardware**: Training and hyperparameter tuning execute in minutes on an 8 GB RAM CPU, and single-pair inference requires <50 ms without GPU acceleration.
4. **Prediction Variance as Uncertainty Proxy**: Individual tree predictions provide an empirical distribution whose variance serves as an intrinsic uncertainty estimate.

---

## 2. Molecular Feature Engineering Pipeline

The input consists of two distinct molecules: a **Donor** ($D$) and an **Acceptor** ($A$). Features are computed using RDKit:

```
[ Donor SMILES ]   ---> [ RDKit Descriptors ] ---> Vector D (dim: 128)
                                                        \
                                                         +---> [ Combiner & Offset Pipeline ] ---> Feature Vector X
                                                        /
[ Acceptor SMILES ] ---> [ RDKit Descriptors ] ---> Vector A (dim: 128)
```

### 2.1 Individual Molecule Descriptors
For both donor and acceptor, RDKit extracts:
- **Physicochemical Scalars**:
  - Molecular Weight (`MolWt`)
  - LogP (Wildman-Crippen partition coefficient, `MolLogP`)
  - Topological Polar Surface Area (`TPSA`)
  - Fraction of sp³ carbons (`FractionCSP3`)
  - Rotatable bond count (`NumRotatableBonds`)
  - Aromatic ring count (`NumAromaticRings`)
  - Hydrogen bond donor count (`NumHBD`) and acceptor count (`NumHBA`)
- **Topological & Electrostatic**:
  - Partial charge estimations (Gasteiger charges)
  - Balaban J, Wiener, and Chi connectivity indices
- **Fingerprints**:
  - Morgan Fingerprints (Radius 2, 512 bits) capturing local sub-structural motifs (e.g., thiophene rings, benzothiadiazole cores, cyano end-groups).

### 2.2 Donor–Acceptor Blend & Interaction Features
Because device operation depends on the donor–acceptor heterojunction interface, composite blend features are constructed:
1. **Difference Features**:
   - $\Delta\text{HOMO} = E_{HOMO}^{Donor} - E_{HOMO}^{Acceptor}$
   - $\Delta\text{LUMO} = E_{LUMO}^{Donor} - E_{LUMO}^{Acceptor}$ (critical for exciton dissociation driving force)
   - Molecular weight difference: $|MW_D - MW_A|$
   - Polarity disparity: $|LogP_D - LogP_A|$
2. **Concatenated Vector**: $[ \mathbf{v}_D \,\|\, \mathbf{v}_A \,\|\, (\mathbf{v}_D - \mathbf{v}_A) \,\|\, (\mathbf{v}_D \odot \mathbf{v}_A) ]$.
3. **Process Metadata Features (When Available)**:
   - Blend ratio (e.g., $1.0 : 1.2 \rightarrow 0.833$)
   - Processing solvent boiling point and solubility parameters
   - Presence of processing additives (e.g., binary indicator for DIO/CN)
   - Thermal annealing temperature (°C).

---

## 3. Training Targets & Experimental Data Sourcing

### 3.1 Candidate Targets
1. **PCE (%) — Primary Target**: Overall device power conversion efficiency under standard 1-sun irradiance (AM1.5G, 100 mW/cm², 25°C).
2. **Voc (V)**: Open-circuit voltage.
3. **Jsc (mA/cm²)**: Short-circuit current density.
4. **FF (%)**: Fill factor.

### 3.2 Data Sources & Scientific Rigor
- Training labels must originate exclusively from validated, peer-reviewed experimental literature (e.g., HOPV15, Harvard Clean Energy Project, or curated journal publications).
- **Zero Fabrication Policy**: No synthetic labels, fake efficiencies, or randomly generated target numbers are tolerated.
- Multi-target prediction can be handled either via Scikit-Learn's native multi-output regression or through independent tuned regressors for each target metric.

---

## 4. Data Cleaning & Preprocessing Protocols

1. **Chemical Canonicalization**:
   - All SMILES strings are stripped of solvent salts and converted to standard canonical forms via `Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)`.
   - Explicit hydrogens are removed; tautomers are standardized.
2. **Range & Sanity Filtering**:
   - Physical boundary verification: $0 < PCE \le 25\%$, $0 < Voc \le 2.0\text{ V}$, $0 < Jsc \le 40\text{ mA/cm}^2$, $20 \le FF \le 90\%$.
   - Duplicates with identical donor, acceptor, and device conditions are averaged or curated for experimental variance.
3. **Missing Value Treatment**:
   - Missing continuous molecular descriptors are rejected if core structure is corrupted.
   - Missing process parameters (e.g., unreported annealing temperature) are assigned explicit category indicators or median-imputed with an accompanying missingness indicator flag, avoiding false synthetic certainty.

---

## 5. Train/Test Separation & Leakage Prevention

In chemoinformatics, random train/test splits lead to **optimistic evaluation bias** because structural analogs (e.g., identical conjugated cores with slight alkyl side-chain variations) appear simultaneously in train and test splits.

To ensure true predictive generalization:
- **Bemis-Murcko Scaffold Splitting**: Splits molecules based on core ring frameworks and linkers. All derivatives of a specific molecular scaffold are assigned exclusively to either the training set or the test set.
- **GroupKFold by Donor / Acceptor Class**: Evaluates the model's ability to extrapolate to completely novel acceptor families (e.g., training on fullerene acceptors and testing on non-fullerene acceptors like Y6).

---

## 6. Evaluation Metrics

Models are benchmarked using three quantitative criteria:

1. **Mean Absolute Error (MAE)**:
   $$MAE = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**:
   $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

A model is considered viable for preliminary virtual screening if $R^2 \ge 0.65$ and $MAE(PCE) \le 1.2\%$ on scaffold-held-out test splits.

---

## 7. Model Persistence & Versioning

- Trained models are serialized using `joblib.dump(pipeline, "ml/models/saved_models/opv_rf_<version>.joblib", compress=3)`.
- Each model artifact is accompanied by a `<version>.meta.json` record containing exact library versions, feature names, scaffold split hashes, and cross-validation scores.

---

## 8. Prediction Uncertainty & Inherent Scientific Limitations

Machine-learning predictions in OPV-PredictNet must always be interpreted in the context of experimental physical chemistry:
1. **Morphology Dependency**: Bulk heterojunction performance is intimately dictated by nanoscale morphology (domain size, phase purity, molecular orientation, face-on vs. edge-on stacking). Identical molecular blends processed from different solvents can vary in PCE by >50%.
2. **Domain of Applicability**: Predictions for chemical scaffolds significantly distant from the training distribution (measured by Tanimoto similarity <0.4) carry substantial uncertainty.
3. **Role of Model**: The model functions as an **accelerated prescreening filter** to prioritize candidate combinations for synthesis, not as a replacement for physical laboratory fabrication.
