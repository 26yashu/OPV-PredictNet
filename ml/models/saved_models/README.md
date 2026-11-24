# Serialized Machine Learning Models

This directory stores trained model artifacts serialized using `joblib`.

## Storage Format
- `<model_name>_<version>.joblib`: Serialized Scikit-learn estimator or pipeline.
- `<model_name>_<version>.meta.json`: Accompanying metadata containing:
  - Model architecture and hyperparameters
  - Feature names and scaling configurations
  - Training dataset hash and date
  - Cross-validation evaluation metrics ($R^2$, MAE, RMSE)
  - Target variable definition (PCE, $V_{OC}$, $J_{SC}$, $FF$)

No model weights or binary artifacts are committed to Git.
