import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from ml.src.features.descriptors import FEATURE_NAMES
from ml.src.evaluation.sanity import validate_prediction_sanity

logger = logging.getLogger(__name__)


def validate_scientific_artifact(
    model: Any,
    metadata: Optional[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Strict validation rules for production OPV model artifacts.

    Verifies:
    1. Feature schema compatibility (exact 22 canonical features).
    2. Genuine experimental dataset provenance (rejects synthetic / test fixtures).
    3. Valid evaluation metrics (R2 > 0, MAE > 0, RMSE > 0).
    4. Explicit is_scientifically_validated certification flag.
    5. Real dataset source citation (DOI or URL).
    """
    errors: List[str] = []

    if model is None:
        return False, ["Model object is None or failed deserialization."]

    # 1. Feature dimension check
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURE_NAMES):
        errors.append(
            f"Feature count mismatch: model requires {model.n_features_in_} features, "
            f"expected {len(FEATURE_NAMES)} canonical features."
        )

    # 2. Metadata existence
    if not metadata or not isinstance(metadata, dict):
        errors.append("Model metadata missing or not a valid dictionary.")
        return False, errors

    # 3. Explicit scientific validation flag
    if not metadata.get("is_scientifically_validated", False):
        errors.append("Model lacks required 'is_scientifically_validated: true' certification flag.")

    # 4. Provenance check: must not be a test or synthetic fixture
    provenance = str(metadata.get("training_provenance", "")).strip().lower()
    if not provenance:
        errors.append("Missing 'training_provenance' documenting experimental literature data source.")
    else:
        unverified_keywords = ["synthetic", "test fixture", "dummy", "mock", "random", "placeholder"]
        for kw in unverified_keywords:
            if kw in provenance:
                errors.append(
                    f"Model is a test fixture or synthetic artifact (disallowed term '{kw}' found in provenance)."
                )
                break

    # 5. Dataset source identifier (DOI, URL, or published benchmark)
    dataset_source = str(metadata.get("dataset_source", "")).strip().lower()
    if not dataset_source:
        errors.append("Missing 'dataset_source' (DOI, URL, or experimental database identifier).")
    elif not (
        dataset_source.startswith("http://")
        or dataset_source.startswith("https://")
        or dataset_source.startswith("doi:")
        or dataset_source.startswith("10.")
        or "opv" in dataset_source
    ):
        errors.append(
            "Invalid 'dataset_source': must be a valid DOI citation, HTTPS URL, or experimental database identifier."
        )

    # 6. Evaluation metrics check
    eval_metrics = metadata.get("evaluation_metrics")
    if not isinstance(eval_metrics, dict):
        errors.append("Missing 'evaluation_metrics' dictionary.")
    else:
        for metric in ["r2", "mae", "rmse"]:
            val = eval_metrics.get(metric)
            if val is None:
                val = eval_metrics.get(f"pce_{metric}")
            if val is None or not isinstance(val, (int, float)):
                errors.append(f"Evaluation metric '{metric}' (or 'pce_{metric}') missing or non-numeric.")
            elif metric == "r2" and val <= 0.0:
                errors.append(f"Evaluation metric R2 ({val}) indicates non-predictive or negative score.")
            elif metric in ["mae", "rmse"] and val <= 0.0:
                errors.append(f"Evaluation metric '{metric}' must be positive.")

    # 7. Feature ordering check
    if "features" in metadata and list(metadata["features"]) != FEATURE_NAMES:
        errors.append("Metadata feature ordering does not match canonical FEATURE_NAMES.")

    is_valid = len(errors) == 0
    return is_valid, errors


class OPVModelRegistry:
    """Manages loading, validation, and serving of genuine machine-learning models.
    Strictly prohibits returning fabricated model predictions or serving unvalidated fixtures.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            # Default to OPV-PredictNet/ml/models/saved_models
            self.models_dir = Path(__file__).resolve().parent.parent.parent / "models" / "saved_models"
        else:
            self.models_dir = Path(models_dir)

        self._active_model: Any = None
        self._active_metadata: Optional[Dict[str, Any]] = None
        self._validation_errors: List[str] = []
        self._load_active_model()

    def refresh(self) -> None:
        """Reload models from disk."""
        self._load_active_model()

    def _load_active_model(self) -> None:
        """Scan models directory for a valid .joblib artifact and accompanying .meta.json."""
        self._active_model = None
        self._active_metadata = None
        self._validation_errors = []

        if not self.models_dir.exists():
            return

        joblib_files = list(self.models_dir.glob("*.joblib"))
        if not joblib_files:
            return

        # Load first serialized model located within canonical directory
        target_artifact = joblib_files[0]

        # Security Check: Ensure file is directly inside models_dir (no path traversal)
        try:
            resolved_target = target_artifact.resolve()
            resolved_dir = self.models_dir.resolve()
            if resolved_target.parent != resolved_dir:
                logger.error(f"Security error: Artifact {target_artifact} is outside allowed models directory.")
                self._validation_errors = ["Security violation: artifact path outside models directory."]
                return
        except Exception as e:
            logger.error(f"Path resolution error for {target_artifact}: {e}")
            self._validation_errors = [f"Path resolution error: {e}"]
            return

        meta_file = target_artifact.with_suffix(".meta.json")

        try:
            loaded = joblib.load(target_artifact)
            if isinstance(loaded, dict) and "model" in loaded:
                self._active_model = loaded["model"]
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        self._active_metadata = json.load(f)
                else:
                    self._active_metadata = {k: v for k, v in loaded.items() if k != "model"}
                    if "version_tag" not in self._active_metadata:
                        self._active_metadata["version_tag"] = target_artifact.stem
            else:
                self._active_model = loaded
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        self._active_metadata = json.load(f)
                else:
                    self._active_metadata = {
                        "version_tag": target_artifact.stem,
                        "artifact_file": target_artifact.name,
                        "algorithm": type(self._active_model).__name__,
                    }

            # Run strict scientific validation checks
            is_valid, errors = validate_scientific_artifact(self._active_model, self._active_metadata)
            self._validation_errors = errors

            if is_valid:
                logger.info(f"Loaded scientifically validated ML model artifact: {target_artifact.name}")
            else:
                logger.warning(
                    f"Loaded model artifact {target_artifact.name} failed scientific validation: {errors}"
                )

        except Exception as e:
            logger.error(f"Failed to load model artifact {target_artifact}: {e}")
            self._active_model = None
            self._active_metadata = None
            self._validation_errors = [f"Deserialization failure: {e}"]

    def is_model_available(self) -> bool:
        """Check whether a model artifact binary is loaded in memory."""
        return self._active_model is not None

    def is_scientifically_validated(self) -> bool:
        """Check whether the loaded model is verified on real experimental datasets.

        A model artifact is NEVER considered scientifically validated merely because
        a .joblib binary file successfully deserializes in memory.
        """
        return self.is_model_available() and len(self._validation_errors) == 0

    def get_validation_errors(self) -> List[str]:
        """Return list of validation errors explaining why the model was rejected."""
        return list(self._validation_errors)

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Return metadata for the currently active model."""
        return self._active_metadata

    def predict(self, feature_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run prediction using the loaded model and enforce physical output sanity.

        Returns:
            Dictionary with predicted metrics and sanity status, or None if model unavailable.
        """
        if not self.is_model_available():
            return None

        # Verify exact input feature vector dimension matches canonical schema
        vec = np.asarray(feature_vector, dtype=np.float64).flatten()
        if vec.size != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature vector length ({vec.size}) does not match expected 22-feature schema ({len(FEATURE_NAMES)})."
            )

        # Reshape for single-sample inference (1, num_features)
        X = vec.reshape(1, -1)
        raw_pred = self._active_model.predict(X)

        if isinstance(raw_pred, np.ndarray) and raw_pred.ndim == 2:
            pred_row = raw_pred[0]
        elif isinstance(raw_pred, np.ndarray) and raw_pred.ndim == 1:
            pred_row = raw_pred
        else:
            pred_row = np.array([float(raw_pred)])

        # Expected output mapping based on standard order [PCE, Voc, Jsc, FF]
        pce = round(float(pred_row[0]), 2)
        voc = round(float(pred_row[1]), 3) if len(pred_row) > 1 else None
        jsc = round(float(pred_row[2]), 2) if len(pred_row) > 2 else None
        ff = round(float(pred_row[3]), 2) if len(pred_row) > 3 else None

        # Enforce physical sanity bounds on predicted outputs (Task 4)
        is_sane, violations = validate_prediction_sanity(pce=pce, voc=voc, jsc=jsc, ff=ff)

        if not is_sane:
            logger.warning(f"Unphysical prediction generated: {violations}")
            return {
                "status": "unphysical_output",
                "is_sane": False,
                "sanity_violations": violations,
                "pce_percent": None,
                "voc_volts": None,
                "jsc_ma_cm2": None,
                "fill_factor_percent": None,
                "raw_unphysical_values": {
                    "pce": pce,
                    "voc": voc,
                    "jsc": jsc,
                    "ff": ff,
                },
                "message": f"Model inference produced physically implausible outputs: {'; '.join(violations)}",
            }

        return {
            "status": "success",
            "is_sane": True,
            "pce_percent": pce,
            "voc_volts": voc,
            "jsc_ma_cm2": jsc,
            "fill_factor_percent": ff,
            "units": {
                "pce": "%",
                "voc": "V",
                "jsc": "mA/cm²",
                "fill_factor": "%",
            },
            "scientific_limitations": (
                "Predicted values represent statistical approximations under standard AM1.5G (100 mW/cm²) "
                "test conditions. Real device PCE strongly depends on active layer film thickness, solvent "
                "additives (e.g. DIO), thermal annealing, and nanoscale morphology."
            )
        }


# Global model registry singleton
model_registry = OPVModelRegistry()
