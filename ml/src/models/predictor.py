import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from ml.src.features.descriptors import FEATURE_NAMES

logger = logging.getLogger(__name__)


class OPVModelRegistry:
    """Manages loading, validation, and serving of genuine machine-learning models.
    Strictly prohibits returning fabricated model predictions when no artifact exists.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            # Default to OPV-PredictNet/ml/models/saved_models
            self.models_dir = Path(__file__).resolve().parent.parent.parent / "models" / "saved_models"
        else:
            self.models_dir = Path(models_dir)

        self._active_model: Any = None
        self._active_metadata: Optional[Dict[str, Any]] = None
        self._load_active_model()

    def refresh(self) -> None:
        """Reload models from disk."""
        self._load_active_model()

    def _load_active_model(self) -> None:
        """Scan models directory for a valid .joblib artifact and accompanying .meta.json."""
        if not self.models_dir.exists():
            return

        joblib_files = list(self.models_dir.glob("*.joblib"))
        if not joblib_files:
            self._active_model = None
            self._active_metadata = None
            return

        # Load first valid serialized model
        target_artifact = joblib_files[0]
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
            # Safety Verification 1: Feature dimension compatibility
            if hasattr(self._active_model, "n_features_in_"):
                if self._active_model.n_features_in_ != len(FEATURE_NAMES):
                    logger.error(
                        f"Incompatible model {target_artifact.name}: expected {len(FEATURE_NAMES)} features, "
                        f"but model requires {self._active_model.n_features_in_}. Rejecting artifact."
                    )
                    self._active_model = None
                    self._active_metadata = None
                    return

            # Safety Verification 2: Feature names and ordering verification
            if self._active_metadata and "features" in self._active_metadata:
                if list(self._active_metadata["features"]) != FEATURE_NAMES:
                    logger.error(
                        f"Feature mismatch in {target_artifact.name}: metadata features do not match "
                        f"canonical FEATURE_NAMES schema. Rejecting artifact."
                    )
                    self._active_model = None
                    self._active_metadata = None
                    return

            logger.info(f"Loaded ML model artifact: {target_artifact.name}")
        except Exception as e:
            logger.error(f"Failed to load model artifact {target_artifact}: {e}")
            self._active_model = None
            self._active_metadata = None

    def is_model_available(self) -> bool:
        """Check whether a real, verified model artifact is loaded in memory."""
        return self._active_model is not None

    def is_scientifically_validated(self) -> bool:
        """Check whether the loaded model is verified on real experimental datasets.

        A model artifact is NEVER considered scientifically validated merely because
        a .joblib binary file successfully deserializes in memory.
        """
        if not self.is_model_available():
            return False
        meta = self._active_metadata or {}
        provenance = str(meta.get("training_provenance", "")).lower()
        if not provenance or "test" in provenance or "synthetic" in provenance:
            return False
        return bool(meta.get("is_scientifically_validated", False))

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Return metadata for the currently active model."""
        return self._active_metadata

    def predict(self, feature_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run genuine prediction using the loaded model.

        Returns:
            Dictionary of predicted metrics (pce, voc, jsc, ff) or None if model unavailable.
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

        # Expected output mapping based on model metadata or standard order [PCE, Voc, Jsc, FF]
        pce = round(float(pred_row[0]), 2)
        voc = round(float(pred_row[1]), 3) if len(pred_row) > 1 else None
        jsc = round(float(pred_row[2]), 2) if len(pred_row) > 2 else None
        ff = round(float(pred_row[3]), 2) if len(pred_row) > 3 else None

        return {
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
            "scientific_limitations": "Predicted values represent statistical approximations under standard AM1.5G (100 mW/cm²) test conditions. Real device PCE strongly depends on active layer film thickness, solvent additives (e.g. DIO), thermal annealing, and nanoscale morphology."
        }


# Global model registry singleton
model_registry = OPVModelRegistry()
