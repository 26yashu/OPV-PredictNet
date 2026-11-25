"""
Phase 9 Prediction Service for OPV-PredictNet.

Loads versioned Phase 9 model artifacts (M0, M1, M2A, M3), the frozen
UnifiedFeaturePipeline state, and the training scaffold registry from
ml/models/saved_models/phase9_v1/.

Enforces:
- Strict SMILES and physical processing validation via UnifiedFeaturePipeline.
- Rejection of BigSMILES, polymer wildcards (*), and multi-component mixtures (.).
- Bemis-Murcko scaffold OOD detection and unsupported dual-scaffold rejection.
- Conformal prediction intervals from Phase 8.5 Strategy C2 calibration.
- Hardcoded research_estimate_only=True and validated_for_production=False.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from ml.src.evaluation.sanity import validate_prediction_sanity
from ml.src.features.processing import (
    FeaturePipelineError,
    InvalidMolecularInputError,
    MissingMolecularIdentityError,
    PhysicalInputValidationError,
    UnifiedFeaturePipeline,
    UnsupportedPolymerOrMixtureError,
    validate_and_parse_smiles,
    validate_processing_record,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE9_MODELS_DIR = PROJECT_ROOT / "ml" / "models" / "saved_models" / "phase9_v1"
SUPPORTED_MODELS = ("M0", "M1", "M2A", "M3")
SUPPORTED_CONFIDENCE_LEVELS = ("80%", "90%", "95%")


class DualScaffoldOODError(ValueError):
    """Raised when both donor and acceptor scaffolds are unseen in training."""


class ModelNotAvailableError(RuntimeError):
    """Raised when the requested Phase 9 model artifact is not loaded."""


class Phase9PredictionService:
    """Thread-safe loader and inference engine for Phase 9 packaged OPV models."""

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else PHASE9_MODELS_DIR
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._pipeline: Optional[UnifiedFeaturePipeline] = None
        self._donor_scaffolds: Set[str] = set()
        self._acceptor_scaffolds: Set[str] = set()
        self._global_metadata: Dict[str, Any] = {}
        self._load_errors: List[str] = []
        self.refresh()

    def refresh(self) -> None:
        """Load or reload Phase 9 artifacts, pipeline state, and scaffold registry."""
        self._artifacts = {}
        self._pipeline = None
        self._donor_scaffolds = set()
        self._acceptor_scaffolds = set()
        self._global_metadata = {}
        self._load_errors = []

        if not self.models_dir.exists():
            self._load_errors.append(f"Phase 9 models directory not found: {self.models_dir}")
            return

        # Load pipeline state
        pipeline_path = self.models_dir / "phase9_pipeline_state.json"
        if pipeline_path.exists():
            with open(pipeline_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._pipeline = UnifiedFeaturePipeline.from_state_dict(state)
        else:
            self._load_errors.append(f"Missing pipeline state: {pipeline_path}")

        # Load scaffold registry
        registry_path = self.models_dir / "phase9_scaffold_registry.json"
        if registry_path.exists():
            with open(registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            self._donor_scaffolds = set(reg.get("donor_scaffolds", []))
            self._acceptor_scaffolds = set(reg.get("acceptor_scaffolds", []))
        else:
            self._load_errors.append(f"Missing scaffold registry: {registry_path}")

        # Load global metadata
        meta_path = self.models_dir / "phase9_model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._global_metadata = json.load(f)

        # Load model artifacts
        resolved_dir = self.models_dir.resolve()
        for model_id in SUPPORTED_MODELS:
            artifact_path = self.models_dir / f"phase9_{model_id}_model.joblib"
            if not artifact_path.exists():
                self._load_errors.append(f"Missing artifact for {model_id}: {artifact_path}")
                continue
            if artifact_path.resolve().parent != resolved_dir:
                self._load_errors.append(f"Security violation: {artifact_path} outside models directory.")
                continue
            try:
                loaded = joblib.load(artifact_path)
                if not isinstance(loaded, dict) or "model" not in loaded:
                    self._load_errors.append(f"Invalid artifact structure for {model_id}.")
                    continue
                if not loaded.get("is_scientifically_validated", False):
                    self._load_errors.append(f"Artifact {model_id} lacks scientific validation flag.")
                    continue
                self._artifacts[model_id] = loaded
            except Exception as exc:
                self._load_errors.append(f"Failed to load {model_id}: {exc}")

    def is_ready(self, model_id: str = "M3") -> bool:
        return (
            self._pipeline is not None
            and model_id in self._artifacts
            and len(self._donor_scaffolds) > 0
            and len(self._acceptor_scaffolds) > 0
        )

    def get_available_models(self) -> Dict[str, Any]:
        models_info = {}
        for model_id in SUPPORTED_MODELS:
            art = self._artifacts.get(model_id)
            if art:
                models_info[model_id] = {
                    "version_tag": art.get("version_tag"),
                    "algorithm": art.get("algorithm"),
                    "n_features": art.get("n_features"),
                    "conformal_quantiles": art.get("conformal_quantiles"),
                    "calibration_strategy": art.get("calibration_strategy"),
                    "evaluation_metrics": art.get("evaluation_metrics"),
                    "research_estimate_only": True,
                    "validated_for_production": False,
                }
        return {
            "models": models_info,
            "n_donor_scaffolds": len(self._donor_scaffolds),
            "n_acceptor_scaffolds": len(self._acceptor_scaffolds),
            "research_estimate_only": True,
            "validated_for_production": False,
            "load_errors": list(self._load_errors),
        }

    def predict(
        self,
        donor_smiles: str,
        acceptor_smiles: str,
        donor_name: str = "Donor Material",
        acceptor_name: str = "Acceptor Material",
        model_id: str = "M3",
        confidence_level: str = "90%",
        processing_params: Optional[Dict[str, Any]] = None,
        allow_dual_scaffold_ood: bool = False,
    ) -> Dict[str, Any]:
        """Run Phase 9 validated PCE prediction with OOD detection and conformal intervals."""
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_id '{model_id}'. Choose from {SUPPORTED_MODELS}.")
        if confidence_level not in SUPPORTED_CONFIDENCE_LEVELS:
            raise ValueError(
                f"Unsupported confidence_level '{confidence_level}'. Choose from {SUPPORTED_CONFIDENCE_LEVELS}."
            )
        if not self.is_ready(model_id):
            raise ModelNotAvailableError(
                f"Phase 9 model '{model_id}' is not loaded. Errors: {'; '.join(self._load_errors)}"
            )

        # 1. Strict molecular identity & SMILES validation (reject wildcards * and mixtures .)
        _, donor_canonical, donor_scaffold, donor_mol_warnings = validate_and_parse_smiles(
            donor_smiles, role="donor", reject_wildcards_or_mixtures=True
        )
        _, acceptor_canonical, acceptor_scaffold, acceptor_mol_warnings = validate_and_parse_smiles(
            acceptor_smiles, role="acceptor", reject_wildcards_or_mixtures=True
        )

        # 2. Processing metadata validation
        params = dict(processing_params or {})
        proc_record = {
            "donor_smiles": donor_canonical,
            "acceptor_smiles": acceptor_canonical,
            "d_a_ratio": params.get("d_a_ratio"),
            "active_layer_thickness": params.get("active_layer_thickness"),
            "annealing_temp": params.get("annealing_temp"),
            "additive_canonical": params.get("additive_canonical"),
            "additive_ratio": params.get("additive_ratio"),
            "solvent_canonical": params.get("solvent_canonical"),
            "device_type": params.get("device_type"),
            "etl_canonical": params.get("etl_canonical"),
            "htl_canonical": params.get("htl_canonical"),
        }
        proc_warnings = validate_processing_record(proc_record, strict_categorical_types=True)

        # 3. Scaffold OOD check
        donor_known = donor_scaffold in self._donor_scaffolds
        acceptor_known = acceptor_scaffold in self._acceptor_scaffolds
        ood_donor = not donor_known
        ood_acceptor = not acceptor_known
        ood_dual = ood_donor and ood_acceptor

        warnings: List[str] = []
        warnings.extend(donor_mol_warnings)
        warnings.extend(acceptor_mol_warnings)
        warnings.extend(proc_warnings)

        if ood_dual:
            msg = (
                "DUAL_SCAFFOLD_OOD_REJECTED: Both donor and acceptor Bemis-Murcko scaffolds are absent "
                "from the training dataset (Strategy C1 R^2~0.08). Dual-scaffold extrapolation is not "
                "scientifically supported by default."
            )
            if not allow_dual_scaffold_ood:
                raise DualScaffoldOODError(msg)
            warnings.append(
                "CRITICAL_DUAL_SCAFFOLD_OOD: Both donor and acceptor scaffolds are unseen in training "
                "(Strategy C1 R^2~0.08); prediction is an unsupported extrapolation."
            )
        else:
            if ood_donor:
                warnings.append(
                    "OOD_DONOR_SCAFFOLD: Donor scaffold is unseen in training (Strategy A); "
                    "only M1 and M3 are exploratory candidates for unseen donor scaffolds."
                )
            if ood_acceptor:
                warnings.append(
                    "OOD_ACCEPTOR_SCAFFOLD: Acceptor scaffold is unseen in training (Strategy B); "
                    "empirical 90% coverage collapses to ~45% in high-PCE regimes."
                )

        # Processing missingness OOD flags (Phase 8.3 integration contract)
        processing_ood_flags: List[str] = []
        if proc_record["etl_canonical"] is None:
            processing_ood_flags.append("etl_missing")
            warnings.append("PROCESSING_ETL_MISSING: ETL not specified; imputed or treated as missing.")
        if proc_record["htl_canonical"] is None:
            processing_ood_flags.append("htl_missing")
            warnings.append("PROCESSING_HTL_MISSING: HTL not specified; imputed or treated as missing.")
        if proc_record["device_type"] is None:
            processing_ood_flags.append("missing_architecture")
            warnings.append("PROCESSING_ARCHITECTURE_MISSING: Device architecture not specified.")

        if ood_dual:
            ood_severity = "critical"
        elif ood_donor or ood_acceptor:
            ood_severity = "warning"
        elif processing_ood_flags or proc_warnings:
            ood_severity = "minor"
        else:
            ood_severity = "none"

        # 4. Feature extraction via UnifiedFeaturePipeline
        df_row = pd.DataFrame([proc_record])
        extractor = {
            "M0": self._pipeline.extract_m0,
            "M1": self._pipeline.extract_m1,
            "M2A": self._pipeline.extract_m2a,
            "M3": self._pipeline.extract_m3,
        }[model_id]
        df_features = extractor(df_row)
        X = df_features.to_numpy(dtype=np.float64 if model_id != "M3" else object)

        # 5. Model inference & physical sanity check
        artifact = self._artifacts[model_id]
        estimator = artifact["model"]
        raw_pred = float(estimator.predict(X)[0])
        predicted_pce = round(raw_pred, 4)

        is_sane, violations = validate_prediction_sanity(pce=predicted_pce)
        if not is_sane:
            raise ValueError(f"Model produced unphysical PCE prediction ({predicted_pce}%): {'; '.join(violations)}")

        # 6. Conformal interval estimate
        quantiles = artifact["conformal_quantiles"]
        q_val = float(quantiles[confidence_level])
        interval_low = round(predicted_pce - q_val, 4)
        interval_high = round(predicted_pce + q_val, 4)
        interval_width = round(2.0 * q_val, 4)

        if ood_dual:
            support_level = "unsupported_dual_scaffold_extrapolation"
        elif ood_acceptor:
            support_level = "exploratory_unseen_acceptor_high_pce_degraded"
        elif ood_donor:
            support_level = (
                "exploratory_unseen_donor_candidate"
                if model_id in ("M1", "M3")
                else "baseline_unseen_donor_reference"
            )
        else:
            support_level = "in_scaffold_research_estimate"

        eval_metrics = artifact.get("evaluation_metrics", {})

        return {
            "model_id": model_id,
            "model_version": artifact.get("version_tag", f"phase9_{model_id}"),
            "algorithm": artifact.get("algorithm", type(estimator).__name__),
            "point_prediction": {
                "predicted_pce": predicted_pce,
                "unit": "%",
            },
            "interval_estimate": {
                "prediction_interval_low": interval_low,
                "prediction_interval_high": interval_high,
                "confidence_level": confidence_level,
                "conformal_quantile_q": q_val,
                "interval_width": interval_width,
                "calibration_strategy": artifact.get(
                    "calibration_strategy", "Phase 8.5 Strategy C2 (N_cal=3863)"
                ),
            },
            "ood_status": {
                "ood_donor_scaffold": ood_donor,
                "ood_acceptor_scaffold": ood_acceptor,
                "ood_dual_scaffold": ood_dual,
                "donor_scaffold_known": donor_known,
                "acceptor_scaffold_known": acceptor_known,
                "donor_scaffold_id": donor_scaffold,
                "acceptor_scaffold_id": acceptor_scaffold,
                "processing_ood_flags": processing_ood_flags,
                "severity": ood_severity,
            },
            "scientific_support": {
                "research_estimate_only": True,
                "validated_for_production": False,
                "support_level": support_level,
                "governance_role": eval_metrics.get(
                    "phase_8_3_governance_role", "Phase 8.3 research model"
                ),
                "benchmark_metrics": {
                    "r2": eval_metrics.get("r2"),
                    "mae": eval_metrics.get("mae"),
                    "rmse": eval_metrics.get("rmse"),
                    "evaluation_partition": eval_metrics.get("evaluation_partition"),
                },
                "limitations": (
                    "Research estimate only; not validated for production device engineering. "
                    "Conformal intervals assume Strategy C2 exchangeability; unseen acceptor scaffolds "
                    "exhibit degraded high-PCE coverage (~45% at 90% nominal) and dual-scaffold "
                    "extrapolation has low predictive power (R^2~0.08)."
                ),
            },
            "warnings": warnings,
            "donor": {
                "name": donor_name,
                "canonical_smiles": donor_canonical,
                "scaffold_id": donor_scaffold,
            },
            "acceptor": {
                "name": acceptor_name,
                "canonical_smiles": acceptor_canonical,
                "scaffold_id": acceptor_scaffold,
            },
            "processing_parameters": params,
            "research_estimate_only": True,
            "validated_for_production": False,
        }


# Singleton instance for FastAPI service
phase9_service = Phase9PredictionService()
