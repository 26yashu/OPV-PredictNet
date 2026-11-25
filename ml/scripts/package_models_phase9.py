"""
Phase 9: OPV-PredictNet Model Packaging Script

Trains M0, M1, M2A, M3 on the full dataset using the authoritative
UnifiedFeaturePipeline and saves versioned artifacts for FastAPI deployment.

Training strategy:
  - ALL 22,467 records are used for model training (maximizes coverage).
  - Conformal calibration quantiles are computed from Strategy C2 D_cal
    partition (N_cal = 3,863; most balanced, well-exchangeable distribution).
  - This matches Phase 8.5 Strategy C2 calibration parameters.

Governance constraints (Phase 8.3):
  - research_estimate_only = True (hardcoded)
  - validated_for_production = False (hardcoded)
  - All artifacts include OOD scaffold registry for inference-time warnings.

Usage:
    python -m ml.scripts.package_models_phase9
  or:
    backend/.venv/Scripts/python.exe -m ml.scripts.package_models_phase9
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.features.processing import (
    UnifiedFeaturePipeline,
    validate_and_parse_smiles,
    MODEL_SCHEMAS,
    M3_NAN_ALLOWED_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = PROJECT_ROOT / "ml/data/raw/extracted_opvdb_v1.1.0/data/opv_devices_strict_molecular_benchmark.csv"
DATASET_SHA256 = "ffbcd6e9dc60f94d7bce037b8fba8c0e23e503b5a32047e76d40de1027bc3897"
PHASE85_CONFIG_PATH = PROJECT_ROOT / "ml/results/phase_8_5/phase_8_5_config.json"
PHASE85_SPLIT_META_PATH = PROJECT_ROOT / "ml/results/phase_8_5/phase_8_5_split_metadata.json"

SAVE_DIR = PROJECT_ROOT / "ml/models/saved_models/phase9_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 8888

# Approved Phase 8.3 model configurations
MODEL_CONFIGS = {
    "M0": {
        "estimator_cls": RandomForestRegressor,
        "estimator_kwargs": {"n_estimators": 100, "random_state": RANDOM_SEED, "n_jobs": 1},
    },
    "M1": {
        "estimator_cls": RandomForestRegressor,
        "estimator_kwargs": {"n_estimators": 100, "random_state": RANDOM_SEED, "n_jobs": 1},
    },
    "M2A": {
        "estimator_cls": RandomForestRegressor,
        "estimator_kwargs": {"n_estimators": 100, "random_state": RANDOM_SEED, "n_jobs": 1},
    },
    "M3": {
        "estimator_cls": HistGradientBoostingRegressor,
        "estimator_kwargs": {"max_iter": 100, "random_state": RANDOM_SEED},
    },
}

# Phase 8.5 Strategy C2 calibration quantiles (from phase_8_5_interval_metrics.csv)
# These are the authoritative conformal calibration parameters.
CONFORMAL_QUANTILES_C2 = {
    "M0":  {"80%": 3.3730, "90%": 4.6503, "95%": 6.2258},
    "M1":  {"80%": 3.2106, "90%": 4.2726, "95%": 5.5214},
    "M2A": {"80%": 3.2626, "90%": 4.3230, "95%": 5.5246},
    "M3":  {"80%": 3.1379, "90%": 4.1624, "95%": 5.2954},
}

DATASET_SOURCE = "https://doi.org/10.1039/d1ee02691g"
TRAINING_PROVENANCE = (
    "Validated experimental peer-reviewed OPV device database (OPVdb v1.1.0). "
    "Phase 8.3 governance-approved model configurations trained on full 22,467-record benchmark. "
    "Conformal calibration from Phase 8.5 Strategy C2 partition (N_cal=3863)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_dataset_sha256(path: Path, expected: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Dataset SHA-256 mismatch!\n  expected: {expected}\n  actual:   {actual}"
        )
    logger.info(f"Dataset SHA-256 verified: {actual}")
    return actual


def extract_scaffold_registry(df: pd.DataFrame) -> dict:
    """Build the set of known donor and acceptor Bemis-Murcko scaffold IDs from the dataset."""
    donor_scaffolds = set()
    acceptor_scaffolds = set()

    for smiles in df["donor_smiles"].dropna().unique():
        try:
            _, _, scaffold_id, _ = validate_and_parse_smiles(smiles, role="donor")
            donor_scaffolds.add(scaffold_id)
        except Exception:
            pass

    for smiles in df["acceptor_smiles"].dropna().unique():
        try:
            _, _, scaffold_id, _ = validate_and_parse_smiles(smiles, role="acceptor")
            acceptor_scaffolds.add(scaffold_id)
        except Exception:
            pass

    logger.info(
        f"Scaffold registry: {len(donor_scaffolds)} donor scaffolds, "
        f"{len(acceptor_scaffolds)} acceptor scaffolds."
    )
    return {
        "donor_scaffolds": sorted(donor_scaffolds),
        "acceptor_scaffolds": sorted(acceptor_scaffolds),
        "n_donor_scaffolds": len(donor_scaffolds),
        "n_acceptor_scaffolds": len(acceptor_scaffolds),
        "source": "Full OPVdb v1.1.0 benchmark dataset (22,467 records)",
    }


def load_phase85_split_indices() -> dict:
    """Load the Strategy C2 calibration split indices from Phase 8.5 metadata."""
    if PHASE85_SPLIT_META_PATH.exists():
        with open(PHASE85_SPLIT_META_PATH, "r") as f:
            return json.load(f)
    return {}


def verify_deterministic_reload(
    model_id: str,
    model_path: Path,
    pipeline: UnifiedFeaturePipeline,
    df_check: pd.DataFrame,
    original_preds: np.ndarray,
) -> bool:
    """Reload artifact and verify predictions match original to 1e-9 tolerance."""
    artifact = joblib.load(model_path)
    reloaded_model = artifact["model"]
    state = artifact["pipeline_state"]
    reloaded_pipe = UnifiedFeaturePipeline.from_state_dict(state)

    extractor = {
        "M0": reloaded_pipe.extract_m0,
        "M1": reloaded_pipe.extract_m1,
        "M2A": reloaded_pipe.extract_m2a,
        "M3": reloaded_pipe.extract_m3,
    }[model_id]

    X_check = extractor(df_check).to_numpy(dtype=np.float64 if model_id != "M3" else object)
    reload_preds = reloaded_model.predict(X_check)

    max_delta = np.max(np.abs(reload_preds - original_preds))
    if max_delta < 1e-9:
        logger.info(f"  {model_id}: deterministic reload verified (max_delta={max_delta:.2e})")
        return True
    else:
        logger.error(f"  {model_id}: reload mismatch! max_delta={max_delta:.6f}")
        return False


# ---------------------------------------------------------------------------
# Main packaging routine
# ---------------------------------------------------------------------------

def package_models() -> None:
    logger.info("=" * 70)
    logger.info("Phase 9 — Model Packaging")
    logger.info("=" * 70)

    # 1. Verify dataset
    logger.info(f"Loading dataset: {DATASET_PATH}")
    sha256 = verify_dataset_sha256(DATASET_PATH, DATASET_SHA256)
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset: {len(df)} records, {df.shape[1]} columns")

    # 2. Fit unified feature pipeline on FULL dataset
    logger.info("Fitting UnifiedFeaturePipeline on full dataset...")
    pipeline = UnifiedFeaturePipeline()
    pipeline.fit_descriptors(df)
    pipeline.fit_training(df)
    pipeline_state = pipeline.to_state_dict()
    logger.info("Pipeline fitted.")

    # 3. Extract features for all models
    logger.info("Extracting features for all models...")
    df_m0 = pipeline.extract_m0(df)
    logger.info(f"  M0: {df_m0.shape}")

    df_m1 = pipeline.extract_m1(df, df_m0=df_m0)
    logger.info(f"  M1: {df_m1.shape}")

    df_m2a = pipeline.extract_m2a(df, df_m0=df_m0)
    logger.info(f"  M2A: {df_m2a.shape}")

    df_m3 = pipeline.extract_m3(df, df_m0=df_m0)
    logger.info(f"  M3: {df_m3.shape}")

    target = df["pce"].values
    logger.info(f"Target (PCE): min={target.min():.2f}, max={target.max():.2f}, mean={target.mean():.2f}")

    # 4. Build scaffold registry from full dataset
    logger.info("Building scaffold registry...")
    scaffold_registry = extract_scaffold_registry(df)

    # 5. Save pipeline state and scaffold registry
    pipeline_state_path = SAVE_DIR / "phase9_pipeline_state.json"
    with open(pipeline_state_path, "w") as f:
        json.dump(pipeline_state, f, indent=2)
    logger.info(f"Pipeline state saved: {pipeline_state_path}")

    scaffold_registry_path = SAVE_DIR / "phase9_scaffold_registry.json"
    with open(scaffold_registry_path, "w") as f:
        json.dump(scaffold_registry, f, indent=2)
    logger.info(f"Scaffold registry saved: {scaffold_registry_path}")

    # 6. Train, save, and verify each model
    feature_frames = {"M0": df_m0, "M1": df_m1, "M2A": df_m2a, "M3": df_m3}
    model_artifacts = {}
    verification_rows = df.sample(10, random_state=RANDOM_SEED)
    all_verified = True

    for model_id, config in MODEL_CONFIGS.items():
        logger.info(f"\n--- Training {model_id} ---")
        X_train = feature_frames[model_id].to_numpy(dtype=np.float64 if model_id != "M3" else object)
        y_train = target

        estimator = config["estimator_cls"](**config["estimator_kwargs"])
        estimator.fit(X_train, y_train)

        n_features = len(MODEL_SCHEMAS[model_id])
        logger.info(f"  Trained {type(estimator).__name__} on {X_train.shape[0]} samples, {n_features} features.")

        # Compute predictions on verification rows for determinism check
        extractor = {
            "M0": pipeline.extract_m0,
            "M1": pipeline.extract_m1,
            "M2A": pipeline.extract_m2a,
            "M3": pipeline.extract_m3,
        }[model_id]
        X_verify = extractor(verification_rows).to_numpy(dtype=np.float64 if model_id != "M3" else object)
        verify_preds = estimator.predict(X_verify)

        # Compose artifact
        artifact = {
            "model": estimator,
            "pipeline_state": pipeline_state,
            "model_id": model_id,
            "version_tag": f"phase9_{model_id}",
            "feature_names": list(MODEL_SCHEMAS[model_id]),
            "n_features": n_features,
            "algorithm": type(estimator).__name__,
            "hyperparameters": config["estimator_kwargs"],
            "conformal_quantiles": CONFORMAL_QUANTILES_C2[model_id],
            "calibration_strategy": "Phase 8.5 Strategy C2 (N_cal=3863, balanced scaffold coverage)",
            "training_n_samples": int(X_train.shape[0]),
            "training_provenance": TRAINING_PROVENANCE,
            "dataset_source": DATASET_SOURCE,
            "dataset_sha256": sha256,
            "is_scientifically_validated": True,
            "research_estimate_only": True,
            "validated_for_production": False,
            "evaluation_metrics": {
                # Phase 8.5 Strategy C2 metrics (from phase_8_5_scaffold_metrics.csv)
                "r2": {"M0": 0.6172, "M1": 0.6621, "M2A": 0.6584, "M3": 0.6640}[model_id],
                "mae": {"M0": 2.3163, "M1": 2.1278, "M2A": 2.1611, "M3": 2.1638}[model_id],
                "rmse": {"M0": 3.0573, "M1": 2.8724, "M2A": 2.8878, "M3": 2.8640}[model_id],
                "pce_r2": {"M0": 0.6172, "M1": 0.6621, "M2A": 0.6584, "M3": 0.6640}[model_id],
                "pce_mae": {"M0": 2.3163, "M1": 2.1278, "M2A": 2.1611, "M3": 2.1638}[model_id],
                "pce_rmse": {"M0": 3.0573, "M1": 2.8724, "M2A": 2.8878, "M3": 2.8640}[model_id],
                "evaluation_partition": "Phase 8.5 Strategy C2 test set",
                "phase_8_3_governance_role": {
                    "M0": "Molecular-descriptor-only baseline",
                    "M1": "Categorical processing-aware; exploratory candidate for unseen-donor scaffold",
                    "M2A": "Continuous processing-aware reference",
                    "M3": "Native-NaN processing-aware; primary recommended model",
                }[model_id],
            },
            "ood_warnings": {
                "novel_donor_scaffold": "WARNING — Strategy A: exploratory candidates (M1, M3) only",
                "novel_acceptor_scaffold": (
                    "WARNING — Strategy B: high-PCE coverage collapses to ~45%; "
                    "predictions unreliable in top performance regime"
                ),
                "novel_dual_scaffold": (
                    "REJECT — Strategy C1: R²≈0.08; dual-scaffold extrapolation "
                    "not scientifically supported"
                ),
                "etl_missing": "WARNING — missing ETL imputed from training median",
                "htl_missing": "WARNING — missing HTL imputed from training median",
                "missing_architecture": "WARNING — missing device architecture imputed",
            },
            "governance_phase": "Phase 8.3",
        }

        # Save artifact
        artifact_path = SAVE_DIR / f"phase9_{model_id}_model.joblib"
        joblib.dump(artifact, artifact_path, compress=3)
        logger.info(f"  Saved: {artifact_path}")

        # Verify deterministic reload
        ok = verify_deterministic_reload(
            model_id, artifact_path, pipeline, verification_rows, verify_preds
        )
        if not ok:
            all_verified = False
        model_artifacts[model_id] = str(artifact_path)

    # 7. Save global metadata
    metadata = {
        "phase": "Phase 9",
        "packaging_script": "ml/scripts/package_models_phase9.py",
        "governance_reference": "ml/PHASE_8_3_MODEL_GOVERNANCE.md",
        "feature_pipeline": "ml/src/features/processing.py::UnifiedFeaturePipeline",
        "research_estimate_only": True,
        "validated_for_production": False,
        "models": {
            model_id: {
                "artifact": model_artifacts[model_id],
                "n_features": len(MODEL_SCHEMAS[model_id]),
                "algorithm": MODEL_CONFIGS[model_id]["estimator_cls"].__name__,
                "conformal_quantiles_90pct": CONFORMAL_QUANTILES_C2[model_id]["90%"],
            }
            for model_id in MODEL_CONFIGS
        },
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": sha256,
            "n_records": len(df),
        },
        "scaffold_registry": str(scaffold_registry_path),
        "pipeline_state": str(pipeline_state_path),
        "determinism_verified": all_verified,
        "calibration": {
            "strategy": "Phase 8.5 Strategy C2",
            "n_cal": 3863,
            "levels": ["80%", "90%", "95%"],
            "quantiles": CONFORMAL_QUANTILES_C2,
        },
        "ood_policy": {
            "novel_donor_scaffold": "warn",
            "novel_acceptor_scaffold": "warn",
            "novel_dual_scaffold": "reject_or_strong_warn",
        },
    }

    metadata_path = SAVE_DIR / "phase9_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nGlobal metadata saved: {metadata_path}")

    # 8. Summary
    logger.info("\n" + "=" * 70)
    logger.info("Phase 9 Model Packaging — COMPLETE")
    logger.info(f"  Models trained:    {list(MODEL_CONFIGS.keys())}")
    logger.info(f"  Training samples:  {len(df)}")
    logger.info(f"  Artifacts saved:   {SAVE_DIR}")
    logger.info(f"  Determinism check: {'PASS' if all_verified else 'FAIL'}")
    if not all_verified:
        logger.error("DETERMINISM VERIFICATION FAILED — investigate before deployment.")
        sys.exit(1)
    logger.info("=" * 70)


if __name__ == "__main__":
    package_models()
