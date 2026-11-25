import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.db.session import get_db
from backend.app.models.model_version import ModelVersion
from backend.app.models.prediction import Prediction
from backend.app.models.user import User
from backend.app.schemas.prediction import (
    MolecularAnalysisRequest,
    MolecularAnalysisResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryResponse,
    PredictionHistoryItem,
    Phase9PredictionRequest,
    Phase9PredictionResponse,
)
from backend.app.services.molecular import analyze_blend
from backend.app.services.phase9_prediction import (
    DualScaffoldOODError,
    ModelNotAvailableError,
    phase9_service,
)
from backend.app.services.prediction import run_prediction_pipeline
from ml.src.features.processing import FeaturePipelineError

router = APIRouter()


@router.post(
    "/analyze",
    response_model=MolecularAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute 2D physicochemical descriptors using RDKit"
)
def analyze_molecules(payload: MolecularAnalysisRequest):
    """Validate SMILES strings and calculate RDKit descriptors for donor and acceptor separately."""
    try:
        blend_data = analyze_blend(
            donor_smiles=payload.donor_smiles,
            acceptor_smiles=payload.acceptor_smiles,
            donor_name=payload.donor_name or "Donor Material",
            acceptor_name=payload.acceptor_name or "Acceptor Material",
        )
        return blend_data
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run prediction pipeline or return model-unavailable state"
)
def create_prediction(
    req: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit donor and acceptor SMILES for OPV performance prediction.

    If a validated model artifact is loaded, generates real predictions and persists the record to SQLite.
    If no validated model exists, returns an explicit model_unavailable response without fabricating predictions.
    """
    try:
        pipeline_result = run_prediction_pipeline(
            donor_smiles=req.donor_smiles,
            acceptor_smiles=req.acceptor_smiles,
            donor_name=req.donor_name or "Donor Material",
            acceptor_name=req.acceptor_name or "Acceptor Material",
            device_parameters=req.device_parameters,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # If genuine model predictions were generated, persist the record in SQLite
    if pipeline_result.get("status") == "success" and pipeline_result.get("predictions"):
        preds = pipeline_result["predictions"]
        model_version_tag = pipeline_result.get("model_metadata", {}).get("version_tag", "v1.0.0")
        # Ensure model version exists in model_versions table to satisfy Foreign Key & NOT NULL constraints
        from backend.app.models.model_version import ModelVersion
        existing_mv = db.query(ModelVersion).filter(ModelVersion.version_tag == model_version_tag).first()
        if not existing_mv:
            existing_mv = ModelVersion(
                version_tag=model_version_tag,
                model_name="OPV-PredictNet-RF",
                algorithm=pipeline_result.get("model_metadata", {}).get("algorithm", "RandomForestRegressor"),
                target_metrics="PCE,Voc,Jsc,FF",
                artifact_path=str(model_version_tag),
                is_active=True,
            )
            db.add(existing_mv)
            db.commit()

        persisted_version = existing_mv.version_tag

        prediction_record = Prediction(
            user_id=current_user.id,
            input_method=req.input_method or "smiles",
            donor_name=req.donor_name or "Donor Material",
            donor_smiles=pipeline_result["donor"]["canonical_smiles"],
            acceptor_name=req.acceptor_name or "Acceptor Material",
            acceptor_smiles=pipeline_result["acceptor"]["canonical_smiles"],
            device_parameters_json=json.dumps(req.device_parameters) if req.device_parameters else None,
            predicted_pce=float(preds["pce_percent"]),
            predicted_voc=float(preds["voc_volts"]) if preds.get("voc_volts") is not None else None,
            predicted_jsc=float(preds["jsc_ma_cm2"]) if preds.get("jsc_ma_cm2") is not None else None,
            predicted_ff=float(preds["fill_factor_percent"]) if preds.get("fill_factor_percent") is not None else None,
            uncertainty_json=None,
            model_version=persisted_version,
        )
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        pipeline_result["id"] = prediction_record.id
        pipeline_result["created_at"] = prediction_record.created_at

    return pipeline_result


@router.get(
    "",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve authenticated user's prediction history"
)
def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve historical prediction records belonging exclusively to the authenticated user."""
    # Defensively resolve limit/offset if called directly as a Python function in tests
    limit_val = limit if isinstance(limit, int) else getattr(limit, "default", 20)
    offset_val = offset if isinstance(offset, int) else getattr(offset, "default", 0)

    query = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
    )
    total = query.count()
    items = query.offset(offset_val).limit(limit_val).all()

    return PredictionHistoryResponse(
        total=total,
        items=[PredictionHistoryItem.model_validate(item) for item in items],
    )


@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
    summary="List packaged Phase 9 OPV models and governance metadata"
)
def list_phase9_models():
    """Return available Phase 9 models (M0, M1, M2A, M3), conformal quantiles, and governance status."""
    return phase9_service.get_available_models()


@router.post(
    "/pce",
    response_model=Phase9PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Phase 9 PCE prediction with unified feature pipeline, conformal intervals, and OOD checks"
)
def create_phase9_prediction(
    req: Phase9PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run Phase 9 PCE prediction using packaged M0/M1/M2A/M3 models and persist history."""
    processing_params = {
        "d_a_ratio": req.d_a_ratio,
        "active_layer_thickness": req.active_layer_thickness,
        "annealing_temp": req.annealing_temp,
        "additive_canonical": req.additive_canonical,
        "additive_ratio": req.additive_ratio,
        "solvent_canonical": req.solvent_canonical,
        "device_type": req.device_type,
        "etl_canonical": req.etl_canonical,
        "htl_canonical": req.htl_canonical,
    }

    try:
        result = phase9_service.predict(
            donor_smiles=req.donor_smiles,
            acceptor_smiles=req.acceptor_smiles,
            donor_name=req.donor_name or "Donor Material",
            acceptor_name=req.acceptor_name or "Acceptor Material",
            model_id=req.model_id,
            confidence_level=req.confidence_level,
            processing_params=processing_params,
            allow_dual_scaffold_ood=req.allow_dual_scaffold_ood,
        )
    except DualScaffoldOODError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except (FeaturePipelineError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    # Persist to prediction history
    version_tag = result["model_version"]
    existing_mv = db.query(ModelVersion).filter(ModelVersion.version_tag == version_tag).first()
    if not existing_mv:
        bench = result["scientific_support"].get("benchmark_metrics", {})
        existing_mv = ModelVersion(
            version_tag=version_tag,
            model_name=f"OPV-PredictNet-{result['model_id']}",
            algorithm=result["algorithm"],
            target_metrics="PCE",
            r2_score=bench.get("r2"),
            mae_score=bench.get("mae"),
            rmse_score=bench.get("rmse"),
            artifact_path=str(phase9_service.models_dir / f"phase9_{result['model_id']}_model.joblib"),
            is_active=True,
        )
        db.add(existing_mv)
        db.commit()

    uncertainty_payload = {
        "model_id": result["model_id"],
        "interval_estimate": result["interval_estimate"],
        "ood_status": result["ood_status"],
        "scientific_support": result["scientific_support"],
        "warnings": result["warnings"],
        "research_estimate_only": True,
        "validated_for_production": False,
    }

    prediction_record = Prediction(
        user_id=current_user.id,
        input_method=req.input_method or "smiles",
        donor_name=req.donor_name or "Donor Material",
        donor_smiles=result["donor"]["canonical_smiles"],
        acceptor_name=req.acceptor_name or "Acceptor Material",
        acceptor_smiles=result["acceptor"]["canonical_smiles"],
        device_parameters_json=json.dumps(processing_params),
        predicted_pce=float(result["point_prediction"]["predicted_pce"]),
        predicted_voc=None,
        predicted_jsc=None,
        predicted_ff=None,
        uncertainty_json=json.dumps(uncertainty_payload),
        model_version=existing_mv.version_tag,
    )
    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    result["id"] = prediction_record.id
    result["created_at"] = prediction_record.created_at
    return result


@router.get(
    "/{prediction_id}",
    response_model=PredictionHistoryItem,
    status_code=status.HTTP_200_OK,
    summary="Retrieve single prediction record"
)
def get_prediction_by_id(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve a specific prediction record owned by the authenticated user."""
    item = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == current_user.id)
        .first()
    )
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction record not found or access denied",
        )
    return PredictionHistoryItem.model_validate(item)
