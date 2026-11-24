from typing import Any, Dict, Optional
from rdkit import Chem
from backend.app.services.molecular import analyze_blend
from ml.src.features.descriptors import extract_features_from_mols, FEATURE_NAMES
from ml.src.models.predictor import model_registry


def run_prediction_pipeline(
    donor_smiles: str,
    acceptor_smiles: str,
    donor_name: str = "Donor Material",
    acceptor_name: str = "Acceptor Material",
    device_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end molecular analysis and model prediction pipeline.

    Args:
        donor_smiles: SMILES string of donor.
        acceptor_smiles: SMILES string of acceptor.
        donor_name: Name of donor.
        acceptor_name: Name of acceptor.
        device_parameters: Optional device stack & fabrication parameters.

    Returns:
        Structured result containing descriptors, features, and either genuine predictions or model_unavailable state.

    Raises:
        ValueError: If chemical validation fails.
    """
    # 1. RDKit Molecular Analysis & Descriptor Computation
    blend_analysis = analyze_blend(
        donor_smiles=donor_smiles,
        acceptor_smiles=acceptor_smiles,
        donor_name=donor_name,
        acceptor_name=acceptor_name,
    )

    donor_desc = blend_analysis["donor"]
    acceptor_desc = blend_analysis["acceptor"]
    blend_features = blend_analysis["blend_features"]

    # 2. Extract Canonical Feature Vector for Model Inference
    donor_mol = Chem.MolFromSmiles(donor_desc["canonical_smiles"])
    acceptor_mol = Chem.MolFromSmiles(acceptor_desc["canonical_smiles"])
    feature_vector, feature_dict = extract_features_from_mols(donor_mol, acceptor_mol)

    # 3. Model Availability and Scientific Validation Checks
    if not model_registry.is_model_available():
        return {
            "status": "model_unavailable",
            "message": (
                "No validated machine learning model artifact found in ml/models/saved_models. "
                "In accordance with scientific integrity rules, predictions cannot be fabricated without a validated training dataset."
            ),
            "donor": donor_desc,
            "acceptor": acceptor_desc,
            "blend_features": blend_features,
            "device_parameters": device_parameters or {},
            "feature_schema": FEATURE_NAMES,
            "feature_values": {k: round(v, 4) for k, v in feature_dict.items()},
            "predictions": None,
            "model_metadata": {
                "algorithm": "RandomForestRegressor",
                "status": "unloaded",
                "training_dataset_status": "pending_experimental_data_validation",
            },
        }

    # Strict Scientific Validation Check (Task 3)
    # A model artifact must have verified experimental provenance, schema matching, and evaluation metrics.
    # Synthetic test fixtures or unvalidated binaries are never permitted to produce production predictions.
    if not model_registry.is_scientifically_validated():
        errors = model_registry.get_validation_errors()
        return {
            "status": "model_unvalidated",
            "message": (
                "Model artifact in ml/models/saved_models failed scientific validation and cannot be used for production inference. "
                f"Validation reasons: {'; '.join(errors)}"
            ),
            "donor": donor_desc,
            "acceptor": acceptor_desc,
            "blend_features": blend_features,
            "device_parameters": device_parameters or {},
            "feature_schema": FEATURE_NAMES,
            "feature_values": {k: round(v, 4) for k, v in feature_dict.items()},
            "predictions": None,
            "model_metadata": {
                "status": "rejected_scientific_validation",
                "validation_errors": errors,
                "metadata": model_registry.get_metadata(),
            },
        }

    # 4. Genuine Inference Execution & Physical Sanity Verification (Task 4)
    predictions = model_registry.predict(feature_vector)
    metadata = model_registry.get_metadata() or {
        "version_tag": "v1.0.0",
        "algorithm": "RandomForestRegressor",
    }

    # Check for unphysical model output (PCE > 34%, Voc > 2.0 V, Jsc > 45 mA/cm², FF > 100%)
    if predictions and (predictions.get("is_sane") is False or predictions.get("status") == "unphysical_output"):
        return {
            "status": "unphysical_prediction",
            "message": predictions.get("message") or "Model generated unphysical figures of merit exceeding thermodynamic or single-junction sanity bounds.",
            "donor": donor_desc,
            "acceptor": acceptor_desc,
            "blend_features": blend_features,
            "device_parameters": device_parameters or {},
            "predictions": None,  # Explicitly None to prevent saving or presenting unphysical data
            "sanity_violations": predictions.get("sanity_violations", []),
            "raw_unphysical_values": predictions.get("raw_unphysical_values", {}),
            "model_metadata": metadata,
        }

    return {
        "status": "success",
        "donor": donor_desc,
        "acceptor": acceptor_desc,
        "blend_features": blend_features,
        "device_parameters": device_parameters or {},
        "predictions": predictions,
        "model_metadata": metadata,
    }
