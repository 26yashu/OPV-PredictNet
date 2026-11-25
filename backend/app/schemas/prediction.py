from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class MolecularAnalysisRequest(BaseModel):
    donor_smiles: str = Field(..., min_length=1, description="SMILES string of donor molecule")
    acceptor_smiles: str = Field(..., min_length=1, description="SMILES string of acceptor molecule")
    donor_name: Optional[str] = Field("Donor Material", description="Display name for donor")
    acceptor_name: Optional[str] = Field("Acceptor Material", description="Display name for acceptor")


class MolecularDescriptorSummary(BaseModel):
    role: str
    canonical_smiles: str
    formula: str
    molecular_weight: float
    exact_mass: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    aromatic_rings: int
    fraction_csp3: float
    heavy_atom_count: int
    ring_count: int
    name: Optional[str] = None


class BlendFeaturesSummary(BaseModel):
    molecular_weight_diff: float
    logp_diff: float
    tpsa_total: float
    aromatic_rings_total: int
    heavy_atom_count_total: int


class MolecularAnalysisResponse(BaseModel):
    donor: Dict[str, Any]
    acceptor: Dict[str, Any]
    blend_features: Dict[str, Any]


class PredictionRequest(BaseModel):
    donor_smiles: str = Field(..., min_length=1, description="SMILES of donor")
    acceptor_smiles: str = Field(..., min_length=1, description="SMILES of acceptor")
    donor_name: Optional[str] = Field("Donor Material", description="Name of donor material")
    acceptor_name: Optional[str] = Field("Acceptor Material", description="Name of acceptor material")
    input_method: Optional[Literal["smiles", "file_upload", "library_selection"]] = Field("smiles", description="Input method: smiles, file_upload, or library_selection")
    device_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Fabrication conditions & device stack. Note: Experimental processing parameters are "
            "recorded for provenance but are not currently included in the 22-feature molecular descriptor model."
        )
    )


class PredictionResponse(BaseModel):
    id: Optional[str] = None
    status: str
    message: Optional[str] = None
    donor: Dict[str, Any]
    acceptor: Dict[str, Any]
    blend_features: Dict[str, Any]
    device_parameters: Optional[Dict[str, Any]] = None
    predictions: Optional[Dict[str, Any]] = None
    model_metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class PredictionHistoryItem(BaseModel):
    id: str
    input_method: str
    donor_name: str
    donor_smiles: str
    acceptor_name: str
    acceptor_smiles: str
    predicted_pce: float
    predicted_voc: Optional[float] = None
    predicted_jsc: Optional[float] = None
    predicted_ff: Optional[float] = None
    uncertainty_json: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionHistoryResponse(BaseModel):
    total: int
    items: List[PredictionHistoryItem]


# =============================================================================
# Phase 9 Prediction Schemas
# =============================================================================

class Phase9PredictionRequest(BaseModel):
    donor_smiles: str = Field(..., min_length=1, description="SMILES string of donor molecule")
    acceptor_smiles: str = Field(..., min_length=1, description="SMILES string of acceptor molecule")
    donor_name: Optional[str] = Field("Donor Material", description="Display name for donor")
    acceptor_name: Optional[str] = Field("Acceptor Material", description="Display name for acceptor")
    input_method: Optional[Literal["smiles", "file_upload", "library_selection"]] = Field("smiles", description="Input method: smiles, file_upload, or library_selection")

    # Optional processing metadata
    d_a_ratio: Optional[str] = Field(None, description="Donor:Acceptor weight ratio, e.g. '1:1'")
    active_layer_thickness: Optional[str] = Field(None, description="Active layer thickness, e.g. '100nm'")
    annealing_temp: Optional[float] = Field(None, description="Thermal annealing temperature in Celsius")
    additive_canonical: Optional[str] = Field(None, description="Solvent additive name, e.g. 'DIO', 'CN'")
    additive_ratio: Optional[str] = Field(None, description="Additive concentration, e.g. '3%'")
    solvent_canonical: Optional[str] = Field(None, description="Host solvent, e.g. 'chlorobenzene'")
    device_type: Optional[str] = Field(None, description="Device architecture: 'conventional' or 'inverted'")
    etl_canonical: Optional[str] = Field(None, description="Electron transport layer, e.g. 'ZnO', 'PDINN'")
    htl_canonical: Optional[str] = Field(None, description="Hole transport layer, e.g. 'PEDOT:PSS', 'MoO3'")

    # Model & uncertainty options
    model_id: Literal["M0", "M1", "M2A", "M3"] = Field("M3", description="Approved Phase 8.3 model ID: M0, M1, M2A, or M3")
    confidence_level: Literal["80%", "90%", "95%"] = Field("90%", description="Conformal interval confidence level: 80%, 90%, or 95%")
    allow_dual_scaffold_ood: bool = Field(
        False,
        description="Whether to permit unsupported dual-scaffold OOD extrapolation (rejected by default)"
    )


class PointPrediction(BaseModel):
    predicted_pce: float
    unit: str = "%"


class QualifiedIntervalEstimate(BaseModel):
    prediction_interval_low: float
    prediction_interval_high: float
    confidence_level: str
    conformal_quantile_q: float
    interval_width: float
    calibration_strategy: str


class OODStatus(BaseModel):
    ood_donor_scaffold: bool
    ood_acceptor_scaffold: bool
    ood_dual_scaffold: bool
    donor_scaffold_known: bool
    acceptor_scaffold_known: bool
    donor_scaffold_id: str
    acceptor_scaffold_id: str
    processing_ood_flags: List[str]
    severity: str


class ScientificSupportStatus(BaseModel):
    research_estimate_only: bool = True
    validated_for_production: bool = False
    support_level: str
    governance_role: str
    benchmark_metrics: Dict[str, Any]
    limitations: str


class Phase9PredictionResponse(BaseModel):
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    model_id: str
    model_version: str
    algorithm: str
    point_prediction: PointPrediction
    interval_estimate: QualifiedIntervalEstimate
    ood_status: OODStatus
    scientific_support: ScientificSupportStatus
    warnings: List[str]
    donor: Dict[str, Any]
    acceptor: Dict[str, Any]
    processing_parameters: Dict[str, Any]
    research_estimate_only: bool = True
    validated_for_production: bool = False
