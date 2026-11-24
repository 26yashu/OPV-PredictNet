from datetime import datetime
from typing import Any, Dict, List, Optional
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
    input_method: Optional[str] = Field("smiles", description="Input method: smiles, file_upload, or library_selection")
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
    model_version: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionHistoryResponse(BaseModel):
    total: int
    items: List[PredictionHistoryItem]
