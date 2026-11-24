/**
 * OPV-PredictNet — Prediction & Molecular Analysis Type Definitions
 * Phase 3: RDKit Chemoinformatics and ML Inference Schema
 */

export interface MolecularDescriptorSummary {
  name?: string;
  role: 'donor' | 'acceptor';
  canonical_smiles: string;
  formula: string;
  molecular_weight: number;
  exact_mass: number;
  logp: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotatable_bonds: number;
  aromatic_rings: number;
  fraction_csp3: number;
  heavy_atom_count: number;
  ring_count: number;
}

export interface BlendFeaturesSummary {
  molecular_weight_diff: number;
  logp_diff: number;
  tpsa_total: number;
  aromatic_rings_total: number;
  heavy_atom_count_total: number;
}

export interface MolecularAnalysisResponse {
  donor: MolecularDescriptorSummary;
  acceptor: MolecularDescriptorSummary;
  blend_features: BlendFeaturesSummary;
}

export interface MolecularAnalysisRequest {
  donor_smiles: string;
  acceptor_smiles: string;
  donor_name?: string;
  acceptor_name?: string;
}

export interface DeviceParameters {
  active_layer_thickness_nm?: number;
  d_a_ratio?: string;
  solvent?: string;
  solvent_additive?: string;
  additive_concentration_vol_pct?: number;
  thermal_annealing_temp_c?: number;
  thermal_annealing_time_min?: number;
  hole_transport_layer?: string;
  electron_transport_layer?: string;
  device_architecture?: string;
  substrate?: string;
}

export interface PredictionFiguresOfMerit {
  pce_percent: number;
  voc_volts?: number | null;
  jsc_ma_cm2?: number | null;
  fill_factor_percent?: number | null;
}

export interface ModelMetadata {
  version_tag: string;
  algorithm: string;
  features_used?: string[];
  training_provenance?: string;
  disclaimer?: string;
  limitations?: string[];
}

export interface PredictionRequest {
  donor_smiles: string;
  acceptor_smiles: string;
  donor_name?: string;
  acceptor_name?: string;
  input_method?: string;
  device_parameters?: DeviceParameters;
}

export interface PredictionResponse {
  id?: string | null;
  status: 'completed' | 'model_unavailable' | 'failed';
  message?: string | null;
  donor: MolecularDescriptorSummary;
  acceptor: MolecularDescriptorSummary;
  blend_features: BlendFeaturesSummary;
  device_parameters?: Record<string, unknown> | null;
  predictions?: PredictionFiguresOfMerit | null;
  model_metadata?: ModelMetadata | null;
  created_at?: string | null;
}

export interface PredictionHistoryItem {
  id: string;
  input_method: string;
  donor_name: string;
  donor_smiles: string;
  acceptor_name: string;
  acceptor_smiles: string;
  predicted_pce: number;
  predicted_voc?: number | null;
  predicted_jsc?: number | null;
  predicted_ff?: number | null;
  model_version?: string | null;
  created_at: string;
}

export interface PredictionHistoryResponse {
  total: number;
  items: PredictionHistoryItem[];
}
