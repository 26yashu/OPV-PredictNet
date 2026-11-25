/**
 * OPV-PredictNet — Prediction & Molecular Analysis Type Definitions
 * Supports both Phase 3 (RDKit Descriptors) and Phase 9 (FastAPI Model Serving)
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

// -----------------------------------------------------------------------------
// Legacy Phase 3 Compatibility Schemas
// -----------------------------------------------------------------------------

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
  status: 'completed' | 'model_unavailable' | 'failed' | 'success' | 'model_unvalidated';
  message?: string | null;
  donor: MolecularDescriptorSummary;
  acceptor: MolecularDescriptorSummary;
  blend_features: BlendFeaturesSummary;
  device_parameters?: Record<string, unknown> | null;
  predictions?: PredictionFiguresOfMerit | null;
  created_at?: string | null;
  feature_schema?: string[];
  feature_values?: Record<string, number>;
  model_metadata?: {
    algorithm?: string;
    version_tag?: string;
    status?: string;
    disclaimer?: string;
    limitations?: string[] | string;
    validation_errors?: string[];
  } | null;
}

// -----------------------------------------------------------------------------
// Phase 9 Authoritative Prediction API Contracts
// -----------------------------------------------------------------------------

export interface Phase9PredictionRequest {
  donor_smiles: string;
  acceptor_smiles: string;
  donor_name?: string;
  acceptor_name?: string;
  input_method?: string;

  // Optional fabrication and device processing metadata
  d_a_ratio?: string;
  active_layer_thickness?: string;
  annealing_temp?: number;
  additive_canonical?: string;
  additive_ratio?: string;
  solvent_canonical?: string;
  device_type?: string;
  etl_canonical?: string;
  htl_canonical?: string;

  // Model & confidence selection
  model_id: 'M0' | 'M1' | 'M2A' | 'M3';
  confidence_level: '80%' | '90%' | '95%';
  allow_dual_scaffold_ood?: boolean;
}

export interface PointPrediction {
  predicted_pce: number;
  unit: string;
}

export interface QualifiedIntervalEstimate {
  prediction_interval_low: number;
  prediction_interval_high: number;
  confidence_level: string;
  conformal_quantile_q: number;
  interval_width: number;
  calibration_strategy: string;
}

export interface OODStatus {
  ood_donor_scaffold: boolean;
  ood_acceptor_scaffold: boolean;
  ood_dual_scaffold: boolean;
  donor_scaffold_known: boolean;
  acceptor_scaffold_known: boolean;
  donor_scaffold_id: string;
  acceptor_scaffold_id: string;
  processing_ood_flags: string[];
  severity: 'none' | 'minor' | 'warning' | 'critical';
}

export interface ScientificSupportStatus {
  research_estimate_only: boolean;
  validated_for_production: boolean;
  support_level: string;
  governance_role: string;
  benchmark_metrics: {
    r2?: number;
    mae?: number;
    rmse?: number;
    evaluation_partition?: string;
    [key: string]: unknown;
  };
  limitations: string;
}

export interface Phase9PredictionResponse {
  id?: string | null;
  created_at?: string | null;
  model_id: 'M0' | 'M1' | 'M2A' | 'M3';
  model_version: string;
  algorithm: string;
  point_prediction: PointPrediction;
  interval_estimate: QualifiedIntervalEstimate;
  ood_status: OODStatus;
  scientific_support: ScientificSupportStatus;
  warnings: string[];
  donor: {
    name: string;
    canonical_smiles: string;
    scaffold_id: string;
  };
  acceptor: {
    name: string;
    canonical_smiles: string;
    scaffold_id: string;
  };
  processing_parameters: Record<string, unknown>;
  research_estimate_only: boolean;
  validated_for_production: boolean;
}

export interface ModelDetail {
  version_tag: string;
  algorithm: string;
  n_features: number;
  conformal_quantiles: Record<string, number>;
  calibration_strategy: string;
  evaluation_metrics: {
    r2: number;
    mae: number;
    rmse: number;
    pce_r2?: number;
    pce_mae?: number;
    pce_rmse?: number;
    evaluation_partition?: string;
    phase_8_3_governance_role?: string;
  };
  research_estimate_only: boolean;
  validated_for_production: boolean;
}

export interface ModelsListResponse {
  models: {
    M0?: ModelDetail;
    M1?: ModelDetail;
    M2A?: ModelDetail;
    M3?: ModelDetail;
    [key: string]: ModelDetail | undefined;
  };
  n_donor_scaffolds: number;
  n_acceptor_scaffolds: number;
  research_estimate_only: boolean;
  validated_for_production: boolean;
  load_errors: string[];
}

// -----------------------------------------------------------------------------
// History Schemas
// -----------------------------------------------------------------------------

export interface ParsedUncertainty {
  model_id: string;
  interval_estimate: QualifiedIntervalEstimate;
  ood_status: OODStatus;
  scientific_support: ScientificSupportStatus;
  warnings: string[];
  research_estimate_only: boolean;
  validated_for_production: boolean;
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
  uncertainty_json?: string | null;
  model_version?: string | null;
  created_at: string;
}

export interface PredictionHistoryResponse {
  total: number;
  items: PredictionHistoryItem[];
}
