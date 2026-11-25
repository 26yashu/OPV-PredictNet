/**
 * OPV-PredictNet — Prediction & Molecular Analysis Service
 * Phase 10: Client for RDKit molecular analysis, Phase 9 FastAPI Model Serving, and History
 */

import type {
  MolecularAnalysisRequest,
  MolecularAnalysisResponse,
  PredictionRequest,
  PredictionResponse,
  Phase9PredictionRequest,
  Phase9PredictionResponse,
  ModelsListResponse,
  PredictionHistoryResponse,
  PredictionHistoryItem,
} from '../types/prediction';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

interface ApiErrorResponse {
  detail?: string | Array<{ loc: string[]; msg: string; type: string }>;
}

async function parseErrorMessage(response: Response, defaultMessage: string): Promise<string> {
  try {
    const errorData: ApiErrorResponse = await response.json();
    if (typeof errorData.detail === 'string') {
      return errorData.detail;
    }
    if (Array.isArray(errorData.detail) && errorData.detail.length > 0) {
      return errorData.detail.map((d) => d.msg).join(', ');
    }
  } catch {
    // Non-JSON response
  }
  return `${defaultMessage} (HTTP ${response.status})`;
}

/**
 * Validate SMILES and calculate 2D RDKit physicochemical descriptors for Donor and Acceptor.
 * Public endpoint — does not require authentication.
 */
export async function analyzeMolecules(payload: MolecularAnalysisRequest): Promise<MolecularAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predictions/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Molecular analysis failed');
    throw new Error(errorMsg);
  }

  return response.json();
}

/**
 * Run full OPV prediction pipeline (Phase 3 legacy contract).
 */
export async function submitPrediction(
  payload: PredictionRequest,
  token?: string | null
): Promise<PredictionResponse> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}/api/predictions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Prediction request failed');
    throw new Error(errorMsg);
  }

  return response.json();
}

/**
 * Fetch available Phase 9 packaged models (M0, M1, M2A, M3), metadata, and quantiles.
 */
export async function fetchPhase9Models(): Promise<ModelsListResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predictions/models`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Failed to fetch model catalog');
    throw new Error(errorMsg);
  }

  return response.json();
}

/**
 * Run Phase 9 PCE prediction with UnifiedFeaturePipeline, conformal uncertainty intervals, and OOD checks.
 * Requires authenticated user JWT bearer token.
 */
export async function submitPhase9Prediction(
  payload: Phase9PredictionRequest,
  token: string
): Promise<Phase9PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predictions/pce`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Prediction calculation failed');
    throw new Error(errorMsg);
  }

  return response.json();
}

/**
 * Fetch authenticated user's prediction history.
 */
export async function getPredictionHistory(
  token: string,
  limit: number = 30,
  offset: number = 0
): Promise<PredictionHistoryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predictions?limit=${limit}&offset=${offset}`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Failed to fetch prediction history');
    throw new Error(errorMsg);
  }

  return response.json();
}

/**
 * Fetch a single prediction record by ID.
 */
export async function getPredictionById(
  predictionId: string,
  token: string
): Promise<PredictionHistoryItem> {
  const response = await fetch(`${API_BASE_URL}/api/predictions/${encodeURIComponent(predictionId)}`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Failed to retrieve prediction details');
    throw new Error(errorMsg);
  }

  return response.json();
}
