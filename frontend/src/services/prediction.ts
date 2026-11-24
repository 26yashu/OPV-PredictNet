/**
 * OPV-PredictNet — Prediction & Molecular Analysis Service
 * Phase 3: Client for RDKit molecular analysis, inference pipeline, and prediction history
 */

import type {
  MolecularAnalysisRequest,
  MolecularAnalysisResponse,
  PredictionRequest,
  PredictionResponse,
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
 * Run full OPV prediction pipeline. If an auth token is provided, persists the run to the user's history.
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
 * Fetch authenticated user's prediction history.
 */
export async function getPredictionHistory(
  token: string,
  limit: number = 20,
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
