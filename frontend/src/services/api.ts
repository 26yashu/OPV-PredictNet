import type { BackendHealthResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Fetches the operational health status from the backend FastAPI service.
 */
export async function checkBackendHealth(): Promise<BackendHealthResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Backend returned HTTP status ${response.status}: ${response.statusText}`);
    }

    const data: BackendHealthResponse = await response.json();
    return data;
  } catch (error: unknown) {
    clearTimeout(timeoutId);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('An unknown network error occurred while connecting to backend.');
  }
}
