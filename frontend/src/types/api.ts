/**
 * API response contract for the health check endpoint.
 */
export interface BackendHealthResponse {
  status: 'healthy' | 'unhealthy' | string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
}

/**
 * Frontend representation of connection state.
 */
export type ServiceConnectionStatus = 'checking' | 'connected' | 'error';
