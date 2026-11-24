import type {
  UserProfile,
  AuthTokenResponse,
  RegisterPayload,
  LoginPayload,
} from '../types/auth';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const TOKEN_STORAGE_KEY = 'opv_predictnet_auth_token';

export function getStoredToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setStoredToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } catch {
    // Storage quota or sandboxed iframe
  }
}

export function removeStoredToken(): void {
  try {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // Storage quota or sandboxed iframe
  }
}

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

export async function registerUser(payload: RegisterPayload): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Registration failed');
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function loginUser(payload: LoginPayload): Promise<AuthTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Authentication failed');
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function fetchUserProfile(token: string): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const errorMsg = await parseErrorMessage(response, 'Failed to retrieve profile');
    throw new Error(errorMsg);
  }

  return response.json();
}
