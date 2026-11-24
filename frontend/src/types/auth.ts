export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  institution?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface RegisterPayload {
  email: string;
  password: string;
  full_name: string;
  institution?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}
