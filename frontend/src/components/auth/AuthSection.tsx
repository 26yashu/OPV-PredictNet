import React, { useState, useEffect, useCallback } from 'react';
import type { UserProfile, RegisterPayload, LoginPayload } from '../../types/auth';
import {
  registerUser,
  loginUser,
  fetchUserProfile,
  getStoredToken,
  setStoredToken,
  removeStoredToken,
} from '../../services/auth';

export const AuthSection: React.FC = () => {
  const [token, setToken] = useState<string | null>(getStoredToken());
  const [user, setUser] = useState<UserProfile | null>(null);
  const [mode, setMode] = useState<'login' | 'register'>('login');

  // Form states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [institution, setInstitution] = useState('');

  // Status and feedback states
  const [loading, setLoading] = useState(false);
  const [sessionChecking, setSessionChecking] = useState<boolean>(() => Boolean(getStoredToken()));
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Validate session on manual re-check
  const verifySession = useCallback(async (authToken: string) => {
    setSessionChecking(true);
    setErrorMessage(null);
    try {
      const profile = await fetchUserProfile(authToken);
      setUser(profile);
      setSuccessMessage(null);
    } catch (err: unknown) {
      removeStoredToken();
      setToken(null);
      setUser(null);
      if (err instanceof Error) {
        setErrorMessage(`Session expired or invalid: ${err.message}`);
      }
    } finally {
      setSessionChecking(false);
    }
  }, []);

  // Restore authenticated session on mount without cascading synchronous setState
  useEffect(() => {
    const savedToken = getStoredToken();
    if (!savedToken) return;

    let isMounted = true;
    fetchUserProfile(savedToken)
      .then((profile) => {
        if (isMounted) {
          setUser(profile);
          setSessionChecking(false);
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          removeStoredToken();
          setToken(null);
          setUser(null);
          setSessionChecking(false);
          if (err instanceof Error) {
            setErrorMessage(`Session expired or invalid: ${err.message}`);
          }
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleLogout = () => {
    removeStoredToken();
    setToken(null);
    setUser(null);
    setSuccessMessage('You have been signed out successfully.');
    setErrorMessage(null);
    setPassword('');
  };

  const validateRegistrationInputs = (): string | null => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      return 'Please enter a valid email address (e.g., user@domain.com).';
    }
    if (!fullName.trim()) {
      return 'Full name is required.';
    }
    if (password.length < 8) {
      return 'Password must be at least 8 characters long.';
    }
    if (!/[A-Z]/.test(password)) {
      return 'Password must contain at least one uppercase letter (A-Z).';
    }
    if (!/[a-z]/.test(password)) {
      return 'Password must contain at least one lowercase letter (a-z).';
    }
    if (!/[0-9]/.test(password)) {
      return 'Password must contain at least one number (0-9).';
    }
    return null;
  };

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);

    const validationError = validateRegistrationInputs();
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }

    setLoading(true);
    try {
      const payload: RegisterPayload = {
        email: email.trim(),
        password,
        full_name: fullName.trim(),
        institution: institution.trim() || undefined,
      };
      await registerUser(payload);
      setSuccessMessage('Registration successful! You may now sign in with your credentials.');
      setMode('login');
      setPassword('');
    } catch (err: unknown) {
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('An unexpected error occurred during registration.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);

    if (!email.trim() || !password) {
      setErrorMessage('Please provide both email and password.');
      return;
    }

    setLoading(true);
    try {
      const payload: LoginPayload = {
        email: email.trim(),
        password,
      };
      const tokenResponse = await loginUser(payload);
      setStoredToken(tokenResponse.access_token);
      setToken(tokenResponse.access_token);
      await verifySession(tokenResponse.access_token);
      setPassword('');
      setSuccessMessage('Authentication successful. Welcome to OPV-PredictNet.');
    } catch (err: unknown) {
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Authentication failed.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="auth-section">
      <div className="auth-header">
        <h3 className="section-title" style={{ marginBottom: '0.25rem' }}>
          Researcher Authentication &amp; Profile (Phase 2)
        </h3>
        <p className="auth-subtitle">
          Protected by Argon2 password hashing &amp; signed JWT access tokens with SQLite persistence.
        </p>
      </div>

      {/* Global Alerts */}
      {errorMessage && (
        <div className="auth-alert error">
          <strong>Error:</strong> {errorMessage}
        </div>
      )}
      {successMessage && (
        <div className="auth-alert success">
          <strong>Notice:</strong> {successMessage}
        </div>
      )}

      {/* Case 1: Authenticated User Profile */}
      {user && (
        <div className="profile-card">
          <div className="profile-header">
            <div>
              <span className="profile-badge">Active Session</span>
              <h4 className="profile-name">{user.full_name}</h4>
              <p className="profile-email">{user.email}</p>
            </div>
            <button className="auth-button outline danger" onClick={handleLogout}>
              Sign Out
            </button>
          </div>

          <div className="profile-details-grid">
            <div className="profile-field">
              <span className="field-label">User ID</span>
              <span className="field-val mono">{user.id}</span>
            </div>
            <div className="profile-field">
              <span className="field-label">Institution</span>
              <span className="field-val">{user.institution || 'Independent Researcher'}</span>
            </div>
            <div className="profile-field">
              <span className="field-label">Account Status</span>
              <span className="field-val status-active">
                {user.is_active ? 'Active & Verified' : 'Deactivated'}
              </span>
            </div>
            <div className="profile-field">
              <span className="field-label">Registered On</span>
              <span className="field-val">{new Date(user.created_at).toLocaleString()}</span>
            </div>
          </div>

          <div className="profile-footer">
            <span className="token-indicator">
              Authenticated via <code>GET /api/auth/me</code> (Bearer JWT)
            </span>
            <button
              className="refresh-session-btn"
              onClick={() => token && verifySession(token)}
              disabled={sessionChecking}
            >
              {sessionChecking ? 'Re-verifying...' : 'Re-verify Session'}
            </button>
          </div>
        </div>
      )}

      {/* Case 2: Unauthenticated - Login or Register Forms */}
      {!user && (
        <div className="auth-card">
          <div className="auth-tabs">
            <button
              className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
              onClick={() => {
                setMode('login');
                setErrorMessage(null);
              }}
            >
              Sign In
            </button>
            <button
              className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
              onClick={() => {
                setMode('register');
                setErrorMessage(null);
              }}
            >
              Register New Account
            </button>
          </div>

          {mode === 'login' ? (
            <form onSubmit={handleLoginSubmit} className="auth-form">
              <div className="form-group">
                <label htmlFor="login-email">Email Address</label>
                <input
                  id="login-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="researcher@laboratory.org"
                  required
                  autoComplete="email"
                  disabled={loading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="login-password">Password</label>
                <input
                  id="login-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  autoComplete="current-password"
                  disabled={loading}
                />
              </div>

              <button type="submit" className="auth-button primary" disabled={loading}>
                {loading ? 'Authenticating...' : 'Sign In with JWT'}
              </button>
            </form>
          ) : (
            <form onSubmit={handleRegisterSubmit} className="auth-form">
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="reg-name">Full Name</label>
                  <input
                    id="reg-name"
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="Dr. Eleanor Vance"
                    required
                    disabled={loading}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="reg-institution">Institution (Optional)</label>
                  <input
                    id="reg-institution"
                    type="text"
                    value={institution}
                    onChange={(e) => setInstitution(e.target.value)}
                    placeholder="Institute of Solar Energy"
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="reg-email">Email Address</label>
                <input
                  id="reg-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="researcher@laboratory.org"
                  required
                  autoComplete="email"
                  disabled={loading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="reg-password">Password</label>
                <input
                  id="reg-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Min 8 chars, 1 uppercase, 1 lowercase, 1 digit"
                  required
                  autoComplete="new-password"
                  disabled={loading}
                />
                <span className="field-hint">
                  Must contain &ge;8 characters including uppercase, lowercase, and numeric digits. Hashed with Argon2id.
                </span>
              </div>

              <button type="submit" className="auth-button primary" disabled={loading}>
                {loading ? 'Creating Account...' : 'Create Account & Hash Password'}
              </button>
            </form>
          )}
        </div>
      )}
    </section>
  );
};
