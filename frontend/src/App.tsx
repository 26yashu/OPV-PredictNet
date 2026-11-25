import { useEffect, useState, useCallback } from 'react';
import './App.css';
import { checkBackendHealth } from './services/api';
import type { BackendHealthResponse, ServiceConnectionStatus } from './types/api';
import { AuthSection } from './components/auth/AuthSection';
import { LandingPage } from './components/landing/LandingPage';
import { PredictionDashboard } from './components/dashboard/PredictionDashboard';
import { getStoredToken } from './services/auth';

type AppView = 'landing' | 'workspace' | 'auth';

function App() {
  const [currentView, setCurrentView] = useState<AppView>('landing');
  const [backendStatus, setBackendStatus] = useState<ServiceConnectionStatus>('checking');
  const [healthData, setHealthData] = useState<BackendHealthResponse | null>(null);
  const [token, setToken] = useState<string | null>(getStoredToken());
  const [showAuthModal, setShowAuthModal] = useState(false);

  const fetchHealth = useCallback(async () => {
    setBackendStatus('checking');
    try {
      const data = await checkBackendHealth();
      setHealthData(data);
      setBackendStatus('connected');
    } catch {
      setHealthData(null);
      setBackendStatus('error');
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    const poll = async () => {
      try {
        const data = await checkBackendHealth();
        if (isMounted) {
          setHealthData(data);
          setBackendStatus('connected');
        }
      } catch {
        if (isMounted) {
          setHealthData(null);
          setBackendStatus('error');
        }
      }
    };

    poll();
    const intervalId = setInterval(poll, 15000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  // Sync token state
  useEffect(() => {
    const handleStorage = () => {
      setToken(getStoredToken());
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  return (
    <div className="app-root-layout">
      {/* Universal Top Navigation Header */}
      <header className="global-navbar">
        <div className="nav-container">
          <div className="nav-brand" onClick={() => setCurrentView('landing')}>
            <span className="brand-logo-icon">&#x2600;</span>
            <div className="brand-text-block">
              <span className="brand-name">OPV-PredictNet</span>
              <span className="brand-tagline">Phase 10 &bull; Scientific Platform</span>
            </div>
          </div>

          <nav className="nav-links">
            <button
              type="button"
              className={`nav-link-btn ${currentView === 'landing' ? 'active' : ''}`}
              onClick={() => setCurrentView('landing')}
            >
              Overview &amp; Science
            </button>
            <button
              type="button"
              className={`nav-link-btn ${currentView === 'workspace' ? 'active' : ''}`}
              onClick={() => setCurrentView('workspace')}
            >
              Prediction Workspace
            </button>
          </nav>

          <div className="nav-right-actions">
            {/* Backend Health Status Badge */}
            <div
              className={`health-badge-indicator ${backendStatus}`}
              title={
                backendStatus === 'connected'
                  ? `FastAPI ${healthData?.version || 'Online'} (Click to re-probe)`
                  : 'Backend Disconnected (Click to re-probe)'
              }
              onClick={fetchHealth}
            >
              <span className={`status-dot ${backendStatus === 'connected' ? 'online-dot' : 'offline-dot'}`} />
              <span className="health-badge-label">
                {backendStatus === 'connected' ? 'API Online' : backendStatus === 'checking' ? 'Probing...' : 'API Offline'}
              </span>
            </div>

            {/* Auth / Account Button */}
            <button
              type="button"
              className="nav-auth-btn"
              onClick={() => setShowAuthModal(true)}
            >
              {token ? 'Researcher Profile' : 'Sign In / Register'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content-flow">
        {currentView === 'landing' && (
          <LandingPage
            onOpenWorkspace={() => setCurrentView('workspace')}
            backendOnline={backendStatus === 'connected'}
          />
        )}

        {currentView === 'workspace' && (
          <PredictionDashboard
            authToken={token}
            onRequireAuth={() => setShowAuthModal(true)}
          />
        )}
      </main>

      {/* Auth Modal Drawer */}
      {showAuthModal && (
        <div className="auth-modal-overlay" onClick={() => setShowAuthModal(false)}>
          <div className="auth-modal-window" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Researcher Authentication</h3>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => {
                  setShowAuthModal(false);
                  setToken(getStoredToken());
                }}
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              <AuthSection />
            </div>
          </div>
        </div>
      )}

      {/* Global Scientific Footer */}
      <footer className="global-scientific-footer">
        <div className="footer-container">
          <div className="footer-left">
            <span className="footer-brand">OPV-PredictNet &bull; Phase 10</span>
            <p className="footer-statement">
              Open-source AI-driven molecular screening and power conversion efficiency prediction platform for Organic Photovoltaics.
            </p>
          </div>
          <div className="footer-center">
            <div className="footer-pills">
              <span>RDKit Chemoinformatics</span>
              <span>Bemis&ndash;Murcko Scaffolds</span>
              <span>Unified Feature Pipeline (Phase 8.4)</span>
              <span>FastAPI Model Serving (Phase 9)</span>
              <span>Conformal Uncertainty</span>
            </div>
          </div>
          <div className="footer-right">
            <span className="footer-disclaimer">
              Research estimates only &bull; Not validated for production device engineering &bull; MIT License
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
