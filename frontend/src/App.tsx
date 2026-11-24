import { useEffect, useState, useCallback } from 'react';
import './App.css';
import { checkBackendHealth } from './services/api';
import type { BackendHealthResponse, ServiceConnectionStatus } from './types/api';
import { AuthSection } from './components/auth/AuthSection';
import { PredictionSection } from './components/prediction/PredictionSection';

function App() {
  const [backendStatus, setBackendStatus] = useState<ServiceConnectionStatus>('checking');
  const [healthData, setHealthData] = useState<BackendHealthResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<string | null>(null);

  const fetchHealth = useCallback(async () => {
    setBackendStatus('checking');
    setErrorMessage(null);
    try {
      const data = await checkBackendHealth();
      setHealthData(data);
      setBackendStatus('connected');
      setLastChecked(new Date().toLocaleTimeString());
    } catch (err: unknown) {
      setHealthData(null);
      setBackendStatus('error');
      setLastChecked(new Date().toLocaleTimeString());
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to connect to backend service.');
      }
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
          setErrorMessage(null);
          setLastChecked(new Date().toLocaleTimeString());
        }
      } catch (err: unknown) {
        if (isMounted) {
          setHealthData(null);
          setBackendStatus('error');
          setLastChecked(new Date().toLocaleTimeString());
          if (err instanceof Error) {
            setErrorMessage(err.message);
          } else {
            setErrorMessage('Failed to connect to backend service.');
          }
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

  return (
    <div className="app-container">
      {/* Header & Scientific Objective */}
      <header className="header-section">
        <span className="badge-tag">Phase 3 &bull; Chemoinformatics &amp; ML Inference</span>
        <h1 className="project-title">OPV-PredictNet</h1>
        <h2 className="project-subtitle">
          AI-Driven Molecular Screening &amp; Performance Optimization for Organic Photovoltaics
        </h2>
        <p className="project-description">
          A modular, open-source scientific platform coupling chemoinformatics (RDKit) and machine learning
          (Random Forest regression) to model bulk-heterojunction donor–acceptor active layers.
          Designed to predict device-level figures of merit (PCE, V<sub>OC</sub>, J<sub>SC</sub>, FF) under standard AM1.5G test conditions.
        </p>
      </header>

      {/* Real-Time Operational Service Health */}
      <section>
        <h3 className="section-title">System Status &amp; Integration Health</h3>
        <div className="status-grid">
          {/* Frontend Service Card */}
          <div className="status-card">
            <div>
              <div className="status-card-header">
                <h4 className="status-card-title">Frontend Client</h4>
                <span className="status-indicator online">
                  <span className="status-dot"></span>
                  Active
                </span>
              </div>
              <div className="status-card-body">
                <div className="status-meta-row">
                  <span className="status-meta-label">Framework</span>
                  <span className="status-meta-val">React 19 + TypeScript</span>
                </div>
                <div className="status-meta-row">
                  <span className="status-meta-label">Bundler</span>
                  <span className="status-meta-val">Vite</span>
                </div>
                <div className="status-meta-row">
                  <span className="status-meta-label">Environment</span>
                  <span className="status-meta-val">Development</span>
                </div>
                <div className="status-meta-row">
                  <span className="status-meta-label">Hardware Profile</span>
                  <span className="status-meta-val">8 GB RAM Optimized</span>
                </div>
              </div>
            </div>
          </div>

          {/* Backend Service Card */}
          <div className="status-card">
            <div>
              <div className="status-card-header">
                <h4 className="status-card-title">Backend REST API</h4>
                {backendStatus === 'connected' && (
                  <span className="status-indicator online">
                    <span className="status-dot"></span>
                    Online
                  </span>
                )}
                {backendStatus === 'checking' && (
                  <span className="status-indicator checking">
                    <span className="status-dot"></span>
                    Connecting...
                  </span>
                )}
                {backendStatus === 'error' && (
                  <span className="status-indicator offline">
                    <span className="status-dot"></span>
                    Offline
                  </span>
                )}
              </div>
              <div className="status-card-body">
                {backendStatus === 'connected' && healthData && (
                  <>
                    <div className="status-meta-row">
                      <span className="status-meta-label">Service Name</span>
                      <span className="status-meta-val">{healthData.service}</span>
                    </div>
                    <div className="status-meta-row">
                      <span className="status-meta-label">API Version</span>
                      <span className="status-meta-val">{healthData.version}</span>
                    </div>
                    <div className="status-meta-row">
                      <span className="status-meta-label">Environment</span>
                      <span className="status-meta-val">{healthData.environment}</span>
                    </div>
                    <div className="status-meta-row">
                      <span className="status-meta-label">Server Timestamp</span>
                      <span className="status-meta-val">
                        {new Date(healthData.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </>
                )}

                {backendStatus === 'error' && (
                  <div className="error-box">
                    <strong>Connection Unavailable:</strong> {errorMessage}
                    <div style={{ marginTop: '0.4rem', fontSize: '0.8rem' }}>
                      To start the backend server, run:
                      <br />
                      <code>uvicorn backend.app.main:app --port 8000 --reload</code>
                    </div>
                  </div>
                )}

                {backendStatus === 'checking' && (
                  <p style={{ margin: '0.5rem 0' }}>Probing <code>http://127.0.0.1:8000/api/health</code>...</p>
                )}

                {lastChecked && (
                  <div style={{ marginTop: '0.6rem', fontSize: '0.78rem', color: 'var(--text)' }}>
                    Last probed: {lastChecked}
                  </div>
                )}
              </div>
            </div>

            <button
              className="refresh-button"
              onClick={fetchHealth}
              disabled={backendStatus === 'checking'}
            >
              {backendStatus === 'checking' ? 'Probing...' : 'Re-check Connection'}
            </button>
          </div>
        </div>
      </section>

      {/* Researcher Authentication & Session Management (Phase 2) */}
      <AuthSection />

      {/* Chemoinformatics Analysis & ML Inference (Phase 3) */}
      <PredictionSection />

      {/* Scientific Domain Foundations */}
      <section>
        <h3 className="section-title">Core Scientific Distinctions</h3>
        <div className="concept-grid">
          <div className="concept-card">
            <div className="concept-number">Pillar 1</div>
            <div className="concept-heading">Molecular Candidate</div>
            <div className="concept-body">
              An isolated chemical species (donor or acceptor) defined by chemical graph topology and canonical SMILES.
              Possesses intrinsic properties like HOMO/LUMO levels and molecular weight, but <em>does not have a device efficiency in isolation</em>.
            </div>
          </div>

          <div className="concept-card">
            <div className="concept-number">Pillar 2</div>
            <div className="concept-heading">Material Blend</div>
            <div className="concept-body">
              A donor–acceptor combination forming a bulk-heterojunction active layer.
              Governed by frontier orbital alignment (&Delta;HOMO, &Delta;LUMO), absorption complementarity, and blend stoichiometry (e.g. 1:1.2).
            </div>
          </div>

          <div className="concept-card">
            <div className="concept-number">Pillar 3</div>
            <div className="concept-heading">Physical OPV Device</div>
            <div className="concept-body">
              A physically fabricated multilayer stack (Substrate / Anode / Buffer / Active Layer / Interlayer / Cathode).
              Real PCE is strongly morphology-dependent, dictated by solvent additives (DIO, CN), film thickness, and thermal annealing.
            </div>
          </div>

          <div className="concept-card">
            <div className="concept-number">Pillar 4</div>
            <div className="concept-heading">Predicted Performance</div>
            <div className="concept-body">
              Data-driven statistical regressions (PCE, V<sub>OC</sub>, J<sub>SC</sub>, FF) trained on peer-reviewed experimental device distributions.
              Accompanied by domain-of-applicability constraints and prediction intervals.
            </div>
          </div>
        </div>
      </section>

      {/* Engineering Roadmap */}
      <section className="phase-roadmap">
        <h3 className="section-title" style={{ fontSize: '1.2rem', marginBottom: '0.75rem' }}>
          Engineering Execution Roadmap
        </h3>
        <div className="phase-item">
          <span className="phase-pill active">Phase 1 (Complete)</span>
          <div className="phase-content">
            <h4>Repository Initialization &amp; Architectural Contracts</h4>
            <p>
              Directory structures, API contracts, SQLite schema design, Random Forest ML specification,
              minimal FastAPI backend starter, and real-time health probe.
            </p>
          </div>
        </div>
        <div className="phase-item">
          <span className="phase-pill active">Phase 2 (Completed)</span>
          <div className="phase-content">
            <h4>Database Implementation &amp; JWT Authentication</h4>
            <p>
              SQLAlchemy ORM models, Argon2 password hashing, user registration, JWT access token issuance,
              and authenticated user profiles.
            </p>
          </div>
        </div>
        <div className="phase-item">
          <span className="phase-pill upcoming">Phase 3 (Upcoming)</span>
          <div className="phase-content">
            <h4>RDKit Feature Pipeline &amp; Random Forest Serving</h4>
            <p>
              Scaffold-split dataset curation, RDKit 2D descriptor &amp; blend feature extraction,
              Joblib model serving, and normalized prediction endpoint.
            </p>
          </div>
        </div>
        <div className="phase-item">
          <span className="phase-pill upcoming">Phase 4 (Upcoming)</span>
          <div className="phase-content">
            <h4>Offline Knowledge Assistant &amp; 3D Molecular Visualization</h4>
            <p>
              Local Python scientific knowledge chatbot without paid APIs, 3Dmol.js/Three.js molecular canvas,
              and Recharts performance visualization.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        OPV-PredictNet &bull; Phase 1 Repository Initialization &bull; All dependencies free and open-source.
      </footer>
    </div>
  );
}

export default App;
