import React, { useState, useEffect, useCallback } from 'react';
import type {
  Phase9PredictionRequest,
  Phase9PredictionResponse,
  ModelsListResponse,
  PredictionHistoryItem,
  MolecularAnalysisResponse,
  ParsedUncertainty,
} from '../../types/prediction';
import {
  fetchPhase9Models,
  submitPhase9Prediction,
  analyzeMolecules,
  getPredictionHistory,
} from '../../services/prediction';
import { getStoredToken } from '../../services/auth';

interface PredictionDashboardProps {
  authToken: string | null;
  onRequireAuth: () => void;
}

interface BenchmarkPreset {
  name: string;
  category: string;
  donorName: string;
  donorSmiles: string;
  acceptorName: string;
  acceptorSmiles: string;
  daRatio: string;
  thickness: string;
  annealingTemp: number;
  solvent: string;
  additive: string;
  additiveRatio: string;
  deviceType: string;
  htl: string;
  etl: string;
}

const BENCHMARK_PRESETS: BenchmarkPreset[] = [
  {
    name: 'PM6 : Y6 (High Efficiency Fluorinated Benchmark)',
    category: 'High Efficiency NFA',
    donorName: 'PM6 (PBDB-T-2F Repeat Unit)',
    donorSmiles: 'Cc1cc(C)c(-c2ccc(-c3nnc4sccc34)s2)s1',
    acceptorName: 'Y6 (BTP-4F Core)',
    acceptorSmiles: 'O=C1c2ccccc2C(=O)C1=Cc1ccc(-c2ccc(C=C3C(=O)c4ccccc4C3=O)s2)s1',
    daRatio: '1:1.2',
    thickness: '100nm',
    annealingTemp: 110.0,
    solvent: 'chloroform',
    additive: 'CN',
    additiveRatio: '0.5%',
    deviceType: 'conventional',
    htl: 'PEDOT:PSS',
    etl: 'PDINN',
  },
  {
    name: 'D18 : Y6 (High Mobility Oligomer Blend)',
    category: 'High Performance',
    donorName: 'D18 (DTTC Core Subunit)',
    donorSmiles: 'c1sc2c(c1)-c1cc3sc4c5scc(C)c5sc4c3cc1C2',
    acceptorName: 'Y6 (BTP-4F Core)',
    acceptorSmiles: 'O=C1c2ccccc2C(=O)C1=Cc1ccc(-c2ccc(C=C3C(=O)c4ccccc4C3=O)s2)s1',
    daRatio: '1:1',
    thickness: '105nm',
    annealingTemp: 100.0,
    solvent: 'chloroform',
    additive: 'none',
    additiveRatio: '0%',
    deviceType: 'inverted',
    htl: 'MoO3',
    etl: 'ZnO',
  },
  {
    name: 'P3HT : PC61BM (Classic Polymer-Fullerene Benchmark)',
    category: 'Fullerene Reference',
    donorName: 'P3HT (3-Hexylthiophene Repeat Unit)',
    donorSmiles: 'CCCCCCc1cccs1',
    acceptorName: 'PC61BM Core',
    acceptorSmiles: 'c1ccc2nsnc2c1',
    daRatio: '1:1',
    thickness: '120nm',
    annealingTemp: 150.0,
    solvent: 'chlorobenzene',
    additive: 'none',
    additiveRatio: '0%',
    deviceType: 'conventional',
    htl: 'PEDOT:PSS',
    etl: 'Ca',
  },
  {
    name: 'Novel / OOD Candidate (Demonstrates OOD Detection)',
    category: 'Novel Scaffold',
    donorName: 'Norbornane Test Scaffold (Unseen)',
    donorSmiles: 'C1CC2CCC1C2',
    acceptorName: 'Subphthalocyanine Core (Known)',
    acceptorSmiles: 'B1n2c3c4ccccc4c2N=c2c4ccccc4c(n21)=NC1=NC(=N3)c2ccccc21',
    daRatio: '1:1',
    thickness: '95nm',
    annealingTemp: 80.0,
    solvent: 'chlorobenzene',
    additive: 'DIO',
    additiveRatio: '1%',
    deviceType: 'inverted',
    htl: 'MoO3',
    etl: 'ZnO',
  },
];

export const PredictionDashboard: React.FC<PredictionDashboardProps> = ({
  authToken,
  onRequireAuth,
}) => {
  // Model catalog
  const [modelCatalog, setModelCatalog] = useState<ModelsListResponse | null>(null);
  const [loadingModels, setLoadingModels] = useState(true);

  // Form inputs
  const [donorSmiles, setDonorSmiles] = useState(BENCHMARK_PRESETS[0].donorSmiles);
  const [donorName, setDonorName] = useState(BENCHMARK_PRESETS[0].donorName);
  const [acceptorSmiles, setAcceptorSmiles] = useState(BENCHMARK_PRESETS[0].acceptorSmiles);
  const [acceptorName, setAcceptorName] = useState(BENCHMARK_PRESETS[0].acceptorName);

  // Model & confidence
  const [modelId, setModelId] = useState<'M0' | 'M1' | 'M2A' | 'M3'>('M3');
  const [confidenceLevel, setConfidenceLevel] = useState<'80%' | '90%' | '95%'>('90%');
  const [allowDualScaffoldOod, setAllowDualScaffoldOod] = useState(false);

  // Fabrication metadata
  const [showAdvancedParams, setShowAdvancedParams] = useState(true);
  const [daRatio, setDaRatio] = useState('1:1.2');
  const [thickness, setThickness] = useState('100nm');
  const [annealingTemp, setAnnealingTemp] = useState<number | ''>(110);
  const [solvent, setSolvent] = useState('chloroform');
  const [additive, setAdditive] = useState('CN');
  const [additiveRatio, setAdditiveRatio] = useState('0.5%');
  const [deviceType, setDeviceType] = useState('conventional');
  const [etl, setEtl] = useState('PDINN');
  const [htl, setHtl] = useState('PEDOT:PSS');

  // Execution states
  const [isPredicting, setIsPredicting] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResponse, setPredictionResponse] = useState<Phase9PredictionResponse | null>(null);
  const [analysisResponse, setAnalysisResponse] = useState<MolecularAnalysisResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'prediction' | 'descriptors' | 'history'>('prediction');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // History state
  const [historyItems, setHistoryItems] = useState<PredictionHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  // Load models on mount
  useEffect(() => {
    let mounted = true;
    fetchPhase9Models()
      .then((cat) => {
        if (mounted) {
          setModelCatalog(cat);
          setLoadingModels(false);
        }
      })
      .catch((err: unknown) => {
        if (mounted) {
          console.warn('Could not fetch model catalog:', err);
          setLoadingModels(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, []);

  // Fetch prediction history on mount or when token updates
  useEffect(() => {
    if (!authToken) {
      return;
    }

    let mounted = true;
    getPredictionHistory(authToken, 25, 0)
      .then((res) => {
        if (mounted) {
          setHistoryItems(res.items);
          setHistoryLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (mounted) {
          if (err instanceof Error) {
            setHistoryError(err.message);
          } else {
            setHistoryError('Failed to fetch prediction history.');
          }
          setHistoryLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, [authToken]);

  // Manual reload for button click or post-submission
  const reloadHistory = useCallback(async () => {
    const token = authToken || getStoredToken();
    if (!token) return;

    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await getPredictionHistory(token, 25, 0);
      setHistoryItems(res.items);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setHistoryError(err.message);
      } else {
        setHistoryError('Failed to fetch prediction history.');
      }
    } finally {
      setHistoryLoading(false);
    }
  }, [authToken]);

  const handleApplyPreset = (p: BenchmarkPreset) => {
    setDonorName(p.donorName);
    setDonorSmiles(p.donorSmiles);
    setAcceptorName(p.acceptorName);
    setAcceptorSmiles(p.acceptorSmiles);
    setDaRatio(p.daRatio);
    setThickness(p.thickness);
    setAnnealingTemp(p.annealingTemp);
    setSolvent(p.solvent);
    setAdditive(p.additive);
    setAdditiveRatio(p.additiveRatio);
    setDeviceType(p.deviceType);
    setHtl(p.htl);
    setEtl(p.etl);
    setErrorMessage(null);
  };

  const handlePreAnalyze = async () => {
    if (!donorSmiles.trim() || !acceptorSmiles.trim()) {
      setErrorMessage('Please provide both Donor and Acceptor SMILES strings.');
      return;
    }
    setErrorMessage(null);
    setIsAnalyzing(true);
    try {
      const res = await analyzeMolecules({
        donor_smiles: donorSmiles.trim(),
        acceptor_smiles: acceptorSmiles.trim(),
        donor_name: donorName.trim() || undefined,
        acceptor_name: acceptorName.trim() || undefined,
      });
      setAnalysisResponse(res);
      setActiveTab('descriptors');
    } catch (err: unknown) {
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Molecular descriptor calculation failed.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRunPrediction = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    const token = authToken || getStoredToken();
    if (!token) {
      onRequireAuth();
      return;
    }

    if (!donorSmiles.trim() || !acceptorSmiles.trim()) {
      setErrorMessage('Donor and Acceptor SMILES are required.');
      return;
    }

    setIsPredicting(true);
    const reqPayload: Phase9PredictionRequest = {
      donor_smiles: donorSmiles.trim(),
      acceptor_smiles: acceptorSmiles.trim(),
      donor_name: donorName.trim() || undefined,
      acceptor_name: acceptorName.trim() || undefined,
      d_a_ratio: daRatio.trim() || undefined,
      active_layer_thickness: thickness.trim() || undefined,
      annealing_temp: annealingTemp !== '' ? Number(annealingTemp) : undefined,
      additive_canonical: additive.trim() !== 'none' ? additive.trim() : undefined,
      additive_ratio: additiveRatio.trim() || undefined,
      solvent_canonical: solvent.trim() || undefined,
      device_type: deviceType.trim() || undefined,
      etl_canonical: etl.trim() || undefined,
      htl_canonical: htl.trim() || undefined,
      model_id: modelId,
      confidence_level: confidenceLevel,
      allow_dual_scaffold_ood: allowDualScaffoldOod,
    };

    try {
      const res = await submitPhase9Prediction(reqPayload, token);
      setPredictionResponse(res);
      setActiveTab('prediction');
      reloadHistory();
    } catch (err: unknown) {
      setPredictionResponse(null);
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Inference pipeline execution failed.');
      }
    } finally {
      setIsPredicting(false);
    }
  };

  const parseHistoryUncertainty = (rawJson?: string | null): ParsedUncertainty | null => {
    if (!rawJson) return null;
    try {
      return JSON.parse(rawJson);
    } catch {
      return null;
    }
  };

  return (
    <div className="dashboard-container">
      {/* Workspace Header */}
      <div className="workspace-header">
        <div className="workspace-title-group">
          <div className="badge-row">
            <span className="badge-tag cyan-glow">Scientific Workspace</span>
            <span className="badge-tag slate-tag">Phase 9 FastAPI Serving</span>
            {!authToken && (
              <span className="badge-tag warning-badge">
                Sign-In Required for Live Inference
              </span>
            )}
          </div>
          <h2 className="workspace-title">OPV Molecular Screening &amp; PCE Inference</h2>
          <p className="workspace-subtitle">
            Configure molecular pairs, select governed predictive models, and inspect conformal uncertainty intervals.
          </p>
        </div>

        {/* Quick Tabs */}
        <div className="workspace-nav-tabs">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'prediction' ? 'active' : ''}`}
            onClick={() => setActiveTab('prediction')}
          >
            Prediction Diagnostics
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'descriptors' ? 'active' : ''}`}
            onClick={() => setActiveTab('descriptors')}
          >
            RDKit Descriptors {analysisResponse && '✓'}
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            History ({historyItems.length})
          </button>
        </div>
      </div>

      {/* Preset Molecules Selector */}
      <div className="preset-selector-card">
        <div className="preset-selector-header">
          <span className="preset-title">Load Peer-Reviewed Reference Blend:</span>
          <span className="preset-hint">Populates verified canonical SMILES &amp; fabrication conditions</span>
        </div>
        <div className="preset-buttons-row">
          {BENCHMARK_PRESETS.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              className="preset-chip"
              onClick={() => handleApplyPreset(preset)}
            >
              <span className="preset-chip-cat">{preset.category}</span>
              <span className="preset-chip-name">{preset.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Error & Warning Alert Banner */}
      {errorMessage && (
        <div className="dashboard-alert error-alert">
          <div className="alert-icon">&#x26A0;</div>
          <div className="alert-content">
            <h4 className="alert-title">Input Validation or Scientific Constraint Notice</h4>
            <p className="alert-desc">{errorMessage}</p>
          </div>
        </div>
      )}

      {/* Main Two-Column Scientific Layout */}
      <div className="dashboard-grid">
        {/* Left Column: Input Formulation */}
        <div className="dashboard-form-col">
          <form onSubmit={handleRunPrediction} className="scientific-form">
            {/* Molecular Pair Inputs */}
            <div className="form-card">
              <div className="form-card-header">
                <span className="card-step-badge">1</span>
                <div>
                  <h3 className="card-title">Molecular Structure Inputs (SMILES)</h3>
                  <p className="card-sub">Chemical topology evaluated with strict RDKit validation</p>
                </div>
              </div>

              {/* Donor Input */}
              <div className="input-group">
                <div className="label-row">
                  <label htmlFor="donor-smiles">Donor Canonical SMILES</label>
                  <span className="field-type">Polymer / Small Molecule</span>
                </div>
                <input
                  id="donor-smiles"
                  type="text"
                  className="code-input"
                  value={donorSmiles}
                  onChange={(e) => setDonorSmiles(e.target.value)}
                  placeholder="e.g. CCCCCc1cccs1"
                  required
                />
                <input
                  type="text"
                  className="sub-input"
                  value={donorName}
                  onChange={(e) => setDonorName(e.target.value)}
                  placeholder="Donor Material Label (e.g. PM6 / P3HT)"
                />
              </div>

              {/* Acceptor Input */}
              <div className="input-group" style={{ marginTop: '1.2rem' }}>
                <div className="label-row">
                  <label htmlFor="acceptor-smiles">Acceptor Canonical SMILES</label>
                  <span className="field-type">NFA / Fullerene</span>
                </div>
                <input
                  id="acceptor-smiles"
                  type="text"
                  className="code-input"
                  value={acceptorSmiles}
                  onChange={(e) => setAcceptorSmiles(e.target.value)}
                  placeholder="e.g. c1ccc2nsnc2c1"
                  required
                />
                <input
                  type="text"
                  className="sub-input"
                  value={acceptorName}
                  onChange={(e) => setAcceptorName(e.target.value)}
                  placeholder="Acceptor Material Label (e.g. Y6 / ITIC / PCBM)"
                />
              </div>

              <div className="form-actions-inline">
                <button
                  type="button"
                  className="secondary-action-btn"
                  onClick={handlePreAnalyze}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? 'Computing RDKit Descriptors...' : 'Analyze 2D Descriptors & Blend'}
                </button>
              </div>
            </div>

            {/* Model & Uncertainty Selection */}
            <div className="form-card">
              <div className="form-card-header">
                <span className="card-step-badge">2</span>
                <div>
                  <h3 className="card-title">Governed Model &amp; Uncertainty Calibration</h3>
                  <p className="card-sub">Select Phase 8.3 approved model configuration</p>
                </div>
              </div>

              <div className="model-select-grid">
                {[
                  {
                    id: 'M0' as const,
                    name: 'M0 Baseline',
                    features: '22 Feats',
                    desc: 'Molecular descriptors only (baseline)',
                  },
                  {
                    id: 'M1' as const,
                    name: 'M1 One-Hot',
                    features: '50 Feats',
                    desc: 'Categorical & continuous processing',
                  },
                  {
                    id: 'M2A' as const,
                    name: 'M2A Continuous',
                    features: '37 Feats',
                    desc: 'Continuous processing reference',
                  },
                  {
                    id: 'M3' as const,
                    name: 'M3 Native-NaN (Recommended)',
                    features: '44 Feats',
                    desc: 'HistGradientBoosting with native missingness',
                    recommended: true,
                  },
                ].map((m) => (
                  <label
                    key={m.id}
                    className={`model-option-card ${modelId === m.id ? 'selected' : ''} ${
                      m.recommended ? 'recommended' : ''
                    }`}
                  >
                    <input
                      type="radio"
                      name="modelSelection"
                      value={m.id}
                      checked={modelId === m.id}
                      onChange={() => setModelId(m.id)}
                    />
                    <div className="model-option-content">
                      <div className="model-option-head">
                        <span className="model-option-name">{m.name}</span>
                        <span className="model-option-feat">{m.features}</span>
                      </div>
                      <p className="model-option-desc">{m.desc}</p>
                    </div>
                  </label>
                ))}
              </div>

              {/* Confidence Level */}
              <div className="confidence-level-row">
                <label className="confidence-label">Conformal Prediction Confidence Level:</label>
                <div className="confidence-buttons">
                  {(['80%', '90%', '95%'] as const).map((lvl) => (
                    <button
                      key={lvl}
                      type="button"
                      className={`conf-btn ${confidenceLevel === lvl ? 'active' : ''}`}
                      onClick={() => setConfidenceLevel(lvl)}
                    >
                      {lvl} Nominal
                    </button>
                  ))}
                </div>
              </div>

              {/* Allow Dual-Scaffold OOD Toggle */}
              <div className="dual-ood-toggle-box">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={allowDualScaffoldOod}
                    onChange={(e) => setAllowDualScaffoldOod(e.target.checked)}
                  />
                  <span>
                    Permit Dual-Scaffold OOD Extrapolation (Strategy C1, $R^2 \approx 0.08$)
                  </span>
                </label>
                <p className="toggle-explanation">
                  By default, the Phase 9 backend rejects inputs where <em>both</em> donor and acceptor scaffolds are absent from training (HTTP 422). Checking this bypasses rejection with a critical warning.
                </p>
              </div>
            </div>

            {/* Device Fabrication Conditions */}
            <div className="form-card">
              <div
                className="form-card-header clickable"
                onClick={() => setShowAdvancedParams(!showAdvancedParams)}
              >
                <span className="card-step-badge">3</span>
                <div style={{ flex: 1 }}>
                  <h3 className="card-title">Device Fabrication &amp; Processing Parameters</h3>
                  <p className="card-sub">Layer thickness, thermal annealing, solvents &amp; interlayers</p>
                </div>
                <span className="collapse-arrow">{showAdvancedParams ? '▲' : '▼'}</span>
              </div>

              {showAdvancedParams && (
                <div className="processing-grid">
                  <div className="field-col">
                    <label>D:A Weight Ratio</label>
                    <input
                      type="text"
                      value={daRatio}
                      onChange={(e) => setDaRatio(e.target.value)}
                      placeholder="e.g. 1:1.2"
                    />
                  </div>

                  <div className="field-col">
                    <label>Active Layer Thickness</label>
                    <input
                      type="text"
                      value={thickness}
                      onChange={(e) => setThickness(e.target.value)}
                      placeholder="e.g. 100nm"
                    />
                  </div>

                  <div className="field-col">
                    <label>Annealing Temperature (°C)</label>
                    <input
                      type="number"
                      step="1"
                      value={annealingTemp}
                      onChange={(e) =>
                        setAnnealingTemp(e.target.value === '' ? '' : Number(e.target.value))
                      }
                      placeholder="e.g. 110"
                    />
                  </div>

                  <div className="field-col">
                    <label>Host Solvent</label>
                    <select value={solvent} onChange={(e) => setSolvent(e.target.value)}>
                      <option value="chloroform">Chloroform (BP 61.2°C)</option>
                      <option value="chlorobenzene">Chlorobenzene (BP 131.6°C)</option>
                      <option value="o-dcb">o-Dichlorobenzene (BP 180.5°C)</option>
                      <option value="toluene">Toluene (BP 110.6°C)</option>
                      <option value="o-xylene">o-Xylene (BP 144.4°C)</option>
                    </select>
                  </div>

                  <div className="field-col">
                    <label>Solvent Additive</label>
                    <select value={additive} onChange={(e) => setAdditive(e.target.value)}>
                      <option value="none">None / As-cast</option>
                      <option value="DIO">1,8-Diiodooctane (DIO)</option>
                      <option value="CN">1-Chloronaphthalene (CN)</option>
                      <option value="other">Other Additive</option>
                    </select>
                  </div>

                  <div className="field-col">
                    <label>Additive Concentration</label>
                    <input
                      type="text"
                      value={additiveRatio}
                      onChange={(e) => setAdditiveRatio(e.target.value)}
                      placeholder="e.g. 0.5% or 3%"
                    />
                  </div>

                  <div className="field-col">
                    <label>Device Architecture</label>
                    <select value={deviceType} onChange={(e) => setDeviceType(e.target.value)}>
                      <option value="conventional">Conventional (Anode on Substrate)</option>
                      <option value="inverted">Inverted (Cathode on Substrate)</option>
                    </select>
                  </div>

                  <div className="field-col">
                    <label>Hole Transport Layer (HTL)</label>
                    <select value={htl} onChange={(e) => setHtl(e.target.value)}>
                      <option value="PEDOT:PSS">PEDOT:PSS (WF 5.00 eV)</option>
                      <option value="MoO3">MoO3 (WF 5.30 eV)</option>
                      <option value="2PACz">2PACz (WF 5.20 eV)</option>
                      <option value="other">Other HTL</option>
                    </select>
                  </div>

                  <div className="field-col">
                    <label>Electron Transport Layer (ETL)</label>
                    <select value={etl} onChange={(e) => setEtl(e.target.value)}>
                      <option value="PDINN">PDINN (WF 4.00 eV)</option>
                      <option value="ZnO">ZnO (WF 4.30 eV)</option>
                      <option value="Ca">Calcium (WF 2.87 eV)</option>
                      <option value="PFN-Br">PFN-Br (WF 4.20 eV)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Run Prediction Button */}
            <div className="submit-section">
              <button
                type="submit"
                className="predict-cta-btn"
                disabled={isPredicting}
              >
                {isPredicting ? (
                  <span className="btn-loading-content">
                    <span className="spinner"></span> Running Phase 9 Inference &amp; Conformal Analysis...
                  </span>
                ) : (
                  <span>Run Governed PCE Prediction &rarr;</span>
                )}
              </button>
              {!authToken && (
                <p className="auth-nag-text">
                  Note: A guest session or signed-in account is required by the FastAPI backend to persist results.
                </p>
              )}
            </div>
          </form>
        </div>

        {/* Right Column: Prediction Diagnostics & Results */}
        <div className="dashboard-results-col">
          {activeTab === 'prediction' && (
            <div className="results-wrapper">
              {predictionResponse ? (
                <div className="prediction-report-card">
                  {/* Research Estimate Notice */}
                  <div className="result-caveat-banner">
                    <span className="caveat-tag">RESEARCH ESTIMATE ONLY</span>
                    <span>Not an experimentally guaranteed device efficiency.</span>
                  </div>

                  {/* Main Point Estimate & Interval Card */}
                  <div className="primary-metric-box">
                    <div className="metric-header">
                      <div>
                        <span className="metric-eyebrow">
                          Model {predictionResponse.model_id} &bull; {predictionResponse.algorithm}
                        </span>
                        <h3 className="metric-headline">Predicted Power Conversion Efficiency</h3>
                      </div>
                      <span className="version-tag">{predictionResponse.model_version}</span>
                    </div>

                    <div className="pce-hero-display">
                      <span className="pce-number">
                        {predictionResponse.point_prediction.predicted_pce.toFixed(2)}
                      </span>
                      <span className="pce-unit">%</span>
                    </div>

                    {/* Conformal Uncertainty Interval Display */}
                    <div className="interval-gauge-card">
                      <div className="interval-head">
                        <span className="interval-title">
                          Qualified Conformal Interval ({predictionResponse.interval_estimate.confidence_level} Confidence)
                        </span>
                        <span className="interval-q-val">
                          Quantile $q = {predictionResponse.interval_estimate.conformal_quantile_q.toFixed(4)}$
                        </span>
                      </div>

                      <div className="interval-range-bar">
                        <div className="range-track">
                          <div className="range-bracket" />
                          <div className="range-center-dot" />
                        </div>
                        <div className="range-labels">
                          <span className="range-low">
                            Lower: {predictionResponse.interval_estimate.prediction_interval_low.toFixed(2)}%
                          </span>
                          <span className="range-mid">
                            Point: {predictionResponse.point_prediction.predicted_pce.toFixed(2)}%
                          </span>
                          <span className="range-high">
                            Upper: {predictionResponse.interval_estimate.prediction_interval_high.toFixed(2)}%
                          </span>
                        </div>
                      </div>

                      <div className="interval-meta">
                        <span>
                          Interval Width: <strong>{predictionResponse.interval_estimate.interval_width.toFixed(2)} pp</strong>
                        </span>
                        <span>
                          Calibration Strategy: <strong>{predictionResponse.interval_estimate.calibration_strategy}</strong>
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Out of Distribution (OOD) Analysis */}
                  <div className="diagnostic-card">
                    <div className="diag-header">
                      <h4 className="diag-title">Out-of-Distribution (OOD) &amp; Scaffold Status</h4>
                      <span className={`severity-badge ${predictionResponse.ood_status.severity}`}>
                        Severity: {predictionResponse.ood_status.severity.toUpperCase()}
                      </span>
                    </div>

                    <div className="scaffold-status-grid">
                      <div className="scaffold-item">
                        <span className="scaffold-role">Donor Molecule</span>
                        <span className={`status-pill ${predictionResponse.ood_status.donor_scaffold_known ? 'known' : 'unseen'}`}>
                          {predictionResponse.ood_status.donor_scaffold_known ? 'Known Training Scaffold' : 'Unseen Novel Scaffold'}
                        </span>
                        <code className="scaffold-id-text">{predictionResponse.ood_status.donor_scaffold_id}</code>
                      </div>

                      <div className="scaffold-item">
                        <span className="scaffold-role">Acceptor Molecule</span>
                        <span className={`status-pill ${predictionResponse.ood_status.acceptor_scaffold_known ? 'known' : 'unseen'}`}>
                          {predictionResponse.ood_status.acceptor_scaffold_known ? 'Known Training Scaffold' : 'Unseen Novel Scaffold'}
                        </span>
                        <code className="scaffold-id-text">{predictionResponse.ood_status.acceptor_scaffold_id}</code>
                      </div>
                    </div>

                    {predictionResponse.ood_status.processing_ood_flags.length > 0 && (
                      <div className="processing-flags-box">
                        <span className="flag-label">Processing Missingness Indicators:</span>
                        <div className="flag-pills">
                          {predictionResponse.ood_status.processing_ood_flags.map((flag, idx) => (
                            <span key={idx} className="flag-pill">{flag}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Warnings Alert Panel */}
                  {predictionResponse.warnings.length > 0 && (
                    <div className="warnings-alert-card">
                      <h4 className="warnings-title">&#x26A0; Model Governance &amp; Domain Warnings</h4>
                      <ul className="warnings-list">
                        {predictionResponse.warnings.map((w, idx) => (
                          <li key={idx}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Scientific Support & Benchmark Metrics */}
                  <div className="diagnostic-card">
                    <h4 className="diag-title">Scientific Support Status &amp; Benchmark Performance</h4>
                    <p className="support-desc">
                      <strong>Governance Role:</strong> {predictionResponse.scientific_support.governance_role}
                    </p>
                    <div className="benchmark-pills">
                      <span>Strategy C2 $R^2$: <strong>{predictionResponse.scientific_support.benchmark_metrics.r2 ?? 'N/A'}</strong></span>
                      <span>MAE: <strong>{predictionResponse.scientific_support.benchmark_metrics.mae ?? 'N/A'}%</strong></span>
                      <span>RMSE: <strong>{predictionResponse.scientific_support.benchmark_metrics.rmse ?? 'N/A'}%</strong></span>
                      <span>Partition: <strong>{predictionResponse.scientific_support.benchmark_metrics.evaluation_partition ?? 'Held-out C2'}</strong></span>
                    </div>
                    <p className="limitations-quote">
                      <em>&ldquo;{predictionResponse.scientific_support.limitations}&rdquo;</em>
                    </p>
                  </div>
                </div>
              ) : (
                <div className="empty-results-card">
                  <div className="empty-icon">&#x269B;</div>
                  <h3 className="empty-title">Awaiting Inference Execution</h3>
                  <p className="empty-desc">
                    Configure donor and acceptor SMILES on the left and run governed prediction to inspect the point estimate, conformal uncertainty bounds, scaffold OOD flags, and scientific limitations.
                  </p>
                  <div className="catalog-preview">
                    {loadingModels ? (
                      <span>Loading model catalog...</span>
                    ) : (
                      modelCatalog && (
                        <span>
                          Loaded {Object.keys(modelCatalog.models).length} Governed Models &bull;{' '}
                          {modelCatalog.n_donor_scaffolds} Donor Scaffolds &bull;{' '}
                          {modelCatalog.n_acceptor_scaffolds} Acceptor Scaffolds
                        </span>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'descriptors' && (
            <div className="descriptors-tab-content">
              {analysisResponse ? (
                <div className="descriptors-container">
                  <h3 className="tab-section-title">RDKit 2D Physicochemical Descriptors</h3>

                  {/* Donor Card */}
                  <div className="desc-card">
                    <h4 className="desc-card-title">Donor: {analysisResponse.donor.name || 'Donor Candidate'}</h4>
                    <code className="desc-smiles">{analysisResponse.donor.canonical_smiles}</code>
                    <div className="desc-grid">
                      <div><span>Formula:</span> <strong>{analysisResponse.donor.formula}</strong></div>
                      <div><span>MW:</span> <strong>{analysisResponse.donor.molecular_weight.toFixed(2)} g/mol</strong></div>
                      <div><span>LogP:</span> <strong>{analysisResponse.donor.logp.toFixed(2)}</strong></div>
                      <div><span>TPSA:</span> <strong>{analysisResponse.donor.tpsa.toFixed(2)} &Aring;&sup2;</strong></div>
                      <div><span>Aromatic Rings:</span> <strong>{analysisResponse.donor.aromatic_rings}</strong></div>
                      <div><span>Fraction CSP3:</span> <strong>{analysisResponse.donor.fraction_csp3.toFixed(3)}</strong></div>
                    </div>
                  </div>

                  {/* Acceptor Card */}
                  <div className="desc-card">
                    <h4 className="desc-card-title">Acceptor: {analysisResponse.acceptor.name || 'Acceptor Candidate'}</h4>
                    <code className="desc-smiles">{analysisResponse.acceptor.canonical_smiles}</code>
                    <div className="desc-grid">
                      <div><span>Formula:</span> <strong>{analysisResponse.acceptor.formula}</strong></div>
                      <div><span>MW:</span> <strong>{analysisResponse.acceptor.molecular_weight.toFixed(2)} g/mol</strong></div>
                      <div><span>LogP:</span> <strong>{analysisResponse.acceptor.logp.toFixed(2)}</strong></div>
                      <div><span>TPSA:</span> <strong>{analysisResponse.acceptor.tpsa.toFixed(2)} &Aring;&sup2;</strong></div>
                      <div><span>Aromatic Rings:</span> <strong>{analysisResponse.acceptor.aromatic_rings}</strong></div>
                      <div><span>Fraction CSP3:</span> <strong>{analysisResponse.acceptor.fraction_csp3.toFixed(3)}</strong></div>
                    </div>
                  </div>

                  {/* Blend Features */}
                  <div className="desc-card">
                    <h4 className="desc-card-title">Composite Blend Features</h4>
                    <div className="desc-grid">
                      <div><span>&Delta;MW:</span> <strong>{analysisResponse.blend_features.molecular_weight_diff.toFixed(2)} g/mol</strong></div>
                      <div><span>&Delta;LogP:</span> <strong>{analysisResponse.blend_features.logp_diff.toFixed(2)}</strong></div>
                      <div><span>Total TPSA:</span> <strong>{analysisResponse.blend_features.tpsa_total.toFixed(2)} &Aring;&sup2;</strong></div>
                      <div><span>Total Aromatic Rings:</span> <strong>{analysisResponse.blend_features.aromatic_rings_total}</strong></div>
                      <div><span>Total Heavy Atoms:</span> <strong>{analysisResponse.blend_features.heavy_atom_count_total}</strong></div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-results-card">
                  <div className="empty-icon">&#x2697;</div>
                  <h3 className="empty-title">No Descriptors Computed Yet</h3>
                  <p className="empty-desc">
                    Click &ldquo;Analyze 2D Descriptors &amp; Blend&rdquo; on the left to extract RDKit physicochemical properties without running the full ML pipeline.
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'history' && (
            <div className="history-tab-content">
              <div className="history-header">
                <h3 className="tab-section-title">Prediction Run History (SQLite Backed)</h3>
                <button
                  type="button"
                  className="refresh-btn"
                  onClick={reloadHistory}
                  disabled={historyLoading}
                >
                  {historyLoading ? 'Refreshing...' : 'Refresh History'}
                </button>
              </div>

              {historyError && (
                <div className="dashboard-alert error-alert" style={{ marginBottom: '1rem' }}>
                  <p>{historyError}</p>
                </div>
              )}

              {historyItems.length > 0 ? (
                <div className="history-list">
                  {historyItems.map((item) => {
                    const unc = parseHistoryUncertainty(item.uncertainty_json);
                    return (
                      <div key={item.id} className="history-item-card">
                        <div className="history-card-top">
                          <div>
                            <span className="history-donor">{item.donor_name}</span>
                            <span className="history-divider"> / </span>
                            <span className="history-acceptor">{item.acceptor_name}</span>
                          </div>
                          <div className="history-pce-badge">
                            PCE: <strong>{item.predicted_pce.toFixed(2)}%</strong>
                          </div>
                        </div>

                        <div className="history-card-meta">
                          <span>Model: <code>{item.model_version || 'Phase 9'}</code></span>
                          <span>ID: <code>{item.id}</code></span>
                          <span>Timestamp: {new Date(item.created_at).toLocaleString()}</span>
                        </div>

                        {unc?.interval_estimate && (
                          <div className="history-interval-summary">
                            <span>
                              {unc.interval_estimate.confidence_level} Interval:{' '}
                              <strong>
                                [{unc.interval_estimate.prediction_interval_low.toFixed(2)}%,{' '}
                                {unc.interval_estimate.prediction_interval_high.toFixed(2)}%]
                              </strong>{' '}
                              (Width: {unc.interval_estimate.interval_width.toFixed(2)} pp)
                            </span>
                            {unc.ood_status && (
                              <span className={`mini-ood-pill ${unc.ood_status.severity}`}>
                                OOD: {unc.ood_status.severity.toUpperCase()}
                              </span>
                            )}
                          </div>
                        )}

                        <div className="history-card-actions">
                          <button
                            type="button"
                            className="history-reload-btn"
                            onClick={() => {
                              setDonorName(item.donor_name);
                              setDonorSmiles(item.donor_smiles);
                              setAcceptorName(item.acceptor_name);
                              setAcceptorSmiles(item.acceptor_smiles);
                              setActiveTab('prediction');
                            }}
                          >
                            Load Into Form &rarr;
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-results-card">
                  <div className="empty-icon">&#x1F4DC;</div>
                  <h3 className="empty-title">No Prediction Records Yet</h3>
                  <p className="empty-desc">
                    {authToken
                      ? 'Submit a prediction run to save records to your authenticated session.'
                      : 'Please sign in to persist and view your personal prediction history.'}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
