import React, { useState, useEffect, useCallback } from 'react';
import type {
  MolecularAnalysisResponse,
  PredictionResponse,
  PredictionHistoryItem,
  DeviceParameters,
} from '../../types/prediction';
import {
  analyzeMolecules,
  submitPrediction,
  getPredictionHistory,
} from '../../services/prediction';
import { getStoredToken } from '../../services/auth';

interface PredictionSectionProps {
  authToken?: string | null;
}

const PRESET_PAIRS = [
  {
    name: 'P3HT Monomer Subunit / Benzothiadiazole (BT) Core',
    donorName: '3-Hexylthiophene (P3HT Monomer Unit)',
    donorSmiles: 'CCCCCCc1cccs1',
    acceptorName: '2,1,3-Benzothiadiazole (BT Core Fragment)',
    acceptorSmiles: 'c1ccc2nsnc2c1',
    ratio: '1:1',
    thickness: 100,
    solvent: 'Chlorobenzene',
  },
  {
    name: 'BDT-T Core Fragment / ITIC (IDT) Core Fragment',
    donorName: 'BDT-T Core Fragment (PBDB-T Repeat Scaffold)',
    donorSmiles: 'c1sc(-c2cccs2)c2c1ccc1c2sc(-c2cccs2)c1',
    acceptorName: 'Indacenodithiophene (IDT Core Scaffold)',
    acceptorSmiles: 'c1sc2c(c1)-c1cc3c4c(cc1C2)sc1c4sc(c1)-c3',
    ratio: '1:1.2',
    thickness: 110,
    solvent: 'Chloroform',
  },
];

export const PredictionSection: React.FC<PredictionSectionProps> = ({ authToken: propToken }) => {
  // Input states
  const [donorSmiles, setDonorSmiles] = useState('CCCCCCc1cccs1');
  const [donorName, setDonorName] = useState('3-Hexylthiophene (P3HT Monomer Unit)');
  const [acceptorSmiles, setAcceptorSmiles] = useState('c1ccc2nsnc2c1');
  const [acceptorName, setAcceptorName] = useState('2,1,3-Benzothiadiazole (BT Core Fragment)');

  // Device parameter states
  const [showDeviceParams, setShowDeviceParams] = useState(false);
  const [thickness, setThickness] = useState<number>(100);
  const [daRatio, setDaRatio] = useState<string>('1:1.2');
  const [solvent, setSolvent] = useState<string>('Chlorobenzene');
  const [solventAdditive, setSolventAdditive] = useState<string>('DIO');
  const [additiveConcentration, setAdditiveConcentration] = useState<number>(0.5);
  const [annealingTemp, setAnnealingTemp] = useState<number>(100);
  const [annealingTime, setAnnealingTime] = useState<number>(10);

  // Result & UI states
  const [analyzing, setAnalyzing] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<MolecularAnalysisResponse | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // History state
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const getActiveToken = useCallback((): string | null => {
    return propToken !== undefined ? propToken : getStoredToken();
  }, [propToken]);

  const fetchHistory = useCallback(async (authToken: string) => {
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const data = await getPredictionHistory(authToken, 15, 0);
      setHistory(data.items);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setHistoryError(err.message);
      } else {
        setHistoryError('Failed to load prediction history.');
      }
    } finally {
      setLoadingHistory(false);
    }
  }, []);

  // Fetch history on mount or when token updates
  useEffect(() => {
    const activeToken = propToken !== undefined ? propToken : getStoredToken();
    if (!activeToken) {
      return;
    }

    let isMounted = true;
    getPredictionHistory(activeToken, 15, 0)
      .then((data) => {
        if (isMounted) {
          setHistory(data.items);
        }
      })
      .catch((err: unknown) => {
        if (isMounted && err instanceof Error) {
          setHistoryError(err.message);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [propToken]);

  const handleApplyPreset = (preset: typeof PRESET_PAIRS[0]) => {
    setDonorSmiles(preset.donorSmiles);
    setDonorName(preset.donorName);
    setAcceptorSmiles(preset.acceptorSmiles);
    setAcceptorName(preset.acceptorName);
    setDaRatio(preset.ratio);
    setThickness(preset.thickness);
    setSolvent(preset.solvent);
    setErrorMessage(null);
    setAnalysisResult(null);
    setPredictionResult(null);
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setAnalyzing(true);

    try {
      const result = await analyzeMolecules({
        donor_smiles: donorSmiles.trim(),
        acceptor_smiles: acceptorSmiles.trim(),
        donor_name: donorName.trim() || undefined,
        acceptor_name: acceptorName.trim() || undefined,
      });
      setAnalysisResult(result);
    } catch (err: unknown) {
      setAnalysisResult(null);
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to analyze molecular structures.');
      }
    } finally {
      setAnalyzing(false);
    }
  };

  const handlePredict = async () => {
    setErrorMessage(null);
    setPredicting(true);

    const deviceParams: DeviceParameters = {
      active_layer_thickness_nm: thickness,
      d_a_ratio: daRatio,
      solvent: solvent,
      solvent_additive: solventAdditive,
      additive_concentration_vol_pct: additiveConcentration,
      thermal_annealing_temp_c: annealingTemp,
      thermal_annealing_time_min: annealingTime,
      hole_transport_layer: 'PEDOT:PSS',
      electron_transport_layer: 'PDINN',
      device_architecture: 'conventional',
      substrate: 'Glass/ITO',
    };

    try {
      const activeToken = getActiveToken();
      const result = await submitPrediction(
        {
          donor_smiles: donorSmiles.trim(),
          acceptor_smiles: acceptorSmiles.trim(),
          donor_name: donorName.trim() || undefined,
          acceptor_name: acceptorName.trim() || undefined,
          device_parameters: deviceParams,
        },
        activeToken
      );
      setPredictionResult(result);
      if (result.donor && result.acceptor && result.blend_features) {
        setAnalysisResult({
          donor: result.donor,
          acceptor: result.acceptor,
          blend_features: result.blend_features,
        });
      }
      if (activeToken) {
        fetchHistory(activeToken);
      }
    } catch (err: unknown) {
      setPredictionResult(null);
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to run OPV prediction pipeline.');
      }
    } finally {
      setPredicting(false);
    }
  };

  return (
    <section className="prediction-section">
      <div className="section-header-row">
        <div>
          <span className="badge-tag">Phase 3 &bull; Chemoinformatics &amp; ML Inference</span>
          <h3 className="section-title" style={{ marginTop: '0.25rem' }}>
            Molecular Analysis &amp; OPV Performance Inference
          </h3>
          <p className="section-subtitle">
            Validate donor/acceptor SMILES with RDKit, compute 2D physicochemical descriptors, and query the leakage-aware Random Forest inference pipeline.
          </p>
        </div>
      </div>

      {/* Preset Candidate Pairs */}
      <div className="preset-bar">
        <span className="preset-label">Load Reference Pair (Monomer / Core Fragment Level):</span>
        {PRESET_PAIRS.map((pair, idx) => (
          <button
            key={idx}
            type="button"
            className="preset-btn"
            onClick={() => handleApplyPreset(pair)}
          >
            {pair.name}
          </button>
        ))}
      </div>
      <p style={{ margin: '-0.75rem 0 1.25rem', fontSize: '0.8rem', color: '#64748b' }}>
        <em>Note: Presets represent monomer repeat units and central conjugated cores for descriptor calculation, not full polymer chains or bulk devices.</em>
      </p>

      {/* Error Banner */}
      {errorMessage && (
        <div className="auth-alert error" style={{ marginBottom: '1.5rem' }}>
          <strong>Scientific Validation / Pipeline Notice:</strong>
          <p style={{ margin: '0.35rem 0 0', fontSize: '0.9rem' }}>{errorMessage}</p>
        </div>
      )}

      {/* Input Formulation */}
      <form onSubmit={handleAnalyze} className="molecular-form">
        <div className="molecular-grid">
          {/* Donor Column */}
          <div className="chem-card donor-card">
            <div className="chem-card-header">
              <span className="role-badge donor">Donor Material</span>
              <span className="role-sub">Electron Donor • Monomer Unit / Core Fragment / Small Molecule</span>
            </div>
            <div className="chem-input-group">
              <label className="input-label" htmlFor="donor-name-input">Donor Label (Monomer Unit or Small Molecule)</label>
              <input
                id="donor-name-input"
                type="text"
                className="text-input"
                value={donorName}
                onChange={(e) => setDonorName(e.target.value)}
                placeholder="e.g. 3-Hexylthiophene (P3HT Monomer Unit)"
              />
            </div>
            <div className="chem-input-group">
              <label className="input-label" htmlFor="donor-smiles-input">
                Donor SMILES (Monomer Unit or Molecular Structure) <span className="req">*</span>
              </label>
              <textarea
                id="donor-smiles-input"
                className="text-input textarea mono"
                rows={3}
                value={donorSmiles}
                onChange={(e) => setDonorSmiles(e.target.value)}
                placeholder="e.g. CCCCCc1cccs1"
                required
              />
            </div>
          </div>

          {/* Acceptor Column */}
          <div className="chem-card acceptor-card">
            <div className="chem-card-header">
              <span className="role-badge acceptor">Acceptor Material</span>
              <span className="role-sub">Electron Acceptor • Small Molecule / NF-Acceptor Core Fragment</span>
            </div>
            <div className="chem-input-group">
              <label className="input-label" htmlFor="acceptor-name-input">Acceptor Label (Small Molecule or Core Fragment)</label>
              <input
                id="acceptor-name-input"
                type="text"
                className="text-input"
                value={acceptorName}
                onChange={(e) => setAcceptorName(e.target.value)}
                placeholder="e.g. 2,1,3-Benzothiadiazole (BT Core Fragment)"
              />
            </div>
            <div className="chem-input-group">
              <label className="input-label" htmlFor="acceptor-smiles-input">
                Acceptor SMILES (Full Molecule or Core Fragment) <span className="req">*</span>
              </label>
              <textarea
                id="acceptor-smiles-input"
                className="text-input textarea mono"
                rows={3}
                value={acceptorSmiles}
                onChange={(e) => setAcceptorSmiles(e.target.value)}
                placeholder="e.g. c1ccc2nsnc2c1"
                required
              />
            </div>
          </div>
        </div>

        {/* Collapsible Device Metadata Parameters */}
        <div className="device-params-box">
          <button
            type="button"
            className="toggle-device-params-btn"
            onClick={() => setShowDeviceParams(!showDeviceParams)}
          >
            <span>{showDeviceParams ? '▼ Hide' : '▶ Show'} Experimental Device Processing Parameters (Pillar 3)</span>
            <span className="params-hint">Thickness, D:A stoichiometry, solvent additives, thermal annealing</span>
          </button>

          {showDeviceParams && (
            <div className="device-params-container">
              <div
                style={{
                  margin: '0.75rem 0',
                  padding: '0.6rem 0.85rem',
                  borderRadius: '6px',
                  backgroundColor: 'rgba(30, 41, 59, 0.6)',
                  border: '1px solid rgba(148, 163, 184, 0.2)',
                  fontSize: '0.85rem',
                  color: '#94a3b8',
                }}
              >
                <strong style={{ color: '#38bdf8' }}>Scope Notice:</strong> Experimental processing parameters are recorded for provenance but are not currently included in the 22-feature molecular descriptor model.
              </div>
              <div className="device-params-grid">
              <div className="param-field">
                <label className="input-label">D:A Weight Ratio</label>
                <input
                  type="text"
                  className="text-input"
                  value={daRatio}
                  onChange={(e) => setDaRatio(e.target.value)}
                  placeholder="e.g. 1:1.2"
                />
              </div>
              <div className="param-field">
                <label className="input-label">Active Layer Thickness (nm)</label>
                <input
                  type="number"
                  className="text-input"
                  value={thickness}
                  onChange={(e) => setThickness(Number(e.target.value))}
                  min={10}
                  max={500}
                />
              </div>
              <div className="param-field">
                <label className="input-label">Host Solvent</label>
                <input
                  type="text"
                  className="text-input"
                  value={solvent}
                  onChange={(e) => setSolvent(e.target.value)}
                  placeholder="e.g. Chlorobenzene, Chloroform"
                />
              </div>
              <div className="param-field">
                <label className="input-label">Solvent Additive</label>
                <input
                  type="text"
                  className="text-input"
                  value={solventAdditive}
                  onChange={(e) => setSolventAdditive(e.target.value)}
                  placeholder="e.g. DIO, CN"
                />
              </div>
              <div className="param-field">
                <label className="input-label">Additive Concentration (vol %)</label>
                <input
                  type="number"
                  step="0.1"
                  className="text-input"
                  value={additiveConcentration}
                  onChange={(e) => setAdditiveConcentration(Number(e.target.value))}
                  min={0}
                  max={10}
                />
              </div>
              <div className="param-field">
                <label className="input-label">Annealing Temperature (°C)</label>
                <input
                  type="number"
                  className="text-input"
                  value={annealingTemp}
                  onChange={(e) => setAnnealingTemp(Number(e.target.value))}
                  min={20}
                  max={250}
                />
              </div>
              <div className="param-field">
                <label className="input-label">Annealing Time (min)</label>
                <input
                  type="number"
                  className="text-input"
                  value={annealingTime}
                  onChange={(e) => setAnnealingTime(Number(e.target.value))}
                  min={1}
                  max={120}
                />
              </div>
            </div>
            </div>
          )}
        </div>

        {/* Action Controls */}
        <div className="action-buttons-row">
          <button
            type="submit"
            className="action-btn secondary"
            disabled={analyzing || predicting}
          >
            {analyzing ? 'Validating Chemistry...' : '1. Analyze Molecules (RDKit Descriptors)'}
          </button>
          <button
            type="button"
            className="action-btn primary"
            onClick={handlePredict}
            disabled={analyzing || predicting}
          >
            {predicting ? 'Evaluating Inference Pipeline...' : '2. Run OPV Performance Inference'}
          </button>
        </div>
      </form>

      {/* RDKit Molecular Descriptors Display */}
      {analysisResult && (
        <div className="analysis-results-container">
          <h4 className="results-heading">
            Computed RDKit Physicochemical Descriptors
          </h4>
          <div className="descriptors-comparison-grid">
            {/* Donor Descriptors */}
            <div className="descriptor-panel donor">
              <div className="panel-header">
                <h5 className="panel-title">{analysisResult.donor.name || 'Donor'}</h5>
                <span className="canonical-formula mono">{analysisResult.donor.formula}</span>
              </div>
              <p className="canonical-smiles mono">
                <strong>Canonical SMILES:</strong> {analysisResult.donor.canonical_smiles}
              </p>
              <div className="metrics-table">
                <div className="metric-row">
                  <span>Molecular Weight:</span>
                  <span className="metric-val">{analysisResult.donor.molecular_weight.toFixed(2)} g/mol</span>
                </div>
                <div className="metric-row">
                  <span>Calculated LogP (MolLogP):</span>
                  <span className="metric-val">{analysisResult.donor.logp.toFixed(3)}</span>
                </div>
                <div className="metric-row">
                  <span>Topological Polar Surface Area (TPSA):</span>
                  <span className="metric-val">{analysisResult.donor.tpsa.toFixed(1)} Å²</span>
                </div>
                <div className="metric-row">
                  <span>H-Bond Donors / Acceptors:</span>
                  <span className="metric-val">{analysisResult.donor.hbd} / {analysisResult.donor.hba}</span>
                </div>
                <div className="metric-row">
                  <span>Rotatable Bonds:</span>
                  <span className="metric-val">{analysisResult.donor.rotatable_bonds}</span>
                </div>
                <div className="metric-row">
                  <span>Aromatic Rings / Total Rings:</span>
                  <span className="metric-val">{analysisResult.donor.aromatic_rings} / {analysisResult.donor.ring_count}</span>
                </div>
                <div className="metric-row">
                  <span>Fraction Csp3:</span>
                  <span className="metric-val">{analysisResult.donor.fraction_csp3.toFixed(3)}</span>
                </div>
              </div>
            </div>

            {/* Acceptor Descriptors */}
            <div className="descriptor-panel acceptor">
              <div className="panel-header">
                <h5 className="panel-title">{analysisResult.acceptor.name || 'Acceptor'}</h5>
                <span className="canonical-formula mono">{analysisResult.acceptor.formula}</span>
              </div>
              <p className="canonical-smiles mono">
                <strong>Canonical SMILES:</strong> {analysisResult.acceptor.canonical_smiles}
              </p>
              <div className="metrics-table">
                <div className="metric-row">
                  <span>Molecular Weight:</span>
                  <span className="metric-val">{analysisResult.acceptor.molecular_weight.toFixed(2)} g/mol</span>
                </div>
                <div className="metric-row">
                  <span>Calculated LogP (MolLogP):</span>
                  <span className="metric-val">{analysisResult.acceptor.logp.toFixed(3)}</span>
                </div>
                <div className="metric-row">
                  <span>Topological Polar Surface Area (TPSA):</span>
                  <span className="metric-val">{analysisResult.acceptor.tpsa.toFixed(1)} Å²</span>
                </div>
                <div className="metric-row">
                  <span>H-Bond Donors / Acceptors:</span>
                  <span className="metric-val">{analysisResult.acceptor.hbd} / {analysisResult.acceptor.hba}</span>
                </div>
                <div className="metric-row">
                  <span>Rotatable Bonds:</span>
                  <span className="metric-val">{analysisResult.acceptor.rotatable_bonds}</span>
                </div>
                <div className="metric-row">
                  <span>Aromatic Rings / Total Rings:</span>
                  <span className="metric-val">{analysisResult.acceptor.aromatic_rings} / {analysisResult.acceptor.ring_count}</span>
                </div>
                <div className="metric-row">
                  <span>Fraction Csp3:</span>
                  <span className="metric-val">{analysisResult.acceptor.fraction_csp3.toFixed(3)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Prediction Response View */}
      {predictionResult && (
        <div className="prediction-response-card">
          <div className="prediction-header">
            <div>
              <span className={`status-pill ${predictionResult.status}`}>
                Status: {predictionResult.status === 'model_unavailable' ? 'Model Pending (No Fabrication)' : predictionResult.status}
              </span>
              <h4 className="prediction-title" style={{ marginTop: '0.4rem' }}>
                Pipeline Evaluation Summary
              </h4>
              <p className="prediction-msg">{predictionResult.message}</p>
            </div>
            {predictionResult.id && (
              <div className="prediction-id-tag">
                <span className="mono">ID: {predictionResult.id.slice(0, 8)}...</span>
              </div>
            )}
          </div>

          {/* Genuine Predictions (if model artifact exists) */}
          {predictionResult.predictions && (
            <div className="metrics-cards-grid">
              <div className="metric-stat-card">
                <span className="stat-label">Power Conversion Efficiency (PCE)</span>
                <span className="stat-number">{predictionResult.predictions.pce_percent.toFixed(2)}</span>
                <span className="stat-unit">%</span>
              </div>
              <div className="metric-stat-card">
                <span className="stat-label">Open-Circuit Voltage (Voc)</span>
                <span className="stat-number">
                  {predictionResult.predictions.voc_volts !== null && predictionResult.predictions.voc_volts !== undefined
                    ? predictionResult.predictions.voc_volts.toFixed(3)
                    : 'N/A'}
                </span>
                <span className="stat-unit">V</span>
              </div>
              <div className="metric-stat-card">
                <span className="stat-label">Short-Circuit Current Density (Jsc)</span>
                <span className="stat-number">
                  {predictionResult.predictions.jsc_ma_cm2 !== null && predictionResult.predictions.jsc_ma_cm2 !== undefined
                    ? predictionResult.predictions.jsc_ma_cm2.toFixed(2)
                    : 'N/A'}
                </span>
                <span className="stat-unit">mA/cm²</span>
              </div>
              <div className="metric-stat-card">
                <span className="stat-label">Fill Factor (FF)</span>
                <span className="stat-number">
                  {predictionResult.predictions.fill_factor_percent !== null && predictionResult.predictions.fill_factor_percent !== undefined
                    ? predictionResult.predictions.fill_factor_percent.toFixed(1)
                    : 'N/A'}
                </span>
                <span className="stat-unit">%</span>
              </div>
            </div>
          )}

          {/* Integrity & Scientific Limitations Banner */}
          <div className="limitations-box">
            <span className="limitation-title">Scientific Limitations &amp; Reproducibility Notice:</span>
            <ul className="limitation-list">
              {predictionResult.model_metadata?.limitations && predictionResult.model_metadata.limitations.length > 0 ? (
                predictionResult.model_metadata.limitations.map((lim, i) => <li key={i}>{lim}</li>)
              ) : (
                <>
                  <li>Isolated molecular SMILES alone do not determine device efficiency. Thin-film morphology and nanophase separation are required.</li>
                  <li>Inference is strictly bounded by the chemical domain of applicability of experimental training data.</li>
                </>
              )}
            </ul>
            <p className="disclaimer-text">
              <em>{predictionResult.model_metadata?.disclaimer || 'Data-driven regression for screening purposes only. Real devices require morphological verification.'}</em>
            </p>
          </div>
        </div>
      )}

      {/* Authenticated User Prediction History */}
      <div className="history-section">
        {(() => {
          const activeToken = getActiveToken();
          return (
            <>
              <div className="history-header">
                <div>
                  <h4 className="history-title">Saved Prediction Runs (User History)</h4>
                  <span className="history-sub">
                    Persisted in SQLite for authenticated researchers. Strictly isolated per user.
                  </span>
                </div>
                {activeToken && (
                  <button
                    type="button"
                    className="refresh-btn"
                    onClick={() => fetchHistory(activeToken)}
                    disabled={loadingHistory}
                  >
                    {loadingHistory ? 'Refreshing...' : 'Refresh History'}
                  </button>
                )}
              </div>

              {!activeToken ? (
                <div className="history-guest-notice">
                  <p>
                    <strong>Session Notice:</strong> You are currently working as a guest. Authenticate in the session panel above to persist your prediction runs to your private researcher history.
                  </p>
                </div>
              ) : historyError ? (
                <div className="auth-alert error">
                  <p>{historyError}</p>
                </div>
              ) : history.length === 0 ? (
                <div className="empty-history-box">
                  <p>No historical prediction runs recorded yet for your account. Run a prediction above to persist your first run.</p>
                </div>
              ) : (
                <div className="history-table-wrapper">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Donor</th>
                        <th>Acceptor</th>
                        <th>Predicted PCE</th>
                        <th>Model Version</th>
                        <th>Prediction ID</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map((item) => (
                        <tr key={item.id}>
                          <td>{new Date(item.created_at).toLocaleString()}</td>
                          <td className="mono" title={item.donor_smiles}>
                            {item.donor_name || item.donor_smiles.slice(0, 16) + '...'}
                          </td>
                          <td className="mono" title={item.acceptor_smiles}>
                            {item.acceptor_name || item.acceptor_smiles.slice(0, 16) + '...'}
                          </td>
                          <td className="mono">
                            {item.predicted_pce ? `${item.predicted_pce.toFixed(2)} %` : 'N/A'}
                          </td>
                          <td className="mono">{item.model_version || 'N/A'}</td>
                          <td className="mono">{item.id.slice(0, 8)}...</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          );
        })()}
      </div>
    </section>
  );
};
