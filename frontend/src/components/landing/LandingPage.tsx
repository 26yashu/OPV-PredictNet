import React from 'react';
import { DeviceMorphologyCanvas } from '../visualization/DeviceMorphologyCanvas';

interface LandingPageProps {
  onOpenWorkspace: () => void;
  backendOnline: boolean;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onOpenWorkspace, backendOnline }) => {
  return (
    <div className="landing-page">
      {/* Top Scientific Caveat Status Banner */}
      <div className="governance-top-banner">
        <div className="governance-top-content">
          <span className="gov-pill warning-pill">RESEARCH ESTIMATE ONLY</span>
          <span className="gov-banner-text">
            Predictions are computational approximations calibrated on Strategy C2 benchmark data (N_cal = 3,863) &mdash; not guaranteed experimental efficiencies.
          </span>
          <span className="gov-status-dot-wrapper">
            <span className={`status-dot ${backendOnline ? 'online-dot' : 'offline-dot'}`} />
            <span className="gov-status-label">{backendOnline ? 'FastAPI Serving Active' : 'Connecting API...'}</span>
          </span>
        </div>
      </div>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-badge-row">
          <span className="badge-tag cyan-glow">Phase 10 &bull; Scientific Web Platform</span>
          <span className="badge-tag slate-tag">Governed under Phase 8.3 Scientific Contract</span>
        </div>

        <h1 className="hero-title">
          OPV-PredictNet
          <span className="hero-title-sub">
            AI-Driven Molecular Screening &amp; Device-Aware Photovoltaic Inference
          </span>
        </h1>

        <p className="hero-lead">
          A rigorous machine learning framework coupling RDKit chemoinformatics, the authoritative 44-feature Unified Feature Pipeline, and finite-sample conformal uncertainty quantification to predict Organic Photovoltaic Power Conversion Efficiency (PCE).
        </p>

        <div className="hero-actions">
          <button
            type="button"
            className="cta-primary-btn"
            onClick={onOpenWorkspace}
          >
            Launch Prediction Workspace &rarr;
          </button>
          <a href="#methodology" className="cta-secondary-btn">
            Explore Scientific Architecture
          </a>
          <a href="#governance" className="cta-secondary-btn outline-btn">
            Governance &amp; Limitations
          </a>
        </div>

        {/* Benchmark Highlights Bar */}
        <div className="hero-metrics-grid">
          <div className="hero-metric-card">
            <span className="metric-val">22,467</span>
            <span className="metric-label">Strict Benchmark Devices</span>
            <span className="metric-sub">OPVdb v1.1.0 Verified</span>
          </div>
          <div className="hero-metric-card">
            <span className="metric-val">1,073</span>
            <span className="metric-label">Known Donor Scaffolds</span>
            <span className="metric-sub">Bemis&ndash;Murcko Topology</span>
          </div>
          <div className="hero-metric-card">
            <span className="metric-val">812</span>
            <span className="metric-label">Known Acceptor Scaffolds</span>
            <span className="metric-sub">NFAs &amp; Fullerenes</span>
          </div>
          <div className="hero-metric-card">
            <span className="metric-val">4</span>
            <span className="metric-label">Governed ML Models</span>
            <span className="metric-sub">M0, M1, M2A, M3 (Native-NaN)</span>
          </div>
          <div className="hero-metric-card">
            <span className="metric-val">90%</span>
            <span className="metric-label">Conformal Coverage</span>
            <span className="metric-sub">Strategy C2 Calibrated</span>
          </div>
        </div>

        {/* Cinematic 3D Device Visualization */}
        <div className="hero-visualization-wrapper">
          <DeviceMorphologyCanvas />
        </div>
      </section>

      {/* 4-Stage Scientific Pipeline */}
      <section id="methodology" className="workflow-section">
        <div className="section-title-block">
          <span className="section-eyebrow">Rigorous End-to-End Methodology</span>
          <h2 className="section-heading">Deterministic Four-Stage Inference Architecture</h2>
          <p className="section-subtext">
            Unlike black-box models, OPV-PredictNet separates molecular graph validation, fabrication physics, feature transformation, and calibrated uncertainty quantification.
          </p>
        </div>

        <div className="workflow-grid">
          <div className="workflow-card">
            <div className="workflow-step-num">01</div>
            <h3 className="workflow-card-title">Molecular Graph &amp; Scaffold Auditing</h3>
            <p className="workflow-card-body">
              Strict canonical SMILES validation with RDKit. Rejects BigSMILES, ambiguous polymer wildcards (<code>*</code>), and multi-component mixtures (<code>.</code>). Extracts Bemis&ndash;Murcko cyclic scaffolds and pseudo-scaffolds to monitor chemical novelty.
            </p>
            <div className="workflow-tags">
              <span>RDKit Sanitization</span>
              <span>Murcko Scaffolds</span>
              <span>Pseudo-Scaffolds</span>
            </div>
          </div>

          <div className="workflow-card">
            <div className="workflow-step-num">02</div>
            <h3 className="workflow-card-title">Device &amp; Interface Physics</h3>
            <p className="workflow-card-body">
              Incorporates fabrication realities beyond isolated chemistry: donor:acceptor weight ratios (D:A), active layer thickness (15&ndash;600 nm), thermal annealing temperatures, host solvent boiling points (61.2&ndash;180.5 &deg;C), and electrode interlayer work functions.
            </p>
            <div className="workflow-tags">
              <span>Film Thickness</span>
              <span>Thermal Annealing</span>
              <span>Work Functions</span>
            </div>
          </div>

          <div className="workflow-card">
            <div className="workflow-step-num">03</div>
            <h3 className="workflow-card-title">Authoritative Unified Feature Pipeline</h3>
            <p className="workflow-card-body">
              Frozen feature extraction established in Phase 8.4: 22 molecular descriptors, continuous physical transformations, 7 missingness indicators, and 7 non-imputed physical regimes. Supports M0 (22), M1 (50), M2A (37), and M3 (44) with zero script drift.
            </p>
            <div className="workflow-tags">
              <span>Phase 8.4 Pipeline</span>
              <span>Native NaNs (M3)</span>
              <span>Zero Leakage</span>
            </div>
          </div>

          <div className="workflow-card">
            <div className="workflow-step-num">04</div>
            <h3 className="workflow-card-title">Conformal Bounds &amp; OOD Detection</h3>
            <p className="workflow-card-body">
              Returns valid prediction intervals at 80%, 90%, and 95% confidence calibrated on the held-out Strategy C2 partition (N_cal = 3,863). Detects novel donor or acceptor scaffolds and enforces strict rejection on unsupported dual-scaffold extrapolation (R&sup2; &asymp; 0.08).
            </p>
            <div className="workflow-tags">
              <span>Conformal Intervals</span>
              <span>Scaffold OOD Flags</span>
              <span>Dual-OOD Rejection</span>
            </div>
          </div>
        </div>
      </section>

      {/* Governed Model Roster */}
      <section className="model-roster-section">
        <div className="section-title-block">
          <span className="section-eyebrow">Phase 8.3 Governed Model Roster</span>
          <h2 className="section-heading">Purpose-Built Model Roles</h2>
          <p className="section-subtext">
            Each configuration serves a distinct scientific purpose in research screening.
          </p>
        </div>

        <div className="roster-grid">
          <div className="roster-card">
            <div className="roster-header">
              <span className="model-code">M0</span>
              <span className="roster-role-pill">Baseline Reference</span>
            </div>
            <h3 className="roster-title">Molecular-Descriptor-Only Baseline</h3>
            <p className="roster-desc">
              22 RDKit physicochemical descriptors (MW, LogP, TPSA, aromatic rings, heavy atoms). Evaluates chemical baseline without processing parameters.
            </p>
            <div className="roster-stats">
              <div><span>Features:</span> <strong>22</strong></div>
              <div><span>Algorithm:</span> <strong>Random Forest</strong></div>
              <div><span>Strategy C2 MAE:</span> <strong>2.316%</strong></div>
              <div><span>90% Interval:</span> <strong>&plusmn;4.650 pp</strong></div>
            </div>
          </div>

          <div className="roster-card">
            <div className="roster-header">
              <span className="model-code">M1</span>
              <span className="roster-role-pill exploratory-pill">Exploratory Candidate</span>
            </div>
            <h3 className="roster-title">Categorical &amp; Continuous Processing</h3>
            <p className="roster-desc">
              50 features including 23 one-hot encoded categorical columns for solvent, additive, architecture, HTL, and ETL. Demonstrates superior unseen donor scaffold handling.
            </p>
            <div className="roster-stats">
              <div><span>Features:</span> <strong>50</strong></div>
              <div><span>Algorithm:</span> <strong>Random Forest</strong></div>
              <div><span>Strategy C2 MAE:</span> <strong>2.128%</strong></div>
              <div><span>90% Interval:</span> <strong>&plusmn;4.273 pp</strong></div>
            </div>
          </div>

          <div className="roster-card">
            <div className="roster-header">
              <span className="model-code">M2A</span>
              <span className="roster-role-pill">Continuous Reference</span>
            </div>
            <h3 className="roster-title">Continuous Processing Ablation</h3>
            <p className="roster-desc">
              37 features with continuous variables, physical regimes, and interface constants, but without the 7 missingness indicator columns.
            </p>
            <div className="roster-stats">
              <div><span>Features:</span> <strong>37</strong></div>
              <div><span>Algorithm:</span> <strong>Random Forest</strong></div>
              <div><span>Strategy C2 MAE:</span> <strong>2.161%</strong></div>
              <div><span>90% Interval:</span> <strong>&plusmn;4.323 pp</strong></div>
            </div>
          </div>

          <div className="roster-card highlighted-roster-card">
            <div className="roster-header">
              <span className="model-code highlight-code">M3</span>
              <span className="roster-role-pill recommended-pill">Primary Recommended</span>
            </div>
            <h3 className="roster-title">Native-Missing-Value Processing</h3>
            <p className="roster-desc">
              44 features using HistGradientBoosting with native missing value support. Avoids synthetic imputation artifacts and achieves the lowest Strategy A and B MAE.
            </p>
            <div className="roster-stats">
              <div><span>Features:</span> <strong>44</strong></div>
              <div><span>Algorithm:</span> <strong>HistGradientBoosting</strong></div>
              <div><span>Strategy C2 MAE:</span> <strong>2.164%</strong></div>
              <div><span>90% Interval:</span> <strong>&plusmn;4.162 pp</strong></div>
            </div>
          </div>
        </div>
      </section>

      {/* Scientific Governance, Integrity & Limitations */}
      <section id="governance" className="governance-section">
        <div className="governance-card">
          <div className="governance-header">
            <span className="gov-icon">&#x26A0;</span>
            <div>
              <h2 className="governance-title">Scientific Integrity &amp; Strict Limitations Registry</h2>
              <p className="governance-sub">Mandatory disclosures under OPV-PredictNet Phase 8.3 &amp; Phase 9 governance</p>
            </div>
          </div>

          <div className="governance-body-grid">
            <div className="gov-point">
              <h4>1. Research Estimate Only</h4>
              <p>
                All figures of merit (PCE, interval bounds) represent statistical estimates calibrated on peer-reviewed experimental device archives (AM1.5G, 100 mW/cm&sup2;). They are <strong>never guaranteed laboratory results</strong>.
              </p>
            </div>

            <div className="gov-point">
              <h4>2. Dual-Scaffold Extrapolation Rejection</h4>
              <p>
                When both donor and acceptor molecules contain Bemis&ndash;Murcko scaffolds absent from training (Strategy C1), the benchmark $R^2$ collapses to approximately $0.08$. The API <strong>rejects dual-scaffold OOD inputs by default with HTTP 422</strong>.
              </p>
            </div>

            <div className="gov-point">
              <h4>3. Acceptor Novelty High-PCE Degradation</h4>
              <p>
                Under Strategy B (unseen acceptor scaffolds), while overall 90% coverage remains near nominal, empirical coverage in the top-performing quartile collapses to ~45%. Extreme predictions must be verified experimentally.
              </p>
            </div>

            <div className="gov-point">
              <h4>4. Nanoscale Morphology Sensitivity</h4>
              <p>
                Real bulk-heterojunction performance is acutely sensitive to thin-film morphology, batch polydispersity, solvent additive quality (e.g. DIO purity), and thermal history that tabular descriptors cannot fully capture.
              </p>
            </div>
          </div>

          <div className="governance-footer">
            <button
              type="button"
              className="cta-primary-btn"
              onClick={onOpenWorkspace}
            >
              Open Interactive Prediction Workspace &rarr;
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
