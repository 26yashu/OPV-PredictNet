import React, { useEffect, useRef, useState } from 'react';

type VisualizationMode = 'nanomorphology' | 'device_stack' | 'band_alignment';

export const DeviceMorphologyCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mode, setMode] = useState<VisualizationMode>('nanomorphology');
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let time = 0;

    // Handle high DPI displays
    const handleResize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    // Particle simulation for excitons and charges in nanomorphology
    interface Particle {
      x: number;
      y: number;
      type: 'exciton' | 'electron' | 'hole';
      life: number;
      maxLife: number;
      vx: number;
      vy: number;
    }

    const particles: Particle[] = [];
    for (let i = 0; i < 35; i++) {
      particles.push({
        x: Math.random() * 600,
        y: Math.random() * 320,
        type: Math.random() > 0.4 ? 'exciton' : Math.random() > 0.5 ? 'electron' : 'hole',
        life: Math.random() * 100,
        maxLife: 80 + Math.random() * 60,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8,
      });
    }

    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      if (mode === 'nanomorphology') {
        renderNanomorphology(ctx, w, h, time, particles);
      } else if (mode === 'device_stack') {
        renderDeviceStack(ctx, w, h, time);
      } else {
        renderBandAlignment(ctx, w, h, time);
      }

      if (isPlaying) {
        time += 0.02;
      }
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [mode, isPlaying]);

  return (
    <div className="morphology-container">
      <div className="morphology-toolbar">
        <div className="toolbar-left">
          <span className="morphology-badge">
            <span className="pulse-dot"></span> 3D Device Physics &amp; Interface Dynamics
          </span>
        </div>
        <div className="toolbar-controls">
          <button
            type="button"
            className={`toolbar-btn ${mode === 'nanomorphology' ? 'active' : ''}`}
            onClick={() => setMode('nanomorphology')}
          >
            BHJ Nanomorphology
          </button>
          <button
            type="button"
            className={`toolbar-btn ${mode === 'device_stack' ? 'active' : ''}`}
            onClick={() => setMode('device_stack')}
          >
            Multi-Layer Stack
          </button>
          <button
            type="button"
            className={`toolbar-btn ${mode === 'band_alignment' ? 'active' : ''}`}
            onClick={() => setMode('band_alignment')}
          >
            HOMO/LUMO Alignment
          </button>
          <button
            type="button"
            className="toolbar-btn icon-btn"
            title={isPlaying ? 'Pause Animation' : 'Play Animation'}
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
        </div>
      </div>

      <div className="canvas-wrapper">
        <canvas ref={canvasRef} className="morphology-canvas" />
      </div>

      <div className="morphology-caption">
        {mode === 'nanomorphology' && (
          <p>
            <strong>Bulk Heterojunction (BHJ) Nanoscale Active Layer:</strong> Interpenetrating bicontinuous network of donor polymers (cyan domains) and non-fullerene acceptors (amber domains). Illustrated are exciton diffusion, interfacial charge separation (&Delta;E &gt; exciton binding energy), and bipolar charge transport to electrodes.
          </p>
        )}
        {mode === 'device_stack' && (
          <p>
            <strong>Inverted &amp; Conventional Multilayer Architectures:</strong> Substrate / Transparent Anode (ITO) / HTL (PEDOT:PSS, MoO3, 2PACz) / BHJ Active Layer (100–120 nm) / ETL (ZnO, PDINN, Ca, PFN-Br) / Metal Cathode (Ag/Al). Work functions &amp; energy offsets govern charge extraction.
          </p>
        )}
        {mode === 'band_alignment' && (
          <p>
            <strong>Frontier Orbital Alignment &amp; V<sub>OC</sub> Thermodynamics:</strong> Open-circuit voltage is governed by the effective donor HOMO to acceptor LUMO energy offset minus energetic disorder losses: V<sub>OC</sub> &approx; (1/q)(|E<sub>HOMO,D</sub>| - |E<sub>LUMO,A</sub>|) - &Delta;V<sub>loss</sub>.
          </p>
        )}
      </div>
    </div>
  );
};

// -----------------------------------------------------------------------------
// Canvas Render Modes
// -----------------------------------------------------------------------------

function renderNanomorphology(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  time: number,
  particles: Array<{ x: number; y: number; type: string; life: number; maxLife: number; vx: number; vy: number }>
) {
  // Deep background
  const bgGrad = ctx.createLinearGradient(0, 0, w, h);
  bgGrad.addColorStop(0, '#090D16');
  bgGrad.addColorStop(1, '#0F172A');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  // Subtle grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 1;
  const gridSize = 40;
  for (let x = 0; x < w; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw donor and acceptor domains with smooth sine waves
  const numDomains = 6;
  for (let i = 0; i < numDomains; i++) {
    const isDonor = i % 2 === 0;
    const baseOffset = (i / numDomains) * w;
    const color = isDonor ? 'rgba(6, 182, 212, 0.16)' : 'rgba(245, 158, 11, 0.14)';
    const stroke = isDonor ? 'rgba(6, 182, 212, 0.5)' : 'rgba(245, 158, 11, 0.5)';

    ctx.fillStyle = color;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const startX = baseOffset + Math.sin(time + i) * 20;
    ctx.moveTo(startX, 0);

    for (let y = 0; y <= h; y += 20) {
      const wobble = Math.sin(y * 0.02 + time * 1.5 + i * 1.2) * 35 + Math.cos(y * 0.04 - time) * 15;
      ctx.lineTo(startX + wobble + (isDonor ? 25 : -25), y);
    }

    ctx.lineTo(startX + (isDonor ? 70 : -70), h);
    ctx.lineTo(startX + (isDonor ? 70 : -70), 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // Draw simulated interpenetrating polymer network fibrils
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
  ctx.lineWidth = 1.2;
  for (let f = 0; f < 5; f++) {
    ctx.beginPath();
    const startY = 40 + f * 55;
    ctx.moveTo(20, startY);
    for (let x = 20; x < w - 20; x += 30) {
      const py = startY + Math.sin(x * 0.03 + time * 2 + f) * 18;
      ctx.lineTo(x, py);
    }
    ctx.stroke();
  }

  // Animate particles (excitons: cyan/white, electrons: blue, holes: amber)
  particles.forEach((p) => {
    p.x += p.vx;
    p.y += p.vy;
    p.life += 0.8;

    if (p.x < 10) p.x = w - 10;
    if (p.x > w - 10) p.x = 10;
    if (p.y < 10) p.y = h - 10;
    if (p.y > h - 10) p.y = 10;
    if (p.life > p.maxLife) {
      p.life = 0;
      p.x = Math.random() * w;
      p.y = Math.random() * h;
    }

    const alpha = Math.sin((p.life / p.maxLife) * Math.PI);
    ctx.beginPath();
    if (p.type === 'exciton') {
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.9})`;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    } else if (p.type === 'electron') {
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(6, 182, 212, ${alpha * 0.85})`;
      ctx.fill();
    } else {
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(245, 158, 11, ${alpha * 0.85})`;
      ctx.fill();
    }
  });

  // Floating labels
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillStyle = 'rgba(56, 189, 248, 0.8)';
  ctx.fillText('Donor Polymer Nanofibril (P3HT / PM6 / D18)', 30, 30);
  ctx.fillStyle = 'rgba(245, 158, 11, 0.8)';
  ctx.fillText('Acceptor Aggregation Domain (Y6 / ITIC / PCBM)', w - 310, 30);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText('• Exciton (Diffusion length L_D ~10 nm)', 30, h - 20);
  ctx.fillStyle = 'rgba(6, 182, 212, 0.8)';
  ctx.fillText('• Electron (e⁻)', 280, h - 20);
  ctx.fillStyle = 'rgba(245, 158, 11, 0.8)';
  ctx.fillText('• Hole (h⁺)', 390, h - 20);
}

function renderDeviceStack(ctx: CanvasRenderingContext2D, w: number, h: number, time: number) {
  const bgGrad = ctx.createLinearGradient(0, 0, w, h);
  bgGrad.addColorStop(0, '#0A0F1D');
  bgGrad.addColorStop(1, '#0F172A');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  const layers = [
    { name: 'Top Metal Electrode (Ag / Al, ~100 nm)', color: 'rgba(148, 163, 184, 0.35)', height: 26, stroke: '#94a3b8' },
    { name: 'Electron Transport Interlayer (ZnO / PDINN / PFN-Br, ~10 nm)', color: 'rgba(14, 165, 233, 0.28)', height: 24, stroke: '#0ea5e9' },
    { name: 'Bulk Heterojunction Active Layer (Donor:Acceptor Blend, 100 nm)', color: 'rgba(20, 184, 166, 0.38)', height: 85, stroke: '#14b8a6', isBhj: true },
    { name: 'Hole Transport Interlayer (PEDOT:PSS / MoO3, ~35 nm)', color: 'rgba(245, 158, 11, 0.28)', height: 26, stroke: '#f59e0b' },
    { name: 'Transparent Anode (ITO on Glass, ~150 nm)', color: 'rgba(99, 102, 241, 0.3)', height: 32, stroke: '#6366f1' },
    { name: 'Glass Substrate (~1.1 mm)', color: 'rgba(51, 65, 85, 0.4)', height: 36, stroke: '#475569' },
  ];

  const totalHeight = layers.reduce((acc, l) => acc + l.height + 6, 0);
  let startY = (h - totalHeight) / 2;

  const padX = 60;
  const layerWidth = w - padX * 2;

  layers.forEach((l) => {
    ctx.fillStyle = l.color;
    ctx.strokeStyle = l.stroke;
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.roundRect(padX, startY, layerWidth, l.height, 4);
    ctx.fill();
    ctx.stroke();

    // BHJ internal texture
    if (l.isBhj) {
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(padX, startY, layerWidth, l.height, 4);
      ctx.clip();
      for (let x = padX; x < padX + layerWidth; x += 25) {
        const wave = Math.sin(x * 0.04 + time) * 12;
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
        ctx.beginPath();
        ctx.moveTo(x, startY);
        ctx.lineTo(x + wave, startY + l.height);
        ctx.stroke();
      }
      ctx.restore();
    }

    ctx.font = '12px ui-monospace, monospace';
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(l.name, padX + 16, startY + l.height / 2 + 4);

    startY += l.height + 6;
  });

  // Photon arrow entering through glass
  const arrowX = padX + layerWidth * 0.82;
  ctx.strokeStyle = '#facc15';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(arrowX, h - 15);
  ctx.lineTo(arrowX, startY - 145);
  ctx.stroke();

  // Arrow head
  ctx.fillStyle = '#facc15';
  ctx.beginPath();
  ctx.moveTo(arrowX - 6, startY - 140);
  ctx.lineTo(arrowX + 6, startY - 140);
  ctx.lineTo(arrowX, startY - 155);
  ctx.closePath();
  ctx.fill();

  ctx.font = '11px ui-monospace, monospace';
  ctx.fillStyle = '#facc15';
  ctx.fillText('Incident AM1.5G Sunlight (100 mW/cm²)', arrowX - 110, h - 8);
}

function renderBandAlignment(ctx: CanvasRenderingContext2D, w: number, h: number, _time: number) {
  const bgGrad = ctx.createLinearGradient(0, 0, w, h);
  bgGrad.addColorStop(0, '#090D18');
  bgGrad.addColorStop(1, '#0E1726');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2;
  const colW = 120;
  const dX = cx - colW - 40;
  const aX = cx + 40;

  // Energy levels (eV)
  const dLumo = 100;
  const dHomo = 220;
  const aLumo = 145;
  const aHomo = 250;

  // Donor block
  ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(dX, dLumo, colW, dHomo - dLumo, 6);
  ctx.fill();
  ctx.stroke();

  // Acceptor block
  ctx.fillStyle = 'rgba(245, 158, 11, 0.15)';
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(aX, aLumo, colW, aHomo - aLumo, 6);
  ctx.fill();
  ctx.stroke();

  // Energy labels
  ctx.font = '12px ui-monospace, monospace';
  ctx.fillStyle = '#38bdf8';
  ctx.fillText('Donor (e.g. PM6)', dX + 10, dLumo - 20);
  ctx.fillText('LUMO: -3.50 eV', dX + 10, dLumo - 5);
  ctx.fillText('HOMO: -5.45 eV', dX + 10, dHomo + 18);

  ctx.fillStyle = '#fbbf24';
  ctx.fillText('Acceptor (e.g. Y6)', aX + 10, aLumo - 20);
  ctx.fillText('LUMO: -4.10 eV', aX + 10, aLumo - 5);
  ctx.fillText('HOMO: -5.65 eV', aX + 10, aHomo + 18);

  // Offset arrows
  // Delta LUMO
  ctx.strokeStyle = '#e2e8f0';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(dX + colW, dLumo);
  ctx.lineTo(aX, dLumo);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(aX - 15, dLumo);
  ctx.lineTo(aX - 15, aLumo);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#f8fafc';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText('ΔLUMO (Driving force for e⁻ transfer)', cx - 90, dLumo + 25);

  // V_OC thermodynamics bracket
  const vocX = cx;
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(vocX, dHomo);
  ctx.lineTo(vocX, aLumo);
  ctx.stroke();

  ctx.fillStyle = '#34d399';
  ctx.font = 'bold 12px ui-monospace, monospace';
  ctx.fillText('q·V_OC potential ~1.35 eV', vocX + 10, (dHomo + aLumo) / 2);

  // Axis
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(35, 40);
  ctx.lineTo(35, h - 30);
  ctx.stroke();
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('Energy (Vacuum 0 eV ↓)', 15, 30);
  ctx.fillText('-3.0 eV', 45, 80);
  ctx.fillText('-4.0 eV', 45, 140);
  ctx.fillText('-5.0 eV', 45, 200);
  ctx.fillText('-6.0 eV', 45, 260);
}
