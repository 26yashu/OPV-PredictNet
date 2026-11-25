"""
OPV-PredictNet — Essential Academic Report Diagrams Generator.
Generates 5 publication-ready scientific and architectural diagrams as 300 DPI PNG and SVG.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = PROJECT_ROOT / "docs" / "report_assets"
SCI_DIR = ASSETS_DIR / "scientific_diagrams"
ARCH_DIR = ASSETS_DIR / "architecture_diagrams"

SCI_DIR.mkdir(parents=True, exist_ok=True)
ARCH_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# -----------------------------------------------------------------------------
# 1. OPV Working Principle (4-Step Photophysical Process)
# -----------------------------------------------------------------------------
print("Generating 01_opv_working_principle...")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

steps = [
    ("1. Light Absorption\n& Exciton Generation", "Photons ($h\\nu$) excite electrons from\nHOMO to LUMO, forming bound excitons ($e^--h^+$).", 1.2, "#3a86ff"),
    ("2. Exciton Diffusion", "Excitons diffuse toward D:A interface\n(diffusion length $L_D \\approx 10\\text{--}20\\text{ nm}$).", 3.7, "#8338ec"),
    ("3. Charge Dissociation", "Energy offset ($\\Delta E$) drives electron\ntransfer to acceptor; exciton dissociates.", 6.2, "#ff006e"),
    ("4. Charge Collection", "Free carriers drift to electrodes:\nholes to anode (ITO), electrons to cathode.", 8.7, "#fb5607")
]

for title, desc, cx, col in steps:
    # Card box
    box = patches.FancyBboxPatch((cx - 1.05, 1.2), 2.1, 3.8, boxstyle="round,pad=0.2",
                                facecolor="#f8f9fa", edgecolor=col, linewidth=2.5)
    ax.add_patch(box)
    # Header badge
    badge = patches.FancyBboxPatch((cx - 1.0, 4.0), 2.0, 0.8, boxstyle="round,pad=0.1",
                                  facecolor=col, edgecolor="none")
    ax.add_patch(badge)
    ax.text(cx, 4.4, title, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(cx, 2.6, desc, ha="center", va="center", color="#222", fontsize=9, wrap=True)

# Connecting arrows
for i in range(len(steps) - 1):
    ax.annotate("", xy=(steps[i+1][2] - 1.15, 3.1), xytext=(steps[i][2] + 1.15, 3.1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2.0))

ax.set_title("Diagram 1: Photophysical Working Principle of Organic Photovoltaic (OPV) Solar Cells",
             fontsize=13, fontweight="bold", pad=12)
fig.savefig(SCI_DIR / "01_opv_working_principle.png")
fig.savefig(SCI_DIR / "01_opv_working_principle.svg")
plt.close(fig)

# -----------------------------------------------------------------------------
# 2. OPV Device Architecture (Multilayer Thin-Film Stack)
# -----------------------------------------------------------------------------
print("Generating 02_opv_device_architecture...")
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

layers = [
    ("Top Metal Electrode (Ag / Al)", "Cathode / Top Contact (~100 nm)", "#b0bec5", 0.5),
    ("Electron Transport Layer (ZnO / PDINN)", "Facilitates electron extraction (~5-20 nm)", "#90caf9", 0.6),
    ("BHJ Active Layer (Donor Polymer : Non-Fullerene Acceptor)", "Light absorption, exciton dissociation (~100 nm)", "#ffab91", 1.8),
    ("Hole Transport Layer (PEDOT:PSS / MoO3)", "Selective hole collection (~30-40 nm)", "#ce93d8", 0.6),
    ("Transparent Conductive Oxide (ITO)", "Anode contact layer (~120 nm)", "#80deea", 0.6),
    ("Glass Substrate", "Mechanical support & optical window (~1 mm)", "#e0e0e0", 0.7)
]

y_pos = 1.0
for name, role, color, height in layers[::-1]:
    rect = patches.Rectangle((1.5, y_pos), 7.0, height, facecolor=color, edgecolor="#37474f", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5.0, y_pos + height / 2, f"{name}\n({role})", ha="center", va="center",
            fontsize=9.5, fontweight="bold" if "Active" in name else "normal",
            color="#212121")
    y_pos += height + 0.08

# Light incoming arrow
ax.annotate("Incident Solar Light\n(AM1.5G, 100 mW/cm²)", xy=(5.0, 0.9), xytext=(5.0, 0.1),
            arrowprops=dict(facecolor="#fbc02d", edgecolor="#f57f17", width=3, headwidth=10),
            ha="center", va="top", fontsize=10, fontweight="bold", color="#e65100")

ax.set_title("Diagram 2: Multilayer Bulk-Heterojunction (BHJ) OPV Device Architecture",
             fontsize=13, fontweight="bold", pad=12)
fig.savefig(SCI_DIR / "02_opv_device_architecture.png")
fig.savefig(SCI_DIR / "02_opv_device_architecture.svg")
plt.close(fig)

# -----------------------------------------------------------------------------
# 3. OPV-PredictNet System Architecture
# -----------------------------------------------------------------------------
print("Generating 01_system_architecture...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Presentation Tier
tier1 = patches.FancyBboxPatch((0.5, 3.8), 2.6, 1.8, boxstyle="round,pad=0.2", facecolor="#e3f2fd", edgecolor="#1976d2", linewidth=2)
ax.add_patch(tier1)
ax.text(1.8, 5.0, "Frontend Layer\n(React 19 + Vite)", ha="center", va="center", fontweight="bold", color="#0d47a1")
ax.text(1.8, 4.2, "- Chemical preset cards\n- RDKit 2D viewer\n- Conformal gauge\n- Responsive UI", ha="center", va="center", fontsize=8.5)

# API Gateway & Backend Tier
tier2 = patches.FancyBboxPatch((3.7, 3.8), 2.6, 1.8, boxstyle="round,pad=0.2", facecolor="#e8f5e9", edgecolor="#388e3c", linewidth=2)
ax.add_patch(tier2)
ax.text(5.0, 5.0, "Backend API Layer\n(FastAPI / Python 3.11)", ha="center", va="center", fontweight="bold", color="#1b5e20")
ax.text(5.0, 4.2, "- Argon2id + JWT Auth\n- RDKit Chemoinformatics\n- Unified Feature Pipeline\n- Scaffold OOD Detector", ha="center", va="center", fontsize=8.5)

# Storage Tier
tier3 = patches.FancyBboxPatch((6.9, 3.8), 2.6, 1.8, boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#f57c00", linewidth=2)
ax.add_patch(tier3)
ax.text(8.2, 5.0, "Storage & Persistence\n(SQLite + Registry)", ha="center", va="center", fontweight="bold", color="#e65100")
ax.text(8.2, 4.2, "- User isolation\n- Predictions logging\n- Scaffold hash registry\n- Conformal quantiles", ha="center", va="center", fontsize=8.5)

# Model Artifacts Box
m_box = patches.FancyBboxPatch((2.0, 0.8), 6.0, 1.8, boxstyle="round,pad=0.2", facecolor="#ede7f6", edgecolor="#512da8", linewidth=2)
ax.add_patch(m_box)
ax.text(5.0, 2.0, "Packaged Research Model Suite (Phase 9 v1)", ha="center", va="center", fontweight="bold", color="#311b92")
ax.text(5.0, 1.3, "M0: Molecular (22f)  |  M1: One-Hot (50f)  |  M2A: Continuous (37f)  |  M3: Champion (44f)\nGoverned by research_estimate_only=True & validated_for_production=False",
        ha="center", va="center", fontsize=8.5)

# Inter-tier connectors
ax.annotate("", xy=(3.6, 4.7), xytext=(3.2, 4.7), arrowprops=dict(arrowstyle="<->", color="#333", lw=2))
ax.annotate("", xy=(6.8, 4.7), xytext=(6.4, 4.7), arrowprops=dict(arrowstyle="<->", color="#333", lw=2))
ax.annotate("", xy=(5.0, 2.7), xytext=(5.0, 3.7), arrowprops=dict(arrowstyle="<->", color="#333", lw=2))

ax.set_title("Diagram 3: OPV-PredictNet Full-Stack System Architecture", fontsize=13, fontweight="bold", pad=12)
fig.savefig(ARCH_DIR / "01_system_architecture.png")
fig.savefig(ARCH_DIR / "01_system_architecture.svg")
plt.close(fig)

# -----------------------------------------------------------------------------
# 4. Machine Learning Training & Evaluation Workflow
# -----------------------------------------------------------------------------
print("Generating 02_ml_workflow...")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

boxes = [
    ("1. Data Curation", "OPVdb (N=8,739)\nZenodo CC BY 4.0\nQuality Filtering", 1.0, "#0077b6"),
    ("2. Chemoinformatics", "RDKit SMILES Normalization\nBemis-Murcko Scaffolds\n22 2D Descriptors", 3.0, "#0096c7"),
    ("3. Disjoint Partitioning", "Scaffold-Cluster Splits\nStrategy A (Unseen Donor)\nStrategy B (Unseen Acceptor)\nStrategy C2 (In-Scaffold)", 5.0, "#00b4d8"),
    ("4. Model Fitting", "M0, M1, M2A (RF)\nM3 (HistGradBoost)\nNative Missingness", 7.0, "#48cae4"),
    ("5. Uncertainty Calibration", "Strategy C2 Partition\n(N=3,863)\nConformal Quantiles\n80%, 90%, 95%", 9.0, "#90e0ef")
]

for title, desc, cx, col in boxes:
    b = patches.FancyBboxPatch((cx - 0.85, 1.2), 1.7, 3.8, boxstyle="round,pad=0.15",
                               facecolor="#ffffff", edgecolor=col, linewidth=2)
    ax.add_patch(b)
    h = patches.FancyBboxPatch((cx - 0.8, 4.0), 1.6, 0.8, boxstyle="round,pad=0.1",
                              facecolor=col, edgecolor="none")
    ax.add_patch(h)
    ax.text(cx, 4.4, title, ha="center", va="center", color="white" if "00" in col else "#111", fontweight="bold", fontsize=9)
    ax.text(cx, 2.5, desc, ha="center", va="center", color="#222", fontsize=8.5)

for i in range(len(boxes) - 1):
    ax.annotate("", xy=(boxes[i+1][2] - 0.9, 3.1), xytext=(boxes[i][2] + 0.9, 3.1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.8))

ax.set_title("Diagram 4: Machine Learning Pipeline, Training & Conformal Calibration Workflow",
             fontsize=13, fontweight="bold", pad=12)
fig.savefig(ARCH_DIR / "02_ml_workflow.png")
fig.savefig(ARCH_DIR / "02_ml_workflow.svg")
plt.close(fig)

# -----------------------------------------------------------------------------
# 5. End-to-End Application Workflow
# -----------------------------------------------------------------------------
print("Generating 03_end_to_end_workflow...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

steps = [
    ("User Input", "Donor & Acceptor SMILES\n+ Fabrication Conditions", 1.2, "#2b2d42"),
    ("Validation Guard", "RDKit SMILES Parsing\nMixture & Bounds Check\nScaffold Extraction", 3.7, "#8d99ae"),
    ("OOD Evaluation", "Bemis-Murcko Match\nDual-OOD Check\n(Reject 422 vs Warning)", 6.2, "#ef233c"),
    ("Prediction Output", "PCE Point Estimate\nConformal Interval (q)\nGovernance Status", 8.7, "#d90429")
]

for title, desc, cx, col in steps:
    b = patches.FancyBboxPatch((cx - 1.05, 1.2), 2.1, 3.8, boxstyle="round,pad=0.2",
                               facecolor="#f8f9fa", edgecolor=col, linewidth=2)
    ax.add_patch(b)
    h = patches.FancyBboxPatch((cx - 1.0, 4.0), 2.0, 0.8, boxstyle="round,pad=0.1",
                              facecolor=col, edgecolor="none")
    ax.add_patch(h)
    ax.text(cx, 4.4, title, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(cx, 2.6, desc, ha="center", va="center", color="#222", fontsize=9)

for i in range(len(steps) - 1):
    ax.annotate("", xy=(steps[i+1][2] - 1.15, 3.1), xytext=(steps[i][2] + 1.15, 3.1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2.0))

ax.set_title("Diagram 5: End-to-End Prediction Request Lifecycle & Governance Pipeline",
             fontsize=13, fontweight="bold", pad=12)
fig.savefig(ARCH_DIR / "03_end_to_end_workflow.png")
fig.savefig(ARCH_DIR / "03_end_to_end_workflow.svg")
plt.close(fig)

print("[OK] All 5 essential scientific and architectural diagrams created!")
