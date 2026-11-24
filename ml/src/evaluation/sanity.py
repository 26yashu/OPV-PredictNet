"""
OPV-PredictNet — Single-Junction OPV Output Sanity Bounds
Defines broad physical and engineering plausibility checks for predicted device figures of merit.
"""

from typing import Dict, List, Optional, Tuple

# Broad engineering sanity limits for single-junction OPV devices under AM1.5G (100 mW/cm²)
# These represent screening guardrails, not universal thermodynamic proofs.
SANITY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "pce": (0.0, 34.0),     # Power Conversion Efficiency in % (SQ thermodynamic ceiling ~33.7%)
    "voc": (0.0, 2.0),      # Open-Circuit Voltage in Volts (typical single-junction OPV < 1.5 V)
    "jsc": (0.0, 45.0),     # Short-Circuit Current Density in mA/cm²
    "ff": (0.0, 100.0),     # Fill Factor in % (must be between 0% and 100%)
}


def validate_prediction_sanity(
    pce: float,
    voc: Optional[float] = None,
    jsc: Optional[float] = None,
    ff: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """Validate predicted device figures of merit against physical sanity boundaries.

    Does NOT silently clip or alter values. If any metric is outside plausibility bounds,
    reports the exact violation so that unphysical model outputs can be rejected explicitly.

    Args:
        pce: Predicted Power Conversion Efficiency (%).
        voc: Predicted Open-Circuit Voltage (V), or None.
        jsc: Predicted Short-Circuit Current Density (mA/cm²), or None.
        ff: Predicted Fill Factor (%), or None.

    Returns:
        Tuple of (is_sane: bool, violations: List[str]).
    """
    violations: List[str] = []

    # 1. PCE bounds
    min_pce, max_pce = SANITY_BOUNDS["pce"]
    if pce < min_pce or pce > max_pce:
        violations.append(
            f"Predicted PCE ({pce:.2f}%) exceeds physical sanity range [{min_pce}%, {max_pce}%]."
        )

    # 2. VOC bounds (if predicted)
    if voc is not None:
        min_voc, max_voc = SANITY_BOUNDS["voc"]
        if voc < min_voc or voc > max_voc:
            violations.append(
                f"Predicted Voc ({voc:.3f} V) exceeds physical sanity range [{min_voc} V, {max_voc} V]."
            )

    # 3. JSC bounds (if predicted)
    if jsc is not None:
        min_jsc, max_jsc = SANITY_BOUNDS["jsc"]
        if jsc < min_jsc or jsc > max_jsc:
            violations.append(
                f"Predicted Jsc ({jsc:.2f} mA/cm²) exceeds physical sanity range [{min_jsc} mA/cm², {max_jsc} mA/cm²]."
            )

    # 4. Fill Factor bounds (if predicted)
    if ff is not None:
        min_ff, max_ff = SANITY_BOUNDS["ff"]
        # Normalize if passed as fraction (e.g. 0.65 instead of 65%)
        ff_pct = ff * 100.0 if (0.0 < ff <= 1.0) else ff
        if ff_pct < min_ff or ff_pct > max_ff:
            violations.append(
                f"Predicted Fill Factor ({ff_pct:.1f}%) exceeds physical sanity range [{min_ff}%, {max_ff}%]."
            )

    is_sane = len(violations) == 0
    return is_sane, violations
