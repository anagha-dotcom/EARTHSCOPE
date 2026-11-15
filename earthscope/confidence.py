"""Evidence ("confidence") score: transparent, reproducible, NOT a prediction probability.

score (0-100) = completeness points + significance points + strength points
    completeness  : weight 20 * (valid months / expected months)
    significance  : weight 50 * clip((P_ZERO - p) / (P_ZERO - P_FULL), 0, 1)   (Mann-Kendall p, annual means)
    strength      : weight 30 * clip(|Kendall tau| / TAU_FULL, 0, 1)

Bands: <40 Weak, 40-69 Moderate, >=70 Strong. If the trend is not significant at alpha the label is
capped at "Weak" regardless of the numeric score. All weights/cut-offs are design choices (config.py)
and are described in the interface and docs/METHODOLOGY.md.
"""
from __future__ import annotations

import numpy as np

from . import config
from .models import ConfidenceResult, TrendResult


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_confidence(completeness: float, trend: TrendResult) -> ConfidenceResult:
    if not trend.ok or trend.mk_p is None or trend.mk_tau is None:
        return ConfidenceResult(
            variable=trend.variable, score=None, label="Not assessable", components=[],
            note=trend.reason or "Trend could not be computed.",
        )
    sig_scaled = _clip01((config.CONF_P_ZERO - trend.mk_p) / (config.CONF_P_ZERO - config.CONF_P_FULL))
    str_scaled = _clip01(abs(trend.mk_tau) / config.CONF_TAU_FULL)
    comp_scaled = _clip01(completeness)
    components = [
        dict(name="Data completeness", raw=completeness, raw_text=f"{completeness * 100:.1f}% of months valid",
             scaled=comp_scaled, weight=config.CONF_W_COMPLETENESS, points=comp_scaled * config.CONF_W_COMPLETENESS),
        dict(name="Statistical significance", raw=trend.mk_p, raw_text=f"Mann-Kendall p = {trend.mk_p:.4f}",
             scaled=sig_scaled, weight=config.CONF_W_SIGNIFICANCE, points=sig_scaled * config.CONF_W_SIGNIFICANCE),
        dict(name="Trend strength", raw=abs(trend.mk_tau), raw_text=f"|Kendall tau| = {abs(trend.mk_tau):.2f}",
             scaled=str_scaled, weight=config.CONF_W_STRENGTH, points=str_scaled * config.CONF_W_STRENGTH),
    ]
    score = float(sum(c["points"] for c in components))
    if score >= config.CONF_BAND_STRONG:
        label = "Strong"
    elif score >= config.CONF_BAND_MODERATE:
        label = "Moderate"
    else:
        label = "Weak"
    note = ""
    if not trend.significant and label != "Weak":
        label = "Weak"
        note = ("Label capped at 'Weak' because the trend is not statistically significant at "
                f"the {int(trend.alpha * 100)}% level, even though the numeric score is higher.")
    if trend.serial_corr_flag:
        note = (note + " " if note else "") + (
            "Residuals are serially correlated, which can make the Mann-Kendall p-value too optimistic.")
    return ConfidenceResult(variable=trend.variable, score=score, label=label, components=components, note=note)
