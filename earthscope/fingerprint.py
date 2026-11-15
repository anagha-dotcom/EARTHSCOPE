"""Environmental Change Fingerprint: rule-based combination of the per-variable trend results.

Each variable is reduced to a direction code using the annual Mann-Kendall result:
    '+' significant increase, '-' significant decrease, '0' no significant trend, 'NA' not assessable.
A fixed rule table maps combinations of (temperature, precipitation, humidity) to a named pattern.
No machine learning, no external service. The pattern is a *description of co-occurring trends*;
it never asserts that one variable caused another.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .models import ConfidenceResult, FingerprintResult, TrendResult

T, P, H = "T2M", "PRECTOTCORR", "RH2M"

# (id, label, description, predicate on direction dict)
_RULES = [
    ("warming_drying", "Warming–drying pattern",
     "Temperature is rising while both precipitation and relative humidity are falling. "
     "This combination is consistent with increasingly dry conditions.",
     lambda d: d[T] == "+" and d[P] == "-" and d[H] == "-"),
    ("warming_less_rain", "Warming with decreasing precipitation",
     "Temperature is rising and precipitation is falling; humidity shows no significant trend.",
     lambda d: d[T] == "+" and d[P] == "-" and d[H] == "0"),
    ("warming_drier_air", "Warming with drier air",
     "Temperature is rising and relative humidity is falling; precipitation shows no significant trend. "
     "Warmer air can hold more moisture, so falling relative humidity can reflect a drying tendency.",
     lambda d: d[T] == "+" and d[P] == "0" and d[H] == "-"),
    ("drying", "Drying without significant warming",
     "Precipitation and relative humidity are both falling while temperature shows no significant trend.",
     lambda d: d[T] == "0" and d[P] == "-" and d[H] == "-"),
    ("warming_wetting", "Warming–wetting pattern",
     "Temperature, precipitation and relative humidity are all rising.",
     lambda d: d[T] == "+" and d[P] == "+" and d[H] == "+"),
    ("warming_more_rain", "Warming with increasing precipitation",
     "Temperature and precipitation are rising; humidity shows no significant trend.",
     lambda d: d[T] == "+" and d[P] == "+" and d[H] == "0"),
    ("warming_more_humid", "Warming with rising humidity",
     "Temperature and relative humidity are rising; precipitation shows no significant trend.",
     lambda d: d[T] == "+" and d[P] == "0" and d[H] == "+"),
    ("cooling_wetting", "Cooling–wetting pattern",
     "Temperature is falling while precipitation and relative humidity are rising.",
     lambda d: d[T] == "-" and d[P] == "+" and d[H] == "+"),
    ("cooling_drying", "Cooling with decreasing precipitation and humidity",
     "Temperature, precipitation and relative humidity are all falling.",
     lambda d: d[T] == "-" and d[P] == "-" and d[H] == "-"),
    ("wetting", "Wetting without significant warming",
     "Precipitation and relative humidity are rising while temperature shows no significant trend.",
     lambda d: d[T] == "0" and d[P] == "+" and d[H] == "+"),
    ("warming_only", "Warming only",
     "Temperature is rising; precipitation and humidity show no significant trend.",
     lambda d: d[T] == "+" and d[P] == "0" and d[H] == "0"),
    ("cooling_only", "Cooling only",
     "Temperature is falling; precipitation and humidity show no significant trend.",
     lambda d: d[T] == "-" and d[P] == "0" and d[H] == "0"),
]

_PHRASE = {
    (T, "+"): "warming", (T, "-"): "cooling",
    (P, "+"): "increasing precipitation", (P, "-"): "decreasing precipitation",
    (H, "+"): "rising relative humidity", (H, "-"): "falling relative humidity",
}


def direction_code(tr: TrendResult | None) -> str:
    if tr is None or not tr.ok:
        return "NA"
    if not tr.significant:
        return "0"
    return "+" if tr.direction == "increasing" else "-"


def _correlations(annual: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rho = pd.DataFrame(np.nan, index=variables, columns=variables)
    pv = pd.DataFrame(np.nan, index=variables, columns=variables)
    for a in variables:
        for b in variables:
            pair = annual[[a, b]].dropna() if a != b else annual[[a]].dropna()
            if a == b:
                rho.loc[a, b], pv.loc[a, b] = 1.0, 0.0
            elif len(pair) >= config.MIN_ANNUAL_YEARS:
                r, p = stats.spearmanr(pair[a], pair[b])
                rho.loc[a, b], pv.loc[a, b] = float(r), float(p)
    return rho, pv


def build_fingerprint(
    annual_trends: dict[str, TrendResult],
    confidence: dict[str, ConfidenceResult],
    annual: pd.DataFrame,
) -> FingerprintResult:
    variables = [v for v in config.VARIABLE_ORDER if v in annual_trends]
    d = {v: direction_code(annual_trends.get(v)) for v in config.VARIABLE_ORDER}

    std_change = {}
    for v in variables:
        tr = annual_trends[v]
        std_change[v] = (tr.slope_per_decade / tr.std) if tr.ok and tr.std and tr.std > 0 else None

    caveats = [
        "A fingerprint describes trends that occur together; it does not show that one change caused another.",
        "Directions use the annual Mann-Kendall test at the "
        f"{int(config.ALPHA * 100)}% level; 'no significant trend' means the data do not show a clear "
        "monotonic change, not that the variable is unchanged.",
    ]
    missing = [v for v in config.VARIABLE_ORDER if d[v] == "NA"]
    if missing:
        names = ", ".join(config.VARIABLES[v].name for v in missing)
        caveats.append(f"Not assessed (not selected or insufficient data): {names}. "
                       "The pattern is based only on the remaining indicators.")

    pattern_id, label, desc = None, None, None
    if not missing:
        for pid, lab, text, pred in _RULES:
            if pred(d):
                pattern_id, label, desc = pid, lab, text
                break
        if pattern_id is None and all(d[v] == "0" for v in variables):
            pattern_id, label = "no_significant_change", "No statistically significant combined change"
            desc = ("None of the selected indicators shows a statistically significant monotonic trend "
                    "over this period.")
        elif pattern_id is None:
            parts = [_PHRASE[(v, d[v])] for v in config.VARIABLE_ORDER if d[v] in "+-" and (v, d[v]) in _PHRASE]
            pattern_id, label = "mixed", "Mixed change: " + ", ".join(parts)
            desc = ("A combination of significant trends that does not match one of the predefined named "
                    "patterns. Read each indicator's trend individually.")
    else:
        parts = [_PHRASE[(v, d[v])] for v in config.VARIABLE_ORDER if (v, d[v]) in _PHRASE]
        if parts:
            pattern_id, label = "partial", "Partial fingerprint: " + ", ".join(parts)
            desc = "Only some indicators could be assessed, so a named combined pattern is not assigned."
        else:
            pattern_id, label = "partial", "Partial fingerprint: no significant trends in assessed indicators"
            desc = "Only some indicators could be assessed, so a named combined pattern is not assigned."

    sig_scores = [confidence[v].score for v in variables
                  if d[v] in "+-" and v in confidence and confidence[v].score is not None]
    strength = float(np.mean(sig_scores)) if sig_scores else None

    rho, pv = _correlations(annual, variables) if len(variables) >= 2 else (pd.DataFrame(), pd.DataFrame())
    return FingerprintResult(
        pattern_id=pattern_id, label=label, description=desc, directions=d,
        std_change_per_decade=std_change, strength=strength, correlations=rho, correlation_p=pv,
        caveats=caveats,
    )


def pattern_catalog() -> list[tuple[str, str]]:
    """(label, description) of every named pattern, for display in the interface."""
    return [(label, desc) for _, label, desc, _ in _RULES]
