"""Rule/template-based explanations. No AI services: every sentence is built from computed numbers."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config
from .models import (ConfidenceResult, DataQuality, FingerprintResult, Finding, PowerMeta,
                     TrendResult)


def num(x: Optional[float], signed: bool = False) -> str:
    if x is None:
        return "n/a"
    s = f"{x:+.3g}" if signed else f"{x:.3g}"
    return s


def pfmt(p: Optional[float]) -> str:
    if p is None:
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def describe_trend(tr: TrendResult, period: str) -> str:
    info = config.VARIABLES[tr.variable]
    name = info.long_name
    if not tr.ok:
        return f"{name}: not assessed. {tr.reason}"
    ru = info.rate_unit
    rate = f"{num(tr.sens_slope, True)} {ru} per year ({num(tr.slope_per_decade, True)} {ru} per decade)"
    ci = f"95% CI of Sen's slope {num(tr.sens_low, True)} to {num(tr.sens_high, True)} {ru}/year"
    stat = f"Mann-Kendall {pfmt(tr.mk_p)}, Kendall tau = {num(tr.mk_tau, True)}"
    basis = f"{tr.n} annual means, {period}"
    if tr.significant:
        word = "increasing" if tr.direction == "increasing" else "decreasing"
        s = (f"{name} shows a statistically significant {word} trend of {rate} ({basis}). "
             f"{stat}; {ci}.")
        if tr.pct_per_decade is not None:
            s += f" This is about {num(tr.pct_per_decade, True)}% of the period mean per decade."
    else:
        s = (f"{name} shows no statistically significant monotonic trend ({basis}). {stat}. "
             f"The estimated Sen's slope ({num(tr.sens_slope, True)} {ru}/year) cannot be distinguished "
             "from zero at the 5% level. This does not prove that no change occurred; it means the data do "
             "not provide enough evidence of a consistent one.")
    if tr.serial_corr_flag:
        s += (" Caution: year-to-year residuals are serially correlated "
              f"(lag-1 r = {num(tr.lag1_autocorr, True)}), which can make the p-value too optimistic.")
    return s


def describe_seasonal(tr: Optional[TrendResult], by_season: dict[str, TrendResult]) -> str:
    if tr is None or not tr.ok:
        return "Seasonal test not run: " + (tr.reason if tr else "no data")
    name = config.VARIABLES[tr.variable].long_name
    verdict = (f"a significant {tr.direction} trend" if tr.significant else "no significant trend")
    s = (f"Seasonal Mann-Kendall on monthly values (accounts for the annual cycle) finds {verdict} "
         f"for {name} ({pfmt(tr.mk_p)}; seasonal Sen's slope {num(tr.sens_slope, True)} {config.VARIABLES[tr.variable].rate_unit}/year).")
    sig = [f"{k} ({v.direction})" for k, v in by_season.items() if v.ok and v.significant]
    if sig:
        s += " Seasons with a significant trend: " + ", ".join(sig) + "."
    else:
        s += " No individual three-month season shows a significant trend."
    return s


def describe_confidence(c: ConfidenceResult) -> str:
    name = config.VARIABLES[c.variable].name
    if c.score is None:
        return f"Evidence score for {name}: not assessable. {c.note}"
    parts = "; ".join(f"{x['name'].lower()} {x['points']:.0f}/{x['weight']:.0f}" for x in c.components)
    s = f"Evidence score for {name}: {c.score:.0f}/100 ({c.label.lower()} evidence) — {parts}."
    return s + (" " + c.note if c.note else "")


def describe_fingerprint(fp: FingerprintResult) -> str:
    s = f"{fp.label}. {fp.description}"
    if fp.strength is not None:
        s += f" Mean evidence score of the variables with significant trends: {fp.strength:.0f}/100."
    return s + " This describes co-occurring trends and does not establish causation."


def describe_anomalies(flagged: pd.DataFrame, n_obs: int, threshold: float, expected: float) -> str:
    if flagged.empty:
        return (f"No months exceeded |z| ≥ {threshold:g} relative to their calendar-month average "
                f"({n_obs} monthly values checked; about {expected:.0f} would be expected by chance if values were normally distributed).")
    top = flagged.iloc[0]
    name = config.VARIABLES[top["variable"]].name.lower()
    when = pd.Timestamp(top["date"]).strftime("%B %Y")
    return (f"{len(flagged)} unusual months (|z| ≥ {threshold:g}) among {n_obs} monthly values; "
            f"about {expected:.0f} would be expected by chance if values were normally distributed. "
            f"Largest: {name} in {when} was {top['label']} (z = {top['z']:+.1f}; value {top['value']:.2f} vs "
            f"usual {top['clim_mean']:.2f} {config.VARIABLES[top['variable']].unit}).")


def describe_quality(q: DataQuality, meta: PowerMeta) -> str:
    s = f"Overall data completeness {q.overall_completeness * 100:.1f}% of expected months."
    if q.notes:
        s += " " + " ".join(q.notes)
    return s


def build_findings(
    location_name: str,
    period: str,
    trends: dict,
    confidence: dict,
    fingerprint: FingerprintResult,
    flagged: pd.DataFrame,
    n_obs: int,
    quality: DataQuality,
    meta: PowerMeta,
    anomaly_threshold: float,
    expected_chance: float,
) -> list[Finding]:
    f: list[Finding] = []
    f.append(Finding("quality", "Data quality", describe_quality(quality, meta),
                     {"completeness": quality.overall_completeness}))
    for var, vt in trends.items():
        tr = vt.annual
        f.append(Finding("trend", f"{config.VARIABLES[var].name} trend", describe_trend(tr, period),
                         {"p": tr.mk_p, "sens_slope_per_year": tr.sens_slope, "tau": tr.mk_tau, "n": tr.n},
                         variable=var))
        f.append(Finding("seasonal", f"{config.VARIABLES[var].name}: seasonal view",
                         describe_seasonal(vt.seasonal_mk, vt.by_season), {}, variable=var))
        f.append(Finding("confidence", f"{config.VARIABLES[var].name}: evidence score",
                         describe_confidence(confidence[var]),
                         {"score": confidence[var].score, "label": confidence[var].label}, variable=var))
    f.append(Finding("fingerprint", "Environmental fingerprint", describe_fingerprint(fingerprint),
                     {"pattern": fingerprint.pattern_id, "directions": fingerprint.directions}))
    f.append(Finding("anomaly", "Unusual observations",
                     describe_anomalies(flagged, n_obs, anomaly_threshold, expected_chance),
                     {"flagged": int(len(flagged)), "threshold": anomaly_threshold}))
    return f
