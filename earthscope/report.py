"""Downloadable CSV outputs. Pure functions returning DataFrames / CSV text."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from . import config
from .anomalies import flag_anomalies
from .models import InvestigationResult


def _rows_for(res: InvestigationResult, anomaly_threshold: float) -> list[dict]:
    loc = res.location.name
    period = f"{res.start_year}-{res.end_year}"
    rows: list[dict] = []

    def add(section, variable, metric, value, unit="", note=""):
        rows.append(dict(section=section, location=loc, period=period, variable=variable, metric=metric,
                         value=value, unit=unit, note=note))

    m = res.power_meta
    add("provenance", "", "data_source", config.DATA_SOURCE_NAME)
    add("provenance", "", "community", m.community)
    add("provenance", "", "requested_lat", m.requested_lat, "deg")
    add("provenance", "", "requested_lon", m.requested_lon, "deg")
    add("provenance", "", "returned_grid_lat", m.returned_lat, "deg")
    add("provenance", "", "returned_grid_lon", m.returned_lon, "deg")
    add("provenance", "", "elevation", m.elevation_m, "m")
    add("provenance", "", "retrieved_range", f"{m.start_year}-{m.end_year}")
    add("provenance", "", "first_valid_month", m.first_valid)
    add("provenance", "", "last_valid_month", m.last_valid)
    add("provenance", "", "api_version", m.api_version)
    add("provenance", "", "retrieved_at_utc", m.retrieved_at, note="served from local cache" if m.from_cache else "")
    add("provenance", "", "request_url", m.request_url)
    add("provenance", "", "note", config.DATA_PROVENANCE_NOTE)
    for w in res.warnings:
        add("warning", "", "warning", w)

    for v, q in res.quality.per_variable.items():
        u = config.VARIABLES[v].unit
        add("data_quality", v, "valid_months", q.valid_months, "months")
        add("data_quality", v, "expected_months", q.expected_months, "months")
        add("data_quality", v, "completeness", round(q.completeness, 4), "fraction")
        add("data_quality", v, "complete_years", q.years_complete, "years", f"of {q.years_requested}")
        add("data_quality", v, "parameter_unit", u)

    for v, vt in res.trends.items():
        tr, unit = vt.annual, config.VARIABLES[v].rate_unit
        add("trend_annual", v, "status", tr.status, note=tr.reason)
        add("trend_annual", v, "direction", tr.direction)
        if tr.ok:
            add("trend_annual", v, "n_annual_values", tr.n, "years")
            add("trend_annual", v, "mann_kendall_p", tr.mk_p)
            add("trend_annual", v, "kendall_tau", tr.mk_tau)
            add("trend_annual", v, "sens_slope_per_year", tr.sens_slope, f"{unit}/year")
            add("trend_annual", v, "sens_slope_per_decade", tr.slope_per_decade, f"{unit}/decade")
            add("trend_annual", v, "sens_slope_ci95_low_per_year", tr.sens_low, f"{unit}/year")
            add("trend_annual", v, "sens_slope_ci95_high_per_year", tr.sens_high, f"{unit}/year")
            add("trend_annual", v, "ols_slope_per_year", tr.ols_slope, f"{unit}/year")
            add("trend_annual", v, "ols_p", tr.ols_p)
            add("trend_annual", v, "ols_r2", tr.ols_r2)
            add("trend_annual", v, "period_mean", tr.mean, unit)
            add("trend_annual", v, "lag1_autocorrelation", tr.lag1_autocorr,
                note="serial correlation flagged" if tr.serial_corr_flag else "")
        s = vt.seasonal_mk
        if s is not None and s.ok:
            add("trend_seasonal", v, "seasonal_mk_direction", s.direction)
            add("trend_seasonal", v, "seasonal_mk_p", s.mk_p)
            add("trend_seasonal", v, "seasonal_sens_slope_per_year", s.sens_slope, f"{unit}/year")
        for label, st in vt.by_season.items():
            if st.ok:
                add("trend_season", v, f"{label}: direction", st.direction)
                add("trend_season", v, f"{label}: mann_kendall_p", st.mk_p)
                add("trend_season", v, f"{label}: sens_slope_per_year", st.sens_slope, f"{unit}/year")
        c = res.confidence[v]
        add("evidence_score", v, "score_0_100", None if c.score is None else round(c.score, 1))
        add("evidence_score", v, "label", c.label, note=c.note)
        for comp in c.components:
            add("evidence_score", v, f"component: {comp['name']}", round(comp["points"], 2),
                f"of {comp['weight']:.0f} points", comp["raw_text"])

    fp = res.fingerprint
    add("fingerprint", "", "pattern", fp.label)
    add("fingerprint", "", "description", fp.description)
    add("fingerprint", "", "evidence_strength_0_100", None if fp.strength is None else round(fp.strength, 1))
    for v, code in fp.directions.items():
        add("fingerprint", v, "direction_code", code, note="+ increasing, - decreasing, 0 no significant trend, NA not assessed")
    for v, x in fp.std_change_per_decade.items():
        add("fingerprint", v, "change_per_decade_in_interannual_sd", None if x is None else round(x, 3))

    flagged = flag_anomalies(res.anomalies, anomaly_threshold)
    for _, r in flagged.iterrows():
        add("anomaly", r["variable"], pd.Timestamp(r["date"]).strftime("%Y-%m"), round(float(r["value"]), 4),
            config.VARIABLES[r["variable"]].unit,
            f"{r['label']}; z={r['z']:+.2f}; month mean={r['clim_mean']:.3f}; threshold |z|>={anomaly_threshold:g}")
    for f in res.findings:
        add("finding", f.variable or "", f.title, f.text)
    return rows


def build_report(results: Iterable[InvestigationResult], anomaly_threshold: float = config.ANOMALY_Z) -> pd.DataFrame:
    rows: list[dict] = []
    for r in results:
        rows.extend(_rows_for(r, anomaly_threshold))
    return pd.DataFrame(rows, columns=["section", "location", "period", "variable", "metric", "value", "unit", "note"])


def report_csv(results: Iterable[InvestigationResult], anomaly_threshold: float = config.ANOMALY_Z) -> bytes:
    return build_report(results, anomaly_threshold).to_csv(index=False).encode("utf-8-sig")


def monthly_data_csv(results: Iterable[InvestigationResult]) -> bytes:
    """The cleaned monthly values behind the analysis (NaN = missing), with provenance columns."""
    frames = []
    for r in results:
        d = r.monthly.copy()
        d.insert(0, "month", d.index.strftime("%Y-%m"))
        d.insert(0, "location", r.location.name)
        d["source"] = "NASA POWER monthly (MERRA-2 based, model-derived)"
        frames.append(d.reset_index(drop=True))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(index=False).encode("utf-8-sig")
