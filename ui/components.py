"""Reusable Streamlit components shared by all pages."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from earthscope import config
from earthscope.models import Finding, InvestigationResult, TrendResult
from . import state

OVERVIEW = "views/overview.py"


def require_results() -> dict[str, InvestigationResult]:
    """Stop the page with a friendly message if no investigation has been run."""
    state.init()
    ss = st.session_state
    if not ss.investigations:
        st.info("No investigation has been run yet. Choose locations and indicators on the Earth Overview page, "
                "then click **Start Investigation**.")
        if ss.run_errors:
            for name, msg in ss.run_errors.items():
                st.error(f"{name}: {msg}")
        st.page_link(OVERVIEW, label="Go to Earth Overview", icon="🌍")
        st.stop()
    return ss.investigations


def active_result(results: dict[str, InvestigationResult]) -> InvestigationResult:
    """Sidebar selector for the location shown on single-location pages."""
    ss = st.session_state
    names = list(results)
    if ss.active_location not in names:
        ss.active_location = names[0]
    if len(names) > 1:
        choice = st.sidebar.selectbox("Active location", names, index=names.index(ss.active_location), key="w_active")
        ss.active_location = choice
    else:
        st.sidebar.markdown(f"**Location:** {names[0]}")
    return results[ss.active_location]


def sidebar_footer() -> None:
    st.sidebar.divider()
    st.sidebar.caption(
        "Data: NASA POWER (MERRA-2 based, model-derived grid-cell estimates). "
        "Findings describe statistical patterns, not causes."
    )


def direction_text(direction: str) -> str:
    return {
        "increasing": ":orange[▲ Increasing]",
        "decreasing": ":blue[▼ Decreasing]",
        "no significant trend": ":gray[● No significant trend]",
        "insufficient data": ":gray[– Insufficient data]",
    }.get(direction, direction)


def fmt(x: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "–"
    return f"{x:+.{digits}g}" if signed else f"{x:.{digits}g}"


def fmt_p(p: Optional[float]) -> str:
    if p is None or pd.isna(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def provenance_panel(res: InvestigationResult) -> None:
    m = res.power_meta
    with st.expander("Data source and provenance", expanded=False):
        rows = {
            "Source": config.DATA_SOURCE_NAME,
            "Location": f"{res.location.name} — requested {m.requested_lat:.4f}°, {m.requested_lon:.4f}°",
            "NASA grid cell returned": ("–" if m.returned_lat is None else f"{m.returned_lat:.4f}°, {m.returned_lon:.4f}°")
            + ("" if m.elevation_m is None else f"  (elevation {m.elevation_m:.0f} m)"),
            "Analysis period": f"{res.start_year}–{res.end_year}",
            "Data retrieved range": f"{m.start_year}–{m.end_year} (first valid {m.first_valid}, last valid {m.last_valid})",
            "Parameters": "; ".join(
                f"{v} = {config.VARIABLES[v].long_name} [{(m.parameters_info.get(v) or {}).get('units') or config.VARIABLES[v].api_unit}]"
                for v in res.variables),
            "Community / API version": f"{m.community} / {m.api_version or 'not reported'}",
            "Retrieved at (UTC)": m.retrieved_at + ("  (served from local cache)" if m.from_cache else "  (live request)"),
        }
        st.table(pd.Series(rows, name="").to_frame())
        st.caption(config.DATA_PROVENANCE_NOTE)
        st.caption("Precipitation is a mean daily rate (mm/day) for each month, not a monthly total.")
        if m.request_url:
            st.caption(f"Request: {m.request_url}")
    if res.warnings:
        with st.expander(f"⚠️ {len(res.warnings)} data notice(s)", expanded=False):
            for w in res.warnings:
                st.markdown(f"- {w}")


def evidence_card(f: Finding) -> None:
    badge = {"trend": "TREND", "seasonal": "SEASONAL", "confidence": "EVIDENCE SCORE", "fingerprint": "FINGERPRINT",
             "anomaly": "ANOMALY", "quality": "DATA QUALITY"}.get(f.kind, f.kind.upper())
    with st.container(border=True):
        st.caption(badge)
        st.markdown(f"**{f.title}**")
        st.write(f.text)


def confidence_explainer() -> None:
    with st.expander("How the evidence score works (and what it is not)"):
        st.markdown(
            f"""
The **evidence score (0–100)** summarises how strong and complete the *available evidence for a trend* is.
It is **not** a probability that the trend is real, and **not** a forecast.

| Component | Weight | How it is scored |
|---|---|---|
| Data completeness | {config.CONF_W_COMPLETENESS:.0f} | share of expected months with valid data |
| Statistical significance | {config.CONF_W_SIGNIFICANCE:.0f} | Mann-Kendall p ≤ {config.CONF_P_FULL} → full points; p ≥ {config.CONF_P_ZERO} → zero; linear in between |
| Trend strength | {config.CONF_W_STRENGTH:.0f} | \\|Kendall τ\\| ÷ {config.CONF_TAU_FULL}, capped at 1 |

Bands: **Weak** < {config.CONF_BAND_MODERATE:.0f}, **Moderate** {config.CONF_BAND_MODERATE:.0f}–{config.CONF_BAND_STRONG - 1:.0f}, **Strong** ≥ {config.CONF_BAND_STRONG:.0f}.
A trend that is not significant at the {int(config.ALPHA * 100)}% level is always labelled **Weak**.

**Limitations**
- Weights and cut-offs are design choices made for transparency, not established scientific constants.
- A low score for "no trend" means little evidence *of change*; it is not evidence that nothing changed.
- Mann-Kendall assumes roughly independent yearly values; serial correlation can make p-values too optimistic (flagged when detected).
- 25 annual values give limited statistical power; the data are model-derived grid-cell estimates, not station records.
"""
        )


def trend_summary_table(res: InvestigationResult) -> pd.DataFrame:
    rows = []
    for v in res.variables:
        tr: TrendResult = res.trends[v].annual
        c = res.confidence[v]
        info = config.VARIABLES[v]
        rows.append({
            "Indicator": info.name,
            "Trend": ("▲ " if tr.direction == "increasing" else "▼ " if tr.direction == "decreasing" else "● " if tr.ok else "– ") + tr.direction.capitalize(),
            f"Rate per decade": "–" if not tr.ok else f"{tr.slope_per_decade:+.3g} {info.rate_unit}",
            "Mann-Kendall p": fmt_p(tr.mk_p),
            "Valid years": tr.n,
            "Evidence score": "–" if c.score is None else f"{c.score:.0f}/100 ({c.label})",
        })
    return pd.DataFrame(rows)
