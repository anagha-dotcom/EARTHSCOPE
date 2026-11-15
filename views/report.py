"""Page 6 - Investigation Report: findings, evidence cards, statistics, provenance, CSV download."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from earthscope import config, report
from ui import components

results = components.require_results()
ss = st.session_state
rs = ss.run_settings

st.title("📄 Investigation Report")
st.caption(
    f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · period {rs['start']}–{rs['end']} · "
    f"indicators: {', '.join(config.VARIABLES[v].name for v in rs['variables'])} · "
    f"unusual months: |z| ≥ {config.ANOMALY_Z:g}"
)

# ------------------------------------------------------------------ key findings
st.subheader("Key findings")
for name, res in results.items():
    parts = []
    for v in res.variables:
        tr, info = res.trends[v].annual, config.VARIABLES[v]
        if tr.ok and tr.significant:
            word = info.up_word if tr.direction == "increasing" else info.down_word
            parts.append(f"{info.name.lower()} {word} ({tr.slope_per_decade:+.2g} {info.rate_unit}/decade, "
                         f"p {components.fmt_p(tr.mk_p)})")
    trends_text = "; ".join(parts) if parts else "no statistically significant trend in the selected indicators"
    st.markdown(f"- **{name}** — {res.fingerprint.label}. Significant trends: {trends_text}.")
st.caption("Generated automatically from the computed statistics using fixed templates. "
           "Co-occurring trends are not evidence of causation.")

# ------------------------------------------------------------------ evidence cards
st.subheader("Evidence cards")
KIND_ORDER = ["fingerprint", "trend", "seasonal", "confidence", "anomaly", "quality"]
tabs = st.tabs(list(results))
for tab, (name, res) in zip(tabs, results.items()):
    with tab:
        for kind in KIND_ORDER:
            for f in res.findings:
                if f.kind == kind:
                    components.evidence_card(f)

# ------------------------------------------------------------------ statistical results
st.subheader("Statistical results")
frames = []
for name, res in results.items():
    t = components.trend_summary_table(res)
    t.insert(0, "Location", name)
    frames.append(t)
st.dataframe(pd.concat(frames, ignore_index=True), hide_index=True)
components.confidence_explainer()

# ------------------------------------------------------------------ data sources
st.subheader("Data source information")
for name, res in results.items():
    st.markdown(f"**{name}**")
    components.provenance_panel(res)

# ------------------------------------------------------------------ downloads
st.subheader("Download")
report_bytes = report.report_csv(results.values(), config.ANOMALY_Z)
monthly_bytes = report.monthly_data_csv(results.values())
tag = f"{rs['start']}-{rs['end']}"
d1, d2 = st.columns(2)
d1.download_button("Download investigation report (CSV)", data=report_bytes, file_name=f"earthscope_report_{tag}.csv",
                   mime="text/csv", type="primary",
                   help="Provenance, data quality, trends, evidence scores, fingerprint, unusual months and findings.")
d2.download_button("Download cleaned monthly data (CSV)", data=monthly_bytes,
                   file_name=f"earthscope_monthly_data_{tag}.csv", mime="text/csv",
                   help="The monthly values behind the analysis (blank = missing), with the data source.")
with st.expander("Preview of the report CSV"):
    st.dataframe(report.build_report(results.values(), config.ANOMALY_Z).astype(str), hide_index=True)

# ------------------------------------------------------------------ limitations
st.subheader("Limitations")
st.markdown(
    """
- **Model-derived data.** NASA POWER meteorology is gridded (MERRA-2 based) and describes an area of roughly
  50 km, not a point or a weather station.
- **Trends describe, they do not explain.** A fingerprint or a correlation does not show that one change caused
  another, and a statistically significant trend can still have non-climatic causes.
- **Evidence score** reflects data completeness, significance and trend strength using documented weights. It is not a
  probability or a forecast.
- **Limited sample.** Trend tests use at most 25 annual values, so small real changes may go undetected, and
  testing several indicators and seasons means some significant results can occur by chance.
- **Not covered in this version:** vegetation indices, soil moisture, independent cross-dataset verification and
  predictive models.
"""
)
