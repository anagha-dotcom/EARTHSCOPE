"""Page 3 - Environmental Fingerprint: indicators analysed together."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from earthscope import config
from earthscope.fingerprint import pattern_catalog
from ui import charts, components

results = components.require_results()
res = components.active_result(results)
fp = res.fingerprint

st.title("🧬 Environmental Fingerprint")
st.caption(f"**{res.location.name}** · {res.start_year}–{res.end_year}")

with st.container(border=True):
    st.caption("ENVIRONMENTAL FINGERPRINT")
    st.subheader(fp.label)
    st.write(fp.description)
    if fp.strength is not None:
        st.write(f"Mean evidence score of the indicators with significant trends: **{fp.strength:.0f}/100**.")
    else:
        st.write("No indicator has a statistically significant trend, so there is no evidence score to average.")

st.subheader("Direction of each indicator")
cols = st.columns(len(res.variables))
for col, var in zip(cols, res.variables):
    tr = res.trends[var].annual
    info = config.VARIABLES[var]
    with col.container(border=True):
        st.markdown(f"**{info.name}**")
        st.markdown(components.direction_text(tr.direction))
        if tr.ok:
            st.caption(f"{tr.slope_per_decade:+.3g} {info.rate_unit}/decade · Mann-Kendall p {components.fmt_p(tr.mk_p)}")
        else:
            st.caption(tr.reason)

st.subheader("Size of change, on a common scale")
st.plotly_chart(charts.fingerprint_bars(fp, res.trends), key=f"fpbars_{res.location.name}")
st.caption("Each bar is the Sen's slope per decade divided by the indicator's year-to-year standard deviation, so "
           "°C, mm/day and % can be compared. Faded bars are not statistically significant.")

st.subheader("How the indicators move together")
if fp.correlations.empty:
    st.info("Select at least two indicators with enough annual data to compute relationships between them.")
else:
    st.plotly_chart(charts.correlation_heatmap(fp), key=f"fpcorr_{res.location.name}")
    st.caption("Spearman rank correlation between annual series (* = p < 0.05). This measures whether indicators "
               "rise and fall together from year to year. It does not show that one drives another, and it "
               "can reflect shared variability rather than a shared long-term change.")

st.subheader("Explained findings")
for f in res.findings:
    if f.kind == "fingerprint":
        components.evidence_card(f)
for f in res.findings:
    if f.kind == "trend":
        components.evidence_card(f)

st.subheader("Caveats")
for c in fp.caveats:
    st.markdown(f"- {c}")

if len(results) > 1:
    st.subheader("Fingerprints of all investigated locations")
    rows = []
    for name, r in results.items():
        row = {"Location": name, "Fingerprint": r.fingerprint.label}
        for v in config.VARIABLE_ORDER:
            sym = {"+": "▲", "-": "▼", "0": "●", "NA": "–"}[r.fingerprint.directions.get(v, "NA")]
            row[config.VARIABLES[v].name] = sym
        row["Evidence"] = "–" if r.fingerprint.strength is None else f"{r.fingerprint.strength:.0f}/100"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("▲ significant increase · ▼ significant decrease · ● no significant trend · – not assessed.")

with st.expander("All named patterns this tool can recognise"):
    for label, desc in pattern_catalog():
        st.markdown(f"**{label}** — {desc}")
    st.caption("Patterns are defined by fixed rules over the three trend directions. Combinations that match none "
               "of them are reported as 'Mixed change'. No machine learning is involved.")
