"""Page 1 - Earth Overview: introduction, location / indicator / period selection, Start Investigation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from earthscope import config, investigation
from ui import components, state

state.init()
ss = st.session_state
locations = state.all_locations()

# ------------------------------------------------------------------ introduction
st.title("🌍 EARTHSCOPE")
st.markdown(f"**{config.TAGLINE}**")
st.write(
    "EARTHSCOPE investigates how environmental conditions have changed over the past 10–25 years at "
    "locations you choose, using historical **NASA POWER** monthly data. Instead of only drawing graphs, it "
    "tests whether changes are statistically significant, looks at several indicators together, compares "
    "places, flags unusual months and explains every finding in plain language."
)

steps = [
    ("Detect", "Trends in each indicator, with rate of change and a significance test."),
    ("Analyse", "Evidence score, seasonal view and data-quality checks."),
    ("Compare", "Opposite trends across locations and similar change patterns worldwide."),
    ("Connect", "The environmental fingerprint: indicators changing together."),
    ("Explain", "Template-written findings with the numbers behind them."),
]
for col, (title, text) in zip(st.columns(len(steps)), steps):
    with col.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)

with st.expander("What this tool does and does not do"):
    st.markdown(
        """
- Uses **NASA POWER monthly data** (temperature, precipitation, relative humidity). These are
  **model-derived grid-cell estimates** (MERRA-2 based), not ground-station measurements.
- Reports **statistical patterns**. A fingerprint of co-occurring trends is **not** evidence that one change
  caused another.
- The **evidence score** describes how strong and complete the available evidence is. It is not a forecast or a
  probability.
- Vegetation indices, soil moisture and predictions are **not** part of this version.
"""
    )

st.divider()

# ------------------------------------------------------------------ selection
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("1 · Locations")
    # Widgets always receive the persistent copy as their default. This works on every supported Streamlit
    # version: older ones derive a widget's identity from its parameters, so a default that appears only on
    # the first run would make later runs create a different widget and ignore the user's changes.
    st.multiselect(
        f"Locations to investigate (up to {state.MAX_LOCATIONS})", options=list(locations), key="w_sel",
        default=[n for n in ss.selected_locations if n in locations],
        max_selections=state.MAX_LOCATIONS, on_change=state.sync_locations,
        help="Demo locations first, then any custom locations you add, then the 30-location candidate pool.",
    )

    with st.expander("Add a custom location"):
        with st.form("custom_location", clear_on_submit=False):
            st.text_input("Name (optional)", key="w_cname", placeholder="e.g. My town")
            c1, c2 = st.columns(2)
            c1.number_input("Latitude (−90 to 90)", key="w_clat", value=None, min_value=-90.0, max_value=90.0,
                            format="%.4f", step=0.1, placeholder="e.g. 8.5241")
            c2.number_input("Longitude (−180 to 180)", key="w_clon", value=None, min_value=-180.0, max_value=180.0,
                            format="%.4f", step=0.1, placeholder="e.g. 76.9366")
            st.form_submit_button("Add location", on_click=state.add_custom_location)
        msg = ss.get("custom_msg")
        if msg:
            (st.success if msg[0] == "success" else st.error)(msg[1])

    st.subheader("2 · Environmental indicators")
    st.multiselect(
        "Indicators", options=config.VARIABLE_ORDER, key="w_vars", on_change=state.sync_variables,
        default=[v for v in ss.variables if v in config.VARIABLES],
        format_func=lambda v: f"{config.VARIABLES[v].name} ({config.VARIABLES[v].unit})",
    )
    for v in ss.get("w_vars", []):
        st.caption(f"**{config.VARIABLES[v].name}** — {config.VARIABLES[v].description}")
    st.caption("The environmental fingerprint needs all three indicators; with fewer it is reported as partial.")

    st.subheader("3 · Historical period")
    lo, hi = config.MIN_YEAR, config.last_full_year()
    start_default = (max(lo, min(ss.period[0], hi)), max(lo, min(ss.period[1], hi)))
    st.slider("Years (inclusive)", min_value=lo, max_value=hi, value=start_default, key="w_period",
              on_change=state.sync_period)

with right:
    st.subheader("Selected locations")
    sel_names = [n for n in ss.get("w_sel", []) if n in locations]
    if sel_names:
        sel_df = pd.DataFrame([{"Location": n, "lat": locations[n].lat, "lon": locations[n].lon,
                                "Note": locations[n].note} for n in sel_names])
        st.map(sel_df, latitude="lat", longitude="lon", size=30000, zoom=3 if len(sel_names) == 1 else None)
        st.dataframe(sel_df.rename(columns={"lat": "Latitude", "lon": "Longitude"}), hide_index=True)
    else:
        st.info("Select at least one location to see it on the map.")

# ------------------------------------------------------------------ start
st.divider()
variables = [v for v in config.VARIABLE_ORDER if v in ss.get("w_vars", [])]
start, end = ss.get("w_period", start_default)
problems: list[str] = []
if not sel_names:
    problems.append("Select at least one location.")
try:
    investigation.validate_settings(start, end, variables)
except investigation.InvestigationError as exc:
    problems.append(str(exc))

st.subheader("4 · Investigate")
st.write(
    f"**{len(sel_names)}** location(s) · **{', '.join(config.VARIABLES[v].name for v in variables) or 'no indicators'}** "
    f"· **{start}–{end}** ({end - start + 1} years)"
)
for p in problems:
    st.warning(p)
st.caption(
    "Each location needs one NASA POWER request, which can take several seconds. Responses are cached on disk "
    f"for {config.CACHE_TTL_DAYS} days, so repeating an investigation is fast."
)
if st.button("Start Investigation", type="primary", disabled=bool(problems)):
    state.run_investigations([locations[n] for n in sel_names], start, end, variables)

# ------------------------------------------------------------------ latest results
if ss.run_errors:
    for name, message in ss.run_errors.items():
        st.error(f"**{name}** — {message}")

if ss.investigations:
    rs = ss.run_settings
    st.subheader("Latest investigation")
    st.caption(
        f"{rs['start']}–{rs['end']} · indicators: {', '.join(config.VARIABLES[v].name for v in rs['variables'])}. "
        "Change the settings above and click **Start Investigation** to update."
    )
    for name, res in ss.investigations.items():
        with st.container(border=True):
            st.markdown(f"**{name}** — fingerprint: {res.fingerprint.label}")
            st.dataframe(components.trend_summary_table(res), hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.page_link("views/investigation.py", label="Open the investigation", icon="🔎")
    c2.page_link("views/fingerprint.py", label="See the fingerprint", icon="🧬")
    c3.page_link("views/report.py", label="Read the report", icon="📄")
