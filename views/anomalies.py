"""Page 5 - Anomalies and Timeline: unusual observations kept separate from long-term trends."""
from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st

from earthscope import config
from earthscope.anomalies import expected_by_chance, flag_anomalies
from earthscope.narratives import describe_anomalies
from ui import charts, components

results = components.require_results()
res = components.active_result(results)

st.title("📈 Anomalies and Timeline")
st.caption(f"**{res.location.name}** · {res.start_year}–{res.end_year}")

c1, c2 = st.columns([2, 1])
var = c1.radio("Indicator", res.variables, format_func=lambda v: config.VARIABLES[v].name, horizontal=True)
thr = c2.slider("Unusual-month threshold (|z| ≥)", 1.5, 3.0, float(config.ANOMALY_Z), 0.25,
                help="z = how many standard deviations a month is from the average of the same calendar month.")
info = config.VARIABLES[var]

flagged = flag_anomalies(res.anomalies, thr)
sub = flagged[flagged["variable"] == var]
n_obs = int(res.anomalies.loc[res.anomalies["variable"] == var, "z"].notna().sum())
st.info(describe_anomalies(sub, n_obs, thr, expected_by_chance(n_obs, thr)))

st.subheader("Environmental change timeline")
st.plotly_chart(charts.monthly_timeline_chart(res, var, thr), key=f"tl_{res.location.name}_{var}_{thr}")
st.caption("Grey band: usual range for each calendar month (average ± threshold × standard deviation). Dotted line: "
           "the same-month historical average for the selected period. Diamonds: unusual months. Dashed line: "
           "long-term Sen's slope trend.")

st.subheader("Unusual observations")
if sub.empty:
    st.write("No months exceed the threshold for this indicator.")
else:
    st.dataframe(pd.DataFrame({
        "Month": pd.to_datetime(sub["date"]).dt.strftime("%Y-%m"),
        "What happened": sub["label"],
        f"Value ({info.unit})": sub["value"].round(3),
        f"Usual for that month ({info.unit})": sub["clim_mean"].round(3),
        f"Departure ({info.unit})": sub["deviation"].round(3),
        "z-score": sub["z"].round(2),
    }), hide_index=True)

st.plotly_chart(charts.anomaly_chart(res, var, thr), key=f"an_{res.location.name}_{var}_{thr}")
st.caption("Bars: z-score of every month. Highlighted bars pass the threshold (and, for precipitation, a minimum "
           "absolute departure so trivial differences in near-zero months are not flagged).")

per_year = sub.groupby("year").size().reindex(range(res.start_year, res.end_year + 1), fill_value=0)
with st.expander("Unusual months per year"):
    st.bar_chart(per_year.rename("Unusual months"))
    st.caption("If an indicator has a strong long-term trend, unusual months tend to cluster at one end of the period, "
               "because they are measured against the period average rather than against the trend line.")

st.subheader("Comparison with historical averages")
st.plotly_chart(charts.departure_chart(res, var), key=f"dep_{res.location.name}_{var}")
st.caption("Annual mean minus the mean of all valid years in the period. Years with missing months are left out.")

an = res.anomalies[res.anomalies["variable"] == var]
clim = an.groupby(an["date"].dt.month).agg(avg=("clim_mean", "mean"), sd=("clim_std", "mean"), n=("value", "count"))
with st.expander("Historical average for each calendar month"):
    st.dataframe(pd.DataFrame({
        "Month": [calendar.month_abbr[m] for m in clim.index],
        f"Average ({info.unit})": clim["avg"].round(3).to_numpy(),
        f"Standard deviation ({info.unit})": clim["sd"].round(3).to_numpy(),
        "Years with data": clim["n"].to_numpy(),
    }), hide_index=True)

st.caption(
    "Anomalies and trends answer different questions. A trend asks whether values drift over the whole period; an "
    "anomaly asks whether one month is unusual for its time of year. Some unusual months are expected by chance."
)
