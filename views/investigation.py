"""Page 2 - Environmental Investigation: time series, trend, rate of change, significance, evidence score."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from earthscope import config
from earthscope.models import InvestigationResult
from ui import charts, components

results = components.require_results()
res = components.active_result(results)


def finding_text(r: InvestigationResult, kind: str, var: str) -> str:
    return next((f.text for f in r.findings if f.kind == kind and f.variable == var), "")


st.title("🔎 Environmental Investigation")
st.caption(f"**{res.location.name}** · {res.start_year}–{res.end_year}")
components.provenance_panel(res)

st.subheader("Summary of trends")
st.dataframe(components.trend_summary_table(res), hide_index=True)
st.caption(
    "Trends use annual means of complete years. Direction and significance come from the Mann-Kendall test "
    f"(α = {config.ALPHA}); the rate of change is Sen's slope per decade. Significance and magnitude are "
    "reported separately: a small change can be significant and a large one can fail the test."
)
components.confidence_explainer()

st.subheader("Indicator details")
tabs = st.tabs([config.VARIABLES[v].name for v in res.variables])
for tab, var in zip(tabs, res.variables):
    info = config.VARIABLES[var]
    vt = res.trends[var]
    tr = vt.annual
    conf = res.confidence[var]
    with tab:
        st.plotly_chart(charts.timeseries_chart(res, var), key=f"ts_{res.location.name}_{var}")
        st.caption("Thin line: monthly values · points: annual means · dashed line: Sen's slope trend "
                   "(dotted if not significant).")

        if not tr.ok:
            st.warning(f"Trend not assessed: {tr.reason}")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown("**Trend**")
            m1.markdown(components.direction_text(tr.direction))
            m2.metric(f"Rate of change ({info.rate_unit}/decade)", f"{tr.slope_per_decade:+.3g}",
                      help="Sen's slope × 10. Robust to outliers.")
            m3.metric("Mann-Kendall p-value", components.fmt_p(tr.mk_p),
                      help=f"Significant when below {config.ALPHA}. Says nothing about how large the change is.")
            m4.metric("Evidence score", "–" if conf.score is None else f"{conf.score:.0f}/100",
                      help="Strength and completeness of the evidence, not a probability.")
            st.write(finding_text(res, "trend", var))

            with st.expander("Detailed statistics"):
                rows = {
                    "Valid annual values": f"{tr.n} ({tr.first_year}–{tr.last_year})",
                    "Sen's slope": f"{tr.sens_slope:+.4g} {info.rate_unit}/year "
                                   f"(95% CI {tr.sens_low:+.4g} to {tr.sens_high:+.4g})",
                    "Total change over the period (Sen's slope × years)": f"{tr.total_change:+.3g} {info.rate_unit}",
                    "Kendall tau": components.fmt(tr.mk_tau, 3, signed=True),
                    "Mann-Kendall p": components.fmt_p(tr.mk_p),
                    "Period mean ± SD (annual)": f"{tr.mean:.3g} ± {tr.std:.2g} {info.unit}",
                    "OLS slope (cross-check)": "–" if tr.ols_slope is None else
                    f"{tr.ols_slope:+.4g} {info.rate_unit}/year (p = {components.fmt_p(tr.ols_p)}, R² = {tr.ols_r2:.2f})",
                    "Lag-1 autocorrelation of residuals": components.fmt(tr.lag1_autocorr, 2, signed=True)
                    + ("  ⚠️ serial correlation: p-value may be too optimistic" if tr.serial_corr_flag else ""),
                }
                if tr.pct_per_decade is not None:
                    rows["Change as % of mean per decade"] = f"{tr.pct_per_decade:+.3g}%"
                st.table(pd.Series(rows, name="").to_frame())

        # ---- seasonal view
        st.markdown("**Seasonal view** (supporting evidence)")
        srows = []
        sm = vt.seasonal_mk
        if sm is not None and sm.ok:
            srows.append({"View": "All months (seasonal Mann-Kendall)", "Trend": sm.direction.capitalize(),
                          f"Rate per decade ({info.rate_unit})": f"{sm.slope_per_decade:+.3g}",
                          "p": components.fmt_p(sm.mk_p), "Years": sm.n})
        for label, st_res in vt.by_season.items():
            if st_res.ok:
                srows.append({"View": label, "Trend": st_res.direction.capitalize(),
                              f"Rate per decade ({info.rate_unit})": f"{st_res.slope_per_decade:+.3g}",
                              "p": components.fmt_p(st_res.mk_p), "Years": st_res.n})
            else:
                srows.append({"View": label, "Trend": "Not assessed",
                              f"Rate per decade ({info.rate_unit})": "–", "p": "–", "Years": st_res.n})
        if srows:
            st.dataframe(pd.DataFrame(srows), hide_index=True)
        st.write(finding_text(res, "seasonal", var))
        st.caption("Season labels are month ranges (hemisphere-neutral). Each season is tested on its own, so with "
                   "four seasons about one in five results could be significant by chance alone.")

        # ---- evidence score
        st.markdown("**Evidence score breakdown**")
        if conf.score is None:
            st.info(f"Not assessable. {conf.note}")
        else:
            st.dataframe(pd.DataFrame([
                {"Component": c["name"], "Observed": c["raw_text"], "Points": f"{c['points']:.1f} / {c['weight']:.0f}"}
                for c in conf.components
            ]), hide_index=True)
            st.markdown(f"**Total: {conf.score:.0f}/100 — {conf.label} evidence.** {conf.note}")

st.subheader("Data quality")
q = res.quality
st.dataframe(pd.DataFrame([{
    "Indicator": config.VARIABLES[v].name, "Valid months": f"{pq.valid_months} / {pq.expected_months}",
    "Completeness": f"{pq.completeness * 100:.1f}%", "Complete years": f"{pq.years_complete} / {pq.years_requested}",
    "Years with missing months": ", ".join(map(str, pq.incomplete_years)) or "none",
} for v, pq in q.per_variable.items()]), hide_index=True)
st.caption("Years with any missing month are excluded from annual means (so a missing season cannot bias them). "
           "Missing values are never filled in or estimated.")
