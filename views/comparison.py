"""Page 4 - Regional Comparison: trend comparison, Opposite Trend Finder, Similar Earth Regions Discovery."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from earthscope import comparison, config, investigation, pool
from ui import charts, components

results = components.require_results()
ss = st.session_state
settings = ss.run_settings
variables = [v for v in config.VARIABLE_ORDER if v in settings["variables"]]
start, end = settings["start"], settings["end"]
SYM = {"+": "▲", "-": "▼", "0": "●", "NA": "–"}
MAX_PAIRS_SHOWN = 10


def var_label(v: str) -> str:
    return f"{config.VARIABLES[v].name} ({config.VARIABLES[v].unit})"


def scan_pool() -> None:
    """Fetch the fixed candidate pool (full period, all indicators, so cache keys match the investigations)."""
    bar = st.progress(0.0, text="Starting pool retrieval…")

    def cb(i: int, n: int, msg: str) -> None:
        bar.progress(min(i / n, 1.0), text=msg)

    power, errors = pool.fetch_pool(config.MIN_YEAR, config.last_full_year(), config.VARIABLE_ORDER, progress=cb)
    bar.empty()
    ss.pool_power, ss.pool_fetch_errors, ss.pool_analysis = power, errors, None


def pool_results() -> dict | None:
    """Pool analysed for the investigation period (recomputed only when the settings change)."""
    if not ss.pool_power:
        return None
    key = (start, end, tuple(variables))
    pa = ss.pool_analysis
    if pa is None or pa["key"] != key:
        res, errs = investigation.analyze_pool(ss.pool_power, start, end, variables, pool.CANDIDATE_POOL)
        pa = ss.pool_analysis = {"key": key, "results": res, "errors": errs}
    return pa["results"]


def display_trend_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in table.iterrows():
        info = config.VARIABLES[r["variable"]]
        ok = r["status"] == "ok"
        rows.append({
            "Location": r["location"], "Indicator": info.name,
            "Trend": f"{charts.DIR_SYMBOL[r['direction']]} {r['direction'].capitalize()}",
            f"Rate per decade": f"{r['slope_per_decade']:+.3g} {info.rate_unit}" if ok else "–",
            "95% CI per decade": f"{r['slope_low_per_decade']:+.3g} to {r['slope_high_per_decade']:+.3g}" if ok else "–",
            "Mann-Kendall p": components.fmt_p(r["p_value"]),
            "Evidence score": "–" if pd.isna(r["evidence_score"]) else f"{r['evidence_score']:.0f}/100",
        })
    return pd.DataFrame(rows)


st.title("🗺️ Regional Comparison")
st.caption(f"{start}–{end} · indicators: {', '.join(config.VARIABLES[v].name for v in variables)} · "
           f"{len(results)} investigated location(s)")

tab_trends, tab_opposite, tab_similar = st.tabs(["Trend comparison", "Opposite Trend Finder", "Similar Earth Regions"])

# ------------------------------------------------------------------ tab 1: trend comparison
with tab_trends:
    table = comparison.trend_table(results, variables)
    if len(results) < 2:
        st.info("Only one location has been investigated. Add more on the Earth Overview page to compare them side by side.")
    for var in variables:
        st.markdown(f"#### {var_label(var)}")
        if table[(table["variable"] == var) & (table["status"] == "ok")].empty:
            st.info("No location has enough valid data for this indicator.")
            continue
        st.plotly_chart(charts.comparison_bars(table, var), key=f"cmp_{var}")
    st.caption("Bars show Sen's slope per decade with its 95% confidence interval. Faded bars are not statistically "
               "significant at the 5% level. Compare the units of each indicator separately.")
    st.dataframe(display_trend_table(table), hide_index=True)

# ------------------------------------------------------------------ tab 2: opposite trends
with tab_opposite:
    st.write("Does the same indicator change in **opposite directions** in different places? Only trends that are "
             "statistically significant are counted as increasing or decreasing; everything else is 'no significant trend'.")
    pr = pool_results()
    include_pool = False
    if pr:
        include_pool = st.checkbox("Also include the 30-location candidate pool", value=False,
                                   help="Uses the pool retrieved on the 'Similar Earth Regions' tab.")
    elif len(results) < 2:
        st.info("Investigate at least two locations, or scan the candidate pool on the 'Similar Earth Regions' tab, "
                "to look for opposite trends.")
    pool_part = {k: v for k, v in (pr or {}).items() if k not in results} if include_pool else {}
    t_all = comparison.trend_table({**results, **pool_part}, variables)
    opp = comparison.find_opposite_trends(t_all)
    for _, row in opp.iterrows():
        var = row["variable"]
        with st.container(border=True):
            st.markdown(f"**{var_label(var)}**")
            if row["opposite"]:
                st.warning(f"Opposite significant trends: {row['n_increasing']} location(s) increasing and "
                           f"{row['n_decreasing']} decreasing.")
            elif row["n_increasing"] or row["n_decreasing"]:
                st.success("No opposite significant trends: all significant trends point the same way.")
            else:
                st.info("No location shows a statistically significant trend.")
            for label, key in (("▲ Significant increase", "increasing"), ("▼ Significant decrease", "decreasing"),
                               ("● No significant trend", "no_significant")):
                names = row[key]
                st.markdown(f"{label}: " + (", ".join(names) if names else "none"))
            if row["n_not_assessed"]:
                st.caption(f"{row['n_not_assessed']} location(s) could not be assessed for this indicator.")
            if row["opposite"]:
                pairs = comparison.opposite_pairs(t_all, var)
                pairs = pairs.assign(_gap=pairs["increasing_per_decade"] - pairs["decreasing_per_decade"]) \
                             .sort_values("_gap", ascending=False)
                u = config.VARIABLES[var].rate_unit
                st.dataframe(pairs.head(MAX_PAIRS_SHOWN).drop(columns="_gap").rename(columns={
                    "increasing_location": "Increasing at", "decreasing_location": "Decreasing at",
                    "increasing_per_decade": f"Increase ({u}/decade)", "decreasing_per_decade": f"Decrease ({u}/decade)",
                }).round(3), hide_index=True)
                if len(pairs) > MAX_PAIRS_SHOWN:
                    st.caption(f"Showing the {MAX_PAIRS_SHOWN} pairs with the largest contrast, out of {len(pairs)}.")
    map_var = st.selectbox("Map indicator", variables, format_func=var_label, key="w_cmp_map_var")
    st.plotly_chart(charts.direction_map(t_all, map_var, highlight=set(results)), key=f"dirmap_{map_var}_{include_pool}")
    st.caption("Large outlined symbols are your investigated locations. Symbols and labels, not colour alone, show direction.")

# ------------------------------------------------------------------ tab 3: similar regions
with tab_similar:
    st.write(
        "Which places in the world have a **change pattern** most like the selected location? Each location is "
        "described by its Sen's slopes per decade for the selected indicators, standardised across a fixed pool of "
        "30 locations; locations are ranked by the root-mean-square difference between those standardised slopes "
        "(smaller = more similar). Plain mathematics, no AI."
    )
    st.caption("Similar change patterns do not mean similar climates, and nothing here implies a common cause.")

    if len(variables) < 2:
        st.warning("Similarity needs at least two indicators. Select more on the Earth Overview page and re-run.")
        st.stop()

    n_pool = len(pool.CANDIDATE_POOL)
    if not ss.pool_power:
        st.info(f"The candidate pool ({n_pool} locations) has not been retrieved yet. This makes up to {n_pool} NASA "
                "POWER requests one after another with a short pause, which can take a few minutes the first time. "
                f"Results are cached on disk for {config.CACHE_TTL_DAYS} days.")
        if st.button(f"Retrieve candidate pool ({n_pool} locations)", type="primary"):
            scan_pool()
            st.rerun()
        st.stop()

    got, failed = len(ss.pool_power), ss.pool_fetch_errors
    st.write(f"Pool retrieved: **{got}** of {n_pool} locations" + (f" · **{len(failed)} failed**" if failed else ""))
    if failed:
        with st.expander("Locations that could not be retrieved"):
            for name, msg in failed.items():
                st.markdown(f"- **{name}** — {msg}")
        if st.button("Retry failed locations"):
            scan_pool()
            st.rerun()

    pr = pool_results() or {}
    if ss.pool_analysis and ss.pool_analysis["errors"]:
        with st.expander("Pool locations that could not be analysed"):
            for name, msg in ss.pool_analysis["errors"].items():
                st.markdown(f"- **{name}** — {msg}")

    c1, c2 = st.columns([2, 1])
    qname = c1.selectbox("Find regions similar to", list(results), key="w_sim_query")
    top_k = c2.slider("Regions to show", 3, 10, config.SIMILAR_TOP_K, key="w_sim_k")
    query = results[qname]
    ranking, notes = comparison.similar_regions(query, pr, variables, top_k=top_k)

    if ranking.empty:
        for n in notes:
            st.warning(n)
    else:
        coords = {loc.name: (loc.lat, loc.lon) for loc in pool.CANDIDATE_POOL}
        st.plotly_chart(charts.similar_map(qname, query.location.lat, query.location.lon, ranking, coords),
                        key=f"simmap_{qname}_{top_k}")
        shown = [v for v in variables if f"{v}_slope_per_decade" in ranking.columns]

        def cell(slope, code, v) -> str:
            return "–" if pd.isna(slope) else f"{SYM.get(code, '–')} {slope:+.3g}"

        rows = [{"Rank": "★", "Location": f"{qname} (selected)", "Distance": "–",
                 **{f"{config.VARIABLES[v].name} ({config.VARIABLES[v].rate_unit}/decade)":
                    cell(query.trends[v].annual.slope_per_decade, query.fingerprint.directions[v], v) for v in shown},
                 "Same direction": "–"}]
        for _, r in ranking.iterrows():
            rows.append({
                "Rank": str(int(r["rank"])), "Location": r["location"], "Distance": f"{r['distance']:.2f}",
                **{f"{config.VARIABLES[v].name} ({config.VARIABLES[v].rate_unit}/decade)":
                   cell(r[f"{v}_slope_per_decade"], r[f"{v}_direction"], v) for v in shown},
                "Same direction": f"{int(r['same_direction_count'])} of {int(r['shared_variables'])}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("▲ significant increase · ▼ significant decrease · ● slope not statistically significant. "
                   "'Same direction' counts indicators whose slope has the same sign as the selected location's, "
                   "significant or not.")
        for n in notes:
            st.caption(n)

    if pr:
        with st.expander("World map of trend direction (pool + investigated locations)"):
            t_map = comparison.trend_table({**pr, **results}, variables)
            mv = st.selectbox("Indicator", variables, format_func=var_label, key="w_sim_map_var")
            st.plotly_chart(charts.direction_map(t_map, mv, highlight=set(results)), key=f"poolmap_{mv}")
