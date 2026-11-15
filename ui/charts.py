"""Plotly figure builders. One y-axis per chart, thin marks, colour = indicator identity.

Colours come from a palette validated for colour-vision deficiency and contrast (see docs/).
Direction is never encoded by colour alone: arrows/symbols and words accompany it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from earthscope import config
from earthscope.anomalies import flag_anomalies
from earthscope.models import FingerprintResult, InvestigationResult

GRID = "rgba(128,128,128,0.18)"
MUTED = "rgba(128,128,128,0.55)"
DIR_SYMBOL = {"increasing": "▲", "decreasing": "▼", "no significant trend": "●", "insufficient data": "–"}
DIR_COLOR = {"increasing": "#eb6834", "decreasing": "#2a78d6", "no significant trend": "#8a8a85", "insufficient data": "#bbbbb5"}


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # older Streamlit or no theme info
        return False


def var_color(code: str) -> str:
    v = config.VARIABLES[code]
    return v.color_dark if is_dark() else v.color_light


def _trend_color() -> str:
    return "#c3c2b7" if is_dark() else "#52514e"


def _base(fig: go.Figure, height: int = 360, ytitle: str = "", legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0) if legend else None,
        showlegend=legend, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def _year_dates(years) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{int(y)}-07-01" for y in years])


def _trend_trace(res: InvestigationResult, var: str) -> Optional[go.Scatter]:
    tr = res.trends[var].annual
    if not tr.ok:
        return None
    yrs = np.array([tr.first_year, tr.last_year], dtype=float)
    name = "Sen's slope trend" if tr.significant else "Sen's slope (not significant)"
    return go.Scatter(
        x=_year_dates(yrs), y=tr.sens_intercept + tr.sens_slope * yrs, mode="lines", name=name,
        line=dict(color=_trend_color(), width=2, dash="dash" if tr.significant else "dot"),
        hovertemplate="%{y:.3g}<extra>trend</extra>",
    )


def timeseries_chart(res: InvestigationResult, var: str) -> go.Figure:
    info, color = config.VARIABLES[var], var_color(var)
    fig = go.Figure()
    m = res.monthly[var]
    fig.add_trace(go.Scatter(x=m.index, y=m, mode="lines", name="Monthly values",
                             line=dict(color=color, width=1), opacity=0.45, connectgaps=False,
                             hovertemplate="%{y:.3g} " + info.unit + "<extra>monthly</extra>"))
    a = res.annual[var].dropna()
    fig.add_trace(go.Scatter(x=_year_dates(a.index), y=a, mode="lines+markers", name="Annual mean",
                             line=dict(color=color, width=2), marker=dict(size=8, color=color, line=dict(width=2, color="rgba(255,255,255,0.9)")),
                             hovertemplate="%{y:.3g} " + info.unit + "<extra>annual mean</extra>"))
    t = _trend_trace(res, var)
    if t is not None:
        fig.add_trace(t)
    return _base(fig, 380, f"{info.name} ({info.unit})")


def annual_chart(res: InvestigationResult, var: str) -> go.Figure:
    info, color = config.VARIABLES[var], var_color(var)
    fig = go.Figure()
    a = res.annual[var].dropna()
    fig.add_trace(go.Scatter(x=_year_dates(a.index), y=a, mode="lines+markers", name="Annual mean",
                             line=dict(color=color, width=2), marker=dict(size=8, color=color),
                             hovertemplate="%{y:.3g} " + info.unit + "<extra>annual mean</extra>"))
    if len(a):
        fig.add_hline(y=float(a.mean()), line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text="period mean", annotation_position="top left")
    t = _trend_trace(res, var)
    if t is not None:
        fig.add_trace(t)
    return _base(fig, 340, f"{info.name} ({info.unit})")


def departure_chart(res: InvestigationResult, var: str) -> go.Figure:
    """Annual departure from the period mean (comparison with the historical average)."""
    info, color = config.VARIABLES[var], var_color(var)
    a = res.annual[var].dropna()
    dep = a - a.mean()
    fig = go.Figure(go.Bar(
        x=a.index.astype(int), y=dep, marker=dict(color=[color if d >= 0 else MUTED for d in dep]),
        hovertemplate="%{x}: %{y:+.3g} " + info.unit + " vs period mean<extra></extra>", name="Departure",
    ))
    fig.update_layout(bargap=0.35)
    return _base(fig, 280, f"Departure from period mean ({info.unit})", legend=False)


def monthly_timeline_chart(res: InvestigationResult, var: str, threshold: float) -> go.Figure:
    """Monthly values, the usual range for each calendar month (mean ± threshold·sd), and unusual months."""
    info, color = config.VARIABLES[var], var_color(var)
    an = res.anomalies[res.anomalies["variable"] == var].sort_values("date")
    fig = go.Figure()
    upper = an["clim_mean"] + threshold * an["clim_std"]
    lower = an["clim_mean"] - threshold * an["clim_std"]
    fig.add_trace(go.Scatter(x=an["date"], y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=an["date"], y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(128,128,128,0.18)", name=f"Usual range (mean ± {threshold:g} sd)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=an["date"], y=an["clim_mean"], mode="lines", name="Same-month historical average",
                             line=dict(color=MUTED, width=1, dash="dot"),
                             hovertemplate="%{y:.3g}<extra>historical avg</extra>"))
    fig.add_trace(go.Scatter(x=an["date"], y=an["value"], mode="lines", name="Monthly value",
                             line=dict(color=color, width=1.5), connectgaps=False,
                             hovertemplate="%{y:.3g} " + info.unit + "<extra>monthly</extra>"))
    fl = flag_anomalies(res.anomalies, threshold)
    fl = fl[fl["variable"] == var]
    if len(fl):
        fig.add_trace(go.Scatter(
            x=fl["date"], y=fl["value"], mode="markers", name="Unusual month",
            marker=dict(size=10, symbol="diamond", color=color, line=dict(width=2, color="rgba(255,255,255,0.95)")),
            customdata=np.stack([fl["label"], fl["z"]], axis=-1),
            hovertemplate="%{y:.3g} " + info.unit + "<br>%{customdata[0]} (z=%{customdata[1]:+.1f})<extra>unusual</extra>"))
    t = _trend_trace(res, var)
    if t is not None:
        fig.add_trace(t)
    return _base(fig, 400, f"{info.name} ({info.unit})")


def anomaly_chart(res: InvestigationResult, var: str, threshold: float) -> go.Figure:
    info, color = config.VARIABLES[var], var_color(var)
    an = res.anomalies[res.anomalies["variable"] == var].sort_values("date").copy()
    fl = flag_anomalies(res.anomalies, threshold)
    flagged_dates = set(fl[fl["variable"] == var]["date"])
    cols = [color if d in flagged_dates else MUTED for d in an["date"]]
    fig = go.Figure(go.Bar(
        x=an["date"], y=an["z"], marker=dict(color=cols), name="z-score",
        customdata=np.stack([an["value"], an["clim_mean"]], axis=-1),
        hovertemplate="z = %{y:+.2f}<br>value %{customdata[0]:.3g} vs usual %{customdata[1]:.3g} " + info.unit + "<extra></extra>",
    ))
    for s in (threshold, -threshold):
        fig.add_hline(y=s, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_layout(bargap=0.1)
    return _base(fig, 300, "z-score vs same-month average", legend=False)


def fingerprint_bars(fp: FingerprintResult, trends: dict) -> go.Figure:
    names, vals, cols, opac, text = [], [], [], [], []
    for var in config.VARIABLE_ORDER:
        x = fp.std_change_per_decade.get(var)
        if x is None:
            continue
        d = fp.directions[var]
        names.append(config.VARIABLES[var].name)
        vals.append(x)
        cols.append(var_color(var))
        opac.append(1.0 if d in "+-" else 0.35)
        text.append(f"{x:+.2f} SD  {'▲' if d == '+' else '▼' if d == '-' else '● n.s.'}")
    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation="h", marker=dict(color=cols, opacity=opac), text=text, textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:+.2f} interannual SD per decade<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(bargap=0.4)
    fig = _base(fig, 240, "", legend=False)
    lim = max([abs(v) for v in vals] + [0.5]) * 1.6          # head-room so the outside labels are never clipped
    fig.update_xaxes(title_text="Change per decade, in units of the variable's year-to-year standard deviation",
                     gridcolor=GRID, zeroline=False, range=[-lim, lim])
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(hovermode="closest")
    return fig


def correlation_heatmap(fp: FingerprintResult) -> go.Figure:
    rho, pv = fp.correlations, fp.correlation_p
    labels = [config.VARIABLES[v].name for v in rho.index]
    text = [[("–" if np.isnan(rho.iloc[i, j]) else f"{rho.iloc[i, j]:+.2f}" + ("" if i == j or pv.iloc[i, j] >= config.ALPHA else " *"))
             for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=rho.values, x=labels, y=labels, zmin=-1, zmax=1, text=text, texttemplate="%{text}",
        colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#eb6834"]], colorbar=dict(title="Spearman ρ"),
        hovertemplate="%{y} vs %{x}: ρ = %{z:+.2f}<extra></extra>",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed")
    return fig


def comparison_bars(table: pd.DataFrame, var: str) -> go.Figure:
    info, color = config.VARIABLES[var], var_color(var)
    g = table[(table["variable"] == var) & (table["status"] == "ok")]
    err_hi = (g["slope_high_per_decade"] - g["slope_per_decade"]).to_numpy(dtype=float)
    err_lo = (g["slope_per_decade"] - g["slope_low_per_decade"]).to_numpy(dtype=float)
    text = [f"{DIR_SYMBOL[d]} {s:+.3g}" for d, s in zip(g["direction"], g["slope_per_decade"])]
    fig = go.Figure(go.Bar(
        x=g["location"], y=g["slope_per_decade"], marker=dict(color=color, opacity=[1.0 if s else 0.35 for s in g["significant"]]),
        error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, color=MUTED, thickness=1.2),
        text=text, textposition="outside",
        customdata=np.stack([g["p_value"], g["direction"]], axis=-1),
        hovertemplate="%{x}<br>%{y:+.3g} " + info.rate_unit + "/decade<br>MK p = %{customdata[0]:.3f} (%{customdata[1]})<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(bargap=0.4, hovermode="closest")
    return _base(fig, 340, f"Sen's slope per decade ({info.rate_unit})", legend=False)


def _geo_layout(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_geos(
        projection_type="natural earth", showland=True, landcolor="rgba(128,128,128,0.16)", showocean=False,
        showcountries=True, countrycolor="rgba(128,128,128,0.35)", showcoastlines=True,
        coastlinecolor="rgba(128,128,128,0.45)", bgcolor="rgba(0,0,0,0)", lataxis_range=[-60, 80],
    )
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0))
    return fig


_MARKER = {"increasing": "triangle-up", "decreasing": "triangle-down", "no significant trend": "circle", "insufficient data": "x"}


def direction_map(table: pd.DataFrame, var: str, highlight: Optional[set[str]] = None) -> go.Figure:
    """World map of trend direction for one indicator (symbol + colour + legend)."""
    g = table[table["variable"] == var]
    fig = go.Figure()
    for d in ("increasing", "decreasing", "no significant trend", "insufficient data"):
        s = g[g["direction"] == d]
        if s.empty:
            continue
        hl = [loc in (highlight or set()) for loc in s["location"]]
        fig.add_trace(go.Scattergeo(
            lat=s["lat"], lon=s["lon"], mode="markers", name=f"{DIR_SYMBOL[d]} {d.capitalize()} ({len(s)})",
            marker=dict(symbol=_MARKER[d], size=[17 if h else 11 for h in hl], color=DIR_COLOR[d],
                        line=dict(width=[2.5 if h else 1 for h in hl], color=["#111" if h else "rgba(255,255,255,0.9)" for h in hl])),
            text=s["location"], customdata=np.stack([s["slope_per_decade"].fillna(np.nan), s["p_value"].fillna(np.nan)], axis=-1),
            hovertemplate="%{text}<br>%{customdata[0]:+.3g} /decade, p = %{customdata[1]:.3f}<extra></extra>",
        ))
    return _geo_layout(fig)


def similar_map(query_name: str, q_lat: float, q_lon: float, ranking: pd.DataFrame, coords: dict[str, tuple[float, float]]) -> go.Figure:
    fig = go.Figure()
    lats = [coords[n][0] for n in ranking["location"]]
    lons = [coords[n][1] for n in ranking["location"]]
    for lat, lon in zip(lats, lons):
        fig.add_trace(go.Scattergeo(lat=[q_lat, lat], lon=[q_lon, lon], mode="lines", showlegend=False,
                                    line=dict(width=1, color="rgba(128,128,128,0.5)"), hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, mode="markers+text", name="Most similar change pattern",
        text=[str(r) for r in ranking["rank"]], textposition="top center",
        marker=dict(size=13, color="#1baf7a", line=dict(width=1.5, color="rgba(255,255,255,0.95)")),
        hovertext=[f"#{r} {n} (distance {d:.2f})" for r, n, d in zip(ranking["rank"], ranking["location"], ranking["distance"])],
        hoverinfo="text"))
    fig.add_trace(go.Scattergeo(lat=[q_lat], lon=[q_lon], mode="markers", name=query_name,
                                marker=dict(size=17, symbol="star", color="#eb6834", line=dict(width=1.5, color="#111")),
                                hovertext=[query_name], hoverinfo="text"))
    return _geo_layout(fig, 420)
