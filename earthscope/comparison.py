"""Regional comparison: trend table, Opposite Trend Finder, Similar Earth Regions Discovery.

Similarity method (plain mathematics, no AI):
    1. For every location take the annual Sen's slope per decade of each variable.
    2. Standardise each variable across the candidate pool (z-score of the slope), so that
       °C/decade, mm/day/decade and %/decade become comparable.
    3. Distance = root-mean-square difference between two locations' standardised slope vectors
       (Euclidean distance divided by sqrt(number of shared variables)).
    4. Rank pool locations by distance (smaller = more similar change pattern).
Similarity describes the pattern of *change*, not that the climates themselves are alike.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from . import config
from .models import InvestigationResult


def trend_row(res: InvestigationResult, var: str) -> dict:
    tr = res.trends[var].annual
    return {
        "location": res.location.name, "lat": res.location.lat, "lon": res.location.lon,
        "variable": var, "indicator": config.VARIABLES[var].name, "unit": config.VARIABLES[var].unit,
        "status": tr.status, "direction": tr.direction,
        "slope_per_year": tr.sens_slope, "slope_per_decade": tr.slope_per_decade,
        "slope_low_per_decade": None if tr.sens_low is None else tr.sens_low * 10,
        "slope_high_per_decade": None if tr.sens_high is None else tr.sens_high * 10,
        "p_value": tr.mk_p, "tau": tr.mk_tau, "n_years": tr.n,
        "significant": bool(tr.significant),
        "evidence_score": res.confidence[var].score,
    }


def trend_table(results: Mapping[str, InvestigationResult], variables: list[str]) -> pd.DataFrame:
    rows = [trend_row(r, v) for r in results.values() for v in variables if v in r.trends]
    return pd.DataFrame(rows)


def find_opposite_trends(table: pd.DataFrame) -> pd.DataFrame:
    """For each variable list locations with significant increases vs significant decreases.

    Returns one row per variable: n_up, n_down, n_none, up/down/none location lists and `opposite`
    (True when at least one significant increase and one significant decrease exist).
    """
    rows = []
    for var, g in table.groupby("variable", sort=False):
        ok = g[g["status"] == "ok"]
        up = ok[ok["significant"] & (ok["direction"] == "increasing")]["location"].tolist()
        down = ok[ok["significant"] & (ok["direction"] == "decreasing")]["location"].tolist()
        none = ok[~ok["significant"]]["location"].tolist()
        na = g[g["status"] != "ok"]["location"].tolist()
        rows.append({
            "variable": var, "indicator": config.VARIABLES[var].name,
            "n_increasing": len(up), "n_decreasing": len(down), "n_no_significant": len(none),
            "n_not_assessed": len(na), "opposite": bool(up and down),
            "increasing": up, "decreasing": down, "no_significant": none,
        })
    return pd.DataFrame(rows)


def opposite_pairs(table: pd.DataFrame, var: str) -> pd.DataFrame:
    g = table[(table["variable"] == var) & (table["status"] == "ok") & table["significant"]]
    up = g[g["direction"] == "increasing"]
    down = g[g["direction"] == "decreasing"]
    rows = [
        {"increasing_location": u["location"], "increasing_per_decade": u["slope_per_decade"],
         "decreasing_location": d["location"], "decreasing_per_decade": d["slope_per_decade"]}
        for _, u in up.iterrows() for _, d in down.iterrows()
    ]
    return pd.DataFrame(rows)


def similar_regions(
    query: InvestigationResult,
    pool: Mapping[str, InvestigationResult],
    variables: list[str],
    top_k: int = config.SIMILAR_TOP_K,
) -> tuple[pd.DataFrame, list[str]]:
    """Rank pool locations by similarity of change pattern to `query`.

    Returns (ranking table, notes). The ranking has the pool slopes, direction flags and distance.
    """
    notes: list[str] = []
    cols = [v for v in variables if v in query.trends and query.trends[v].annual.ok]
    if len(cols) < 2:
        return pd.DataFrame(), ["At least two assessable indicators are needed for a similarity comparison."]

    slopes = {}
    for name, r in pool.items():
        if r.location.name == query.location.name:
            continue
        slopes[name] = {v: (r.trends[v].annual.slope_per_decade if v in r.trends and r.trends[v].annual.ok else np.nan)
                        for v in cols}
    pool_df = pd.DataFrame.from_dict(slopes, orient="index", columns=cols)
    if len(pool_df.dropna(thresh=2)) < config.SIMILAR_MIN_POOL:
        return pd.DataFrame(), [f"Fewer than {config.SIMILAR_MIN_POOL} pool locations have usable trends; "
                                "similarity is not computed."]
    mu, sd = pool_df.mean(), pool_df.std(ddof=1)
    usable = [v for v in cols if sd[v] and sd[v] > 0]
    if len(usable) < 2:
        return pd.DataFrame(), ["Trends do not vary enough across the pool to standardise."]
    if len(usable) < len(cols):
        notes.append("Variables with no spread across the pool were left out: " + ", ".join(set(cols) - set(usable)))

    z_pool = (pool_df[usable] - mu[usable]) / sd[usable]
    q_slopes = {v: query.trends[v].annual.slope_per_decade for v in usable}
    z_q = pd.Series({v: (q_slopes[v] - mu[v]) / sd[v] for v in usable})

    diff = z_pool.sub(z_q, axis=1)
    n_shared = diff.notna().sum(axis=1)
    dist = np.sqrt((diff ** 2).sum(axis=1, skipna=True)) / np.sqrt(n_shared.where(n_shared >= 2))
    out = pool_df.copy()
    out.columns = [f"{v}_slope_per_decade" for v in cols]
    out["distance"] = dist
    out["shared_variables"] = n_shared
    out["same_direction_count"] = [
        int(sum(np.sign(pool_df.loc[n, v]) == np.sign(q_slopes[v]) for v in usable if pd.notna(pool_df.loc[n, v])))
        for n in out.index
    ]
    sig_cols = {}
    for v in cols:
        sig_cols[v] = [pool[n].fingerprint.directions.get(v, "NA") for n in out.index]
        out[f"{v}_direction"] = sig_cols[v]
    out = out.dropna(subset=["distance"]).sort_values("distance").head(top_k)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.index.name = "location"
    notes.append(
        "Distance is the root-mean-square difference of pool-standardised Sen's slopes per decade "
        f"({', '.join(config.VARIABLES[v].name for v in usable)}); smaller = more similar change pattern. "
        "Slopes are used whether or not individually significant; significance flags are shown alongside."
    )
    return out.reset_index(), notes
