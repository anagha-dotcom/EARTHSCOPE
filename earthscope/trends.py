"""Trend detection: Mann-Kendall + Sen's slope (primary), OLS (secondary), seasonal analysis.

Primary basis: annual means (one value per complete year) -> one independent-ish observation per
year, which keeps the Mann-Kendall test close to its independence assumption.
Seasonal analysis is supporting evidence:
    * Seasonal Mann-Kendall on monthly values (Hirsch-Slack, corrects for inter-month correlation)
    * Mann-Kendall on each three-month season's yearly means
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pymannkendall as pmk
from scipy import stats

from . import config
from .models import TrendResult, VariableTrends


def _lag1(resid: np.ndarray) -> Optional[float]:
    if len(resid) < 4 or np.std(resid) == 0:
        return None
    return float(np.corrcoef(resid[:-1], resid[1:])[0, 1])


def _insufficient(base: dict, reason: str) -> TrendResult:
    return TrendResult(status="insufficient", reason=reason, direction="insufficient data", **base)


def analyze_series(
    series: pd.Series,
    *,
    variable: str,
    method: str = "annual",
    alpha: float = config.ALPHA,
    min_n: int = config.MIN_ANNUAL_YEARS,
) -> TrendResult:
    """Trend analysis of a yearly series (index = year). NaNs are dropped, gaps are allowed."""
    info = config.VARIABLES[variable]
    s = series.dropna()
    n = int(len(s))
    base = dict(variable=variable, label=info.name, unit=info.unit, method=method, n=n, alpha=alpha)
    if n < min_n:
        return _insufficient(base, f"Only {n} valid yearly values; at least {min_n} are required for a trend test.")

    x = s.index.to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)

    mk = pmk.original_test(y, alpha=alpha)                    # p, z, tau, S use ranks only (gap-safe)
    slope, intercept, lo, hi = stats.theilslopes(y, x, alpha=1 - alpha)  # uses real years (gap-safe)

    ols_slope = ols_p = ols_r2 = None
    if np.std(y) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = stats.linregress(x, y)
        ols_slope, ols_p, ols_r2 = float(lr.slope), float(lr.pvalue), float(lr.rvalue ** 2)

    significant = bool(mk.p < alpha)
    if significant:
        direction = "increasing" if mk.z > 0 else "decreasing"
    else:
        direction = "no significant trend"

    resid = y - (intercept + slope * x)
    lag1 = _lag1(resid)
    serial = bool(lag1 is not None and abs(lag1) > 1.96 / np.sqrt(n))

    mean = float(np.mean(y))
    pct = None
    if info.kind != "temperature" and abs(mean) > 1e-9:
        pct = 100.0 * slope * 10.0 / abs(mean)

    return TrendResult(
        **base, first_year=int(x[0]), last_year=int(x[-1]), mean=mean, std=float(np.std(y, ddof=1)),
        mk_p=float(mk.p), mk_z=float(mk.z), mk_tau=float(mk.Tau), mk_s=float(mk.s),
        sens_slope=float(slope), sens_low=float(lo), sens_high=float(hi), sens_intercept=float(intercept),
        ols_slope=ols_slope, ols_p=ols_p, ols_r2=ols_r2, direction=direction, significant=significant,
        lag1_autocorr=lag1, serial_corr_flag=serial, pct_per_decade=pct,
    )


def seasonal_mk(
    monthly_series: pd.Series,
    *,
    variable: str,
    alpha: float = config.ALPHA,
    min_years: int = config.MIN_ANNUAL_YEARS,
) -> TrendResult:
    """Seasonal Mann-Kendall (correlated version) on monthly values from complete years.

    The slope is the Hirsch seasonal Kendall estimator: median of all within-month pairwise
    slopes, computed with real year differences (so dropped incomplete years do not distort it).
    """
    info = config.VARIABLES[variable]
    s = monthly_series
    df = pd.DataFrame({"v": s.to_numpy(), "year": s.index.year, "month": s.index.month})
    counts = df.groupby("year")["v"].apply(lambda c: int(c.notna().sum()))
    full_years = counts[counts == 12].index
    base = dict(variable=variable, label=info.name, unit=info.unit, method="seasonal_mk",
                n=int(len(full_years)), alpha=alpha)
    if len(full_years) < min_years:
        return _insufficient(base, f"Only {len(full_years)} complete years of monthly data; "
                                   f"at least {min_years} are required for the seasonal test.")
    d = df[df["year"].isin(full_years)].sort_values(["year", "month"])
    res = pmk.correlated_seasonal_test(d["v"].to_numpy(dtype=float), period=12, alpha=alpha)

    pivot = d.pivot(index="year", columns="month", values="v")
    yrs = pivot.index.to_numpy(dtype=float)
    iu, ju = np.triu_indices(len(yrs), 1)
    slopes = np.concatenate([(pivot[m].to_numpy()[ju] - pivot[m].to_numpy()[iu]) / (yrs[ju] - yrs[iu])
                             for m in pivot.columns])
    sen = float(np.median(slopes))

    significant = bool(res.p < alpha)
    direction = ("increasing" if res.z > 0 else "decreasing") if significant else "no significant trend"
    return TrendResult(
        **base, first_year=int(yrs[0]), last_year=int(yrs[-1]), mean=float(d["v"].mean()),
        std=float(d["v"].std(ddof=1)), mk_p=float(res.p), mk_z=float(res.z), mk_tau=float(res.Tau),
        mk_s=float(res.s), sens_slope=sen, direction=direction, significant=significant,
    )


def analyze_variable(
    variable: str,
    annual: pd.Series,
    monthly: pd.Series,
    seasonal: dict[str, pd.DataFrame],
    *,
    alpha: float = config.ALPHA,
) -> VariableTrends:
    by_season = {
        label: analyze_series(tbl[variable], variable=variable, method=f"season:{label}", alpha=alpha)
        for label, tbl in seasonal.items()
    }
    return VariableTrends(
        variable=variable,
        annual=analyze_series(annual, variable=variable, alpha=alpha),
        seasonal_mk=seasonal_mk(monthly, variable=variable, alpha=alpha),
        by_season=by_season,
    )
