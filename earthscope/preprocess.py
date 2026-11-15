"""Cleaning and aggregation: data-quality report, day-weighted annual and seasonal means."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .models import DataQuality, VariableQuality


def slice_period(monthly: pd.DataFrame, start_year: int, end_year: int, variables: list[str]) -> pd.DataFrame:
    """Monthly data for [start_year, end_year] restricted to `variables`, on a complete month index."""
    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    cols = [v for v in variables if v in monthly.columns]
    return monthly.reindex(idx)[cols].astype(float)


def assess_quality(monthly: pd.DataFrame, start_year: int, end_year: int) -> DataQuality:
    per: dict[str, VariableQuality] = {}
    n_years = end_year - start_year + 1
    for var in monthly.columns:
        s = monthly[var]
        valid = int(s.notna().sum())
        expected = len(s)
        counts = s.groupby(s.index.year).count()
        incomplete = [int(y) for y, c in counts.items() if c < 12]
        per[var] = VariableQuality(
            variable=var, expected_months=expected, valid_months=valid,
            completeness=valid / expected if expected else 0.0, years_requested=n_years,
            years_complete=int((counts == 12).sum()), incomplete_years=incomplete,
        )
    overall = float(np.mean([q.completeness for q in per.values()])) if per else 0.0
    notes = []
    for var, q in per.items():
        if q.incomplete_years:
            notes.append(
                f"{var}: {len(q.incomplete_years)} year(s) have missing months "
                f"({', '.join(map(str, q.incomplete_years[:6]))}{'…' if len(q.incomplete_years) > 6 else ''}) "
                "and are excluded from annual means."
            )
    return DataQuality(per_variable=per, overall_completeness=overall, notes=notes)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def annual_means(monthly: pd.DataFrame) -> pd.DataFrame:
    """Day-weighted annual mean per variable. A year with any missing month gives NaN.

    Requiring all 12 months avoids biasing the annual mean toward whichever season is present.
    Day-weighting matters mainly for PRECTOTCORR, which is a mean daily rate per month.
    """
    out: dict[str, dict[int, float]] = {v: {} for v in monthly.columns}
    days = monthly.index.days_in_month.to_numpy()
    years = monthly.index.year.to_numpy()
    for year in np.unique(years):
        mask = years == year
        for var in monthly.columns:
            vals = monthly[var].to_numpy()[mask]
            out[var][int(year)] = _weighted_mean(vals, days[mask]) if (np.isfinite(vals).all() and mask.sum() == 12) else np.nan
    df = pd.DataFrame(out)
    df.index.name = "year"
    return df


def seasonal_means(monthly: pd.DataFrame, start_year: int, end_year: int) -> dict[str, pd.DataFrame]:
    """Day-weighted mean per season and year. Dec-Feb is assigned to the year of its Jan/Feb.

    A season-year needs all three months valid; the first Dec-Feb (no preceding December inside
    the period) and any incomplete season are NaN.
    """
    result: dict[str, pd.DataFrame] = {}
    days = pd.Series(monthly.index.days_in_month, index=monthly.index)
    for label, months in config.SEASONS.items():
        season_year = pd.Series(
            monthly.index.year + np.where(monthly.index.month == 12, 1, 0) if 12 in months else monthly.index.year,
            index=monthly.index,
        )
        in_season = pd.Series(monthly.index.month.isin(months), index=monthly.index)
        table = {}
        for var in monthly.columns:
            vals = {}
            for y in range(start_year, end_year + 1):
                sel = in_season & (season_year == y)
                v = monthly.loc[sel, var].to_numpy()
                w = days[sel].to_numpy()
                vals[y] = _weighted_mean(v, w) if (len(v) == len(months) and np.isfinite(v).all()) else np.nan
            table[var] = vals
        df = pd.DataFrame(table)
        df.index.name = "year"
        result[label] = df
    return result
