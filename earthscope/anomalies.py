"""Anomaly detection: unusual months relative to the same calendar month's historical average.

Anomalies are kept separate from trends: they are computed against a fixed climatology for the
selected period (mean and standard deviation of each calendar month), not against the trend line.
Consequence: if a variable has a strong trend, flagged months will cluster at the ends of the
period. That is expected and is shown to the user rather than hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def compute_monthly_anomalies(monthly: pd.DataFrame, min_obs: int = config.MIN_MONTH_OBS) -> pd.DataFrame:
    """Long-format table: date, variable, value, clim_mean, clim_std, z (NaN if not computable)."""
    frames = []
    for var in monthly.columns:
        s = monthly[var]
        grp = s.groupby(s.index.month)
        mean = grp.transform("mean")
        std = grp.transform(lambda x: x.std(ddof=1))
        n = grp.transform("count")
        z = (s - mean) / std
        z = z.where((n >= min_obs) & (std > 1e-9))
        frames.append(pd.DataFrame({
            "date": s.index, "variable": var, "value": s.to_numpy(), "clim_mean": mean.to_numpy(),
            "clim_std": std.to_numpy(), "z": z.to_numpy(),
        }))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "variable", "value", "clim_mean", "clim_std", "z"])
    out["deviation"] = out["value"] - out["clim_mean"]
    return out


def flag_anomalies(anoms: pd.DataFrame, threshold: float = config.ANOMALY_Z) -> pd.DataFrame:
    """Return only unusual months with a plain-language label.

    A month is flagged when |z| >= threshold AND its absolute deviation from the monthly mean is at
    least the variable's `min_abs_dev` (guards against flagging trivial rainfall in near-zero months).
    """
    if anoms.empty:
        return anoms.assign(label=[], year=[], month=[])
    df = anoms.copy()
    min_dev = df["variable"].map(lambda v: config.VARIABLES[v].min_abs_dev)
    mask = (df["z"].abs() >= threshold) & (df["deviation"].abs() >= min_dev)
    df = df[mask].copy()
    df["label"] = [
        config.VARIABLES[v].high_label if z > 0 else config.VARIABLES[v].low_label
        for v, z in zip(df["variable"], df["z"])
    ]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.sort_values("z", key=np.abs, ascending=False).reset_index(drop=True)


def expected_by_chance(n_obs: int, threshold: float) -> float:
    """Expected number of |z| >= threshold months if values were normally distributed."""
    from scipy.stats import norm
    return float(n_obs * 2 * (1 - norm.cdf(threshold)))
