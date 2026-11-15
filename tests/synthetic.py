"""TEST-ONLY synthetic data builders.

!!! These payloads are NOT NASA data. They are generated from the *documented* NASA POWER JSON
layout with known, injected trends so the parser and statistics can be tested against known
answers. The application never imports this module and has no fallback to synthetic data.
Passing tests here verify our code - they say nothing about the real NASA service.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_monthly(start: int, end: int, *, t_trend=0.0, p_trend=0.0, rh_trend=0.0, seed=0,
                 noise=(0.4, 0.15, 1.0)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    yrs = (idx.year - start) + (idx.month - 1) / 12.0
    m = idx.month.to_numpy()
    t = 25 + 5 * np.sin(2 * np.pi * (m - 4) / 12) + t_trend * yrs + rng.normal(0, noise[0], len(idx))
    p = np.clip(3 + 2.5 * np.sin(2 * np.pi * (m - 5) / 12) + p_trend * yrs + rng.normal(0, noise[1], len(idx)), 0.05, None)
    rh = np.clip(65 + 15 * np.sin(2 * np.pi * (m - 6) / 12) + rh_trend * yrs + rng.normal(0, noise[2], len(idx)), 1, 99)
    return pd.DataFrame({"T2M": t, "PRECTOTCORR": p, "RH2M": rh}, index=idx)


def make_payload(lat=10.0, lon=76.0, start=2001, end=2020, *, missing=None, elevation=85.5,
                 unit_overrides=None, drop_param=None, **kw) -> dict:
    """Build a payload shaped like the documented NASA POWER monthly response."""
    df = make_monthly(start, end, **kw)
    missing = set(missing or [])
    params: dict[str, dict[str, float]] = {}
    for var in df.columns:
        if var == drop_param:
            continue
        d: dict[str, float] = {}
        for ts, v in df[var].items():
            key = f"{ts.year}{ts.month:02d}"
            d[key] = -999.0 if (var, key) in missing else round(float(v), 3)
        for y in range(start, end + 1):
            sub = df[var][df.index.year == y]
            d[f"{y}13"] = round(float(np.average(sub.to_numpy(), weights=sub.index.days_in_month)), 3)
        params[var] = d
    units = {"T2M": "C", "PRECTOTCORR": "mm/day", "RH2M": "%"}
    units.update(unit_overrides or {})
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, elevation]},
        "properties": {"parameter": params},
        "header": {"title": "TEST FIXTURE (not NASA data)", "api": {"version": "test", "name": "fixture"},
                   "sources": ["fixture"], "fill_value": -999.0, "start": start, "end": end},
        "messages": [],
        "parameters": {v: {"units": units[v], "longname": v} for v in df.columns},
    }
