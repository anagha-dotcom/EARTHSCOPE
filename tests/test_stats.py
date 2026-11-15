"""Statistics tests against synthetic series with KNOWN injected trends (not NASA data)."""
import numpy as np
import pandas as pd
import pymannkendall as pmk
import pytest

from earthscope import config
from earthscope.anomalies import compute_monthly_anomalies, expected_by_chance, flag_anomalies
from earthscope.confidence import compute_confidence
from earthscope.preprocess import annual_means, assess_quality, seasonal_means, slice_period
from earthscope.trends import analyze_series, analyze_variable, seasonal_mk
from tests.synthetic import make_monthly


def _annual(**kw):
    m = make_monthly(2001, 2025, **kw)
    return m, annual_means(m)


def test_annual_means_are_day_weighted_and_need_full_years():
    idx = pd.date_range("2001-01-01", "2002-12-01", freq="MS")
    vals = np.arange(24, dtype=float)
    m = pd.DataFrame({"T2M": vals}, index=idx)
    m.loc["2002-03-01", "T2M"] = np.nan
    a = annual_means(m)
    expected_2001 = np.average(vals[:12], weights=idx[:12].days_in_month)
    assert a.loc[2001, "T2M"] == pytest.approx(expected_2001)
    assert np.isnan(a.loc[2002, "T2M"])                      # one missing month -> year excluded


def test_known_increasing_trend_recovered():
    _, a = _annual(t_trend=0.04, seed=1)
    r = analyze_series(a["T2M"], variable="T2M")
    assert r.ok and r.significant and r.direction == "increasing"
    assert r.sens_slope == pytest.approx(0.04, abs=0.01)
    assert r.sens_low <= r.sens_slope <= r.sens_high
    assert r.sens_low <= 0.04 <= r.sens_high
    assert r.ols_slope == pytest.approx(r.sens_slope, abs=0.01)
    assert r.slope_per_decade == pytest.approx(r.sens_slope * 10)
    assert r.pct_per_decade is None                          # temperature: no percent-of-mean


def test_known_decreasing_trend_recovered():
    _, a = _annual(p_trend=-0.03, seed=2)
    r = analyze_series(a["PRECTOTCORR"], variable="PRECTOTCORR")
    assert r.significant and r.direction == "decreasing" and r.sens_slope < 0
    assert r.pct_per_decade is not None and r.pct_per_decade < 0


def test_no_trend_is_not_significant():
    _, a = _annual(seed=3)
    r = analyze_series(a["RH2M"], variable="RH2M")
    assert not r.significant and r.direction == "no significant trend"


def test_sens_slope_matches_pymannkendall_on_gap_free_series():
    _, a = _annual(t_trend=0.03, seed=4)
    y = a["T2M"].to_numpy()
    ref = pmk.original_test(y)
    r = analyze_series(a["T2M"], variable="T2M")
    assert r.sens_slope == pytest.approx(ref.slope)
    assert r.mk_p == pytest.approx(ref.p)


def test_gaps_use_real_years_for_slope():
    _, a = _annual(t_trend=0.05, seed=5)
    s = a["T2M"].copy()
    s.loc[[2005, 2006, 2007, 2015]] = np.nan
    r = analyze_series(s, variable="T2M")
    assert r.n == 21 and r.sens_slope == pytest.approx(0.05, abs=0.015)


def test_insufficient_data_is_reported_not_computed():
    _, a = _annual(t_trend=0.05)
    r = analyze_series(a["T2M"].iloc[:8], variable="T2M")
    assert not r.ok and r.direction == "insufficient data" and r.sens_slope is None
    assert "required" in r.reason


def test_constant_series_does_not_crash():
    s = pd.Series(5.0, index=range(2001, 2021))
    r = analyze_series(s, variable="T2M")
    assert r.ok and not r.significant and r.sens_slope == 0.0


def test_seasonal_mk_recovers_slope_and_handles_dropped_year():
    m = make_monthly(2001, 2025, t_trend=0.04, seed=6)
    r = seasonal_mk(m["T2M"], variable="T2M")
    assert r.ok and r.significant and r.direction == "increasing"
    assert r.sens_slope == pytest.approx(0.04, abs=0.01)
    m2 = m.copy()
    m2.loc["2010-05-01", "T2M"] = np.nan                       # whole year 2010 dropped
    r2 = seasonal_mk(m2["T2M"], variable="T2M")
    assert r2.n == 24 and r2.sens_slope == pytest.approx(0.04, abs=0.01)


def test_seasonal_mk_insufficient():
    m = make_monthly(2001, 2008)
    assert not seasonal_mk(m["T2M"], variable="T2M").ok


def test_seasonal_means_dec_feb_year_assignment():
    idx = pd.date_range("2001-01-01", "2003-12-01", freq="MS")
    m = pd.DataFrame({"T2M": np.where(idx.month.isin([12, 1, 2]), 10.0, 0.0)}, index=idx)
    s = seasonal_means(m, 2001, 2003)["Dec–Feb"]
    assert np.isnan(s.loc[2001, "T2M"])                        # Dec 2000 not in period
    assert s.loc[2002, "T2M"] == pytest.approx(10.0)           # Dec 2001 + Jan/Feb 2002
    assert s.loc[2003, "T2M"] == pytest.approx(10.0)


def test_analyze_variable_end_to_end():
    m, a = _annual(t_trend=0.04, seed=7)
    vt = analyze_variable("T2M", a["T2M"], m["T2M"], seasonal_means(m, 2001, 2025))
    assert vt.annual.significant and vt.seasonal_mk.significant
    assert set(vt.by_season) == set(config.SEASONS)
    assert vt.by_season["Mar–May"].n == 25


# --------------------------------------------------------------------------- quality
def test_quality_report():
    m = make_monthly(2001, 2010)
    m.loc["2004-02-01", "T2M"] = np.nan
    q = assess_quality(m, 2001, 2010)
    assert q.per_variable["T2M"].valid_months == 119
    assert q.per_variable["T2M"].incomplete_years == [2004]
    assert q.per_variable["T2M"].years_complete == 9
    assert q.per_variable["RH2M"].completeness == 1.0
    assert q.notes and "2004" in q.notes[0]


def test_slice_period_reindexes():
    m = make_monthly(2001, 2010)
    s = slice_period(m, 2005, 2012, ["T2M"])
    assert len(s) == 96 and s.loc["2011-01-01":].isna().all().all()


# --------------------------------------------------------------------------- anomalies
def test_injected_anomaly_is_flagged_with_correct_label():
    m = make_monthly(2001, 2020, seed=8, noise=(0.3, 0.1, 0.5))
    m.loc["2010-07-01", "T2M"] += 6.0
    m.loc["2012-03-01", "PRECTOTCORR"] = 0.05
    flagged = flag_anomalies(compute_monthly_anomalies(m), 2.5)
    row = flagged[(flagged.variable == "T2M") & (flagged.year == 2010) & (flagged.month == 7)]
    assert len(row) == 1 and row.iloc[0]["label"] == "unusually warm" and row.iloc[0]["z"] > 4
    dry = flagged[(flagged.variable == "PRECTOTCORR") & (flagged.year == 2012) & (flagged.month == 3)]
    assert len(dry) == 1 and dry.iloc[0]["label"] == "unusually dry"


def test_min_abs_dev_guard_for_precipitation():
    idx = pd.date_range("2001-01-01", "2020-12-01", freq="MS")
    rng = np.random.default_rng(0)
    p = np.abs(rng.normal(0.02, 0.01, len(idx)))              # near-zero rainfall climate
    p[idx == "2010-01-01"] = 0.12                             # huge z but only ~0.1 mm/day above mean
    m = pd.DataFrame({"PRECTOTCORR": p}, index=idx)
    an = compute_monthly_anomalies(m)
    assert an.loc[an.date == "2010-01-01", "z"].iloc[0] > 3
    assert flag_anomalies(an, 2.0).empty


def test_anomaly_z_nan_when_too_few_observations_or_zero_std():
    idx = pd.date_range("2001-01-01", "2005-12-01", freq="MS")        # 5 obs per month < 8
    m = pd.DataFrame({"T2M": np.arange(60.0)}, index=idx)
    assert compute_monthly_anomalies(m)["z"].isna().all()


def test_expected_by_chance():
    assert expected_by_chance(300, 2.0) == pytest.approx(13.67, abs=0.05)


# --------------------------------------------------------------------------- confidence
def test_confidence_components_and_bands():
    _, a = _annual(t_trend=0.06, seed=9, noise=(0.1, 0.1, 0.1))
    tr = analyze_series(a["T2M"], variable="T2M")
    c = compute_confidence(1.0, tr)
    assert c.label == "Strong" and 70 <= c.score <= 100
    assert sum(x["weight"] for x in c.components) == pytest.approx(100)
    assert c.score == pytest.approx(sum(x["points"] for x in c.components))
    low = compute_confidence(0.5, tr)
    assert low.score == pytest.approx(c.score - 10)           # halving completeness removes 10 points


def test_confidence_not_significant_is_capped_weak():
    _, a = _annual(seed=11)
    tr = analyze_series(a["T2M"], variable="T2M")
    assert not tr.significant
    c = compute_confidence(1.0, tr)
    assert c.label == "Weak"


def test_confidence_not_assessable():
    _, a = _annual()
    tr = analyze_series(a["T2M"].iloc[:5], variable="T2M")
    c = compute_confidence(1.0, tr)
    assert c.score is None and c.label == "Not assessable"
