"""Fingerprint, comparison, pool, orchestrator and report tests (synthetic inputs, NOT NASA data)."""
import io

import numpy as np
import pandas as pd
import pytest

from earthscope import comparison, config, investigation, pool, power_client, report
from earthscope.fingerprint import build_fingerprint, direction_code
from earthscope.models import Location, TrendResult
from earthscope.power_client import parse_response
from tests.synthetic import make_payload

VARS = config.VARIABLE_ORDER
END = config.last_full_year()


def _power(lat=10.0, lon=76.0, **kw):
    p = make_payload(lat, lon, 2001, END, **kw)
    return parse_response(p, VARS, 2001, END, req_lat=lat, req_lon=lon)


def _tr(var, direction, p=0.01):
    sig = direction in ("increasing", "decreasing")
    return TrendResult(variable=var, label=var, unit="", method="annual", n=20, alpha=0.05,
                       direction=direction, significant=sig, mk_p=p, mk_tau=0.5, sens_slope=0.1 if direction == "increasing" else -0.1,
                       std=1.0, mean=5.0)


def _fp(t, p, h):
    tr = {"T2M": _tr("T2M", t), "PRECTOTCORR": _tr("PRECTOTCORR", p), "RH2M": _tr("RH2M", h)}
    return build_fingerprint(tr, {}, pd.DataFrame({v: np.arange(20.0) for v in VARS}))


# --------------------------------------------------------------------------- fingerprint
def test_spec_example_warming_drying():
    fp = _fp("increasing", "decreasing", "decreasing")
    assert fp.pattern_id == "warming_drying" and fp.directions == {"T2M": "+", "PRECTOTCORR": "-", "RH2M": "-"}
    assert "causation" in " ".join(fp.caveats).lower() or "caused" in " ".join(fp.caveats).lower()


def test_no_significant_change():
    assert _fp("no significant trend", "no significant trend", "no significant trend").pattern_id == "no_significant_change"


def test_every_direction_combination_gets_a_label():
    dirs = ["increasing", "decreasing", "no significant trend"]
    for t in dirs:
        for p in dirs:
            for h in dirs:
                fp = _fp(t, p, h)
                assert fp.pattern_id and fp.label and fp.description


def test_mixed_pattern_fallback_and_partial():
    assert _fp("increasing", "increasing", "decreasing").pattern_id == "mixed"
    tr = {"T2M": _tr("T2M", "increasing"), "PRECTOTCORR": _tr("PRECTOTCORR", "decreasing")}
    fp = build_fingerprint(tr, {}, pd.DataFrame({v: np.arange(20.0) for v in VARS}))
    assert fp.pattern_id == "partial" and fp.directions["RH2M"] == "NA"
    assert any("Not assessed" in c for c in fp.caveats)


def test_direction_code_insufficient():
    assert direction_code(TrendResult(variable="T2M", label="", unit="", method="annual", n=3, alpha=.05,
                                      status="insufficient")) == "NA"


# --------------------------------------------------------------------------- orchestrator
def test_analyze_recovers_injected_warming_drying():
    pw = _power(t_trend=0.05, p_trend=-0.04, rh_trend=-0.3, seed=1)
    res = investigation.analyze(pw, Location("Test A", 10, 76), 2001, END, VARS)
    assert res.fingerprint.pattern_id == "warming_drying"
    assert res.trends["T2M"].annual.sens_slope == pytest.approx(0.05, abs=0.01)
    assert all(res.confidence[v].score is not None for v in VARS)
    assert res.quality.overall_completeness == 1.0
    kinds = {f.kind for f in res.findings}
    assert {"trend", "seasonal", "confidence", "fingerprint", "anomaly", "quality"} <= kinds
    assert not res.fingerprint.correlations.empty


def test_analyze_period_subset_and_variable_subset():
    pw = _power(t_trend=0.05, seed=2)
    res = investigation.analyze(pw, Location("T", 10, 76), 2011, 2020, ["T2M", "RH2M"])
    assert res.variables == ["T2M", "RH2M"] and res.monthly.shape == (120, 2)
    assert res.fingerprint.directions["PRECTOTCORR"] == "NA"


def test_settings_validation():
    pw = _power()
    loc = Location("T", 10, 76)
    with pytest.raises(investigation.InvestigationError):
        investigation.analyze(pw, loc, 2001, 2005, VARS)          # < 10 years
    with pytest.raises(investigation.InvestigationError):
        investigation.analyze(pw, loc, 2001, END, [])
    with pytest.raises(investigation.InvestigationError):
        investigation.analyze(pw, loc, 1995, END, VARS)


def test_missing_data_reduces_completeness_and_is_reported():
    miss = [("T2M", f"2010{m:02d}") for m in range(1, 13)]        # entire 2010 missing for T2M
    pw = _power(missing=miss, t_trend=0.05)
    res = investigation.analyze(pw, Location("T", 10, 76), 2001, END, VARS)
    assert res.trends["T2M"].annual.n == (END - 2001 + 1) - 1
    assert res.quality.per_variable["T2M"].completeness < 1
    assert res.confidence["T2M"].components[0]["scaled"] < 1
    assert any("2010" in w for w in res.warnings)


def test_run_investigation_uses_client_and_progress(monkeypatch):
    calls = []
    monkeypatch.setattr(power_client, "fetch_monthly", lambda *a, **k: (calls.append(a), _power(t_trend=0.03, seed=3))[1])
    seen = []
    res = investigation.run_investigation(Location("T", 10, 76), 2001, END, VARS, progress=lambda i, n, m: seen.append(m))
    assert calls[0][2:4] == (config.MIN_YEAR, END)               # always fetches the full range
    assert seen[0].startswith("Retrieving") and seen[-1] == "Investigation complete"
    assert res.location.name == "T"


def test_power_errors_propagate(monkeypatch):
    def boom(*a, **k):
        raise power_client.PowerAPIError("down")
    monkeypatch.setattr(power_client, "fetch_monthly", boom)
    with pytest.raises(power_client.PowerError):
        investigation.run_investigation(Location("T", 10, 76), 2001, END, VARS)


# --------------------------------------------------------------------------- comparison
@pytest.fixture(scope="module")
def trio():
    specs = {"Warm": dict(t_trend=0.06, seed=1), "Cool": dict(t_trend=-0.06, seed=2), "Flat": dict(seed=3)}
    out = {}
    for i, (name, kw) in enumerate(specs.items()):
        loc = Location(name, 10 + i, 76 + i)
        out[name] = investigation.analyze(_power(10 + i, 76 + i, **kw), loc, 2001, END, VARS)
    return out


def test_opposite_trend_finder(trio):
    table = comparison.trend_table(trio, VARS)
    opp = comparison.find_opposite_trends(table).set_index("variable")
    assert bool(opp.loc["T2M", "opposite"])
    assert opp.loc["T2M", "increasing"] == ["Warm"] and opp.loc["T2M", "decreasing"] == ["Cool"]
    assert "Flat" in opp.loc["T2M", "no_significant"]
    assert not bool(opp.loc["RH2M", "opposite"])
    pairs = comparison.opposite_pairs(table, "T2M")
    assert len(pairs) == 1 and pairs.iloc[0]["increasing_location"] == "Warm"


def _fake_pool(n=10):
    """Pool of synthetic locations whose slopes are known, to check similarity ranking."""
    res = {}
    rng = np.random.default_rng(0)
    for i in range(n):
        t, p, rh = rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04), rng.uniform(-0.3, 0.3)
        loc = Location(f"P{i}", 20 + i, 10 + i)
        res[loc.name] = (loc, dict(t_trend=t, p_trend=p, rh_trend=rh, seed=100 + i))
    return res


def test_similar_regions_ranks_nearest_slopes_first():
    specs = _fake_pool(12)
    # make P3 a near-copy of the query's change pattern
    query_kw = dict(t_trend=0.05, p_trend=-0.04, rh_trend=-0.3, seed=7)
    specs["P3"] = (specs["P3"][0], dict(t_trend=0.051, p_trend=-0.039, rh_trend=-0.29, seed=8))
    pool_res = {n: investigation.analyze(_power(l.lat, l.lon, **kw), l, 2001, END, VARS) for n, (l, kw) in specs.items()}
    q = investigation.analyze(_power(**query_kw), Location("Query", 10, 76), 2001, END, VARS)
    ranking, notes = comparison.similar_regions(q, pool_res, VARS, top_k=3)
    assert ranking.iloc[0]["location"] == "P3"
    assert list(ranking["rank"]) == [1, 2, 3] and ranking["distance"].is_monotonic_increasing
    assert any("root-mean-square" in n for n in notes)


def test_similar_regions_excludes_self_and_handles_small_pool():
    q = investigation.analyze(_power(t_trend=0.03, seed=1), Location("Query", 10, 76), 2001, END, VARS)
    ranking, notes = comparison.similar_regions(q, {"Query": q}, VARS)
    assert ranking.empty and notes


def test_pool_definition_is_sane():
    names = [l.name for l in pool.CANDIDATE_POOL]
    assert 20 <= len(names) <= 30 and len(set(names)) == len(names)
    assert all(-90 <= l.lat <= 90 and -180 <= l.lon <= 180 for l in pool.CANDIDATE_POOL)
    assert len({(round(l.lat), round(l.lon)) for l in pool.CANDIDATE_POOL}) == len(names)   # no duplicates
    assert pool.is_same_place(pool.DEMO_LOCATIONS[1], Location("x", 26.9, 75.8))


def test_fetch_pool_continues_after_failures(monkeypatch):
    def fake(lat, lon, s, e, v=None, **k):
        if lat < 0:
            raise power_client.PowerAPIError("simulated failure")
        return _power(lat, lon, seed=int(abs(lat)))
    monkeypatch.setattr(power_client, "fetch_monthly", fake)
    monkeypatch.setattr(power_client, "is_cached", lambda *a, **k: True)       # skip delays
    prog = []
    results, errors = pool.fetch_pool(2001, END, progress=lambda i, n, m: prog.append((i, n)), sleep=lambda s: None)
    assert errors and results and len(results) + len(errors) == len(pool.CANDIDATE_POOL)
    assert prog[-1] == (len(pool.CANDIDATE_POOL), len(pool.CANDIDATE_POOL))
    res, errs = investigation.analyze_pool(results, 2001, END, VARS, pool.CANDIDATE_POOL)
    assert len(res) == len(results) and not errs


# --------------------------------------------------------------------------- report
def test_report_csv_roundtrip_and_sections(trio):
    data = report.report_csv(trio.values())
    df = pd.read_csv(io.BytesIO(data))
    assert set(df.columns) == {"section", "location", "period", "variable", "metric", "value", "unit", "note"}
    assert {"provenance", "data_quality", "trend_annual", "evidence_score", "fingerprint", "finding"} <= set(df.section)
    assert set(df.location) == {"Warm", "Cool", "Flat"}
    row = df[(df.location == "Warm") & (df.variable == "T2M") & (df.metric == "sens_slope_per_year")]
    assert float(row.iloc[0]["value"]) == pytest.approx(trio["Warm"].trends["T2M"].annual.sens_slope)
    assert "causation" in " ".join(df[df.section == "finding"]["note"].fillna("") + df[df.section == "finding"]["value"].astype(str)).lower()
    m = pd.read_csv(io.BytesIO(report.monthly_data_csv(trio.values())))
    assert len(m) == 3 * (END - 2001 + 1) * 12 and "T2M" in m.columns
