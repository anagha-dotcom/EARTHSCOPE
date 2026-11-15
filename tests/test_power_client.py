"""Parser / client tests. Uses a synthetic payload in the documented format (NOT NASA data)."""
import numpy as np
import pytest
import requests

from earthscope import config, power_client as pc
from earthscope.power_client import (InvalidLocationError, PowerAPIError, PowerDataError,
                                     parse_response, validate_location, validate_period)
from tests.synthetic import make_payload

VARS = ["T2M", "PRECTOTCORR", "RH2M"]


def _parse(payload, start=2001, end=2020):
    return parse_response(payload, VARS, start, end, req_lat=10.0, req_lon=76.0)


def test_parse_shapes_and_annual_key_excluded():
    res = _parse(make_payload())
    assert res.monthly.shape == (240, 3)                      # 20 years x 12, annual (13) not in monthly
    assert res.monthly.index[0].strftime("%Y-%m") == "2001-01"
    assert res.monthly.notna().all().all()
    assert len(res.annual_reported) == 20                      # annual values kept separately
    assert res.meta.returned_lat == 10.0 and res.meta.returned_lon == 76.0
    assert res.meta.elevation_m == 85.5
    assert res.meta.first_valid == "2001-01" and res.meta.last_valid == "2020-12"


def test_fill_value_becomes_nan():
    res = _parse(make_payload(missing=[("T2M", "200507"), ("RH2M", "201012")]))
    assert np.isnan(res.monthly.loc["2005-07-01", "T2M"])
    assert np.isnan(res.monthly.loc["2010-12-01", "RH2M"])
    assert res.monthly.isna().sum().sum() == 2


def test_missing_parameter_raises():
    with pytest.raises(PowerDataError, match="RH2M"):
        _parse(make_payload(drop_param="RH2M"))


def test_unexpected_structure_raises():
    with pytest.raises(PowerDataError):
        _parse({"foo": "bar"})
    with pytest.raises(PowerDataError):
        _parse([1, 2, 3])


def test_all_missing_raises():
    p = make_payload(start=2001, end=2002)
    for var in p["properties"]["parameter"].values():
        for k in var:
            var[k] = -999.0
    with pytest.raises(PowerDataError, match="no valid values"):
        _parse(p, 2001, 2002)


def test_unrecognised_keys_raise_clear_error():
    p = make_payload(start=2001, end=2002)
    p["properties"]["parameter"] = {v: {"Jan-2001": 1.0} for v in VARS}
    with pytest.raises(PowerDataError, match="date keys"):
        _parse(p, 2001, 2002)


def test_unit_mismatch_warns():
    res = _parse(make_payload(unit_overrides={"PRECTOTCORR": "mm/month"}))
    assert any("PRECTOTCORR" in w and "mm/month" in w for w in res.meta.warnings)


def test_implausible_values_become_nan_with_warning():
    p = make_payload(start=2001, end=2010)
    p["properties"]["parameter"]["RH2M"]["200501"] = 250.0
    res = _parse(p, 2001, 2010)
    assert np.isnan(res.monthly.loc["2005-01-01", "RH2M"])
    assert any("plausible" in w for w in res.meta.warnings)


def test_requested_months_absent_from_response_are_nan():
    p = make_payload(start=2001, end=2010)
    res = parse_response(p, VARS, 2001, 2012, req_lat=10, req_lon=76)   # asked for 2 more years
    assert res.monthly.loc["2011-01-01":].isna().all().all()
    assert res.meta.last_valid == "2010-12"


@pytest.mark.parametrize("lat,lon", [(91, 0), (-91, 0), (0, 181), (0, -181), ("x", 1), (float("nan"), 1)])
def test_invalid_locations(lat, lon):
    with pytest.raises(InvalidLocationError):
        validate_location(lat, lon)


def test_validate_period():
    assert validate_period(2001, 2020) == (2001, 2020)
    with pytest.raises(InvalidLocationError):
        validate_period(2020, 2001)
    with pytest.raises(InvalidLocationError):
        validate_period(1970, 2000)


# ------------------------------------------------------------------ fetch path (HTTP mocked)
class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text
        self.url = "https://example.test/request"

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, *a, **k):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "BACKOFF_BASE_S", 0.0)


def test_fetch_success_then_cache_hit():
    s = _Session([_Resp(200, make_payload(start=2001, end=2020))])
    r1 = pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=s)
    assert not r1.meta.from_cache and s.calls == 1
    r2 = pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=_Session([]))   # would crash if it hit network
    assert r2.meta.from_cache
    assert r2.monthly.equals(r1.monthly)


def test_retry_then_success():
    s = _Session([requests.Timeout("slow"), _Resp(503, {"message": "busy"}), _Resp(200, make_payload())])
    r = pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=s)
    assert s.calls == 3 and r.monthly.shape[0] == 240


def test_retries_exhausted_gives_readable_error():
    s = _Session([requests.ConnectionError("down")] * config.MAX_RETRIES)
    with pytest.raises(PowerAPIError, match="Could not reach NASA POWER"):
        pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=s)


def test_client_error_not_retried():
    s = _Session([_Resp(422, {"messages": ["bad parameter"]})])
    with pytest.raises(PowerAPIError, match="rejected"):
        pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=s)
    assert s.calls == 1


def test_invalid_json_body():
    with pytest.raises(PowerDataError, match="not valid JSON"):
        pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=_Session([_Resp(200, None)]))


def test_bad_response_is_not_cached():
    s = _Session([_Resp(200, {"properties": {"parameter": {}}})])
    with pytest.raises(PowerDataError):
        pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=s)
    assert not pc.is_cached(10.0, 76.0, 2001, 2020)


def test_corrupt_cache_is_discarded_and_refetched():
    p = pc._cache_path(10.0, 76.0, 2001, 2020, VARS)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    s = _Session([_Resp(200, make_payload())])
    r = pc.fetch_monthly(10.0, 76.0, 2001, 2020, session=s)
    assert not r.meta.from_cache and s.calls == 1
