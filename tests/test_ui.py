"""UI smoke tests: run the real Streamlit pages headlessly with Streamlit's AppTest.

!!! Test-only synthetic data. `power_client.fetch_monthly` is patched to build NASA-POWER-shaped payloads
from tests/synthetic.py (known, injected trends) and push them through the real parser. Nothing here says
anything about the real NASA service; it verifies that our pages run, wire up correctly and handle errors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from earthscope import config, power_client
from tests.synthetic import make_payload

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["views/investigation.py", "views/fingerprint.py", "views/comparison.py",
         "views/anomalies.py", "views/report.py"]
FAIL_LAT = 47.8864          # Ulaanbaatar (pool location) - made to fail in the pool-error test


def _fake_fetch(fail_lats=(), fail_all=False):
    def fetch(lat, lon, start_year, end_year, variables=None, **kw):
        variables = list(variables or config.VARIABLE_ORDER)
        lat, lon = power_client.validate_location(lat, lon)
        if fail_all or any(abs(lat - f) < 0.01 for f in fail_lats):
            raise power_client.PowerAPIError("Simulated NASA POWER outage (test).")
        h = int(hashlib.sha1(f"{lat},{lon}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        payload = make_payload(
            lat, lon, start_year, end_year, seed=h % 1000,
            t_trend=float(rng.uniform(-0.03, 0.08)), p_trend=float(rng.uniform(-0.04, 0.03)),
            rh_trend=float(rng.uniform(-0.3, 0.2)),
        )
        return power_client.parse_response(payload, variables, start_year, end_year, req_lat=lat, req_lon=lon,
                                           request_url="synthetic://test-fixture")
    return fetch


@pytest.fixture
def make_app(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "POLITE_DELAY_S", 0.0)

    def build(**fetch_kw):
        monkeypatch.setattr(power_client, "fetch_monthly", _fake_fetch(**fetch_kw))
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        return at
    return build


def _click(at: AppTest, label_prefix: str) -> AppTest:
    btn = next(b for b in at.button if b.label.startswith(label_prefix))
    return btn.click().run()


def _text(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.info] + [e.value for e in at.warning]
    parts += [e.value for e in at.error] + [e.value for e in at.success] + [e.value for e in at.caption]
    parts += [e.value for e in at.subheader] + [e.value for e in at.title]
    return "\n".join(str(p) for p in parts)


def test_overview_renders_and_defaults(make_app):
    at = make_app()
    assert not at.exception
    assert "EARTHSCOPE" in _text(at)
    assert at.multiselect[0].value == ["Palakkad, Kerala", "Jaipur, Rajasthan"]      # demo locations preselected
    assert at.multiselect[1].value == config.VARIABLE_ORDER
    assert at.session_state["period"] == (config.MIN_YEAR, config.last_full_year())
    assert not next(b for b in at.button if b.label == "Start Investigation").disabled


def test_start_investigation_then_every_page_renders(make_app):
    at = make_app()
    at = _click(at, "Start Investigation")
    assert not at.exception
    inv = at.session_state["investigations"]
    assert set(inv) == {"Palakkad, Kerala", "Jaipur, Rajasthan"}
    assert not at.session_state["run_errors"]
    for page in PAGES:
        at = at.switch_page(page).run()
        assert not at.exception, f"{page}: {[e.message for e in at.exception]}"
    # the last page is the report: it must offer both CSV downloads and the key-findings section
    assert "Key findings" in _text(at)


def test_pages_without_results_show_guidance_not_errors(make_app):
    at = make_app()
    for page in PAGES:
        at = at.switch_page(page).run()
        assert not at.exception
        assert any("No investigation has been run yet" in i.value for i in at.info), page


def test_short_period_disables_start_button(make_app):
    at = make_app()
    at.slider[0].set_value((2020, 2025)).run()
    assert not at.exception
    assert next(b for b in at.button if b.label == "Start Investigation").disabled
    assert any("at least" in w.value for w in at.warning)


def test_no_indicator_or_location_disables_start_button(make_app):
    at = make_app()
    at.multiselect[1].set_value([]).run()
    assert next(b for b in at.button if b.label == "Start Investigation").disabled
    at.multiselect[0].set_value([]).run()
    assert any("Select at least one location" in w.value for w in at.warning)


def test_nasa_failure_is_reported_not_crashed(make_app):
    at = make_app(fail_all=True)
    at = _click(at, "Start Investigation")
    assert not at.exception
    assert not at.session_state["investigations"]
    assert len(at.session_state["run_errors"]) == 2
    assert any("Simulated NASA POWER outage" in e.value for e in at.error)
    at = at.switch_page("views/report.py").run()
    assert not at.exception and any("No investigation has been run yet" in i.value for i in at.info)


def test_partial_failure_keeps_successful_location(make_app):
    at = make_app(fail_lats=(26.9124,))                      # Jaipur fails
    at = _click(at, "Start Investigation")
    assert not at.exception
    assert list(at.session_state["investigations"]) == ["Palakkad, Kerala"]
    assert list(at.session_state["run_errors"]) == ["Jaipur, Rajasthan"]
    for page in PAGES:
        at = at.switch_page(page).run()
        assert not at.exception, page


def test_two_indicators_gives_partial_fingerprint_and_pages_still_render(make_app):
    at = make_app()
    at.multiselect[1].set_value(["T2M", "PRECTOTCORR"]).run()
    at = _click(at, "Start Investigation")
    res = next(iter(at.session_state["investigations"].values()))
    assert res.fingerprint.pattern_id == "partial"
    for page in PAGES:
        at = at.switch_page(page).run()
        assert not at.exception, page


def test_similar_regions_flow_with_pool_failure(make_app):
    at = make_app(fail_lats=(FAIL_LAT,))
    at = _click(at, "Start Investigation")
    at = at.switch_page("views/comparison.py").run()
    assert not at.exception
    at = _click(at, "Retrieve candidate pool")
    assert not at.exception
    n_pool = 30
    assert len(at.session_state["pool_power"]) == n_pool - 1
    assert list(at.session_state["pool_fetch_errors"]) == ["Ulaanbaatar, Mongolia"]
    ranking_tables = [d.value for d in at.dataframe if "Rank" in d.value.columns]
    assert ranking_tables, "similar-regions ranking table was not rendered"
    table = ranking_tables[0]
    assert table.iloc[0]["Rank"] == "★" and len(table) == 1 + config.SIMILAR_TOP_K
    assert "Ulaanbaatar, Mongolia" not in set(table["Location"])


def test_changing_period_after_pool_scan_recomputes_pool_analysis(make_app):
    at = make_app()
    at = _click(at, "Start Investigation")
    at = at.switch_page("views/comparison.py").run()
    at = _click(at, "Retrieve candidate pool")
    key1 = at.session_state["pool_analysis"]["key"]
    at = at.switch_page("views/overview.py").run()
    at.slider[0].set_value((2005, 2020)).run()
    at = _click(at, "Start Investigation")
    assert not at.exception
    assert at.session_state["pool_power"], "raw pool data must survive a new run"
    at = at.switch_page("views/comparison.py").run()
    assert not at.exception
    assert at.session_state["pool_analysis"]["key"] != key1
    assert at.session_state["pool_analysis"]["key"][:2] == (2005, 2020)


def test_custom_location_validation_and_add(make_app):
    at = make_app()
    at.text_input[0].set_value("Test town")
    # missing coordinates -> friendly error (out-of-range values are blocked by the widgets' min/max)
    _click(at, "Add location")
    assert "custom_msg" in at.session_state
    kind, msg = at.session_state["custom_msg"]
    assert kind == "error" and "numbers" in msg
    # valid coordinates -> added and selected
    at.number_input[0].set_value(8.5241)
    at.number_input[1].set_value(76.9366)
    at = _click(at, "Add location")
    assert at.session_state["custom_msg"][0] == "success"
    assert "Test town" in at.multiselect[0].value
