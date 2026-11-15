"""Tests for scripts/smoke_test_nasa.py against a LOCAL MOCK server (synthetic NASA-shaped payloads).

These verify the smoke-test script's own logic (it passes on a well-formed response, flags gaps as warnings and
fails clearly on HTTP errors / wrong structure). They say nothing about the real NASA service.
"""
from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from earthscope import config
from tests.synthetic import make_payload

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "smoke_test_nasa.py"
END = config.last_full_year()


def _load():
    spec = importlib.util.spec_from_file_location("smoke_test_nasa", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        mode = type(self).mode
        if mode == "403":
            return self._send(403, {"messages": ["forbidden (test)"]})
        if mode == "badshape":
            return self._send(200, {"foo": "bar"})
        kw = {}
        if mode == "gap":
            kw["missing"] = [("T2M", f"{END}11"), ("T2M", f"{END}12")]
        payload = make_payload(float(q["latitude"]), float(q["longitude"]), int(q["start"]), int(q["end"]),
                               t_trend=0.03, p_trend=-0.02, rh_trend=-0.1, **kw)
        self._send(200, payload)


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/api/temporal/monthly/point", srv
    srv.shutdown()
    srv.server_close()
    _Handler.mode = "ok"


@pytest.fixture
def smoke(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "POLITE_DELAY_S", 0.0)
    monkeypatch.setattr(config, "POWER_BASE_URL", config.POWER_BASE_URL)   # restored after the test
    return mod, tmp_path / "report.txt"


def _run(smoke, url, *extra):
    mod, report = smoke
    code = mod.main(["--base-url", url, "--report", str(report), *extra])
    return code, report.read_text(encoding="utf-8")


def test_well_formed_response_passes(server, smoke):
    url, _ = server
    code, text = _run(smoke, url, "--pool", "2")
    assert code == 0, text
    assert "[FAIL]" not in text
    assert "parser accepted the response" in text
    assert "full analysis pipeline ran" in text
    assert "cache round-trip works" in text
    assert "pool: 2 ok, 0 failed" in text
    assert "custom --base-url" in text                  # the mock run is labelled as not describing NASA
    assert "SUMMARY" in text


def test_incomplete_latest_year_is_warning_not_failure(server, smoke):
    url, _ = server
    _Handler.mode = "gap"
    code, text = _run(smoke, url, "--skip-pool")
    assert code == 0, text
    assert f"last valid {END}10" in text and "[WARN]" in text
    assert "latest year is incomplete" in text


def test_http_403_fails_with_clear_message_and_nonzero_exit(server, smoke):
    url, _ = server
    _Handler.mode = "403"
    code, text = _run(smoke, url, "--skip-pool")
    assert code == 1
    assert "expected HTTP 200 but got 403" in text
    assert "Traceback" not in text


def test_wrong_structure_fails_clearly(server, smoke):
    url, _ = server
    _Handler.mode = "badshape"
    code, text = _run(smoke, url, "--skip-pool")
    assert code == 1 and "no properties.parameter block" in text


def test_unreachable_server_fails_without_traceback(smoke, monkeypatch):
    monkeypatch.setattr(config, "REQUEST_TIMEOUT_S", 2)
    mod, report = smoke
    code = mod.main(["--base-url", "http://127.0.0.1:9/none", "--report", str(report), "--skip-pool"])
    assert code == 1
    text = report.read_text(encoding="utf-8")
    assert "request failed" in text and "Traceback" not in text
