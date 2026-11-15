#!/usr/bin/env python3
"""Live NASA POWER smoke test for EARTHSCOPE.

Run this ONCE on your own machine (it needs internet access to power.larc.nasa.gov):

    python scripts/smoke_test_nasa.py                 # 2 demo locations + 3 pool locations
    python scripts/smoke_test_nasa.py --full-pool     # all 30 pool locations (slower; tests throttling)

What it verifies against the REAL service, independently of the application's own parser first:
  1. HTTP status, latency, JSON structure (properties.parameter.<VAR>.<YYYYMM>, month 13 = annual value)
  2. fill value (-999), grid-cell coordinates, elevation, units and long names, API version
  3. completeness: expected vs returned months, missing values, last valid month (is 2025 complete?)
  4. that EARTHSCOPE's parser accepts the response and that our day-weighted annual means agree with
     NASA's own annual values
  5. the whole analysis pipeline on the live data (first REAL results for the demo locations)
  6. disk cache round-trip, and retrieval of candidate-pool locations (throttling / 429 behaviour)

Exit code 0 = no FAIL. Output is also saved to smoke_test_report.txt - send that file back to
whoever is maintaining the code if anything is marked WARN or FAIL.
"""
from __future__ import annotations

import argparse
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from earthscope import config, investigation, pool, power_client  # noqa: E402
from earthscope.preprocess import annual_means  # noqa: E402

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
VARS = list(config.VARIABLE_ORDER)
START, END = config.MIN_YEAR, config.last_full_year()


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []

    def add(self, level: str, msg: str) -> None:
        print(f"[{level}] {msg}", flush=True)
        self.rows.append((level, msg))

    def count(self, level: str) -> int:
        return sum(1 for lv, _ in self.rows if lv == level)

    def text(self) -> str:
        return "\n".join(f"[{lv}] {m}" for lv, m in self.rows) + "\n"


# --------------------------------------------------------------------------- raw request
def raw_request(loc, rep: Report):
    params = {"parameters": ",".join(VARS), "community": config.POWER_COMMUNITY, "longitude": loc.lon,
              "latitude": loc.lat, "start": START, "end": END, "format": "JSON"}
    t0 = time.perf_counter()
    try:
        resp = requests.get(config.POWER_BASE_URL, params=params, timeout=config.REQUEST_TIMEOUT_S,
                            headers={"User-Agent": config.USER_AGENT})
    except requests.RequestException as exc:
        rep.add(FAIL, f"{loc.name}: request failed ({type(exc).__name__}: {exc}). "
                      "Check your internet connection, VPN or firewall.")
        return None
    dt = time.perf_counter() - t0
    rep.add(INFO, f"{loc.name}: HTTP {resp.status_code} in {dt:.1f}s, {len(resp.content) / 1024:.0f} KB")
    if resp.status_code != 200:
        rep.add(FAIL, f"{loc.name}: expected HTTP 200 but got {resp.status_code}. Body: {resp.text[:300]!r}")
        return None
    try:
        payload = resp.json()
    except ValueError:
        rep.add(FAIL, f"{loc.name}: response is not valid JSON. Start of body: {resp.text[:200]!r}")
        return None
    if not isinstance(payload, dict):
        rep.add(FAIL, f"{loc.name}: JSON top level is {type(payload).__name__}, expected an object.")
        return None
    return payload


# --------------------------------------------------------------------------- structure checks
def inspect_payload(payload: dict, loc, rep: Report) -> None:
    name = loc.name
    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    props = payload.get("properties") if isinstance(payload.get("properties"), dict) else {}
    params = props.get("parameter") if isinstance(props.get("parameter"), dict) else None
    rep.add(INFO, f"{name}: top-level keys = {sorted(payload)}")
    if params is None:
        rep.add(FAIL, f"{name}: no properties.parameter block (messages: {payload.get('messages')!r}).")
        return

    fill = header.get("fill_value")
    rep.add(PASS if fill == -999 else WARN, f"{name}: header.fill_value = {fill!r} (client assumes -999)")
    rep.add(INFO, f"{name}: API = {header.get('api')!r}; sources = {header.get('sources')!r}")
    if payload.get("messages"):
        rep.add(WARN, f"{name}: server messages: {payload['messages']!r}")

    coords = (payload.get("geometry") or {}).get("coordinates")
    if isinstance(coords, list) and len(coords) >= 2:
        rep.add(INFO, f"{name}: geometry.coordinates = {coords} (expected [lon, lat, elevation_m]); "
                      f"requested lat={loc.lat}, lon={loc.lon}")
        if abs(coords[0] - loc.lon) > 2 or abs(coords[1] - loc.lat) > 2:
            rep.add(WARN, f"{name}: coordinates look swapped or far from the request - check lon/lat order.")
    else:
        rep.add(WARN, f"{name}: no geometry.coordinates in response.")

    meta = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    n_expected = (END - START + 1) * 12
    for var in VARS:
        series = params.get(var)
        if not isinstance(series, dict) or not series:
            rep.add(FAIL, f"{name}: parameter {var} missing (returned: {sorted(params)}).")
            continue
        keys = list(series)
        bad = [k for k in keys if not re.fullmatch(r"\d{6}", str(k))]
        if bad:
            rep.add(FAIL, f"{name} {var}: {len(bad)} keys are not YYYYMM, e.g. {bad[:3]}.")
            continue
        monthly = [k for k in keys if 1 <= int(k[4:]) <= 12 and START <= int(k[:4]) <= END]
        annual = [k for k in keys if k[4:] == "13"]
        other = [k for k in keys if k not in monthly and k not in annual]
        rep.add(PASS if len(monthly) == n_expected else WARN,
                f"{name} {var}: {len(monthly)} monthly keys (expected {n_expected}); "
                f"{len(annual)} annual (month 13) keys; {len(other)} other keys")
        vals = {k: float(series[k]) for k in monthly}
        filled = [k for k, v in vals.items() if np.isclose(v, -999.0)]
        valid = sorted(k for k in vals if k not in filled)
        rep.add(INFO if not filled else WARN,
                f"{name} {var}: {len(filled)} fill-value (-999) months"
                + (f", e.g. {sorted(filled)[:6]}" if filled else ""))
        if valid:
            last = valid[-1]
            rep.add(PASS if last == f"{END}12" else WARN,
                    f"{name} {var}: first valid {valid[0]}, last valid {last} (expected last {END}12)"
                    + ("" if last == f"{END}12" else " -> the latest year is incomplete; the app will exclude "
                                                     "incomplete years from annual means"))
            arr = np.array([vals[k] for k in valid])
            lo, hi = config.VARIABLES[var].bounds
            outside = int(((arr < lo) | (arr > hi)).sum())
            rep.add(PASS if outside == 0 else WARN,
                    f"{name} {var}: value range {arr.min():.3g} to {arr.max():.3g} "
                    f"(plausible {lo} to {hi}); {outside} outside")
        else:
            rep.add(FAIL, f"{name} {var}: no valid values at all.")
        entry = meta.get(var) if isinstance(meta.get(var), dict) else {}
        units = entry.get("units")
        want = config.VARIABLES[var].api_unit
        same = units is not None and power_client._normalise_unit(units) == power_client._normalise_unit(want)
        rep.add(PASS if same else WARN,
                f"{name} {var}: units = {units!r} (expected {want!r}); longname = {entry.get('longname')!r}")


def check_parser_and_pipeline(payload: dict, loc, rep: Report) -> None:
    name = loc.name
    try:
        pr = power_client.parse_response(payload, VARS, START, END, req_lat=loc.lat, req_lon=loc.lon)
    except power_client.PowerError as exc:
        rep.add(FAIL, f"{name}: EARTHSCOPE's parser rejected the live response: {exc}")
        return
    rep.add(PASS, f"{name}: parser accepted the response; {int(pr.monthly.notna().sum().sum())} valid monthly "
                  f"values, first valid {pr.meta.first_valid}, last valid {pr.meta.last_valid}")
    for w in pr.meta.warnings:
        rep.add(WARN, f"{name}: parser notice: {w}")

    m = pr.meta
    if m.returned_lat is not None:
        off = max(abs(m.returned_lat - loc.lat), abs(m.returned_lon - loc.lon))
        rep.add(PASS if off <= 1.0 else WARN,
                f"{name}: NASA grid cell {m.returned_lat:.4f}, {m.returned_lon:.4f} "
                f"(elevation {m.elevation_m} m), offset from request {off:.2f} deg")

    ours = annual_means(pr.monthly)
    nasa = pr.annual_reported
    if nasa.empty:
        rep.add(WARN, f"{name}: response contained no annual (month 13) values for a cross-check.")
    else:
        tol = {"temperature": 0.1, "precipitation": 0.1, "humidity": 1.0}
        for var in VARS:
            if var not in nasa:
                continue
            both = pd.concat([ours[var], nasa[var]], axis=1, keys=["ours", "nasa"]).dropna()
            if both.empty:
                continue
            d = (both["ours"] - both["nasa"]).abs()
            limit = tol[config.VARIABLES[var].kind]
            rep.add(PASS if d.max() <= limit else WARN,
                    f"{name} {var}: our day-weighted annual mean vs NASA annual value: max |diff| "
                    f"{d.max():.4f}, mean {d.mean():.4f} over {len(both)} years (tolerance {limit})")

    try:
        res = investigation.analyze(pr, loc, START, END, VARS)
    except (investigation.InvestigationError, ValueError) as exc:
        rep.add(FAIL, f"{name}: analysis pipeline failed on live data: {exc}")
        return
    rep.add(PASS, f"{name}: full analysis pipeline ran on the live data ({START}-{END})")
    for var in res.variables:
        tr, conf = res.trends[var].annual, res.confidence[var]
        if tr.ok:
            rep.add(INFO, f"{name} | {config.VARIABLES[var].name}: {tr.direction}, "
                          f"{tr.slope_per_decade:+.3g} {config.VARIABLES[var].rate_unit}/decade, "
                          f"MK p={tr.mk_p:.4f}, n={tr.n}, evidence {conf.score:.0f}/100 ({conf.label})")
        else:
            rep.add(WARN, f"{name} | {config.VARIABLES[var].name}: not assessed - {tr.reason}")
    rep.add(INFO, f"{name} | fingerprint: {res.fingerprint.label}")
    rep.add(INFO, f"{name} | data completeness {res.quality.overall_completeness * 100:.1f}%")
    for w in res.warnings:
        rep.add(WARN, f"{name}: data notice: {w}")


def check_cache(loc, rep: Report) -> None:
    try:
        a = power_client.fetch_monthly(loc.lat, loc.lon, START, END)
        b = power_client.fetch_monthly(loc.lat, loc.lon, START, END)
    except power_client.PowerError as exc:
        rep.add(FAIL, f"{loc.name}: fetch_monthly() failed: {exc}")
        return
    same = np.allclose(a.monthly.to_numpy(), b.monthly.to_numpy(), equal_nan=True)
    if b.meta.from_cache and same:
        rep.add(PASS, f"{loc.name}: cache round-trip works ({config.CACHE_DIR}); the app will now start instantly "
                      "for this location")
    else:
        rep.add(WARN, f"{loc.name}: second request was not served from cache (from_cache={b.meta.from_cache}, "
                      f"identical={same}). Is {config.CACHE_DIR} writable?")


def check_pool(n: int, full: bool, rep: Report) -> None:
    locs = list(pool.CANDIDATE_POOL)
    if not full:
        step = max(1, len(locs) // max(n, 1))
        locs = locs[::step][:n]
    if not locs:
        return
    ok, failed, lat = 0, 0, []
    throttled = False
    for i, loc in enumerate(locs):
        cached = power_client.is_cached(loc.lat, loc.lon, START, END, VARS)
        t0 = time.perf_counter()
        try:
            pr = power_client.fetch_monthly(loc.lat, loc.lon, START, END, VARS)
        except power_client.PowerError as exc:
            failed += 1
            throttled = throttled or "429" in str(exc) or "throttl" in str(exc).lower()
            rep.add(WARN, f"pool {loc.name}: FAILED - {exc}")
        else:
            ok += 1
            dt = time.perf_counter() - t0
            if not cached:
                lat.append(dt)
            comp = pr.monthly.notna().sum().sum() / pr.monthly.size
            rep.add(INFO, f"pool {loc.name}: ok in {dt:.1f}s ({'cache' if cached else 'live'}), "
                          f"completeness {comp * 100:.1f}%")
        if not cached and i < len(locs) - 1:
            time.sleep(config.POLITE_DELAY_S)
    if ok == 0:
        rep.add(FAIL, f"pool: none of {len(locs)} locations could be retrieved.")
    else:
        mean = f", mean live latency {np.mean(lat):.1f}s" if lat else ""
        est = f" Estimated full 30-location scan: about {np.mean(lat) * 30 / 60:.1f} min." if lat else ""
        rep.add(PASS if failed == 0 else WARN, f"pool: {ok} ok, {failed} failed of {len(locs)}{mean}.{est}")
    if throttled:
        rep.add(WARN, "pool: NASA throttled (HTTP 429). Increase POLITE_DELAY_S in earthscope/config.py.")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="Live NASA POWER smoke test for EARTHSCOPE")
    ap.add_argument("--pool", type=int, default=3, help="number of candidate-pool locations to retrieve (default 3)")
    ap.add_argument("--full-pool", action="store_true", help="retrieve all candidate-pool locations")
    ap.add_argument("--skip-pool", action="store_true", help="do not test candidate-pool retrieval")
    ap.add_argument("--base-url", help="override the NASA POWER endpoint (used only by the offline mock-server tests)")
    ap.add_argument("--report", default="smoke_test_report.txt", help="where to save the text report")
    args = ap.parse_args(argv)
    if args.base_url:
        config.POWER_BASE_URL = args.base_url

    rep = Report()
    rep.add(INFO, f"EARTHSCOPE NASA POWER smoke test, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    rep.add(INFO, f"Python {platform.python_version()} on {platform.system()}; requests {requests.__version__}; "
                  f"pandas {pd.__version__}; numpy {np.__version__}")
    rep.add(INFO, f"Endpoint {config.POWER_BASE_URL}; community {config.POWER_COMMUNITY}; period {START}-{END}; "
                  f"cache dir {config.CACHE_DIR}")
    if args.base_url:
        rep.add(WARN, "A custom --base-url is in use: results do NOT describe the real NASA service.")

    for i, loc in enumerate(pool.DEMO_LOCATIONS):
        rep.add(INFO, f"--- demo location: {loc.name} ({loc.lat}, {loc.lon}) ---")
        payload = raw_request(loc, rep)
        if payload is not None:
            inspect_payload(payload, loc, rep)
            check_parser_and_pipeline(payload, loc, rep)
            check_cache(loc, rep)
        if i < len(pool.DEMO_LOCATIONS) - 1:
            time.sleep(config.POLITE_DELAY_S)

    if not args.skip_pool and (args.full_pool or args.pool > 0):
        rep.add(INFO, "--- candidate pool ---")
        check_pool(args.pool, args.full_pool, rep)

    fails, warns = rep.count(FAIL), rep.count(WARN)
    verdict = ("FAIL - the live service does not match what the application expects; see the FAIL lines."
               if fails else "PASS with warnings - read the WARN lines." if warns else "PASS - no problems found.")
    rep.add(INFO, f"SUMMARY: {rep.count(PASS)} pass, {warns} warn, {fails} fail. {verdict}")
    try:
        Path(args.report).write_text(rep.text(), encoding="utf-8")
        print(f"\nReport saved to {Path(args.report).resolve()}")
    except OSError as exc:
        print(f"\nCould not save the report ({exc}); copy the output above instead.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
