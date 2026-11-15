"""NASA POWER monthly-point client: request, validate, parse, cache.

IMPORTANT - verification status
    The response format assumed here follows the NASA POWER documentation:
        properties.parameter.<PARAM>.<YYYYMM>   (MM = 01..12 monthly, 13 = annual value)
        geometry.coordinates = [lon, lat, elevation_m]
        header.fill_value    = -999
    This could NOT be confirmed against the live API from the development sandbox (network
    blocked). `scripts/smoke_test_nasa.py` checks all of it against the real service. The parser
    is deliberately strict: if the structure is not what is expected it raises PowerDataError
    with a descriptive message instead of guessing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from . import config
from .models import PowerMeta, PowerResult

log = logging.getLogger(__name__)

_UNIT_ALIASES = {"c": "C", "degc": "C", "°c": "C", "celsius": "C", "deg c": "C"}


class PowerError(Exception):
    """Base class for all NASA POWER problems (message is safe to show to users)."""


class InvalidLocationError(PowerError):
    """Latitude/longitude/period are not acceptable."""


class PowerAPIError(PowerError):
    """Network problem, HTTP error or throttling."""


class PowerDataError(PowerError):
    """Response arrived but is not usable (unexpected structure, no valid data)."""


# --------------------------------------------------------------------------- validation
def validate_location(lat: float, lon: float) -> tuple[float, float]:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        raise InvalidLocationError("Latitude and longitude must be numbers.") from None
    if not (np.isfinite(lat_f) and np.isfinite(lon_f)):
        raise InvalidLocationError("Latitude and longitude must be finite numbers.")
    if not -90.0 <= lat_f <= 90.0:
        raise InvalidLocationError(f"Latitude {lat_f} is outside the valid range -90 to 90.")
    if not -180.0 <= lon_f <= 180.0:
        raise InvalidLocationError(f"Longitude {lon_f} is outside the valid range -180 to 180.")
    return round(lat_f, 4), round(lon_f, 4)


def validate_period(start_year: int, end_year: int) -> tuple[int, int]:
    try:
        s, e = int(start_year), int(end_year)
    except (TypeError, ValueError):
        raise InvalidLocationError("Start and end year must be whole numbers.") from None
    if s > e:
        raise InvalidLocationError("Start year must not be after end year.")
    if s < 1981:
        raise InvalidLocationError("NASA POWER meteorology starts in 1981.")
    if e > config.last_full_year() + 1:
        raise InvalidLocationError(f"End year {e} is in the future / not yet available.")
    return s, e


# --------------------------------------------------------------------------- parsing (pure)
def _normalise_unit(u: str) -> str:
    return _UNIT_ALIASES.get(str(u).strip().lower(), str(u).strip())


def _to_float(v, fill: float) -> float:
    if v is None or isinstance(v, bool):
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(f) or np.isclose(f, fill):
        return np.nan
    return f


def parse_response(
    payload: dict,
    variables: list[str],
    start_year: int,
    end_year: int,
    *,
    req_lat: float,
    req_lon: float,
    request_url: Optional[str] = None,
    retrieved_at: Optional[str] = None,
    from_cache: bool = False,
) -> PowerResult:
    """Convert a NASA POWER monthly JSON payload to a PowerResult. Pure function (no I/O)."""
    if not isinstance(payload, dict):
        raise PowerDataError("NASA POWER returned something that is not a JSON object.")

    props = payload.get("properties")
    params = props.get("parameter") if isinstance(props, dict) else None
    if not isinstance(params, dict):
        detail = payload.get("messages") or payload.get("message") or payload.get("errors") or ""
        raise PowerDataError(
            "Unexpected NASA POWER response structure (no properties.parameter block)."
            + (f" Server message: {detail}" if detail else "")
        )

    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    fill = header.get("fill_value", -999.0)
    try:
        fill = float(fill)
    except (TypeError, ValueError):
        fill = -999.0

    warnings: list[str] = []
    msgs = payload.get("messages")
    if msgs:
        warnings.append(f"NASA POWER messages: {msgs}")

    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    monthly = pd.DataFrame(index=idx, columns=variables, dtype=float)
    annual_rows: dict[int, dict[str, float]] = {}
    unrecognised = 0
    outside_requested = 0
    bad_range: dict[str, int] = {}

    for var in variables:
        series = params.get(var)
        if not isinstance(series, dict) or not series:
            raise PowerDataError(
                f"Parameter {var} is missing from the NASA POWER response "
                f"(parameters returned: {sorted(params)})."
            )
        info = config.VARIABLES.get(var)
        lo, hi = info.bounds if info else (-np.inf, np.inf)
        for key, raw in series.items():
            m = re.fullmatch(r"(\d{4})(\d{2})", str(key))
            if not m:
                unrecognised += 1
                continue
            year, month = int(m.group(1)), int(m.group(2))
            val = _to_float(raw, fill)
            if month == 13:
                annual_rows.setdefault(year, {})[var] = val
                continue
            if not 1 <= month <= 12:
                unrecognised += 1
                continue
            if not start_year <= year <= end_year:
                outside_requested += 1
                continue
            if np.isfinite(val) and not (lo <= val <= hi):
                bad_range[var] = bad_range.get(var, 0) + 1
                val = np.nan
            monthly.loc[pd.Timestamp(year=year, month=month, day=1), var] = val

    if unrecognised and not annual_rows and monthly.notna().sum().sum() == 0:
        sample = list(next(iter(params.values())))[:5]
        raise PowerDataError(
            f"Could not interpret the date keys in the NASA POWER response (e.g. {sample}); "
            "the API format may have changed."
        )
    if monthly.notna().sum().sum() == 0:
        raise PowerDataError(
            "NASA POWER returned no valid values for this location and period "
            "(all values were missing)."
        )
    if unrecognised:
        warnings.append(f"{unrecognised} response keys had an unrecognised date format and were ignored.")
    if outside_requested:
        warnings.append(f"{outside_requested} values outside the requested years were ignored.")
    for var, cnt in bad_range.items():
        warnings.append(f"{cnt} {var} values were outside the physically plausible range and treated as missing.")

    # --- location metadata
    r_lat = r_lon = elev = None
    coords = (payload.get("geometry") or {}).get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        r_lon, r_lat = float(coords[0]), float(coords[1])
        elev = float(coords[2]) if len(coords) >= 3 and coords[2] is not None else None
    else:
        warnings.append("Response did not include grid-cell coordinates.")

    # --- parameter metadata + unit validation
    p_info: dict[str, dict] = {}
    meta_params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    for var in variables:
        entry = meta_params.get(var) if isinstance(meta_params.get(var), dict) else {}
        p_info[var] = {"units": entry.get("units"), "longname": entry.get("longname")}
        expected = config.VARIABLES.get(var)
        got = entry.get("units")
        if expected and got and _normalise_unit(got) != _normalise_unit(expected.api_unit):
            warnings.append(
                f"{var}: NASA reports unit '{got}' but '{expected.api_unit}' was expected; "
                "results may be mis-scaled."
            )
        if expected and not got:
            warnings.append(f"{var}: unit not stated in the response; assuming {expected.api_unit}.")

    valid_any = monthly.dropna(how="all")
    first_valid = valid_any.index.min().strftime("%Y-%m")
    last_valid = valid_any.index.max().strftime("%Y-%m")
    api = header.get("api") if isinstance(header.get("api"), dict) else {}

    annual = pd.DataFrame.from_dict(annual_rows, orient="index").sort_index()
    annual.index.name = "year"

    meta = PowerMeta(
        requested_lat=req_lat, requested_lon=req_lon, start_year=start_year, end_year=end_year,
        community=config.POWER_COMMUNITY, parameters_requested=list(variables),
        returned_lat=r_lat, returned_lon=r_lon, elevation_m=elev,
        first_valid=first_valid, last_valid=last_valid, fill_value=fill,
        api_version=api.get("version") or header.get("api_version"),
        sources=list(header.get("sources", [])) if isinstance(header.get("sources"), list) else [],
        parameters_info=p_info, request_url=request_url,
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        from_cache=from_cache, warnings=warnings,
    )
    return PowerResult(monthly=monthly, annual_reported=annual, meta=meta)


# --------------------------------------------------------------------------- cache
def _cache_path(lat: float, lon: float, start: int, end: int, variables: list[str]):
    key = f"{config.POWER_COMMUNITY}|{lat:.4f}|{lon:.4f}|{start}|{end}|{','.join(sorted(variables))}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return config.CACHE_DIR / f"power_{lat:.2f}_{lon:.2f}_{start}-{end}_{digest}.json"


def _read_cache(path):
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        saved = datetime.fromisoformat(data["retrieved_at"])
        if datetime.now(timezone.utc) - saved > timedelta(days=config.CACHE_TTL_DAYS):
            return None
        return data
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(path, payload, request_url, retrieved_at):
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"retrieved_at": retrieved_at, "request_url": request_url, "payload": payload}),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError as exc:  # read-only filesystem on some free hosts - not fatal
        log.warning("Could not write cache: %s", exc)


# --------------------------------------------------------------------------- HTTP
def _server_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        for key in ("messages", "message", "errors", "error"):
            if body.get(key):
                return str(body[key])[:400]
    except ValueError:
        pass
    return (resp.text or "")[:300]


def _http_get(params: dict, session: requests.Session):
    last_err = "unknown error"
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = session.get(
                config.POWER_BASE_URL, params=params, timeout=config.REQUEST_TIMEOUT_S,
                headers={"User-Agent": config.USER_AGENT},
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    return resp.json(), resp.url
                except ValueError:
                    raise PowerDataError("NASA POWER returned a response that is not valid JSON.") from None
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code} {_server_message(resp)}"
            elif resp.status_code in (400, 422):
                raise PowerAPIError(
                    f"NASA POWER rejected the request (HTTP {resp.status_code}): {_server_message(resp)}"
                )
            else:
                raise PowerAPIError(
                    f"NASA POWER returned HTTP {resp.status_code}: {_server_message(resp)}"
                )
        if attempt < config.MAX_RETRIES - 1:
            time.sleep(config.BACKOFF_BASE_S * (2 ** attempt))
    raise PowerAPIError(
        f"Could not reach NASA POWER after {config.MAX_RETRIES} attempts ({last_err}). "
        "Check your internet connection or try again in a few minutes; the service may be "
        "throttling or temporarily unavailable."
    )


def fetch_monthly(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    variables: Optional[list[str]] = None,
    *,
    use_cache: bool = True,
    session: Optional[requests.Session] = None,
) -> PowerResult:
    """Retrieve monthly NASA POWER data for one point. Raises PowerError subclasses on failure."""
    variables = list(variables or config.VARIABLE_ORDER)
    lat, lon = validate_location(lat, lon)
    start_year, end_year = validate_period(start_year, end_year)
    path = _cache_path(lat, lon, start_year, end_year, variables)

    if use_cache:
        cached = _read_cache(path)
        if cached:
            try:
                return parse_response(
                    cached["payload"], variables, start_year, end_year, req_lat=lat, req_lon=lon,
                    request_url=cached.get("request_url"), retrieved_at=cached["retrieved_at"],
                    from_cache=True,
                )
            except PowerError:
                log.warning("Discarding unusable cache entry %s", path)
                try:
                    path.unlink()
                except OSError:
                    pass

    params = {
        "parameters": ",".join(variables), "community": config.POWER_COMMUNITY,
        "longitude": lon, "latitude": lat, "start": start_year, "end": end_year, "format": "JSON",
    }
    sess = session or requests.Session()
    payload, url = _http_get(params, sess)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = parse_response(
        payload, variables, start_year, end_year, req_lat=lat, req_lon=lon,
        request_url=url, retrieved_at=retrieved_at,
    )
    if use_cache:  # only valid, parsed responses are cached
        _write_cache(path, payload, url, retrieved_at)
    return result


def is_cached(lat: float, lon: float, start_year: int, end_year: int, variables: Optional[list[str]] = None) -> bool:
    variables = list(variables or config.VARIABLE_ORDER)
    try:
        lat, lon = validate_location(lat, lon)
    except PowerError:
        return False
    return _read_cache(_cache_path(lat, lon, start_year, end_year, variables)) is not None
