"""Session state, location registry and the (UI-side) investigation runner.

Streamlit drops the session value of a widget that is not rendered on the current page, so every
setting has a persistent copy (no prefix) that is mirrored from its widget (`w_` prefix) by callbacks.
"""
from __future__ import annotations

import logging

import streamlit as st

from earthscope import config, investigation, power_client, pool
from earthscope.models import Location

log = logging.getLogger(__name__)
MAX_LOCATIONS = 5


def init() -> None:
    ss = st.session_state
    ss.setdefault("custom_locations", [])
    ss.setdefault("selected_locations", [loc.name for loc in pool.DEMO_LOCATIONS])
    ss.setdefault("variables", list(config.VARIABLE_ORDER))
    ss.setdefault("period", (config.MIN_YEAR, config.last_full_year()))
    ss.setdefault("investigations", {})      # name -> InvestigationResult
    ss.setdefault("run_errors", {})          # name -> message
    ss.setdefault("run_settings", None)
    ss.setdefault("active_location", None)
    ss.setdefault("pool_power", None)        # name -> PowerResult (raw retrieved data)
    ss.setdefault("pool_fetch_errors", {})
    ss.setdefault("pool_analysis", None)     # {"key":..., "results":..., "errors":...}


def all_locations() -> dict[str, Location]:
    """Selectable locations: demo locations, user-added custom locations, candidate pool."""
    out: dict[str, Location] = {}
    for loc in pool.DEMO_LOCATIONS + list(st.session_state.custom_locations) + pool.CANDIDATE_POOL:
        out.setdefault(loc.name, loc)
    return out


# ----------------------------------------------------------------------- widget sync callbacks
def sync_locations() -> None:
    st.session_state.selected_locations = list(st.session_state.get("w_sel", []))


def sync_variables() -> None:
    st.session_state.variables = list(st.session_state.get("w_vars", []))


def sync_period() -> None:
    st.session_state.period = tuple(st.session_state.get("w_period", st.session_state.period))


def add_custom_location() -> None:
    """Callback for the 'add custom location' form (runs before the next script run)."""
    ss = st.session_state
    name = (ss.get("w_cname") or "").strip()
    try:
        lat, lon = power_client.validate_location(ss.get("w_clat"), ss.get("w_clon"))
    except power_client.PowerError as exc:
        ss["custom_msg"] = ("error", str(exc))
        return
    if not name:
        name = f"Custom ({lat:.2f}, {lon:.2f})"
    existing = all_locations()
    if name in existing:
        name = f"{name} (custom)"
    if name in existing:
        ss["custom_msg"] = ("error", f"A location named '{name}' already exists.")
        return
    ss.custom_locations.append(Location(name, lat, lon, "custom"))
    sel = list(ss.get("w_sel", ss.selected_locations))
    if len(sel) < MAX_LOCATIONS:
        sel.append(name)
    ss.selected_locations = sel
    ss.pop("w_sel", None)            # drop the widget's key: it is rebuilt next run from selected_locations
    ss["custom_msg"] = ("success", f"Added {name} ({lat:.4f}, {lon:.4f}).")


# ----------------------------------------------------------------------- runner
def run_investigations(locations: list[Location], start: int, end: int, variables: list[str]) -> None:
    ss = st.session_state
    results: dict = {}
    errors: dict = {}
    with st.status("Running automated investigation…", expanded=True) as status:
        for loc in locations:
            bar = st.progress(0.0, text=f"{loc.name}: starting")

            def cb(i: int, n: int, msg: str, _bar=bar, _name=loc.name) -> None:
                _bar.progress(min(i / n, 1.0), text=f"{_name}: {msg}")

            try:
                results[loc.name] = investigation.run_investigation(loc, start, end, variables, progress=cb)
                bar.progress(1.0, text=f"{loc.name}: complete")
            except (power_client.PowerError, investigation.InvestigationError) as exc:
                errors[loc.name] = str(exc)
                bar.empty()
                st.write(f"❌ **{loc.name}** — {exc}")
            except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
                log.exception("Unexpected failure for %s", loc.name)
                errors[loc.name] = f"Unexpected error ({type(exc).__name__}): {exc}"
                bar.empty()
                st.write(f"❌ **{loc.name}** — {errors[loc.name]}")
        if results and not errors:
            status.update(label="Investigation complete", state="complete", expanded=False)
        elif results:
            status.update(label="Finished with some failures", state="error", expanded=True)
        else:
            status.update(label="Investigation failed", state="error", expanded=True)
    ss.investigations = results
    ss.run_errors = errors
    ss.run_settings = {"start": start, "end": end, "variables": list(variables)}
    ss.active_location = next(iter(results), None)
    ss.pool_analysis = None
    if "w_active" in ss:
        del ss["w_active"]
