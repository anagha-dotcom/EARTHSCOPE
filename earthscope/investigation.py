"""Automated investigation orchestrator: retrieve -> clean -> trends -> significance -> anomalies
-> fingerprint -> summary. Pure Python (no Streamlit); the UI passes an optional progress callback.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import config, power_client
from .anomalies import compute_monthly_anomalies, expected_by_chance, flag_anomalies
from .confidence import compute_confidence
from .fingerprint import build_fingerprint
from .models import InvestigationResult, Location, PowerResult
from .narratives import build_findings
from .preprocess import annual_means, assess_quality, seasonal_means, slice_period
from .trends import analyze_variable

Progress = Callable[[int, int, str], None]
STEPS = [
    "Retrieving NASA POWER data",
    "Cleaning and checking data",
    "Calculating annual and seasonal series",
    "Calculating trends",
    "Testing statistical significance and evidence scores",
    "Detecting unusual observations",
    "Generating environmental fingerprint",
    "Writing investigation summary",
]


class InvestigationError(ValueError):
    """Invalid investigation settings (message is safe to show to users)."""


def validate_settings(start_year: int, end_year: int, variables: list[str]) -> None:
    if not variables:
        raise InvestigationError("Select at least one environmental indicator.")
    unknown = [v for v in variables if v not in config.VARIABLES]
    if unknown:
        raise InvestigationError(f"Unknown indicator(s): {', '.join(unknown)}")
    if end_year - start_year + 1 < config.MIN_PERIOD_YEARS:
        raise InvestigationError(
            f"Choose a period of at least {config.MIN_PERIOD_YEARS} years (trend tests need enough observations).")
    if start_year < config.MIN_YEAR or end_year > config.last_full_year():
        raise InvestigationError(f"Period must lie within {config.MIN_YEAR}–{config.last_full_year()}.")


def analyze(
    power: PowerResult,
    location: Location,
    start_year: int,
    end_year: int,
    variables: list[str],
    *,
    anomaly_threshold: float = config.ANOMALY_Z,
    progress: Optional[Progress] = None,
) -> InvestigationResult:
    """Run all analyses on already-retrieved data (pure computation, deterministic)."""
    validate_settings(start_year, end_year, variables)

    def step(i: int) -> None:
        if progress:
            progress(i, len(STEPS), STEPS[i])

    variables = [v for v in config.VARIABLE_ORDER if v in variables]
    step(1)
    monthly = slice_period(power.monthly, start_year, end_year, variables)
    quality = assess_quality(monthly, start_year, end_year)
    warnings = list(power.meta.warnings) + list(quality.notes)
    if power.meta.returned_lat is not None:
        off = max(abs(power.meta.returned_lat - location.lat), abs(power.meta.returned_lon - location.lon))
        if off > 1.0:
            warnings.append(f"NASA returned a grid cell {off:.2f}° from the requested point; "
                            "check that the location is what you intended.")
    for v in variables:
        if quality.per_variable[v].valid_months == 0:
            warnings.append(f"No valid {v} data in {start_year}–{end_year}.")

    step(2)
    annual = annual_means(monthly)
    seasonal = seasonal_means(monthly, start_year, end_year)

    step(3)
    trends = {v: analyze_variable(v, annual[v], monthly[v], seasonal) for v in variables}

    step(4)
    confidence = {v: compute_confidence(quality.per_variable[v].completeness, trends[v].annual) for v in variables}

    step(5)
    anomalies = compute_monthly_anomalies(monthly)
    flagged = flag_anomalies(anomalies, anomaly_threshold)
    n_obs = int(anomalies["z"].notna().sum())

    step(6)
    fingerprint = build_fingerprint({v: trends[v].annual for v in variables}, confidence, annual)

    step(7)
    period = f"{start_year}–{end_year}"
    findings = build_findings(
        location.name, period, trends, confidence, fingerprint, flagged, n_obs, quality, power.meta,
        anomaly_threshold, expected_by_chance(n_obs, anomaly_threshold),
    )
    return InvestigationResult(
        location=location, start_year=start_year, end_year=end_year, variables=variables,
        power_meta=power.meta, monthly=monthly, annual=annual, quality=quality, trends=trends,
        confidence=confidence, anomalies=anomalies, fingerprint=fingerprint, findings=findings,
        warnings=warnings,
    )


def run_investigation(
    location: Location,
    start_year: int,
    end_year: int,
    variables: list[str],
    *,
    progress: Optional[Progress] = None,
    anomaly_threshold: float = config.ANOMALY_Z,
) -> InvestigationResult:
    """Full pipeline for one location. Raises PowerError / InvestigationError with readable messages.

    Always retrieves the full allowed range (MIN_YEAR..last full year, all three indicators) so one
    cached request serves every period/indicator choice and the candidate-pool comparison.
    """
    validate_settings(start_year, end_year, variables)
    if progress:
        progress(0, len(STEPS), STEPS[0])
    power = power_client.fetch_monthly(location.lat, location.lon, config.MIN_YEAR, config.last_full_year())
    res = analyze(power, location, start_year, end_year, variables,
                  anomaly_threshold=anomaly_threshold, progress=progress)
    if progress:
        progress(len(STEPS), len(STEPS), "Investigation complete")
    return res


def analyze_pool(
    pool_power: dict[str, PowerResult],
    start_year: int,
    end_year: int,
    variables: list[str],
    locations: list[Location],
) -> tuple[dict[str, InvestigationResult], dict[str, str]]:
    """Analyse already-retrieved pool data for the chosen period. Returns (results, errors)."""
    out: dict[str, InvestigationResult] = {}
    errors: dict[str, str] = {}
    by_name = {loc.name: loc for loc in locations}
    for name, pw in pool_power.items():
        try:
            out[name] = analyze(pw, by_name[name], start_year, end_year, variables)
        except (InvestigationError, ValueError, np.linalg.LinAlgError) as exc:
            errors[name] = str(exc)
    return out, errors
