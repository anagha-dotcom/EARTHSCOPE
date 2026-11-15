"""Central configuration for EARTHSCOPE.

Everything that is a design decision (thresholds, weights, bounds) lives here so it can be
documented, reviewed and changed in one place. No Streamlit imports in this package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

APP_NAME = "EARTHSCOPE"
TAGLINE = "Discover Changes. Connect the Signals. Understand Our Planet."

# --------------------------------------------------------------------------- NASA POWER
POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
POWER_COMMUNITY = "AG"          # agroclimatology community: T2M in degC, PRECTOTCORR in mm/day
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3                 # attempts per request (429 / 5xx / timeouts only)
BACKOFF_BASE_S = 2.0            # sleep = BACKOFF_BASE_S * 2**attempt
POLITE_DELAY_S = 0.5            # pause between *network* requests in batch fetches (pool scan)
CACHE_TTL_DAYS = 30
CACHE_DIR = Path(
    os.environ.get("EARTHSCOPE_CACHE_DIR", Path(__file__).resolve().parent.parent / "data_cache")
)
USER_AGENT = "EARTHSCOPE-student-project/1.0 (educational; uses NASA POWER public API)"

DATA_SOURCE_NAME = "NASA POWER (Prediction Of Worldwide Energy Resources), monthly point API"
DATA_PROVENANCE_NOTE = (
    "Values are gridded, model-derived estimates (NASA POWER meteorology is based on the MERRA-2 "
    "reanalysis, roughly 0.5° x 0.625° grid cells). They are not ground-station measurements, and "
    "a grid-cell value describes an area, not an exact point. Precipitation (PRECTOTCORR) is the "
    "MERRA-2 precipitation product with an observation-based bias correction."
)

# --------------------------------------------------------------------------- periods
MIN_YEAR = 2001                 # first year offered in the UI (spec: "such as 2001-2025")
MIN_PERIOD_YEARS = 10           # spec: investigate the past 10-25 years


def last_full_year(today: date | None = None) -> int:
    """Most recent calendar year that can be complete in the monthly product."""
    return (today or date.today()).year - 1


# --------------------------------------------------------------------------- statistics
ALPHA = 0.05                    # significance level for Mann-Kendall
MIN_ANNUAL_YEARS = 10           # minimum valid annual values to run a trend test
ANOMALY_Z = 2.0                 # |z| threshold for "unusual" month
MIN_MONTH_OBS = 8               # min observations per calendar month to build a climatology

# Confidence (evidence) score: weights sum to 100. See docs/METHODOLOGY.md
CONF_W_COMPLETENESS = 20.0
CONF_W_SIGNIFICANCE = 50.0
CONF_W_STRENGTH = 30.0
CONF_P_FULL = 0.01              # p <= this -> full significance points
CONF_P_ZERO = 0.10              # p >= this -> zero significance points
CONF_TAU_FULL = 0.5             # |Kendall tau| >= this -> full strength points
CONF_BAND_MODERATE = 40.0
CONF_BAND_STRONG = 70.0

# Similar regions
SIMILAR_TOP_K = 5
SIMILAR_MIN_POOL = 5            # minimum successfully-retrieved pool locations


# --------------------------------------------------------------------------- variables
@dataclass(frozen=True)
class VariableInfo:
    code: str                   # NASA POWER parameter name
    name: str                   # short display name
    long_name: str
    unit: str
    rate_unit: str              # unit used when quoting a *rate of change* (RH: percentage points)
    api_unit: str               # unit string NASA is expected to return (for validation)
    kind: str                   # temperature | precipitation | humidity
    color_light: str
    color_dark: str
    bounds: tuple[float, float]  # physically plausible range; outside -> treated as missing
    up_word: str                # trend wording
    down_word: str
    high_label: str             # anomaly wording
    low_label: str
    min_abs_dev: float          # extra guard: anomaly must deviate at least this much (unit)
    description: str


VARIABLES: dict[str, VariableInfo] = {
    "T2M": VariableInfo(
        code="T2M", name="Temperature", long_name="Temperature at 2 m", unit="°C", rate_unit="°C", api_unit="C",
        kind="temperature", color_light="#eb6834", color_dark="#d95926", bounds=(-90.0, 60.0),
        up_word="warming", down_word="cooling",
        high_label="unusually warm", low_label="unusually cool", min_abs_dev=0.0,
        description="Mean air temperature at 2 m above the surface (monthly mean).",
    ),
    "PRECTOTCORR": VariableInfo(
        code="PRECTOTCORR", name="Precipitation", long_name="Precipitation (corrected)",
        unit="mm/day", rate_unit="mm/day", api_unit="mm/day", kind="precipitation",
        color_light="#2a78d6", color_dark="#3987e5", bounds=(-0.01, 2000.0),
        up_word="increasing", down_word="decreasing",
        high_label="unusually wet", low_label="unusually dry", min_abs_dev=0.2,
        description="Mean daily precipitation rate for the month (mm/day, NOT a monthly total).",
    ),
    "RH2M": VariableInfo(
        code="RH2M", name="Relative humidity", long_name="Relative humidity at 2 m", unit="%",
        rate_unit="percentage points", api_unit="%", kind="humidity", color_light="#1baf7a", color_dark="#199e70",
        bounds=(0.0, 100.0), up_word="increasing", down_word="decreasing",
        high_label="unusually humid", low_label="unusually low humidity", min_abs_dev=0.0,
        description="Relative humidity of the air at 2 m (monthly mean).",
    ),
}
VARIABLE_ORDER = list(VARIABLES)

# Season definitions (month tuples). Labelled by month range so they are hemisphere-neutral.
SEASONS: dict[str, tuple[int, ...]] = {
    "Dec–Feb": (12, 1, 2),
    "Mar–May": (3, 4, 5),
    "Jun–Aug": (6, 7, 8),
    "Sep–Nov": (9, 10, 11),
}
