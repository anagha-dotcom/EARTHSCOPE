"""Plain dataclasses shared between the analysis package and the Streamlit UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    note: str = ""


@dataclass
class PowerMeta:
    """Provenance of one NASA POWER retrieval."""
    requested_lat: float
    requested_lon: float
    start_year: int
    end_year: int
    community: str
    parameters_requested: list[str]
    returned_lat: Optional[float] = None
    returned_lon: Optional[float] = None
    elevation_m: Optional[float] = None
    first_valid: Optional[str] = None            # 'YYYY-MM' of first non-missing value (any variable)
    last_valid: Optional[str] = None
    fill_value: float = -999.0
    api_version: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    parameters_info: dict[str, dict] = field(default_factory=dict)   # code -> {units, longname}
    request_url: Optional[str] = None
    retrieved_at: str = ""
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class PowerResult:
    monthly: pd.DataFrame            # index: month start (DatetimeIndex); columns: parameter codes
    annual_reported: pd.DataFrame    # NASA-reported annual values (index: year) - cross-check only
    meta: PowerMeta


@dataclass
class VariableQuality:
    variable: str
    expected_months: int
    valid_months: int
    completeness: float              # valid / expected, 0-1
    years_requested: int
    years_complete: int              # years with all 12 months valid (usable for annual means)
    incomplete_years: list[int]


@dataclass
class DataQuality:
    per_variable: dict[str, VariableQuality]
    overall_completeness: float
    notes: list[str] = field(default_factory=list)


@dataclass
class TrendResult:
    variable: str
    label: str
    unit: str
    method: str                      # 'annual' | 'seasonal_mk' | 'season:<label>'
    n: int                           # valid annual values (or complete years for seasonal_mk)
    alpha: float
    status: str = "ok"               # 'ok' | 'insufficient'
    reason: str = ""
    first_year: Optional[int] = None
    last_year: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    mk_p: Optional[float] = None
    mk_z: Optional[float] = None
    mk_tau: Optional[float] = None
    mk_s: Optional[float] = None
    sens_slope: Optional[float] = None       # unit per year
    sens_low: Optional[float] = None         # 95 % CI of Sen's slope (annual method only)
    sens_high: Optional[float] = None
    sens_intercept: Optional[float] = None   # line: y = intercept + slope * year
    ols_slope: Optional[float] = None
    ols_p: Optional[float] = None
    ols_r2: Optional[float] = None
    direction: str = "insufficient data"     # increasing | decreasing | no significant trend | insufficient data
    significant: bool = False
    lag1_autocorr: Optional[float] = None
    serial_corr_flag: bool = False
    pct_per_decade: Optional[float] = None   # % of mean per decade (not for temperature)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def slope_per_decade(self) -> Optional[float]:
        return None if self.sens_slope is None else self.sens_slope * 10.0

    @property
    def total_change(self) -> Optional[float]:
        if self.sens_slope is None or self.first_year is None or self.last_year is None:
            return None
        return self.sens_slope * (self.last_year - self.first_year)


@dataclass
class VariableTrends:
    variable: str
    annual: TrendResult
    seasonal_mk: Optional[TrendResult]
    by_season: dict[str, TrendResult]


@dataclass
class ConfidenceResult:
    variable: str
    score: Optional[float]           # 0-100, None if not assessable
    label: str                       # Weak | Moderate | Strong | Not assessable
    components: list[dict]           # [{name, raw, raw_text, scaled, weight, points}]
    note: str = ""


@dataclass
class FingerprintResult:
    pattern_id: str
    label: str
    description: str
    directions: dict[str, str]       # variable -> '+', '-', '0', 'NA'
    std_change_per_decade: dict[str, Optional[float]]   # slope*10 / interannual std
    strength: Optional[float]        # mean evidence score of variables with a significant trend
    correlations: pd.DataFrame       # Spearman rho between annual series (may be empty)
    correlation_p: pd.DataFrame
    caveats: list[str]


@dataclass
class Finding:
    kind: str                        # trend | seasonal | fingerprint | anomaly | quality | confidence
    title: str
    text: str
    evidence: dict = field(default_factory=dict)
    variable: Optional[str] = None


@dataclass
class InvestigationResult:
    location: Location
    start_year: int
    end_year: int
    variables: list[str]
    power_meta: PowerMeta
    monthly: pd.DataFrame            # cleaned monthly data for the analysis period
    annual: pd.DataFrame
    quality: DataQuality
    trends: dict[str, VariableTrends]
    confidence: dict[str, ConfidenceResult]
    anomalies: pd.DataFrame          # all monthly z-scores (long format); flag with anomalies.flag_anomalies
    fingerprint: FingerprintResult
    findings: list[Finding]
    warnings: list[str]
