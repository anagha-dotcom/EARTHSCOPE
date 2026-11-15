"""Fixed candidate pool for Similar Earth Regions Discovery.

30 land locations chosen for climatic/geographic diversity (all continents except Antarctica,
from desert to rainforest to subarctic). The pool is deliberately fixed so results are reproducible.
Data are fetched live from NASA POWER and cached on disk; nothing here is hard-coded data.
Coordinates are approximate city/site centres (a ~50 km grid cell is used by the data anyway).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from . import config, power_client
from .models import Location, PowerResult

CANDIDATE_POOL: list[Location] = [
    Location("Phoenix, USA", 33.4484, -112.0740, "hot desert"),
    Location("Seattle, USA", 47.6062, -122.3321, "maritime temperate"),
    Location("Anchorage, USA", 61.2181, -149.9003, "subarctic"),
    Location("Mexico City, Mexico", 19.4326, -99.1332, "tropical highland"),
    Location("Manaus, Brazil", -3.1190, -60.0217, "equatorial rainforest"),
    Location("São Paulo, Brazil", -23.5505, -46.6333, "humid subtropical"),
    Location("Lima, Peru", -12.0464, -77.0428, "coastal desert"),
    Location("Quito, Ecuador", -0.1807, -78.4678, "equatorial highland"),
    Location("Buenos Aires, Argentina", -34.6037, -58.3816, "humid temperate"),
    Location("London, UK", 51.5074, -0.1278, "oceanic temperate"),
    Location("Madrid, Spain", 40.4168, -3.7038, "Mediterranean continental"),
    Location("Moscow, Russia", 55.7558, 37.6173, "humid continental"),
    Location("Cairo, Egypt", 30.0444, 31.2357, "hot desert"),
    Location("Niamey, Niger", 13.5116, 2.1254, "Sahel"),
    Location("Lagos, Nigeria", 6.5244, 3.3792, "tropical monsoon"),
    Location("Nairobi, Kenya", -1.2921, 36.8219, "tropical highland"),
    Location("Johannesburg, South Africa", -26.2041, 28.0473, "subtropical highland"),
    Location("Cape Town, South Africa", -33.9249, 18.4241, "Mediterranean"),
    Location("Riyadh, Saudi Arabia", 24.7136, 46.6753, "hot desert"),
    Location("Delhi, India", 28.6139, 77.2090, "semi-arid / monsoon"),
    Location("Chennai, India", 13.0827, 80.2707, "tropical coastal"),
    Location("Guwahati, India", 26.1445, 91.7362, "humid subtropical"),
    Location("Leh, India", 34.1526, 77.5771, "cold high-altitude desert"),
    Location("Beijing, China", 39.9042, 116.4074, "continental monsoon"),
    Location("Ulaanbaatar, Mongolia", 47.8864, 106.9057, "cold steppe"),
    Location("Bangkok, Thailand", 13.7563, 100.5018, "tropical savanna"),
    Location("Jakarta, Indonesia", -6.2088, 106.8456, "equatorial"),
    Location("Tokyo, Japan", 35.6762, 139.6503, "humid subtropical"),
    Location("Alice Springs, Australia", -23.6980, 133.8807, "hot desert"),
    Location("Sydney, Australia", -33.8688, 151.2093, "humid subtropical"),
]

DEMO_LOCATIONS: list[Location] = [
    Location("Palakkad, Kerala", 10.7867, 76.6548, "demo: humid tropical Western Ghats gap"),
    Location("Jaipur, Rajasthan", 26.9124, 75.7873, "demo: semi-arid"),
]


def is_same_place(a: Location, b: Location, tol_deg: float = 0.25) -> bool:
    return abs(a.lat - b.lat) < tol_deg and abs(a.lon - b.lon) < tol_deg


def fetch_pool(
    start_year: int,
    end_year: int,
    variables: Optional[list[str]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, PowerResult], dict[str, str]]:
    """Fetch every pool location sequentially. Returns (results by name, error message by name).

    A failure for one location never aborts the scan. Cached locations are not re-requested and
    do not incur the polite delay.
    """
    variables = list(variables or config.VARIABLE_ORDER)
    results: dict[str, PowerResult] = {}
    errors: dict[str, str] = {}
    total = len(CANDIDATE_POOL)
    for i, loc in enumerate(CANDIDATE_POOL):
        if progress:
            progress(i, total, f"Retrieving {loc.name} ({i + 1}/{total})")
        cached = power_client.is_cached(loc.lat, loc.lon, start_year, end_year, variables)
        try:
            results[loc.name] = power_client.fetch_monthly(loc.lat, loc.lon, start_year, end_year, variables)
        except power_client.PowerError as exc:
            errors[loc.name] = str(exc)
        if not cached and i < total - 1:
            sleep(config.POLITE_DELAY_S)
    if progress:
        progress(total, total, "Pool retrieval finished")
    return results, errors
