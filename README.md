# EARTHSCOPE

**Discover Changes. Connect the Signals. Understand Our Planet.**

EARTHSCOPE is an environmental change detection and investigation platform. Pick a location, choose environmental
indicators and a period (2001 to the last full year), and it retrieves **NASA POWER** monthly data, tests for trends,
looks at the indicators *together* (the **Environmental Fingerprint**), compares places (**Opposite Trend Finder** and
**Similar Earth Regions Discovery**), flags unusual months, scores the strength of the evidence and writes the findings
in plain language, using fixed templates, with no AI service involved.

`Detect → Analyse → Compare → Connect → Explain`

Python · Streamlit · Pandas · NumPy · SciPy · PyMannKendall · Plotly · NASA POWER API. Everything is free and open
source; there are no API keys, paid services or databases.

---

## Honest status

| Area | Status |
|---|---|
| All 9 specified features implemented (6 pages) | ✅ Done |
| Automated tests | ✅ **77 passing** (Streamlit 1.50.0 / pandas 2.2.3 / numpy 2.1.3 / scipy 1.14.1 / plotly 5.24.1 **and** the latest versions, Python 3.13) |
| Pages run end to end in a real headless browser | ✅ With **synthetic** data only |
| **Live NASA POWER response format, grid cell, units, completeness through 2025** | ⚠️ **Not yet verified.** The build sandbox could not reach NASA. **Run `scripts/smoke_test_nasa.py` once** (below) |
| World maps and map tiles | ⚠️ Not visually verified (blocked in the build sandbox); they need internet access in your browser |
| Findings shown in the app | They come only from retrieved data. Nothing is hard-coded, and synthetic data is used **only** inside `tests/` |

The tests that use synthetic data check that our code is correct. They say nothing about what NASA's data will show.

---

## Quick start

Requires Python 3.10+ (developed and tested on 3.13) and an internet connection.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate

# 2. install
pip install -r requirements-dev.txt          # (requirements.txt if you do not need the tests)

# 3. run the tests (no internet needed)
python -m pytest tests

# 4. verify the live NASA service (needs internet), see next section
python scripts/smoke_test_nasa.py

# 5. start the app
streamlit run app.py
```

### Step 4: the NASA smoke test (do this first)

```bash
python scripts/smoke_test_nasa.py             # 2 demo locations + 3 pool locations (7 requests)
python scripts/smoke_test_nasa.py --full-pool # all 30 pool locations; also shows throttling behaviour
```

It checks the live API independently of our parser (HTTP status, JSON structure, `-999` fill value, grid cell,
units, whether 2025 is complete), then runs the whole analysis on the real response and prints the **first real
trend results** for Palakkad and Jaipur. It saves `smoke_test_report.txt`. Exit code `0` means no `FAIL`.
It also warms the on-disk cache, so the app starts instantly afterwards.

If anything is marked `WARN` or `FAIL`, send `smoke_test_report.txt` to whoever maintains the code. The parser is
strict on purpose and will report exactly what differs from expectations.

---

## Demo (Kerala vs Rajasthan)

1. **Earth Overview**: *Palakkad, Kerala* and *Jaipur, Rajasthan* are preselected; keep all three indicators and
   2001–2025; click **Start Investigation**.
2. **Environmental Investigation**: time series, Sen's slope per decade, Mann-Kendall significance, evidence score.
3. **Environmental Fingerprint**: indicators analysed together, named pattern, correlation heat-map, caveats.
4. **Regional Comparison**: trend comparison, Opposite Trend Finder, and **Similar Earth Regions** (click *Retrieve
   candidate pool* once; it makes up to 30 NASA requests, about a few minutes the first time, then cached).
5. **Anomalies and Timeline**: unusual months against each calendar month's historical average.
6. **Investigation Report**: evidence cards, statistics, data provenance, **CSV downloads**.

Any location on Earth can be investigated: add a custom latitude/longitude on the overview page.

## Features ↔ pages

| Specification feature | Where |
|---|---|
| Environmental change detection (trend, rate, significance, time series) | Page 2 |
| Environmental fingerprinting (signature feature) | Page 3 |
| Opposite Trend Finder | Page 4, tab 2 |
| Similar Earth Regions Discovery (fixed pool of 30 locations, plain maths) | Page 4, tab 3 |
| Anomaly detection (kept separate from trends) | Page 5 |
| Evidence ("confidence") score, method explained in the UI | Pages 2 and 6 |
| Automated investigation (retrieve → clean → analyse → report) | **Start Investigation** (Page 1) |
| Explainable findings (template-based) | Pages 2, 3, 6 |
| Environmental change timeline | Page 5 |

## Project structure

```
app.py                  Streamlit entry point: navigation + shared sidebar
views/                  The six pages (overview, investigation, fingerprint, comparison, anomalies, report)
ui/                     Streamlit-side helpers: state.py (session state), charts.py (Plotly), components.py
earthscope/             Analysis package, no Streamlit imports
  config.py             All constants: indicators, thresholds, evidence-score weights, API settings
  power_client.py       NASA POWER client: strict parser, retry/backoff, 30-day disk cache, validation
  preprocess.py         Annual / seasonal series (complete years only, day-weighted)
  trends.py             Mann-Kendall, Sen's slope, seasonal Mann-Kendall, serial-correlation flag
  confidence.py         Transparent evidence score
  fingerprint.py        Multi-indicator fingerprint rules + correlations
  anomalies.py          Per-calendar-month z-scores
  comparison.py         Trend table, opposite trends, similar regions
  pool.py               Demo locations and the fixed 30-location candidate pool
  narratives.py         Template-written findings
  investigation.py      Orchestrator: run_investigation(), analyze(), analyze_pool()
  report.py             CSV report builders
scripts/smoke_test_nasa.py   Live API verification (run on your machine)
tests/                  77 tests; synthetic data is confined to tests/synthetic.py
docs/METHODOLOGY.md     Exactly what is computed and why
docs/PROGRESS.md        Progress record and next steps
```

## Data and provenance

- **Source:** NASA POWER monthly point API, community `AG`. Variables: `T2M` (°C), `PRECTOTCORR` (mm/day, a mean
  daily rate for the month, not a monthly total), `RH2M` (%).
- NASA POWER meteorology is **model-derived** (MERRA-2 based gridded estimates), not ground-station measurements.
  The app shows this, with the requested vs. returned grid cell, period, units, parameter names and request URL.
- Missing values are never filled in; years with any missing month are excluded from annual means.
- Acknowledgement: *These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded
  through the NASA Earth Science/Applied Science Program.* Check <https://power.larc.nasa.gov/docs/> for the current
  citation guidance before submission.

Details of every statistic, threshold and design choice are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Deployment (free)

The app is a standard Streamlit app (`app.py` + `requirements.txt`) and can run on **Streamlit Community Cloud**
(free tier) from a public GitHub repository, or locally. Not deployed or tested from the build environment; follow
Streamlit's current deployment docs. Notes: the disk cache (`data_cache/`) may not persist on hosted platforms, so
the first pool scan after a restart makes live requests again; free tiers have no guaranteed capacity.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Smoke test or app: *HTTP 403 / connection error* | Network, VPN, firewall or proxy blocks `power.larc.nasa.gov`; try another network |
| *HTTP 429 / throttled* | Increase `POLITE_DELAY_S` in `earthscope/config.py`; cached results are reused |
| Maps are blank | The browser needs internet for map tiles and Plotly's world outlines |
| Latest year missing from a series | NASA's final months may be incomplete; incomplete years are excluded (smoke test reports it) |
| Start button disabled | Needs ≥ 1 location, ≥ 1 indicator and a period of ≥ 10 years |
| Want fresh data | Delete the `data_cache/` folder |

## Limitations and future scope

Read the full list in `docs/METHODOLOGY.md` §9. In short: grid-cell model data, at most 25 annual values per
series, no multiple-testing correction, evidence-score weights are design choices, and trends or fingerprints never
establish causation.

**Not implemented in this version** (possible future extensions, not claimed): vegetation indices / global
vegetation-decline mapping, soil-moisture investigations, independent cross-dataset verification, environmental
recovery tracking, predictive early-warning models, complex global hidden-pattern discovery.

## Licence

No licence has been chosen yet. Add one (for example MIT) before publishing the repository.
