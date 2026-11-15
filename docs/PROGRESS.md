# EARTHSCOPE — progress record

Last updated: 2026-11-15 (build chat, after UI completion and compatibility testing)

## Where things stand

| Measure | Value |
|---|---|
| Features implemented (spec §2) | 9 of 9, each with UI and tests |
| Pages (spec §6) | 6 of 6 |
| Automated tests | **77 passed, 0 failed** on both the oldest supported libraries (streamlit 1.50.0, pandas 2.2.3, numpy 2.1.3, scipy 1.14.1, plotly 5.24.1) and the latest (streamlit 1.67.0, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, plotly 7.1.0); Python 3.13 |
| Live NASA POWER verification | **Not done** (build sandbox blocked: HTTP 403 from the egress proxy). Script ready |
| Estimated completion | Code, tests and documentation about **90%**; submission-ready about **75%** (remaining items are mostly yours: live check, demo capture, GitHub, slides) |

## Checklist

- [x] Stage 0: structure, interface contract, requirements
- [x] Stage 1: `power_client` (strict parser, retry/backoff, cache, validation), `preprocess`
- [x] Stage 2: trends, significance, evidence score
- [x] Stage 3: fingerprint, narrative templates
- [x] Stage 4: anomalies, opposite trends, similar regions, fixed 30-location pool
- [x] Stage 5: orchestrator, CSV report
- [x] UI: `app.py`, `ui/` helpers, **six pages in `views/`**
- [x] UI tests with Streamlit AppTest (synthetic data, clearly labelled)
- [x] `scripts/smoke_test_nasa.py` (logic tested against a local mock server)
- [x] `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.streamlit/config.toml`
- [x] README, `docs/METHODOLOGY.md`
- [x] Compatibility check at the declared minimum versions
- [ ] **Run the smoke test against live NASA POWER and fix anything it reports**
- [ ] Visual check of the world maps / map tiles with internet access
- [ ] Demo capture (screenshots or ≤ 30 s video), slides (≤ 7), problem statement
- [ ] Public GitHub repository, licence, optional free deployment

## Feature evidence (spec §2)

| Feature | Code | Evidence |
|---|---|---|
| 2.1 Change detection | `trends.py`, `preprocess.py` | `test_stats.py` (known trends recovered, day-weighted annual means, gaps, insufficient data, agreement with pymannkendall) |
| 2.2 Fingerprint | `fingerprint.py` | `test_pipeline.py` (every direction combination labelled, spec example, mixed/partial), UI render |
| 2.3 Opposite Trend Finder | `comparison.py` | `test_opposite_trend_finder`, UI |
| 2.4 Similar Earth Regions | `comparison.py`, `pool.py` | nearest-first ranking, self-exclusion, small-pool handling, pool continues after failures; UI flow with a failed pool location |
| 2.5 Anomalies | `anomalies.py` | injected anomaly flagged, precipitation minimum-departure guard, too-few-observations |
| 2.6 Evidence score | `confidence.py` | component/band tests, not-significant capped Weak, not assessable |
| 2.7 Automated investigation | `investigation.py` | `test_run_investigation_*`, UI: Start Investigation, partial and total failure |
| 2.8 Explainable findings | `narratives.py` | exercised via pipeline, report and UI tests (templates only) |
| 2.9 Timeline | `charts.py`, `views/anomalies.py` | UI render |

Live-data behaviour of every row above is **unverified** until the smoke test is run.

## Decisions (recorded)

- Similar regions: fixed pool of **30** locations, standardised Sen's slopes, RMS distance, no AI.
- Trends: Mann-Kendall + Sen's slope on annual means of complete years; seasonal Mann-Kendall and per-season tests as supporting evidence.
- Evidence-score weights (20 / 50 / 30), pool and fingerprint rule table are **design choices**, documented as such.
- Demo locations: Palakkad (10.7867, 76.6548), Jaipur (26.9124, 75.7873).
- Widgets use persistent copies as defaults (needed for correct behaviour on Streamlit 1.50; found by the compatibility run).

## Issues found and fixed in this stage

- Streamlit 1.50.0: range slider ignored changes (default given only on first run). Fixed by always passing the persistent copy; verified on 1.50.0 and 1.67.0.
- Custom-location callback caused a Streamlit warning on the latest version; fixed by rebuilding the widget from the persistent copy.
- Mixed-type "Rank" column forced a pyarrow fallback; fixed.
- Clipped labels on the fingerprint bar chart, truncated metric, huge opposite-pairs table; fixed.
- `pool.py` docstring said 27 locations (there are 30); fixed.

## Known limitations of this build

- Live NASA behaviour unverified (see above).
- Map panels need internet in the browser (map tiles; Plotly world outlines load from a CDN).
- No multiple-testing correction; evidence score is not a probability (documented in the app).
- Not tested on Python < 3.13 or on Windows/macOS.

## Exact next steps for the team

1. Unzip, create a virtual environment, `pip install -r requirements-dev.txt`, run `python -m pytest tests` (expect 77 passed).
2. Run `python scripts/smoke_test_nasa.py`; send `smoke_test_report.txt` back if any line is WARN or FAIL.
3. Run `streamlit run app.py`; do the Kerala vs Rajasthan demo; check the maps render; run *Retrieve candidate pool* once.
4. Capture screenshots or a ≤ 30 s screen recording for the submission.
5. Write the problem statement and ≤ 7 slides from README + METHODOLOGY (clearly separate implemented features from future scope).
6. Create the public GitHub repo (add a licence), push, optionally deploy on Streamlit Community Cloud.
