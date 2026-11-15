# EARTHSCOPE — Scientific methodology

This document describes exactly what the application computes, so every number on screen can be reproduced.
Values in **bold monospace** live in `earthscope/config.py`. Items marked **(design choice)** are decisions made
for transparency, not established scientific constants.

## 1. Data

| Item | Value |
|---|---|
| Source | NASA POWER (Prediction Of Worldwide Energy Resources), monthly point API |
| Endpoint | `https://power.larc.nasa.gov/api/temporal/monthly/point` |
| Community | **`AG`** (agroclimatology) |
| Indicators | `T2M` temperature at 2 m (°C); `PRECTOTCORR` precipitation, corrected (**mm/day**, a mean daily rate for the month, *not* a monthly total); `RH2M` relative humidity at 2 m (%) |
| Period offered | **2001** to the last full calendar year (2025 at the time of writing); minimum analysis window **10 years** |
| Resolution | Gridded, MERRA-2 based (about 0.5° × 0.625° cells). A value describes an area, not an exact point |

NASA POWER meteorology is **model-derived** (MERRA-2 reanalysis; precipitation includes an observation-based bias
correction). It is not a ground-station record, and the application states this on every page.

**Retrieval and cleaning**

- One request per location always retrieves the full allowed range and all three indicators, so one cached
  response serves every later choice of period or indicator. Responses are cached on disk for **30 days**.
- The monthly response also contains an annual value (key `YYYY13`). It is kept only as a cross-check and is
  **never** mixed into the monthly series.
- The fill value (`-999`) and values outside a physically plausible range (`bounds` per indicator) are treated as
  missing. Missing values are **never interpolated or filled**.
- The parser is strict: if the response structure is not what is expected it raises a descriptive error rather than
  guessing. Units reported by NASA are compared with the expected units and mismatches are shown as warnings.

> **Verification status.** The response format is implemented from NASA's documentation. It must be confirmed
> against the live service with `scripts/smoke_test_nasa.py`; see the README.

## 2. Annual and seasonal series

- **Annual mean** = day-weighted mean of the 12 monthly values, computed **only for years with all 12 months
  valid**. A year with any missing month is excluded (it would otherwise be biased toward the seasons present).
  Day-weighting matters mainly for precipitation, which is a mean daily rate per month.
- **Seasons** are three-month blocks labelled by month range (hemisphere-neutral): Dec–Feb, Mar–May, Jun–Aug,
  Sep–Nov. A season-year needs all three months valid; Dec–Feb is assigned to the year of its January/February.

## 3. Trend detection

**Primary test — annual means.** One value per complete year keeps the Mann-Kendall test close to its
independence assumption.

| Quantity | Method |
|---|---|
| Direction and significance | Mann-Kendall test (`pymannkendall.original_test`), two-sided, **α = 0.05** |
| Rate of change | Theil–Sen slope (`scipy.stats.theilslopes`) with its 95 % confidence interval, in unit per year; shown per decade (× 10) |
| Cross-check | Ordinary least-squares slope, p-value and R² (displayed, not used for decisions) |
| Serial correlation | Lag-1 autocorrelation of the residuals; flagged when it exceeds 1.96/√n. A flag means the Mann-Kendall p-value may be too optimistic |
| Minimum data | **10** valid annual values, otherwise the trend is reported as *insufficient data* and no test is run |

Direction is *increasing* or *decreasing* only if the test is significant; otherwise *no significant trend*.
Significance and magnitude are reported separately (a small trend can be significant and a large one can fail).
For precipitation and humidity the slope is also quoted as a percentage of the period mean per decade.

**Supporting evidence — seasonal analysis.**
1. *Seasonal Mann-Kendall* on monthly values from complete years (`pymannkendall.correlated_seasonal_test`,
   Hirsch & Slack, which corrects for correlation between months). Its slope is the Hirsch seasonal Kendall
   estimator: the median of all within-month pairwise slopes using real year differences.
2. *Mann-Kendall on each three-month season's yearly means.* Four seasons are tested separately, so some
   significant seasons can occur by chance; the interface says so.

## 4. Evidence score ("confidence") — **(design choice)**

The score summarises how strong and complete the *available evidence for a trend* is. It is **not** a probability
that the trend is real and **not** a forecast.

```
score = 20 × completeness
      + 50 × clip((0.10 − p) / (0.10 − 0.01), 0, 1)      # Mann-Kendall p on annual means
      + 30 × clip(|Kendall tau| / 0.5, 0, 1)
```

- completeness = valid months ÷ expected months in the selected period (0–1).
- Weights **20 / 50 / 30**, p-limits **0.01 / 0.10** and τ-limit **0.5** are design choices (`CONF_*` in config).
- Bands: **Weak** < 40, **Moderate** 40–69, **Strong** ≥ 70. A trend that is not significant at α is always
  labelled **Weak**, whatever the numeric score.
- A low score for "no trend" means little evidence *of change*; it is not evidence that nothing changed.
- The score is reproducible: the interface and the CSV report list the three component points.

## 5. Environmental fingerprint

Each indicator is reduced to a direction code from the annual Mann-Kendall result: `+` significant increase,
`-` significant decrease, `0` no significant trend, `NA` not assessed. A fixed rule table maps the combination of
(temperature, precipitation, humidity) to a named pattern, for example `+ − −` → *Warming–drying pattern*; twelve
patterns are defined. Significant combinations that match no rule are reported as *Mixed change*, and when no
indicator is significant as *No statistically significant combined change*. With fewer than three assessable
indicators a *Partial fingerprint* is reported and no named pattern is assigned.

- Fingerprint strength = mean evidence score of the indicators that have a significant trend.
- The common-scale bar chart divides each Sen's slope per decade by the indicator's interannual standard
  deviation so °C, mm/day and % can be compared.
- The heat-map shows Spearman rank correlations between annual series. They describe whether indicators move
  together from year to year, not whether one drives another.
- **A fingerprint describes trends that occur together. It never asserts causation.**

## 6. Regional comparison

**Opposite Trend Finder.** For each indicator, locations with a significant increase are listed against those with a
significant decrease. "Opposite" requires at least one of each. Results come entirely from the retrieved data.

**Similar Earth Regions Discovery** (plain mathematics, no AI):
1. Each location is described by its Sen's slopes per decade for the selected indicators.
2. Each indicator's slopes are standardised across a **fixed pool of 30 locations** (z-score of the slope), so
   °C, mm/day and % become comparable.
3. Distance = root-mean-square difference between two locations' standardised slope vectors (Euclidean distance
   ÷ √ number of shared indicators); smaller = more similar change pattern.
4. The closest pool locations are ranked. Slopes are used whether or not individually significant; significance
   flags are shown alongside.

The pool is fixed (see `earthscope/pool.py`) so results are reproducible. It was chosen for climatic and geographic
diversity, not statistically; results depend on it. Similar *change patterns* do not mean similar *climates*.
At least 5 pool locations must be retrievable and at least 2 indicators assessable, otherwise no ranking is shown.

## 7. Anomalies — kept separate from trends

For each calendar month, the mean and standard deviation over the selected period are computed
(**≥ 8** observations needed). A month's z-score is `(value − mean) / sd`. A month is flagged as unusual when
**|z| ≥ 2** (adjustable in the interface from 1.5 to 3.0) **and** its absolute departure is at least a minimum
(**0.2 mm/day** for precipitation; this stops near-zero rainfall months being flagged for trivial differences).

Because anomalies are measured against a fixed period climatology and not against the trend line, a strong
trend makes flagged months cluster at one end of the period. This is shown to the user. The interface also states
how many unusual months would be expected by chance if values were normally distributed (about 4.6 % of months at
|z| ≥ 2).

## 8. Explanations

Every sentence in the findings is produced from fixed templates filled with computed numbers
(`earthscope/narratives.py`). No generative or paid AI service is used, and the application has no network
dependency other than NASA POWER (plus the browser fetching map tiles and map outlines).

## 9. Limitations

- Model-derived grid-cell data, not station records; local conditions can differ from the cell average.
- At most 25 annual values per series: small real trends may go undetected; testing several indicators and seasons
  means some significant results can occur by chance (no multiple-testing correction is applied).
- Mann-Kendall assumes roughly independent yearly values; serial correlation is flagged but not corrected in the
  primary test.
- The evidence-score weights and the fingerprint rule table are design choices.
- A trend or fingerprint does not identify a cause, and a significant trend can have non-climatic origins
  (for example changes in the reanalysis inputs over time).
- Not part of this version: vegetation indices, soil moisture, independent cross-dataset verification, recovery
  tracking, predictive early-warning models, global hidden-pattern discovery.

## 10. References (check bibliographic details before citing)

- Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica* 13, 245–259.
- Kendall, M. G. (1975). *Rank Correlation Methods*. Griffin, London.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *J. Am. Stat. Assoc.* 63, 1379–1389.
- Hirsch, R. M., Slack, J. R., Smith, R. A. (1982). Techniques of trend analysis for monthly water quality data.
  *Water Resources Research* 18, 107–121.
- Hirsch, R. M., Slack, J. R. (1984). A nonparametric trend test for seasonal data with serial dependence.
  *Water Resources Research* 20, 727–732.
- Gelaro, R., et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2
  (MERRA-2). *Journal of Climate* 30, 5419–5454.
- Hussain, M., Mahmud, I. (2019). pyMannKendall: a Python package for non-parametric Mann-Kendall family of trend
  tests. *Journal of Open Source Software* 4(39), 1556.
- NASA POWER Project documentation: https://power.larc.nasa.gov/docs/
