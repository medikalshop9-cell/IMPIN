# IMPIN — Project Plan

**Students:** Acheampong Yaw HINNEH · Abiola OKUNSANYA  
**Program:** MIM — Africa Business School, UM6P · 2025–2027  
**Last Updated:** May 2026

---

## Academic Claim

> *"IMPIN is a real-time food price intelligence system for Ghana with three layers:*
> *(1) a live scrape-based price index updated weekly;*
> *(2) a nowcast of the current food CPI component before official GSS release;*
> *(3) a 1–3 month forecast comparing classical time-series and ML models."*

**What we are NOT claiming:** That our index perfectly tracks GSS CPI (we have no multi-week scrape series yet).
**What we ARE claiming:** That the *method* works — we prove it on 4 years of WFP historical data, and show the live scrape extends it to the present.

---

---

## System Architecture

```
─────────────────────────────────────────────────────────────────────────
LAYER 1 · LIVE SCRAPE INDEX  (what is food inflation right now?)
─────────────────────────────────────────────────────────────────────────
  Online Ghana shops (Konzoom, Makola, Big Samps, Shopnaw, KiKUU)
              ↓  scrapers/run_all.py
  1,586 products × prices  →  pipeline/build_index.py
              ↓
  IMPIN Snapshot  (May 2026, W19, base Jan-2026 = 100)
              ↓
  Append as latest data point to historical series  ──────────────────┐
                                                                       │
─────────────────────────────────────────────────────────────────────── │
LAYER 2 · NOWCAST  (before GSS releases the number)                    │
─────────────────────────────────────────────────────────────────────── │
  WFP monthly prices 2019–2023  (48 months, 14 commodities)            │
  GHS/USD implied rate          (WFP-derived)                          │
  Brent crude                   (FRED)                                 │
              ↓  pipeline/build_historical.py                          │
  Historical panel  →  ARIMAX(1,1,0) ✓  →  Point estimate: May 2026 ←─┘
                    →  XGBoost (ML) 🔜   →  (nowcast validation vs scrape)
                    →  Random Forest 🔜
                    →  Prophet 🔜

─────────────────────────────────────────────────────────────────────────
LAYER 3 · LIVE ANOMALY DETECTION  (which products are outliers?)
─────────────────────────────────────────────────────────────────────────
  Scraped products  →  Isolation Forest + Z-score
              ↓
  Flagged anomalies (z > 2.5) per CPI category
              ↓
  "These N items may be driving food CPI up this month"

─────────────────────────────────────────────────────────────────────────
DASHBOARD  →  Streamlit app showing all 3 layers
─────────────────────────────────────────────────────────────────────────
```

---

## Objectives

### Primary
- [x] Build an automated scraping pipeline collecting prices from Ghanaian e-commerce platforms
- [x] Compute IMPIN snapshot index (W19, May 2026) — `pipeline/build_index.py`
- [x] Assemble 48-month historical panel (WFP + GHS/USD + Brent) — `pipeline/build_historical.py`
- [x] Full EDA on historical data — `analysis/eda.py` — 15 plots + PDF report
- [x] Stationarity tests (ADF + KPSS) — all series confirmed I(1)
- [x] Granger causality + VAR — `analysis/granger.py`
- [x] ARIMAX(1,1,0) — best classical model, 83.3% directional accuracy
- [ ] **XGBoost + Random Forest** — ML forecasters for model comparison
- [ ] **Prophet** — Bayesian baseline with seasonality
- [ ] **Nowcast validation** — compare model estimate for May 2026 vs live IMPIN snapshot
- [ ] **Anomaly detection** — Isolation Forest on scraped products
- [ ] **Streamlit dashboard** — show all 3 layers in one app

### Secondary
- [ ] Policy brief targeting Bank of Ghana (2–3 pages)
- [ ] Extend framework to Lagos and Casablanca (roadmap only, not implemented)

---

## 2-Week Sprint Plan

### Week 1 (Now)
| Day | Task | File |
|---|---|---|
| 1–2 | XGBoost + Random Forest forecasters | `models/ml_forecast.py` |
| 3 | Prophet model | `models/prophet_model.py` |
| 4 | Model comparison table | `models/results/comparison.csv` |
| 5 | Anomaly detection on live scrape | `anomaly/detector.py` |

### Week 2
| Day | Task | File |
|---|---|---|
| 1–2 | Nowcast validation (scrape vs model) | `analysis/nowcast_validation.py` |
| 3–4 | Streamlit dashboard | `dashboard/app.py` |
| 5 | Write-up, slides, polish | `docs/` |

---

## Phase Status

### ✅ Phase 1 — Scraping Pipeline (complete)
- Scrapers: Shopify, WooCommerce, KiKUU, Shopnaw, Bolt Food, Hubtel (stub)
- Sources live: **Konzoom (708), Makola (580), Big Samps (214), Shopnaw (74), KiKUU (10)**
- Combined deduplicated dataset: **1,586 products** → `data/raw/scraped_combined.csv`
- Orchestrator: `scrapers/run_all.py` — dispatches by source, saves timestamped CSV

### ✅ Phase 2 — Historical Data (complete)
- WFP VAM: 14 commodities × 48 months → `data/external/wfp_ghana_monthly_national.parquet`
- Proxies: GHS/USD (WFP-implied) + Brent (FRED) → `data/external/proxy_series.parquet`
- Merged panel: 48-month overlap → `data/processed/historical_panel.parquet`
- WFP Food Index built: base 2019-08=100, range 87–340
- Note: FAO FFPI dropped (API down); GSS CPI dropped (PDF-only, no scrape path)

### ✅ Phase 3A — EDA + Stationarity + Granger (complete)
- EDA: 15 plots + `outputs/IMPIN_EDA_Report.pdf` — key findings:
  - WFP Index +211%, GHS/USD +104% over window; Brent correlates at lag 6 (r=0.717)
  - Lean-season spikes Jul–Sep; all series I(1); AR(1) structure expected
- Stationarity: all 3 series I(1) confirmed → `outputs/stationarity_report.csv`
- Granger: VAR(1) stable; no sig. causality at 5% (n=47 limits power; p=0.17 at lag 6)

### ✅ Phase 3B — ARIMAX (complete)
- Best: ARIMAX(1,1,0) — exog: log(GHS/USD), brent_lag6, month dummies
- Train: 2020-02→2022-06 | Test: 2022-07→2023-07
- Directional accuracy **83.3%** vs naive **50%** → model works
- Plots: `outputs/plots/10a–10c`

### 🔜 Phase 3C — ML Models + Model Comparison (next)
- [ ] XGBoost with lag features — `models/ml_forecast.py`
- [ ] Random Forest — same feature set
- [ ] Prophet with seasonality — `models/prophet_model.py`
- [ ] Head-to-head table: ARIMAX vs XGBoost vs RF vs Prophet vs Naive

### 🔜 Phase 4 — Nowcast Validation
- [ ] Compute IMPIN May-2026 index from `scraped_combined.csv`
- [ ] Run best model forward to May 2026
- [ ] Compare: model prediction vs IMPIN live snapshot
- [ ] This is the key validation of the 3-layer claim

### 🔜 Phase 5 — Anomaly Detection
- [ ] Isolation Forest on 1,586 scraped products
- [ ] Z-score cross-check per CPI category
- [ ] Output: flagged anomalies with scores

### 🔜 Phase 6 — Dashboard
- [ ] Streamlit app with 3 tabs: Index | Forecast | Anomalies

### 🔜 Phase 7 — Write-Up
- [ ] Academic paper + policy brief + slide deck

---

## Success Metrics

| Metric | Target | Status |
|---|---|---|
| Directional accuracy (ARIMAX) | > naive 50% | ✅ 83.3% |
| ML model (XGBoost/RF) accuracy | > ARIMAX directional | 🔜 Pending |
| Nowcast validation | IMPIN snapshot within ±20% of model estimate | 🔜 Pending |
| Anomaly detection | ≥5 flagged products per scrape | 🔜 Pending |
| Dashboard | Live Streamlit app | 🔜 Pending |
| Granger (future work) | p < 0.05 with weekly scrape series | 🟡 Null with 47 obs |

---

## Honest Limitations (put these in the paper)

| Limitation | What we say |
|---|---|
| Only 1 scrape date | "This paper presents a pilot; continuous weekly scraping is needed to build an index series" |
| No GSS CPI validation | "We validate against a WFP food price proxy; direct CPI comparison requires GSS access" |
| Granger null result | "Small sample (n=47) limits power; Brent→WFP correlation at lag 6 is consistent with theory" |
| Online prices ≠ market prices | "Online platforms capture a subset of Accra's retail market; coverage gaps are documented" |
