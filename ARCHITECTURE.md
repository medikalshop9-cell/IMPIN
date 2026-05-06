# IMPIN — System Architecture

---

## Overview

IMPIN is a three-layer food price intelligence system for Ghana. It has two data
tracks (historical time-series and live scrape) that converge in a nowcast layer
and are exposed through a single Streamlit dashboard.

---

## Full Data Flow

```
═══════════════════════════════════════════════════════════════════════════════
  DATA INPUTS
═══════════════════════════════════════════════════════════════════════════════

  [A] WFP VAM Ghana          [B] FRED Brent Crude        [C] Online Shops
  7,774 rows, 14 commodities  Monthly 2019–2023           Konzoom, Makola,
  Monthly 2019–2023           ↓                           Big Samps, Shopnaw,
  ↓                           ↓                           KiKUU
  ↓                           ↓                           scrapers/run_all.py
  scrapers/wfp_vam.py         scrapers/proxies.py         ↓
  ↓                           ↓                           1,586 products
  ↓                           ↓                           data/raw/scraped_combined.csv
  └──────────────┬────────────┘                           ↓
                 ↓                                        pipeline/build_index.py
  pipeline/build_historical.py                            ↓
  ↓                                                  LAYER 1: IMPIN Snapshot
  data/processed/historical_panel.parquet            (May 2026, base Jan-2026=100)
  48 months × 3 series                                    │
  (wfp_food_index, ghsusd, brent)                         │ append as latest point
                 │                                        │
                 └──────────────┬─────────────────────────┘
                                ↓

═══════════════════════════════════════════════════════════════════════════════
  LAYER 2: NOWCAST  (estimate current food CPI before GSS releases it)
═══════════════════════════════════════════════════════════════════════════════

  Historical panel + IMPIN snapshot
          ↓
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ ARIMAX      │  │  XGBoost    │  │    Random   │  │   Prophet   │
  │ (1,1,0) ✅  │  │    🔜       │  │   Forest 🔜 │  │    🔜       │
  │ 83.3% DA   │  │  lag feats  │  │  lag feats  │  │ seasonality │
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         └─────────────────┴─────────────────┴────────────────┘
                                    ↓
                      models/results/comparison.csv
                      (RMSE, MAE, directional accuracy)
                                    ↓
              LAYER 2 OUTPUT: Point estimate for May 2026
              ← compared against IMPIN snapshot (Layer 1)

═══════════════════════════════════════════════════════════════════════════════
  LAYER 3: FORECAST  (1–3 months ahead)
═══════════════════════════════════════════════════════════════════════════════

  Best model selected from Layer 2
          ↓
  1-step, 2-step, 3-step ahead forecast with confidence intervals
          ↓
  outputs/plots/forecast_*.png

═══════════════════════════════════════════════════════════════════════════════
  ANOMALY DETECTION  (cross-sectional, on live scrape)
═══════════════════════════════════════════════════════════════════════════════

  scraped_combined.csv (1,586 products)
          ↓
  anomaly/detector.py
  ├─ Isolation Forest (contamination=0.05)
  └─ Z-score per CPI category (|z| > 2.5 = flagged)
          ↓
  outputs/anomaly_report.csv
  (product_name, source, price_ghc, z_score, anomaly_score, category)

═══════════════════════════════════════════════════════════════════════════════
  DASHBOARD
═══════════════════════════════════════════════════════════════════════════════

  app/dashboard.py (Streamlit)
  ├── Tab 1: Live Index  →  IMPIN snapshot chart + basket breakdown
  ├── Tab 2: Forecast    →  historical trend + 1-3 month forecast + CI
  └── Tab 3: Anomalies   →  flagged products table + z-score chart
```

---

## Component Inventory

### Scrapers (`scrapers/`)

| File | Source | Status |
|---|---|---|
| `run_all.py` | Orchestrator | ✅ Done |
| `base.py` | BaseScraper | ✅ Done |
| `shopify.py` | Generic Shopify (`/products.json`) | ✅ Done |
| `woocommerce.py` | Generic WooCommerce (HTML) | ✅ Done |
| `kikuu.py` | KiKUU homepage | ✅ Done |
| `shopnaw.py` | Shopnaw | ✅ Done |
| `bolt_food.py` | Bolt Food (Playwright stub) | ✅ Done |
| `hubtel.py` | Hubtel (app-gated stub) | ✅ Done |
| `wfp_vam.py` | WFP HDX open data | ✅ Done |
| `proxies.py` | FRED Brent + GHS/USD implied | ✅ Done |

### Pipeline (`pipeline/`)

| File | Purpose | Status |
|---|---|---|
| `build_historical.py` | Merge WFP + proxies → historical_panel.parquet | ✅ Done |
| `build_index.py` | Compute IMPIN snapshot from scraped_combined.csv | 🔜 Pending |

### Analysis (`analysis/`)

| File | Purpose | Status |
|---|---|---|
| `eda.py` | 15 plots + EDA PDF | ✅ Done |
| `stationarity.py` | ADF + KPSS — all I(1) | ✅ Done |
| `granger.py` | VAR(1) + Granger — null result | ✅ Done |
| `nowcast_validation.py` | Compare model estimate vs IMPIN snapshot | 🔜 Pending |

### Models (`models/`)

| File | Purpose | Status |
|---|---|---|
| `arimax_model.py` | ARIMAX(1,1,0) — 83.3% directional accuracy | ✅ Done |
| `ml_forecast.py` | XGBoost + Random Forest with lag features | 🔜 Next |
| `prophet_model.py` | Prophet with seasonality + regressors | 🔜 Next |
| `results/comparison.csv` | Head-to-head model table | 🔜 Pending |

### Anomaly Detection (`anomaly/`)

| File | Purpose | Status |
|---|---|---|
| `detector.py` | Isolation Forest + Z-score on scraped_combined.csv | 🔜 Pending |

### Dashboard (`app/`)

| File | Purpose | Status |
|---|---|---|
| `dashboard.py` | Streamlit — 3-tab app | 🔜 Pending |

---

## Key Data Files

| File | Description | Rows |
|---|---|---|
| `data/raw/scraped_combined.csv` | Merged live scrape, May 2026 | 1,586 products |
| `data/processed/historical_panel.parquet` | 48-month historical panel | 48 rows |
| `data/external/wfp_ghana_monthly_national.parquet` | WFP raw | 7,774 rows |
| `data/external/proxy_series.parquet` | GHS/USD + Brent | 48 rows |
| `outputs/stationarity_report.csv` | ADF + KPSS results | 3 series |
| `outputs/granger_results.csv` | VAR(1) Granger output | — |
| `models/results/arimax_comparison.csv` | ARIMAX grid search | 9 models |

---

## ARIMAX Model Specification (Layer 2 baseline)

```
Model:  ARIMAX(1, 1, 0)
Target: wfp_food_index (first-differenced)
Exog:   log_ghsusd, brent_lag6, month_1 … month_11

Train:  2020-02 → 2022-06  (29 observations)
Test:   2022-07 → 2023-07  (13 observations)

Results:
  Directional accuracy:  83.3%  (naive: 50%)
  Test RMSE:             higher than naive persistence
  Note:  multi-step horizon — directional accuracy is the key metric

Artefacts:
  outputs/plots/10a_arimax_forecast.png
  outputs/plots/10b_arimax_comparison.png
  outputs/plots/10c_arimax_residuals.png
  models/results/arimax_comparison.csv
```

---

## ML Feature Set (XGBoost + Random Forest)

```
Features per observation t:
  - wfp_t-1, wfp_t-2, wfp_t-3   (lag features of target)
  - log_ghsusd_t                  (exchange rate)
  - brent_t-6                     (oil price at lag 6)
  - month_1 … month_11            (month dummies for seasonality)

Train/test split: same as ARIMAX
  Train: 2020-02 → 2022-06
  Test:  2022-07 → 2023-07

Evaluation: RMSE, MAE, directional accuracy (same metrics as ARIMAX)
```

### Pipeline

