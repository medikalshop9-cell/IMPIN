# IMPIN — Informal Market Price Intelligence Network

> A real-time food price intelligence system for Ghana.

**Students:** Acheampong Yaw HINNEH · Abiola OKUNSANYA  
**Program:** Master in Management (MIM) — Africa Business School, UM6P  
**Campus:** Rabat, Morocco · **Academic Year:** 2025–2027  
**Track:** Data Science / Machine Learning / Artificial Intelligence  

---

## Academic Claim

> *IMPIN is a real-time food price intelligence system for Ghana with three layers:*
> *(1) a live scrape-based price index updated weekly;*
> *(2) a nowcast of the current food CPI component before official GSS release;*
> *(3) a 1–3 month forecast comparing classical time-series and ML models.*

---

## The Three Layers

### Layer 1 — Live Scrape Index
Automated scrapers collect prices from Ghanaian e-commerce platforms (Konzoom, Makola Stores, Big Samps Market, Shopnaw, KiKUU). These are aggregated into an **IMPIN snapshot index** weighted by the Ghana GSS CPI basket (base period: January 2026 = 100). Current dataset: **1,586 products** across 5 sources.

### Layer 2 — Nowcast
The IMPIN snapshot is appended to a 48-month historical panel (WFP food prices + GHS/USD rate + Brent crude). Four models are trained on the historical data and used to estimate the current month's food CPI component **before GSS releases the official figure**. Models: ARIMAX (classical), XGBoost, Random Forest, Prophet.

### Layer 3 — Forecast
The same models produce **1–3 month ahead forecasts**. This enables proactive policy signals. The best model is selected by directional accuracy on a held-out test set.

---

## What We Found So Far

| Result | Status |
|---|---|
| Scraping pipeline live | ✅ 1,586 products, 5 sources |
| WFP 48-month historical panel | ✅ 14 commodities, 48 months |
| EDA + stationarity confirmed | ✅ All series I(1) |
| Granger causality | 🟡 Null at 5% (n=47 underpowered — publishable) |
| ARIMAX(1,1,0) directional accuracy | ✅ 83.3% vs 50% naive |
| XGBoost / RF / Prophet | 🔜 Next |
| Nowcast validation vs scrape | 🔜 After ML models |
| Anomaly detection | 🔜 Pending |
| Streamlit dashboard | 🔜 Week 2 |

---

## Repository Structure

```
IMPIN/
├── scrapers/           # Per-source scrapers + orchestrator (run_all.py)
├── pipeline/           # build_historical.py, build_index.py
├── models/             # ARIMAX, XGBoost, RF, Prophet + comparison
├── analysis/           # EDA, stationarity, Granger, nowcast validation
├── anomaly/            # Isolation Forest + Z-score detector
├── app/                # Streamlit dashboard
├── data/
│   ├── raw/            # Scraped CSVs (gitignored)
│   ├── processed/      # historical_panel.parquet
│   └── external/       # WFP, proxy series
├── outputs/            # Plots, reports, model results
├── config/             # cpi_basket.yaml
└── models/results/     # arimax_comparison.csv, comparison.csv
```

---

## Quickstart

```bash
# Run all scrapers and collect latest prices
python scrapers/run_all.py

# Build historical panel (WFP + proxies)
python pipeline/build_historical.py

# Run EDA
python analysis/eda.py

# Stationarity tests
python analysis/stationarity.py

# Granger causality
python analysis/granger.py

# ARIMAX model
python models/arimax_model.py

# Launch dashboard (after all models built)
streamlit run app/dashboard.py
```

---

## Data Sources

| Source | Status | Role |
|---|---|---|
| WFP VAM Ghana | ✅ Live | Historical food prices (48 months, 14 commodities) |
| Brent crude (FRED) | ✅ Live | Exogenous predictor for food price model |
| GHS/USD rate | ✅ Implied from WFP | Exchange rate proxy |
| FAO Food Price Index | ❌ API unreachable | Dropped |
| GSS Ghana CPI | ❌ PDF-only, no scrape path | Dropped — WFP used as proxy |
| Konzoom, Makola, etc. | ✅ Live | IMPIN snapshot index (1,586 products) |

---

## Honest Limitations

- **One scrape date** — This is a pilot. A rolling weekly index requires ongoing scraping.
- **No GSS CPI direct validation** — WFP food prices are used as proxy; GSS data is PDF-only.
- **Granger null result** — n=47 is underpowered. Null result is publishable and informative.
- **Online ≠ market** — Online platforms represent a subset of Accra's retail market.

---

## References

- Cavallo, A. (2018). *Scraped Data and Sticky Prices.* Review of Economics and Statistics, 100(1), 105–119.  
- Cavallo, A. & Rigobon, R. (2016). *The Billion Prices Project.* Journal of Economic Perspectives, 30(2), 151–178.  
- Nakamura, E. & Steinsson, J. (2008). *Five Facts About Prices.* Quarterly Journal of Economics, 123(4), 1415–1464.  
- WFP VAM (2023). *Food Security and Market Monitoring in West Africa.*  
- IMF African Department (2022). *Overcoming Data Sparsity.* IMF Working Paper WP/22/88.  

---

## License

Academic research project — Africa Business School, UM6P · 2025–2027.
