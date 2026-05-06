# IMPIN — Data Sources

This document describes every confirmed data source actually used in the IMPIN pipeline.
Sources that were planned but are not operational are marked accordingly.

---

## Summary Table

| Source | Type | Status | Role | Records / Months |
|---|---|---|---|---|
| WFP VAM Ghana | Historical food prices | ✅ Live | Primary historical series | 7,774 rows, 14 commodities, 48 months |
| Brent crude (FRED) | Macroeconomic proxy | ✅ Live | Exogenous predictor (lag 6) | 48 months |
| GHS/USD rate | Macroeconomic proxy | ✅ Implied | Exchange rate proxy | 48 months |
| Konzoom | Shopify e-commerce | ✅ Live | IMPIN snapshot | 708 products |
| Makola Stores | Shopify e-commerce | ✅ Live | IMPIN snapshot | 580 products |
| Big Samps Market | WooCommerce | ✅ Live | IMPIN snapshot | 214 products |
| Shopnaw | Custom scraper | ✅ Live | IMPIN snapshot | 74 products |
| KiKUU | Custom scraper | ✅ Live (limited) | IMPIN snapshot | 10 products |
| FAO Food Price Index | Macroeconomic proxy | ❌ Dropped | Was planned as exogenous | API unreachable |
| Ghana GSS CPI | Official statistic | ❌ Dropped | Was target variable | PDF-only, no scrape path |
| Bolt Food | Food delivery | ❌ Stub only | Playwright needed | 0 |
| Hubtel | Super-app | ❌ Stub only | App-gated, no web access | 0 |

---

## Section 1 — Live Scrape Sources (Layer 1: IMPIN Snapshot)

### 1. Konzoom
- **URL:** https://konzoom.com
- **Type:** Shopify e-commerce (14 product collections, food focus)
- **Scraper:** `scrapers/shopify.py` dispatched via `scrapers/run_all.py`
- **Products collected:** 708
- **CPI Categories:** Food & Beverages (primary), General
- **Scraped:** 2026-05-05 (W19)
- **Notes:** Most comprehensive food source. Shopify `/products.json` API — clean JSON, no JS needed.

---

### 2. Makola Stores
- **URL:** Shopify store (Makola brand)
- **Type:** Shopify e-commerce
- **Scraper:** `scrapers/shopify.py`
- **Products collected:** 580
- **CPI Categories:** Food & Beverages, General
- **Scraped:** 2026-05-05 (W19)

---

### 3. Big Samps Market
- **URL:** https://bigsamps.com
- **Type:** WooCommerce (HTML scrape + pagination)
- **Scraper:** `scrapers/woocommerce.py`
- **Products collected:** 214
- **CPI Categories:** Food & Beverages, Household, General
- **Scraped:** 2026-05-05 (W19)

---

### 4. Shopnaw
- **URL:** https://shopnaw.com
- **Type:** Custom Ghanaian e-commerce
- **Scraper:** `scrapers/shopnaw.py`
- **Products collected:** 74
- **CPI Categories:** General, Personal Care
- **Scraped:** 2026-05-05 (W19)

---

### 5. KiKUU
- **URL:** https://www.kikuu.com/gh/
- **Type:** E-commerce (Africa-focused, Chinese-origin)
- **Scraper:** `scrapers/kikuu.py`
- **Products collected:** 10 (homepage featured items only)
- **CPI Categories:** Clothing, General
- **Scraped:** 2026-05-05 (W19)
- **Notes:** Limited to homepage featured products. Low coverage — included for completeness but not primary.

---

### Combined Dataset

- **File:** `data/raw/scraped_combined.csv`
- **Total products:** 1,586 (deduplicated on source + product_name + price_ghc)
- **Category breakdown:**
  - General: 655 (41.3%)
  - Food & Beverages: 605 (38.1%)
  - Household: 159 (10.0%)
  - Personal Care: 153 (9.6%)
  - Clothing: 14 (0.9%)
- **Columns:** `source, cpi_category, product_name, price_ghc, unit, currency, url, scraped_at`

---

## Section 2 — Historical Data Sources (Layers 2 & 3: Nowcast + Forecast)

### 6. WFP VAM Ghana
- **URL:** https://dataviz.vam.wfp.org / HDX (Humanitarian Data Exchange)
- **Access:** Public, no authentication required
- **Scraper:** `scrapers/wfp_vam.py`
- **Coverage:** National medians — 14 food commodities (tomatoes, peppers, maize, rice, fish, onions, yam, plantain, cassava, groundnut oil, palm oil, beef, chicken, sugar)
- **Period:** 2019-08 → 2023-07 (48 months in the historical panel overlap window)
- **Output:** `data/external/wfp_ghana_monthly_national.parquet`
- **Role in pipeline:** Foundation of `wfp_food_index` (base 2019-08=100); primary target variable for forecasting models
- **Notes:** Raw HDX file has 7,774 rows across all markets. National medians computed from market-level data.

---

### 7. Brent Crude (FRED)
- **URL:** https://fred.stlouisfed.org/series/DCOILBRENTEU
- **Access:** FRED API (free key)
- **Scraper:** `scrapers/proxies.py`
- **Coverage:** Monthly averages, 2019-08 → 2023-07
- **Output:** `data/external/proxy_series.parquet` (column: `brent`)
- **Role in pipeline:** Exogenous regressor in ARIMAX and ML models (at lag 6 — r=0.717 with WFP food index at that lag)
- **Key finding from EDA:** Brent is the strongest cross-correlate of WFP food index (r=0.717 at lag 6 months)

---

### 8. GHS/USD Exchange Rate (implied)
- **Source:** Derived from WFP data (USD-denominated prices cross-referenced with GHS equivalents)
- **Coverage:** Monthly, 2019-08 → 2023-07
- **Output:** `data/external/proxy_series.parquet` (column: `ghsusd`)
- **Role in pipeline:** Exogenous regressor in ARIMAX and ML models (log-transformed)
- **Notes:** Implied rate — not sourced directly from Bank of Ghana. Acknowledged as limitation.

---

## Section 3 — Sources Attempted but Dropped

### FAO Food Price Index (FFPI)
- **Reason dropped:** FAOSTAT API was unreachable at time of development
- **Original role:** Global food inflation proxy — would have been exogenous regressor
- **Replacement:** Brent crude (FRED) used instead — performs well at lag 6

### Ghana Statistical Service (GSS) CPI
- **URL:** https://statsghana.gov.gh/cpi.php → returns 404
- **Reason dropped:** All data published as PDF only; no machine-readable API; web page returned 404
- **Original role:** Target variable (monthly CPI food component)
- **Replacement:** WFP food price index used as proxy target variable. Explicitly stated as limitation in paper.

### Bolt Food
- **Status:** Stub implemented (`scrapers/bolt_food.py`); requires Playwright for JS-rendered menus
- **Products collected:** 0
- **Reason:** Playwright setup not completed in sprint; deprioritised given sufficient data from Shopify/WooCommerce sources

### Hubtel
- **Status:** Stub implemented (`scrapers/hubtel.py`); products behind app authentication
- **Products collected:** 0
- **Reason:** No public web endpoint for product listings

---

## Section 4 — CPI Basket Weights

**Source:** `config/cpi_basket.yaml`  
**Based on:** Ghana GSS 2021 Household Income & Expenditure Survey (HIES) approximation  
**Base period:** January 2026 = 100

| Category | Weight | Notes |
|---|---|---|
| Food & Beverages | 42.5% | Primary category for food inflation analysis |
| General | 40.5% | Broad consumer goods |
| Household | 9.8% | Cleaning, kitchen, furnishings |
| Clothing & Personal Care | 7.2% | Apparel + toiletries |

---
