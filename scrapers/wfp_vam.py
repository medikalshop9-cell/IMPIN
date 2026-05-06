"""
scrapers/wfp_vam.py
===================
Downloads and processes WFP Ghana food price history from the
Humanitarian Data Exchange (HDX) open data portal.

Source : WFP Food Prices for Ghana
URL    : https://data.humdata.org/dataset/wfp-food-prices-for-ghana
File   : wfp_food_prices_gha.csv  (~26 000 rows, 2006–present)
Columns: date, admin1, admin2, market, market_id, latitude, longitude,
         category, commodity, commodity_id, unit, priceflag, pricetype,
         currency, price, usdprice

Output schema (saved to data/external/wfp_ghana_prices.csv):
  date        : YYYY-MM-DD (first of month)
  year_month  : YYYY-MM
  commodity   : cleaned commodity name
  market      : market name
  admin1      : region (e.g. "Greater Accra")
  unit        : unit of measurement
  price_ghc   : price in GHS (local currency)
  price_usd   : price in USD
  cpi_category: mapped CPI basket category

Usage
-----
    python -m scrapers.wfp_vam               # download + save
    python -m scrapers.wfp_vam --summary     # also print summary table
"""

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wfp_vam")

# Direct download URL (HDX open data — no auth required)
_WFP_CSV_URL = (
    "https://data.humdata.org/dataset/626e809c-c4fc-467b-a60c-129acb5e9320"
    "/resource/e877350b-146f-4fa7-8690-db9605eea78c/download/wfp_food_prices_gha.csv"
)

# Commodities to keep — mapped to CPI basket categories
# Keys are regex patterns matching WFP commodity names (case-insensitive)
_COMMODITY_MAP = {
    # Food & Beverages — cereals
    r"maize": ("Maize", "Food & Beverages"),
    r"rice": ("Rice", "Food & Beverages"),
    r"millet": ("Millet", "Food & Beverages"),
    r"sorghum": ("Sorghum", "Food & Beverages"),
    r"wheat\s*flour": ("Wheat flour", "Food & Beverages"),
    r"cassava": ("Cassava", "Food & Beverages"),
    r"yam": ("Yam", "Food & Beverages"),
    r"plantain": ("Plantain", "Food & Beverages"),
    # Food & Beverages — proteins
    r"fish.*smoked|smoked.*fish": ("Fish (smoked)", "Food & Beverages"),
    r"fish.*fresh|fresh.*fish": ("Fish (fresh)", "Food & Beverages"),
    r"chicken|poultry": ("Chicken", "Food & Beverages"),
    r"beef": ("Beef", "Food & Beverages"),
    r"egg": ("Eggs", "Food & Beverages"),
    # Food & Beverages — oils & fats
    r"palm\s*oil": ("Palm oil", "Food & Beverages"),
    r"vegetable\s*oil": ("Vegetable oil", "Food & Beverages"),
    # Food & Beverages — vegetables & fruits
    r"tomato": ("Tomatoes", "Food & Beverages"),
    r"onion": ("Onions", "Food & Beverages"),
    r"pepper|chili|chilli": ("Pepper", "Food & Beverages"),
    r"garden\s*egg": ("Garden egg", "Food & Beverages"),
    r"banana": ("Banana", "Food & Beverages"),
    r"orange": ("Orange", "Food & Beverages"),
    # Food & Beverages — other
    r"sugar": ("Sugar", "Food & Beverages"),
    r"salt": ("Salt", "Food & Beverages"),
    r"bean|cowpea": ("Beans/Cowpea", "Food & Beverages"),
    r"groundnut|peanut": ("Groundnut", "Food & Beverages"),
    # Fuel (non-food but important for CPI)
    r"petrol|gasoline": ("Petrol", "General"),
    r"diesel": ("Diesel", "General"),
    r"kerosene": ("Kerosene", "Household"),
}

_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _ROOT / "data" / "external"


def _download_raw() -> pd.DataFrame:
    """Download WFP Ghana food prices CSV and return raw DataFrame."""
    logger.info("Downloading WFP Ghana food prices from HDX ...")
    response = requests.get(_WFP_CSV_URL, timeout=60)
    response.raise_for_status()
    # First row may be HXL hashtag row — skip if it starts with '#'
    text = response.text
    lines = text.splitlines()
    if lines and lines[1].startswith("#"):
        text = "\n".join([lines[0]] + lines[2:])
    df = pd.read_csv(StringIO(text), low_memory=False)
    logger.info("Downloaded %d rows", len(df))
    return df


def _clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to relevant commodities, standardise columns, map CPI categories."""
    import re

    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price"])

    # Keep retail prices only (pricetype == "Retail")
    if "pricetype" in df.columns:
        df = df[df["pricetype"].str.lower() == "retail"]

    # Rename price columns
    df = df.rename(columns={"price": "price_ghc", "usdprice": "price_usd"})

    # Map commodities
    matched_rows = []
    for pattern, (standard_name, cpi_cat) in _COMMODITY_MAP.items():
        mask = df["commodity"].str.contains(pattern, case=False, na=False, regex=True)
        subset = df[mask].copy()
        subset["commodity_std"] = standard_name
        subset["cpi_category"] = cpi_cat
        matched_rows.append(subset)

    if not matched_rows:
        raise ValueError("No commodities matched — check _COMMODITY_MAP patterns")

    filtered = pd.concat(matched_rows, ignore_index=True)

    # Drop the original WFP commodity column before renaming our mapped column
    if "commodity" in filtered.columns:
        filtered = filtered.drop(columns=["commodity"])
    filtered = filtered.rename(columns={"commodity_std": "commodity"})

    # Drop duplicates caused by overlapping patterns (keep first match)
    filtered = filtered.drop_duplicates(subset=["date", "market", "commodity"])

    # Keep useful columns only
    out_cols = ["date", "admin1", "admin2", "market", "commodity",
                "unit", "price_ghc", "price_usd", "cpi_category"]
    filtered = filtered[[c for c in out_cols if c in filtered.columns]]
    filtered["year_month"] = filtered["date"].dt.strftime("%Y-%m")
    filtered = filtered.sort_values("date").reset_index(drop=True)

    n_rows = len(filtered)
    n_commodities = filtered["commodity"].nunique()
    logger.info(
        "After filtering: %d rows | %d commodities | %s to %s",
        n_rows,
        n_commodities,
        filtered["date"].min().strftime("%Y-%m"),
        filtered["date"].max().strftime("%Y-%m"),
    )
    return filtered


def _build_monthly_national(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to national monthly median per commodity.
    Output: (year_month, commodity, cpi_category, median_price_ghc, n_markets)
    """
    agg = (
        df.groupby(["year_month", "commodity", "cpi_category"])
        .agg(
            median_price_ghc=("price_ghc", "median"),
            n_markets=("market", "nunique"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    logger.info("Monthly national series: %d rows across %d commodities",
                len(agg), agg["commodity"].nunique())
    return agg


def fetch(summary: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: download → clean → filter → aggregate.

    Returns
    -------
    (raw_filtered, monthly_national)
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = _download_raw()
    filtered = _clean_and_filter(raw)
    monthly = _build_monthly_national(filtered)

    # Save
    filtered.to_csv(_OUTPUT_DIR / "wfp_ghana_prices.csv", index=False)
    monthly.to_csv(_OUTPUT_DIR / "wfp_ghana_monthly_national.csv", index=False)
    monthly.to_parquet(_OUTPUT_DIR / "wfp_ghana_monthly_national.parquet", index=False)

    logger.info("Saved → data/external/wfp_ghana_prices.csv")
    logger.info("Saved → data/external/wfp_ghana_monthly_national.parquet")

    if summary:
        print("\n" + "=" * 65)
        print("WFP GHANA — COMMODITY COVERAGE SUMMARY")
        print("=" * 65)
        summary_tbl = (
            monthly.groupby(["commodity", "cpi_category"])
            .agg(
                months=("year_month", "nunique"),
                first=("year_month", "min"),
                last=("year_month", "max"),
                avg_price=("median_price_ghc", "mean"),
            )
            .reset_index()
            .sort_values("months", ascending=False)
        )
        print(summary_tbl.to_string(index=False))

    return filtered, monthly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WFP Ghana food price history")
    parser.add_argument("--summary", action="store_true", help="Print commodity coverage table")
    args = parser.parse_args()
    fetch(summary=args.summary)
