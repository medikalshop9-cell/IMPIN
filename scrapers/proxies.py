"""
scrapers/proxies.py
===================
Downloads macroeconomic proxy series for Granger causality and forecasting:

  1. GHS/USD Exchange Rate — monthly (implied from WFP price data)
     Derived as median(price_ghc / price_usd) per month, 2019-08 → 2023-07

  2. Brent Crude Oil Price — monthly average (USD/barrel)
     Source: FRED series DCOILBRENTEU (free, no key required)

  Note: FAO FFPI was investigated but no free public API is reliably available.
  Global food price signal is proxied by Brent crude + WFP local prices.

Output
------
  data/external/proxy_series.csv     — merged monthly series
  data/external/proxy_series.parquet — same, as Parquet

Usage
-----
    python -m scrapers.proxies               # download + save
    python -m scrapers.proxies --summary     # print coverage table
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
logger = logging.getLogger("proxies")

_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _ROOT / "data" / "external"

# FRED public CSV endpoint (no API key required)
_FRED_BASE  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_FRED_BRENT = _FRED_BASE + "DCOILBRENTEU"   # USD/barrel, daily → resample monthly


def _derive_ghsusd_from_wfp() -> pd.DataFrame:
    """
    Derive implied GHS/USD exchange rate from WFP Ghana price data
    using the ratio: price_ghc / price_usd.
    This is the market-implied rate at informal market level.
    Returns DataFrame with columns: year_month, ghsusd
    """
    wfp_path = _OUTPUT_DIR / "wfp_ghana_prices.csv"
    if not wfp_path.exists():
        logger.warning("WFP prices not found at %s — run scrapers.wfp_vam first", wfp_path)
        return pd.DataFrame(columns=["year_month", "ghsusd"])

    logger.info("Deriving GHS/USD from WFP data ...")
    df = pd.read_csv(wfp_path)
    df["price_ghc"] = pd.to_numeric(df["price_ghc"], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df = df.dropna(subset=["price_ghc", "price_usd"])
    df = df[df["price_usd"] > 0]
    df["implied_fx"] = df["price_ghc"] / df["price_usd"]
    # Reject outliers (GHS/USD range: 1–50 covers 2019–2023)
    df = df[df["implied_fx"].between(1, 50)]
    monthly = (
        df.groupby("year_month")["implied_fx"]
        .median()
        .reset_index()
        .rename(columns={"implied_fx": "ghsusd"})
    )
    logger.info("GHS/USD (implied): %d months (%s → %s)",
                len(monthly), monthly["year_month"].min(), monthly["year_month"].max())
    return monthly


def _fetch_brent() -> pd.DataFrame:
    """
    Download Brent crude daily prices from FRED and aggregate to monthly average.
    Returns DataFrame with columns: year_month, brent
    """
    logger.info("Fetching Brent crude from FRED ...")
    try:
        resp = requests.get(_FRED_BRENT, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        monthly = (
            df.set_index("date")["value"]
            .resample("MS")
            .mean()
            .reset_index()
        )
        monthly["year_month"] = monthly["date"].dt.strftime("%Y-%m")
        monthly = monthly.rename(columns={"value": "brent"})[["year_month", "brent"]]
        logger.info("Brent: %d months (%s → %s)",
                    len(monthly), monthly["year_month"].min(), monthly["year_month"].max())
        return monthly
    except Exception as e:
        logger.warning("Brent FRED fetch failed: %s — will be empty", e)
        return pd.DataFrame(columns=["year_month", "brent"])


def fetch(summary: bool = False) -> pd.DataFrame:
    """
    Download proxy series, merge on year_month, save outputs.

    Returns
    -------
    Merged monthly DataFrame with columns: year_month, ghsusd, brent
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ghsfx = _derive_ghsusd_from_wfp()
    brent = _fetch_brent()

    # Merge on year_month — outer join to keep all available months
    if not ghsfx.empty and not brent.empty:
        merged = ghsfx.merge(brent, on="year_month", how="outer")
    elif not brent.empty:
        merged = brent
    else:
        merged = ghsfx

    merged = merged.sort_values("year_month").reset_index(drop=True)

    # Save
    merged.to_csv(_OUTPUT_DIR / "proxy_series.csv", index=False)
    merged.to_parquet(_OUTPUT_DIR / "proxy_series.parquet", index=False)
    logger.info("Saved → data/external/proxy_series.parquet  (%d rows)", len(merged))

    if summary:
        print("\n" + "=" * 60)
        print("PROXY SERIES COVERAGE")
        print("=" * 60)
        for col in ["ghsusd", "brent"]:
            if col in merged.columns:
                non_null = merged[col].notna()
                first = merged.loc[non_null, "year_month"].min() if non_null.any() else "—"
                last  = merged.loc[non_null, "year_month"].max() if non_null.any() else "—"
                print(f"  {col:12s}  {non_null.sum():4d} months  [{first} → {last}]")
        print(f"\n  Total rows: {len(merged)}")
        print(merged.tail(6).to_string(index=False))

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download macro proxy series")
    parser.add_argument("--summary", action="store_true", help="Print coverage table")
    args = parser.parse_args()
    fetch(summary=args.summary)
