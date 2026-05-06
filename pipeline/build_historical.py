"""
pipeline/build_historical.py
=============================
Merges WFP Ghana commodity prices + proxy macro series into a single
aligned monthly panel dataset for Granger causality and forecasting.

Input files (data/external/)
  wfp_ghana_monthly_national.parquet — WFP commodity medians per month
  proxy_series.parquet               — GHS/USD (implied) + Brent crude

Output (data/processed/)
  historical_panel.parquet           — monthly panel, aligned
  wfp_food_index.parquet             — simple equal-weighted WFP food index

Columns in historical_panel.parquet:
  year_month        : YYYY-MM
  wfp_food_index    : equal-weighted index of WFP commodity prices (2019-08=100)
  n_commodities     : number of commodities contributing to index that month
  ghsusd            : implied GHS/USD exchange rate from WFP
  brent             : Brent crude monthly average (USD/barrel)

Usage
-----
    python -m pipeline.build_historical            # build and save
    python -m pipeline.build_historical --plot     # also print stats table
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_historical")

_ROOT = Path(__file__).parent.parent
_EXT_DIR  = _ROOT / "data" / "external"
_PROC_DIR = _ROOT / "data" / "processed"

# Base period for WFP index normalisation (first full month of data)
_INDEX_BASE = "2019-08"


def _build_wfp_food_index(wfp: pd.DataFrame) -> pd.DataFrame:
    """
    Construct an equal-weighted WFP food price index from monthly national medians.

    Steps:
    1. Pivot to wide format (rows=year_month, cols=commodity)
    2. Normalise each commodity to base period = 100
    3. Average across available commodities → index value
    4. Track n_commodities contributing each month

    Returns DataFrame: year_month, wfp_food_index, n_commodities
    """
    # Pivot: rows = year_month, cols = commodity, values = median_price_ghc
    pivot = wfp.pivot_table(
        index="year_month", columns="commodity", values="median_price_ghc"
    )

    # Normalise each commodity to base period = 100
    base_row = pivot.loc[_INDEX_BASE] if _INDEX_BASE in pivot.index else pivot.iloc[0]
    base_row = base_row.replace(0, float("nan"))  # avoid division by zero
    normalised = pivot.div(base_row) * 100

    # Equal-weighted mean across commodities (ignore NaN)
    index_series = normalised.mean(axis=1)
    n_commodities = normalised.notna().sum(axis=1)

    result = pd.DataFrame({
        "year_month": index_series.index,
        "wfp_food_index": index_series.values,
        "n_commodities": n_commodities.values,
    }).sort_values("year_month").reset_index(drop=True)

    logger.info(
        "WFP food index: %d months | base=%s | "
        "range=%.1f–%.1f | avg commodities=%.1f",
        len(result),
        _INDEX_BASE,
        result["wfp_food_index"].min(),
        result["wfp_food_index"].max(),
        result["n_commodities"].mean(),
    )
    return result


def build(plot: bool = False) -> pd.DataFrame:
    """
    Merge WFP food index + proxy series into a monthly panel.

    Returns
    -------
    historical_panel DataFrame
    """
    _PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Load WFP monthly national
    wfp_path = _EXT_DIR / "wfp_ghana_monthly_national.parquet"
    if not wfp_path.exists():
        raise FileNotFoundError(
            f"{wfp_path} not found — run: python -m scrapers.wfp_vam"
        )
    wfp = pd.read_parquet(wfp_path)
    logger.info("Loaded WFP monthly: %d rows, %d commodities",
                len(wfp), wfp["commodity"].nunique())

    # Load proxy series
    proxy_path = _EXT_DIR / "proxy_series.parquet"
    if not proxy_path.exists():
        raise FileNotFoundError(
            f"{proxy_path} not found — run: python -m scrapers.proxies"
        )
    proxies = pd.read_parquet(proxy_path)
    logger.info("Loaded proxies: %d rows", len(proxies))

    # Build WFP food index
    wfp_index = _build_wfp_food_index(wfp)

    # Save standalone WFP index
    wfp_index.to_parquet(_PROC_DIR / "wfp_food_index.parquet", index=False)
    logger.info("Saved → data/processed/wfp_food_index.parquet")

    # Merge WFP index with proxies
    panel = wfp_index.merge(proxies, on="year_month", how="outer")
    panel = panel.sort_values("year_month").reset_index(drop=True)

    # Forward-fill gaps up to 3 months (handles occasional missing months)
    for col in ["ghsusd"]:
        if col in panel.columns:
            panel[col] = panel[col].ffill(limit=3)

    # Save panel
    panel.to_parquet(_PROC_DIR / "historical_panel.parquet", index=False)
    panel.to_csv(_PROC_DIR / "historical_panel.csv", index=False)
    logger.info(
        "Saved → data/processed/historical_panel.parquet  (%d rows)", len(panel)
    )

    if plot:
        print("\n" + "=" * 70)
        print("HISTORICAL PANEL SUMMARY")
        print("=" * 70)
        print(f"  Date range: {panel['year_month'].min()} → {panel['year_month'].max()}")
        print(f"  Total months: {len(panel)}")
        print()
        for col in ["wfp_food_index", "ghsusd", "brent"]:
            if col in panel.columns:
                s = panel[col].dropna()
                if len(s):
                    print(f"  {col:18s}  {len(s):4d} obs  "
                          f"min={s.min():.2f}  mean={s.mean():.2f}  max={s.max():.2f}")
        print()
        overlap = panel.dropna(subset=["wfp_food_index", "ghsusd", "brent"])
        print(f"  Months with ALL series present: {len(overlap)}")
        print(f"  ({overlap['year_month'].min()} → {overlap['year_month'].max()})")

    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build historical merged panel")
    parser.add_argument("--plot", action="store_true", help="Print stats table")
    args = parser.parse_args()
    build(plot=args.plot)
