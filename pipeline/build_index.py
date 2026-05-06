"""
pipeline/build_index.py
=======================
Constructs the Informal Market Price Index (IMPI) from scraped price data.

Algorithm
---------
1. Load all CSVs from data/raw/ (or a specific file).
2. Enrich with resolved_category + subcategory via commodity_map.
3. Bin scraped_at timestamps into ISO week buckets.
4. For each (week, resolved_category): compute MEDIAN price_ghc.
5. Normalise each category series to base period (default: first available week
   or IMPI_BASE_PERIOD from .env).
6. Compute overall IMPI as weighted average of normalised category indices,
   using weights from config/cpi_basket.yaml.
7. Save weekly series to data/processed/impi_weekly.parquet.

Usage
-----
    python -m pipeline.build_index                          # process all raw CSVs
    python -m pipeline.build_index --file data/raw/x.csv   # specific file
    python -m pipeline.build_index --base-period 2026-W18  # override base week
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

# Resolve project root regardless of cwd
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.commodity_map import enrich_dataframe  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")


def _load_basket_weights() -> dict[str, float]:
    """Return {resolved_category: weight} dict from cpi_basket.yaml."""
    path = ROOT / "config" / "cpi_basket.yaml"
    with open(path, encoding="utf-8") as fh:
        basket = yaml.safe_load(fh)["basket"]
    return {cat: data["weight"] for cat, data in basket.items()}


def _load_raw_data(file: Path | None) -> pd.DataFrame:
    """Load one CSV or concatenate all CSVs in data/raw/."""
    raw_dir = ROOT / "data" / "raw"
    if file:
        paths = [file]
    else:
        paths = sorted(raw_dir.glob("scraped_*.csv"))

    if not paths:
        raise FileNotFoundError(f"No scraped CSV files found in {raw_dir}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        logger.info("Loaded %d rows from %s", len(df), p.name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total rows loaded: %d", len(combined))
    return combined


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing prices; parse timestamps; deduplicate."""
    df = df.copy()
    df["price_ghc"] = pd.to_numeric(df["price_ghc"], errors="coerce")
    df = df.dropna(subset=["price_ghc", "product_name"])
    df = df[df["price_ghc"] > 0]

    # Parse scraped_at
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce")
    df = df.dropna(subset=["scraped_at"])

    # ISO week bucket  e.g. "2026-W18"
    df["week"] = df["scraped_at"].dt.strftime("%G-W%V")

    # Drop duplicates: same product from same source in same week → keep lowest price
    # (conservative — avoids double-counting repriced variants)
    df = (
        df.sort_values("price_ghc")
        .drop_duplicates(subset=["source", "product_name", "week"], keep="first")
    )
    logger.info("After cleaning + dedup: %d rows", len(df))
    return df


def _compute_weekly_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with (week, resolved_category, median_price, n_observations)."""
    grouped = (
        df.groupby(["week", "resolved_category"])["price_ghc"]
        .agg(median_price="median", n_obs="count")
        .reset_index()
    )
    return grouped


def _normalise_to_base(
    weekly: pd.DataFrame,
    base_period: str | None,
    min_obs: int,
) -> pd.DataFrame:
    """
    For each resolved_category, divide all median_price values by the base-period
    median to produce a price relative (index = 100 at base period).

    If base_period is a YYYY-MM date, it's converted to the nearest ISO week.
    """
    weekly = weekly.copy()
    weekly = weekly[weekly["n_obs"] >= min_obs]

    # Determine base week per category
    all_weeks = sorted(weekly["week"].unique())
    if not all_weeks:
        raise ValueError("No data remaining after min_obs filter.")

    if base_period:
        # Convert YYYY-MM → YYYY-WNN (first week of that month)
        if len(base_period) == 7 and "-" in base_period and "W" not in base_period:
            dt = datetime.strptime(base_period + "-01", "%Y-%m-%d")
            base_week = dt.strftime("%G-W%V")
        else:
            base_week = base_period
        logger.info("Base period: %s", base_week)
    else:
        base_week = all_weeks[0]
        logger.info("Base period (auto — first available week): %s", base_week)

    rows = []
    for cat, grp in weekly.groupby("resolved_category"):
        grp = grp.sort_values("week")
        base_row = grp[grp["week"] == base_week]
        if base_row.empty:
            # Fall back to earliest available week for this category
            base_price = grp.iloc[0]["median_price"]
            used_base = grp.iloc[0]["week"]
            logger.warning(
                "[%s] base week %s not found — using %s as base", cat, base_week, used_base
            )
        else:
            base_price = base_row.iloc[0]["median_price"]
            used_base = base_week

        if base_price == 0:
            logger.warning("[%s] base price is 0 — skipping normalisation", cat)
            continue

        grp = grp.copy()
        grp["index_value"] = (grp["median_price"] / base_price) * 100
        grp["base_week"] = used_base
        rows.append(grp)

    if not rows:
        raise ValueError("No category indices could be computed.")

    return pd.concat(rows, ignore_index=True)


def _compute_impi(normalised: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """
    Aggregate normalised category indices into a single IMPI per week.
    Uses basket weights; re-normalises if not all categories are present.
    """
    rows = []
    for week, grp in normalised.groupby("week"):
        grp_cats = set(grp["resolved_category"].unique())
        available_weights = {
            cat: w for cat, w in weights.items() if cat in grp_cats
        }
        total_weight = sum(available_weights.values())
        if total_weight == 0:
            continue

        impi = sum(
            grp.loc[grp["resolved_category"] == cat, "index_value"].iloc[0]
            * (w / total_weight)
            for cat, w in available_weights.items()
            if not grp.loc[grp["resolved_category"] == cat, "index_value"].empty
        )
        rows.append({
            "week": week,
            "impi": round(impi, 4),
            "categories_included": len(available_weights),
            "weight_coverage_pct": round(total_weight, 2),
        })

    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def build_index(
    file: Path | None = None,
    base_period: str | None = None,
    min_obs: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load → clean → enrich → weekly medians → normalise → IMPI.

    Returns
    -------
    (category_index, impi_series)
        category_index : weekly index per resolved_category (long format)
        impi_series    : weekly IMPI values
    """
    import os
    env_base = os.getenv("IMPI_BASE_PERIOD")
    if base_period is None and env_base:
        base_period = env_base

    weights = _load_basket_weights()

    raw = _load_raw_data(file)
    cleaned = _clean(raw)
    enriched = enrich_dataframe(cleaned)

    logger.info("Category distribution after mapping:")
    for cat, cnt in enriched["resolved_category"].value_counts().items():
        logger.info("  %-30s %d", cat, cnt)

    weekly = _compute_weekly_medians(enriched)
    normalised = _normalise_to_base(weekly, base_period, min_obs)
    impi = _compute_impi(normalised, weights)

    # Save outputs
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    normalised.to_parquet(out_dir / "category_index_weekly.parquet", index=False)
    impi.to_parquet(out_dir / "impi_weekly.parquet", index=False)

    normalised.to_csv(out_dir / "category_index_weekly.csv", index=False)
    impi.to_csv(out_dir / "impi_weekly.csv", index=False)

    logger.info("Saved category index → data/processed/category_index_weekly.parquet")
    logger.info("Saved IMPI series   → data/processed/impi_weekly.parquet")

    return normalised, impi


def _print_report(normalised: pd.DataFrame, impi: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("CATEGORY INDEX (latest week per category)")
    print("=" * 60)
    latest = (
        normalised.sort_values("week")
        .groupby("resolved_category")
        .last()
        .reset_index()
        [["resolved_category", "week", "median_price", "index_value", "n_obs"]]
    )
    print(latest.to_string(index=False))

    print("\n" + "=" * 60)
    print("IMPI WEEKLY SERIES")
    print("=" * 60)
    print(impi.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build IMPI from scraped price data")
    parser.add_argument("--file", type=Path, default=None, help="Specific CSV to process")
    parser.add_argument("--base-period", default=None, help="Base period e.g. 2026-01 or 2026-W04")
    parser.add_argument("--min-obs", type=int, default=3, help="Min observations per category per week")
    args = parser.parse_args()

    normalised, impi = build_index(
        file=args.file,
        base_period=args.base_period,
        min_obs=args.min_obs,
    )
    _print_report(normalised, impi)
