"""
run_all.py — IMPIN scraper orchestrator

Reads url.csv, dispatches the correct scraper for each source,
aggregates all price records, and saves them to:
    data/raw/scraped_YYYYMMDD_HHMMSS.csv

Usage:
    python scrapers/run_all.py
    python scrapers/run_all.py --sources Konzoom "Big Samps Market"
    python scrapers/run_all.py --dry-run
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Allow running from project root or from scrapers/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from scrapers.shopify import ShopifyScraper
from scrapers.woocommerce import WooCommerceScraper
from scrapers.kikuu import KiKUUScraper
from scrapers.shopnaw import ShopnawScraper
from scrapers.bolt_food import BoltFoodScraper
from scrapers.hubtel import HubtelScraper
from scrapers.ghbasket import GhBasketScraper
from scrapers.myafrikmart import MyAfrikMartScraper
from scrapers.jumia import JumiaScraper
from scrapers.comilmart import ComilmartScraper

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_all")

# ── Source → scraper class mapping ────────────────────────────────────────────
# Konzoom and Makola run on Shopify / WooCommerce; select by domain.

def _scraper_for(source: str, cpi_category: str):
    """Return an instantiated scraper for the given source name."""
    source_lower = source.lower().strip()

    if source_lower == "konzoom":
        return ShopifyScraper(source, cpi_category)

    if source_lower in ("big samps market", "makola stores"):
        return WooCommerceScraper(source, cpi_category)

    if source_lower == "kikuu":
        return KiKUUScraper(source, cpi_category)

    if source_lower == "shopnaw":
        return ShopnawScraper(source, cpi_category)

    if source_lower == "bolt food":
        return BoltFoodScraper(source, cpi_category)

    if source_lower == "hubtel":
        return HubtelScraper(source, cpi_category)

    if source_lower == "ghbasket":
        return GhBasketScraper(source, cpi_category)

    if source_lower == "myafrikmart":
        return MyAfrikMartScraper(source, cpi_category)

    if source_lower == "jumia":
        return JumiaScraper(source, cpi_category)

    if source_lower == "comilmart":
        return ComilmartScraper(source, cpi_category)

    # Generic fallback — try WooCommerce HTML scraping
    logger.warning("No dedicated scraper for '%s' — using WooCommerce fallback.", source)
    return WooCommerceScraper(source, cpi_category)


# ── URL CSV loader ────────────────────────────────────────────────────────────

def load_urls(csv_path: Path) -> list[dict]:
    """
    Load url.csv, skip comment lines (starting with #), return list of dicts
    with keys: source, cpi_category, url.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip inline comment rows (cells may be None if comment line has no commas)
            source = (row.get("source") or "").strip()
            url = (row.get("url") or "").strip()
            cpi_category = (row.get("cpi_category") or "").strip()
            if not source or source.startswith("#") or not url or url.startswith("#"):
                continue
            rows.append({
                "source": source,
                "cpi_category": cpi_category,
                "url": url,
            })
    return rows


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run(sources_filter: list[str] | None = None, dry_run: bool = False) -> Path | None:
    url_csv = ROOT / "url.csv"
    if not url_csv.exists():
        logger.error("url.csv not found at %s", url_csv)
        sys.exit(1)

    urls = load_urls(url_csv)
    logger.info("Loaded %d URLs from url.csv", len(urls))

    if sources_filter:
        filter_lower = [s.lower() for s in sources_filter]
        urls = [u for u in urls if u["source"].lower() in filter_lower]
        logger.info("Filtered to %d URLs for sources: %s", len(urls), sources_filter)

    if dry_run:
        logger.info("[DRY RUN] Would scrape %d URLs:", len(urls))
        for u in urls:
            logger.info("  %-25s  %s", u["source"], u["url"])
        return None

    all_records: list[dict] = []

    for entry in urls:
        source = entry["source"]
        cpi_category = entry["cpi_category"]
        url = entry["url"]

        scraper = _scraper_for(source, cpi_category)
        logger.info("Scraping %-25s → %s", source, url)

        try:
            records = scraper.scrape(url)
        except Exception as exc:
            logger.error("Unhandled error scraping %s [%s]: %s", source, url, exc)
            records = []

        logger.info("  ✓ %d records from %s", len(records), url)
        all_records.extend(records)

    if not all_records:
        logger.warning("No records collected — check scraper output above.")
        return None

    # ── Save output ───────────────────────────────────────────────────────────
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"scraped_{timestamp}.csv"

    df = pd.DataFrame(all_records)
    # Enforce column order
    col_order = ["source", "cpi_category", "product_name", "price_ghc", "unit", "currency", "url", "scraped_at"]
    df = df[[c for c in col_order if c in df.columns]]

    df.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(
        "Saved %d records to %s  (sources: %s)",
        len(df),
        out_path.relative_to(ROOT),
        ", ".join(df["source"].unique()),
    )

    # Print summary table
    summary = df.groupby("source").size().reset_index(name="records")
    logger.info("\n%s", summary.to_string(index=False))

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMPIN scraper orchestrator")
    parser.add_argument(
        "--sources", nargs="+", metavar="SOURCE",
        help="Only scrape these source(s). E.g. --sources Konzoom 'Big Samps Market'"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List URLs that would be scraped without making any requests."
    )
    args = parser.parse_args()
    run(sources_filter=args.sources, dry_run=args.dry_run)
