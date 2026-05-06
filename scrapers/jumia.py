"""
scrapers/jumia.py
=================
Scraper for Jumia Ghana (https://www.jumia.com.gh) — uses Playwright for
JavaScript-rendered product listings. Chromium headless browser required.

Install:
    pip install playwright
    python -m playwright install chromium
"""

import re
import logging
import time
import random
from datetime import datetime, timezone
from typing import Optional

_log = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    _log.warning("playwright not installed — JumiaScraper will return empty results.")


def _parse_price(raw: str) -> Optional[float]:
    """Strip currency symbols and parse GHS price."""
    cleaned = re.sub(r"[^\d.]", "", re.sub(r"(GH[S₵]|₵)", "", raw).replace(",", ""))
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _extract_unit(title: str) -> str:
    match = re.search(
        r"(\d+\s*(?:kg|g|ml|l|cl|lb|oz|pcs|pack|bag|bundle|sachet|tin|can|box|roll|piece|dozen)s?)",
        title,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


class JumiaScraper:
    """
    Scraper for Jumia Ghana product listing pages.
    Uses Playwright (headless Chromium) to render JS content.

    Pagination: clicks the 'Next' button (or follows ?page=N URL param).
    """

    # Jumia Ghana product card selectors (verified against live HTML, 2025)
    _PRODUCT_CARDS      = "article.prd"
    _PRODUCT_NAME       = "h3.name"
    _PRODUCT_PRICE      = "div.prc"
    _PRODUCT_LINK       = "a.core"
    _NEXT_PAGE_SELECTOR = "a[aria-label='Next Page']"

    def __init__(self, source: str = "Jumia", cpi_category: str = "Food & Beverages") -> None:
        self.source = source
        self.cpi_category = cpi_category
        self.logger = logging.getLogger(self.__class__.__name__)

    def _record(self, *, product_name: str, price_ghc: float,
                unit: str, currency: str, url: str) -> dict:
        return {
            "source":       self.source,
            "product_name": product_name.strip(),
            "price_ghc":    round(price_ghc, 2),
            "unit":         unit,
            "currency":     currency,
            "url":          url,
            "cpi_category": self.cpi_category,
            "scraped_at":   datetime.now(timezone.utc).isoformat(),
        }

    def _scrape_page(self, page, url: str) -> tuple[list[dict], Optional[str]]:
        """Load a page and extract product records + next-page URL."""
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        # Wait for product cards to appear (up to 15 s)
        try:
            page.wait_for_selector(self._PRODUCT_CARDS, timeout=15_000)
        except PlaywrightTimeout:
            self.logger.warning("No product cards found at %s", url)
            return [], None

        # Random polite delay
        time.sleep(random.uniform(1.5, 3.0))

        cards = page.query_selector_all(self._PRODUCT_CARDS)
        records: list[dict] = []
        for card in cards:
            name_el  = card.query_selector(self._PRODUCT_NAME)
            price_el = card.query_selector(self._PRODUCT_PRICE)
            link_el  = card.query_selector(self._PRODUCT_LINK)

            if not name_el or not price_el:
                continue
            title     = name_el.inner_text().strip()
            raw_price = price_el.inner_text().strip()
            price     = _parse_price(raw_price)
            if price is None or price <= 0:
                continue

            product_url = link_el.get_attribute("href") if link_el else url
            if product_url and not product_url.startswith("http"):
                product_url = "https://www.jumia.com.gh" + product_url

            records.append(self._record(
                product_name=title,
                price_ghc=price,
                unit=_extract_unit(title),
                currency="GHS",
                url=product_url or url,
            ))

        # Next page URL
        next_url: Optional[str] = None
        next_el = page.query_selector(self._NEXT_PAGE_SELECTOR)
        if next_el:
            href = next_el.get_attribute("href")
            if href:
                next_url = href if href.startswith("http") else "https://www.jumia.com.gh" + href

        return records, next_url

    def scrape(self, url: str, max_pages: int = 10) -> list[dict]:
        """
        Scrape up to `max_pages` pages from `url`.

        Parameters
        ----------
        url       : Jumia category listing URL (e.g. jumia.com.gh/groceries/)
        max_pages : Safety cap on pagination depth

        Returns
        -------
        List of standardised product dicts.
        """
        if not _PLAYWRIGHT_AVAILABLE:
            self.logger.error("playwright not available — cannot scrape Jumia")
            return []

        all_records: list[dict] = []
        current_url: Optional[str] = url

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
            )
            page = context.new_page()

            for page_num in range(1, max_pages + 1):
                if not current_url:
                    break
                self.logger.info("Jumia page %d: %s", page_num, current_url)
                records, current_url = self._scrape_page(page, current_url)
                all_records.extend(records)
                self.logger.info("  → %d products (total so far: %d)",
                                 len(records), len(all_records))
                if not records:
                    break

            context.close()
            browser.close()

        self.logger.info("Jumia scrape complete: %d products from %s", len(all_records), url)
        return all_records
