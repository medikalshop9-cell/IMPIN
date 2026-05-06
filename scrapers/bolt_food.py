"""
Bolt Food Ghana scraper — food.bolt.eu/en/

Bolt Food requires JavaScript rendering (React SPA).
Uses Playwright (async) to load the page, wait for product cards, then parse.

Install Playwright browsers once:
    playwright install chromium
"""

import re
from typing import Optional

from .base import BaseScraper


def _parse_price(raw: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.]", "", re.sub(r"(GH[S₵]|₵|GHC)", "", raw).replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_unit(title: str) -> str:
    match = re.search(
        r"(\d+\s*(?:kg|g|ml|l|pcs|pack|bag|set|pair|dozen|piece|roll|tin)s?)",
        title, re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


class BoltFoodScraper(BaseScraper):
    """Playwright-based scraper for Bolt Food Ghana."""

    def scrape(self, url: str) -> list[dict]:
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except ImportError:
            self.logger.error(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
            return []

        records: list[dict] = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()

            try:
                page.goto(url, timeout=45_000, wait_until="domcontentloaded")
                # Wait for restaurant/product cards to load
                page.wait_for_selector(
                    "[class*='restaurant'], [class*='product'], [class*='item'], [class*='card']",
                    timeout=20_000,
                )
            except PWTimeout:
                self.logger.warning("[BoltFood] Page load timeout: %s", url)
                browser.close()
                return []

            html = page.content()
            browser.close()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Generic card selectors — Bolt Food uses dynamic class names
        cards = soup.select(
            "[class*='restaurant'], [class*='product-card'], "
            "[class*='menu-item'], [class*='FoodItem']"
        )

        for card in cards:
            title_el = card.select_one("[class*='name'], [class*='title'], h2, h3, h4")
            price_el = card.select_one("[class*='price'], [class*='Price']")
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            raw_price = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(raw_price)
            if price is None or price <= 0:
                continue

            records.append(
                self._record(
                    product_name=title,
                    price_ghc=price,
                    unit=_extract_unit(title),
                    currency="GHS",
                    url=url,
                )
            )

        self.logger.info("[BoltFood] %d records scraped from %s", len(records), url)
        return records
