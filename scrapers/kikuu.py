"""
KiKUU Ghana scraper — kikuu.com.gh

KiKUU is a Ghanaian e-commerce platform with prices in GHS.
Products are rendered client-side; we scrape the listing HTML.
"""

import re
from typing import Optional

from bs4 import BeautifulSoup

from .base import BaseScraper


def _parse_price(raw: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.]", "", re.sub(r"(GH[S₵]|₵|GHC)", "", raw).replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_unit(title: str) -> str:
    match = re.search(
        r"(\d+\s*(?:kg|g|ml|l|pcs|pack|bag|set|pair|dozen|piece|roll)s?)",
        title, re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


class KiKUUScraper(BaseScraper):
    """Scraper for KiKUU Ghana product listings."""

    def scrape(self, url: str) -> list[dict]:
        resp = self._get(url)
        if resp is None:
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        records: list[dict] = []

        # KiKUU product cards — try common selectors
        items = soup.select(".product-item, .goods-item, [class*='product'], [class*='goods']")

        for item in items:
            # Title
            title_el = item.select_one(
                ".product-name, .goods-name, [class*='title'], [class*='name']"
            )
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            if not title:
                continue

            # Price
            price_el = item.select_one(
                ".price, .product-price, [class*='price']"
            )
            raw_price = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(raw_price)
            if price is None or price <= 0:
                continue

            # URL
            link_el = item.select_one("a[href]")
            product_url = link_el["href"] if link_el else url
            if product_url.startswith("/"):
                product_url = "https://www.kikuu.com.gh" + product_url

            records.append(
                self._record(
                    product_name=title,
                    price_ghc=price,
                    unit=_extract_unit(title),
                    currency="GHS",
                    url=product_url,
                )
            )

        self.logger.info("[KiKUU] %d records scraped from %s", len(records), url)
        return records
