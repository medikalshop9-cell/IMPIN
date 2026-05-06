"""
Shopnaw (Shop Ghana Now) scraper — shopghananow.com

General Ghanaian e-commerce site. Scrapes product listings from the homepage
and category pages using BeautifulSoup.
"""

import re
from typing import Optional
from urllib.parse import urljoin

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
        r"(\d+\s*(?:kg|g|ml|l|pcs|pack|bag|set|pair|dozen|piece|roll|tin|can|box)s?)",
        title, re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


class ShopnawScraper(BaseScraper):
    """Scraper for shopghananow.com product listings."""

    _BASE = "https://shopghananow.com"

    def scrape(self, url: str) -> list[dict]:
        resp = self._get(url)
        if resp is None:
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        records: list[dict] = []

        # Try WooCommerce selectors first (site may run WC)
        items = soup.select("li.product, div.product, .product-item, [class*='product']")

        for item in items:
            # Title
            title_el = item.select_one(
                "h2, h3, .woocommerce-loop-product__title, [class*='title'], [class*='name']"
            )
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            if not title:
                continue

            # Price — prefer sale price (ins) over regular
            price_el = item.select_one("ins .woocommerce-Price-amount bdi") or \
                        item.select_one(".woocommerce-Price-amount bdi") or \
                        item.select_one(".price, [class*='price']")
            raw_price = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(raw_price)
            if price is None or price <= 0:
                continue

            link_el = item.select_one("a[href]")
            product_url = urljoin(self._BASE, link_el["href"]) if link_el else url

            records.append(
                self._record(
                    product_name=title,
                    price_ghc=price,
                    unit=_extract_unit(title),
                    currency="GHS",
                    url=product_url,
                )
            )

        self.logger.info("[Shopnaw] %d records scraped from %s", len(records), url)
        return records
