"""
WooCommerce scraper — works with any WooCommerce product-category page.

Scrapes HTML product listings (title + price) with automatic pagination.
Handles both standard and block-based WooCommerce themes.
"""

import re
from typing import Optional

from bs4 import BeautifulSoup

from .base import BaseScraper


def _parse_price(raw: str) -> Optional[float]:
    """Strip currency symbols and parse a GHS price string to float."""
    # Remove GH₵, ₵, GHS, commas, whitespace
    cleaned = re.sub(r"[^\d.]", "", re.sub(r"(GH[S₵]|₵)", "", raw).replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_unit(title: str) -> str:
    match = re.search(
        r"(\d+\s*(?:kg|g|ml|l|cl|lb|oz|pcs|pack|bag|bundle|sachet|tin|can|box|roll|piece|dozen)s?)",
        title,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


class WooCommerceScraper(BaseScraper):
    """Generic scraper for WooCommerce product category pages."""

    # CSS selectors to try in order (covers several common WC themes)
    _TITLE_SELECTORS = [
        "h2.woocommerce-loop-product__title",
        "h3.woocommerce-loop-product__title",
        ".woocommerce-loop-product__title",
        ".product-title",
        "h2.title",
        "h3.title",
        ".title",
        "h2",
        "h3",
    ]
    _PRICE_SELECTORS = [
        "span.price ins span.woocommerce-Price-amount bdi",
        "span.price span.woocommerce-Price-amount bdi",
        "span.woocommerce-Price-amount bdi",
        ".price bdi",
        ".product-price .amount",
        ".price .amount",
        ".amount",
    ]

    def _parse_page(self, html: str, page_url: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        records: list[dict] = []

        # Each product is wrapped in a <li> or <div> with class "product"
        products = soup.select("li.product, div.product")
        if not products:
            # Fallback: any element with woocommerce-loop-product__title
            products = [
                el.find_parent()
                for el in soup.select(".woocommerce-loop-product__title")
                if el.find_parent()
            ]

        for item in products:
            # --- title ---
            title_el = None
            for sel in self._TITLE_SELECTORS:
                title_el = item.select_one(sel)
                if title_el:
                    break
            if not title_el:
                continue
            title = title_el.get_text(strip=True)

            # --- price ---
            price_el = None
            for sel in self._PRICE_SELECTORS:
                price_el = item.select_one(sel)
                if price_el:
                    break
            raw_price = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(raw_price)
            if price is None or price <= 0:
                continue

            # --- product URL ---
            link_el = item.select_one("a.woocommerce-LoopProduct-link, a[href]")
            product_url = link_el["href"] if link_el and link_el.get("href") else page_url

            records.append(
                self._record(
                    product_name=title,
                    price_ghc=price,
                    unit=_extract_unit(title),
                    currency="GHS",
                    url=product_url,
                )
            )

        return records

    def _next_page_url(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Return the URL of the next page, or None if on the last page."""
        next_link = soup.select_one("a.next.page-numbers, .woocommerce-pagination a.next")
        if next_link and next_link.get("href"):
            return next_link["href"]
        return None

    def scrape(self, url: str) -> list[dict]:
        records: list[dict] = []
        current_url: Optional[str] = url
        page = 1

        while current_url:
            resp = self._get(current_url)
            if resp is None:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            page_records = self._parse_page(resp.text, current_url)
            records.extend(page_records)

            self.logger.info(
                "[%s] page %d — %d records scraped so far",
                self.source, page, len(records),
            )

            # Stop paginating if this page had no products (selector mismatch / end of results)
            if not page_records:
                break

            current_url = self._next_page_url(soup, current_url)
            page += 1

        return records
