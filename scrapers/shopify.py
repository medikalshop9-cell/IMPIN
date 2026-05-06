"""
Shopify scraper — works with any Shopify storefront.

Uses the public `/products.json` endpoint (no auth required).
Collection URLs like:
    https://konzoom.shop/collections/fruits-vegetables
are converted to:
    https://konzoom.shop/collections/fruits-vegetables/products.json?limit=250&page=N
"""

import re
from typing import Optional

from .base import BaseScraper


def _parse_price(raw: str) -> Optional[float]:
    """Extract a float from a Shopify price string (e.g. '45.00' or '1,200.00')."""
    cleaned = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_unit(title: str) -> str:
    """
    Best-effort unit extraction from product title.
    e.g. 'Rice 5kg' → '5kg',  'Tomatoes per basket' → 'per basket'
    """
    match = re.search(
        r"(\d+\s*(?:kg|g|ml|l|cl|lb|oz|pcs|pack|bag|bundle|sachet|tin|can|box|roll|piece|dozen)s?)",
        title,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


class ShopifyScraper(BaseScraper):
    """Generic scraper for Shopify stores using the products.json API."""

    def scrape(self, url: str) -> list[dict]:
        """
        Scrape all products from a Shopify collection URL.
        Handles pagination automatically.
        """
        # Convert collection page URL → products.json base
        base = url.rstrip("/")
        if not base.endswith("/products.json"):
            # Handle both collection and plain store URLs
            if "/collections/" in base:
                base = base + "/products.json"
            else:
                base = base + "/products.json"

        records: list[dict] = []
        page = 1

        while True:
            api_url = f"{base}?limit=250&page={page}"
            resp = self._get(api_url, headers={"Accept": "application/json"})
            if resp is None:
                break

            try:
                data = resp.json()
            except ValueError:
                self.logger.error("Non-JSON response from %s", api_url)
                break

            products = data.get("products", [])
            if not products:
                break  # no more pages

            for product in products:
                title: str = product.get("title", "")
                unit = _extract_unit(title)
                product_url = f"https://{resp.url.split('/')[2]}/products/{product.get('handle', '')}"

                for variant in product.get("variants", []):
                    raw_price = variant.get("price", "")
                    price = _parse_price(str(raw_price))
                    if price is None or price <= 0:
                        continue

                    variant_title = variant.get("title", "")
                    full_name = title if variant_title in ("Default Title", "") else f"{title} — {variant_title}"

                    records.append(
                        self._record(
                            product_name=full_name,
                            price_ghc=price,
                            unit=unit or variant_title,
                            currency="GHS",
                            url=product_url,
                        )
                    )

            self.logger.info(
                "[%s] page %d — %d products scraped so far",
                self.source, page, len(records),
            )
            page += 1

            # Shopify returns fewer than 250 on the last page
            if len(products) < 250:
                break

        return records
