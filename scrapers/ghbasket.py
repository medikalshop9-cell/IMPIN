"""
scrapers/ghbasket.py
====================
Scraper for GhBasket (https://ghbasket.com) — a WooCommerce-based Ghanaian
online grocery store. Uses the generic WooCommerce HTML scraper.
"""

from .woocommerce import WooCommerceScraper


class GhBasketScraper(WooCommerceScraper):
    """GhBasket runs on WooCommerce — inherits standard HTML scraping."""
    pass
