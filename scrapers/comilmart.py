"""
scrapers/comilmart.py
=====================
Scraper for Comilmart (https://comilmart.com) — WooCommerce-based Ghanaian
online store. Uses the generic WooCommerce HTML scraper.
"""

from .woocommerce import WooCommerceScraper


class ComilmartScraper(WooCommerceScraper):
    """Comilmart runs on WooCommerce — inherits standard HTML scraping."""
    pass
