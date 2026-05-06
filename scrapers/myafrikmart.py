"""
scrapers/myafrikmart.py
========================
Scraper for MyAfrikMart (https://myafrikmart.com) — WooCommerce-based
Ghanaian marketplace. Inherits the generic WooCommerce HTML scraper.
"""

from .woocommerce import WooCommerceScraper


class MyAfrikMartScraper(WooCommerceScraper):
    """MyAfrikMart runs on WooCommerce — inherits standard HTML scraping."""
    pass
