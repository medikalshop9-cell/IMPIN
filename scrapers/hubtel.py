"""
Hubtel scraper — hubtel.com/app

Hubtel is a super-app with merchant listings rendered in a mobile WebView.
Direct HTML scraping is not reliable; this stub logs the limitation and
returns an empty list. Replace with an API-based approach if Hubtel
exposes a merchant product API or partner data feed.
"""

from .base import BaseScraper


class HubtelScraper(BaseScraper):
    """
    Placeholder scraper for Hubtel.

    Hubtel's merchant product listings are app-gated and not accessible
    via public HTTP. This scraper is a stub that logs and returns [].

    To activate: replace `scrape()` with an implementation that uses
    the Hubtel Partner API (requires merchant credentials).
    """

    def scrape(self, url: str) -> list[dict]:
        self.logger.info(
            "[Hubtel] Skipped — app-gated. Implement Partner API integration to activate."
        )
        return []
