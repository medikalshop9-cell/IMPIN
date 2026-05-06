"""
BaseScraper — abstract base class for all IMPIN scrapers.

Every scraper subclass must implement `scrape(url: str) -> list[dict]`.
Common behaviour provided here:
  - Requests session with exponential back-off retry
  - Randomised inter-request delay (reads SCRAPER_MIN/MAX_DELAY_S from .env)
  - Structured logging
  - Standard record factory (_record)
"""

import os
import random
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


class BaseScraper(ABC):
    """Abstract base for all IMPIN scrapers."""

    def __init__(self, source: str, cpi_category: str) -> None:
        self.source = source
        self.cpi_category = cpi_category
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = self._build_session()

    # ── HTTP session ──────────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=int(os.getenv("SCRAPER_MAX_RETRIES", 3)),
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": random.choice(_USER_AGENTS)})
        return session

    def _delay(self) -> None:
        """Sleep a random interval between requests to avoid rate-limiting."""
        lo = float(os.getenv("SCRAPER_MIN_DELAY_S", 2))
        hi = float(os.getenv("SCRAPER_MAX_DELAY_S", 6))
        time.sleep(random.uniform(lo, hi))

    def _get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """GET a URL with retry; returns None on failure."""
        # Rotate user-agent on every request
        self.session.headers.update({"User-Agent": random.choice(_USER_AGENTS)})
        self._delay()
        try:
            resp = self.session.get(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            self.logger.warning("GET failed [%s]: %s", url, exc)
            return None

    # ── Record factory ────────────────────────────────────────────────────────

    def _record(
        self,
        product_name: str,
        price_ghc: Optional[float],
        unit: str,
        currency: str,
        url: str,
    ) -> dict:
        """Return a standardised price record dict."""
        return {
            "source": self.source,
            "cpi_category": self.cpi_category,
            "product_name": product_name.strip(),
            "price_ghc": price_ghc,
            "unit": unit.strip() if unit else "",
            "currency": currency,
            "url": url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def scrape(self, url: str) -> list[dict]:
        """
        Scrape a single URL and return a list of price records.

        Each record must match the schema produced by `_record()`.
        """
