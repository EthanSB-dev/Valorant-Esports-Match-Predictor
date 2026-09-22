"""
PandaScore API client.

Centralizes authentication, retry/backoff, and pagination logic so that
every extraction script shares one consistent, well-behaved way of talking
to the API instead of re-implementing HTTP handling everywhere.
"""
import os
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pandascore.co"


class PandaScoreClient:
    """
    Thin wrapper around the PandaScore REST API.

    Why a class instead of loose functions:
    - Holds the API key and a requests.Session once, instead of re-reading
      the environment variable and re-building headers on every call.
    - A Session reuses the underlying TCP connection across requests,
      which is faster and more polite to PandaScore's servers than opening
      a brand new connection for every single call.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 5):
        self.api_key = api_key or os.environ["PANDASCORE_API_KEY"]
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _request(self, method: str, path: str, params: Optional[dict] = None) -> requests.Response:
        """
        Makes a single HTTP request with retry/backoff logic.

        Why retries matter here: networks and third-party APIs are
        unreliable. A request can fail because of a dropped connection,
        a temporary server hiccup, or a rate limit (HTTP 429). Instead of
        crashing the whole pipeline on one bad request, we wait and try
        again a few times first.
        """
        url = f"{BASE_URL}{path}"
        attempt = 0

        while True:
            attempt += 1
            response = self.session.request(method, url, params=params, timeout=15)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else 2 ** attempt
                logger.warning(f"Rate limited (429). Waiting {wait}s before retry {attempt}/{self.max_retries}.")
                time.sleep(wait)
            elif response.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(f"Server error ({response.status_code}). Waiting {wait}s before retry {attempt}/{self.max_retries}.")
                time.sleep(wait)
            else:
                response.raise_for_status()
                return response

            if attempt >= self.max_retries:
                response.raise_for_status()
                return response

    def get_matches_page(self, page: int = 1, per_page: int = 50, sort: str = "-end_at",
                          filters: Optional[dict] = None) -> requests.Response:
        """
        Fetch one page of past VALORANT matches.
        """
        params = {"page": page, "per_page": per_page, "sort": sort}
        if filters:
            params.update(filters)
        return self._request("GET", "/valorant/matches/past", params=params)

    def get_upcoming_matches_page(self, page: int = 1, per_page: int = 50,
                                   filters: Optional[dict] = None) -> requests.Response:
        """
        Fetch one page of upcoming (not yet played) VALORANT matches.
        Separate endpoint from get_matches_page — PandaScore splits past
        and upcoming matches into distinct endpoints rather than one
        endpoint with a status filter.
        """
        params = {"page": page, "per_page": per_page}
        if filters:
            params.update(filters)
        return self._request("GET", "/valorant/matches/upcoming", params=params)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = PandaScoreClient()
    resp = client.get_matches_page(page=1, per_page=5)
    matches = resp.json()
    rate_remaining = resp.headers.get("X-Ratelimit-Remaining")
    print(f"Fetched {len(matches)} matches. Rate limit remaining: {rate_remaining}")
    if matches:
        print("Keys on first match:", list(matches[0].keys()))