"""
URA Private Residential Property Transactions API client.

Endpoints (confirmed from URA Data Service API Manual):
  Token : GET https://www.ura.gov.sg/uraDataService/insertNewToken.action
          Header: AccessKey
  Data  : GET https://www.ura.gov.sg/uraDataService/invokeUraDS
          Params: service=PMI_Resi_Transaction, batch=1-4
          Headers: AccessKey, Token

The token is valid for the calendar day (SGT). Batches 1-4 cover
the most recent ~3 years of quarterly data.
"""

import logging
import os
import time
from datetime import date
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_TOKEN_URL = os.getenv(
    "URA_TOKEN_URL",
    "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1",
)
_DATA_URL = os.getenv(
    "URA_DATA_URL",
    "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1",
)
_TOKEN_TTL = 20 * 3600  # refresh before 24-hour expiry


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Content-Type": "application/json"})
    return session


class URAClient:
    """Thread-safe URA Data Service client with automatic daily token refresh."""

    def __init__(self, access_key: str | None = None) -> None:
        self._key = access_key or os.environ["URA_ACCESS_KEY"]
        self._token: str | None = None
        self._token_at: float = 0.0
        self._session = _make_session()

    # ------------------------------------------------------------------
    # Token
    # ------------------------------------------------------------------
    def _token_fresh(self) -> bool:
        return self._token is not None and time.time() - self._token_at < _TOKEN_TTL

    def _refresh_token(self) -> None:
        resp = self._session.get(
            _TOKEN_URL,
            params={"service": "Token"},
            headers={"AccessKey": self._key},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("Status") != "Success":
            raise RuntimeError(
                f"Token generation failed: {body.get('Message', body)}. "
                "Ensure URA_ACCESS_KEY is valid and requests originate from a server IP."
            )
        self._token = body["Result"]
        self._token_at = time.time()
        log.info("URA token refreshed.")

    def token(self) -> str:
        if not self._token_fresh():
            self._refresh_token()
        return self._token  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _fetch_batch(self, batch: int) -> list[dict]:
        resp = self._session.get(
            _DATA_URL,
            params={"service": "PMI_Resi_Transaction", "batch": batch},
            headers={"AccessKey": self._key, "Token": self.token()},
            timeout=90,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("Status") != "Success":
            raise RuntimeError(f"Batch {batch} error: {body.get('Message')}")
        return body.get("Result", [])

    def fetch_ec_transactions(self) -> list[dict]:
        """
        Pull all 4 batches, filter to Executive Condominiums, and return
        a flat list of transaction dicts ready for DB insertion.
        """
        rows: list[dict] = []
        for batch in range(1, 5):
            log.info("Fetching URA batch %d/4 …", batch)
            projects = self._fetch_batch(batch)
            for proj in projects:
                for txn in proj.get("transaction", []):
                    if "executive condominium" not in (txn.get("propertyType") or "").lower():
                        continue
                    rows.append({
                        "project_name": proj.get("project", ""),
                        "street": proj.get("street", ""),
                        "x_coord": _to_float(proj.get("x")),
                        "y_coord": _to_float(proj.get("y")),
                        "market_segment": proj.get("marketSegment", ""),
                        "district": _to_int(txn.get("district")),
                        "area_sqm": _to_float(txn.get("area")),
                        "floor_range": txn.get("floorRange", ""),
                        "no_of_units": _to_int(txn.get("noOfUnits"), 1),
                        "contract_date": parse_contract_date(txn.get("contractDate", "")),
                        "type_of_sale": _to_int(txn.get("typeOfSale")),
                        "price": _to_int(txn.get("price")),
                        "nett_price": _to_int(txn.get("nettPrice")),
                        "property_type": txn.get("propertyType", ""),
                        "tenure": txn.get("tenure", ""),
                    })
        log.info("Fetched %d EC rows across 4 batches.", len(rows))
        return rows

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def parse_contract_date(mmyy: str) -> date | None:
    """'0124' → date(2024, 1, 1)"""
    try:
        m, y = int(mmyy[:2]), int(mmyy[2:])
        return date(2000 + y if y < 100 else y, m, 1)
    except Exception:
        return None


def _to_float(v: Any, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v: Any, default=None):
    try:
        return int(float(str(v)))
    except (TypeError, ValueError):
        return default
