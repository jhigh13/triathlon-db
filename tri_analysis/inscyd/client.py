"""
INSCYD external API client.

Wraps the read endpoints documented by INSCYD:

    GET  /api/external/athlete/all_data/    -> all tests (paginated)
    GET  /api/external/athlete/last_data/   -> latest test only

Auth is via the ``X-Api-Key`` header. Credentials come from ``tri_analysis.config``
(``INSCYD_API_KEY`` / ``INSCYD_HOST`` / ``INSCYD_BASE_URL``), set in ``.env``.

Network calls are made ONLY when you call a method (nothing happens at import or
construction beyond validating that credentials are present). Pulling the full
polynomial payloads is comparatively heavy, so fetch once and cache the result
(see ``tri_analysis.inscyd.storage``); use :meth:`verify_access` for a light
connectivity/auth check first.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .. import config

logger = logging.getLogger(__name__)


class InscydAPIError(RuntimeError):
    """Raised for INSCYD API auth, connectivity, or response errors."""


def _clean_params(**kwargs: Any) -> dict[str, Any]:
    """Drop None-valued query params."""
    return {k: v for k, v in kwargs.items() if v is not None}


class InscydClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        host: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or config.INSCYD_API_KEY
        base = base_url or config.INSCYD_BASE_URL
        if not base:
            host = host or config.INSCYD_HOST
            if host:
                base = f"https://{host}/api/external"
        self.base_url = (base or "").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()

        if not self.api_key:
            raise InscydAPIError(
                "INSCYD_API_KEY is not set. Add it to triathlon-db/.env "
                "(INSCYD_API_KEY=...) or pass api_key=."
            )
        if not self.base_url:
            raise InscydAPIError(
                "INSCYD host/base URL is not set. Add INSCYD_HOST=<host> to "
                "triathlon-db/.env (no scheme, e.g. app.inscyd.com) or pass host=/base_url=."
            )

    # ── internals ────────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key, "Accept": "application/json"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(
                    url, headers=self._headers(), params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:  # network/timeout
                last_exc = exc
                logger.warning("INSCYD request error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                time.sleep(min(2 ** attempt, 5))
                continue

            if resp.status_code in (401, 403):
                raise InscydAPIError(
                    f"INSCYD auth failed ({resp.status_code}). Check X-Api-Key. "
                    f"Response: {resp.text[:300]}"
                )
            if resp.status_code >= 500:
                last_exc = InscydAPIError(f"Server error {resp.status_code}: {resp.text[:300]}")
                logger.warning("INSCYD %d (attempt %d/%d)", resp.status_code, attempt, self.max_retries)
                time.sleep(min(2 ** attempt, 5))
                continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise InscydAPIError(f"INSCYD request failed: {exc} — {resp.text[:300]}") from exc

            try:
                return resp.json()
            except ValueError as exc:
                raise InscydAPIError(f"INSCYD returned non-JSON response: {resp.text[:300]}") from exc

        raise InscydAPIError(f"INSCYD request to {url} failed after {self.max_retries} attempts") from last_exc

    def _paginate(self, path: str, params: dict[str, Any], max_pages: int | None = None) -> list[dict]:
        """Collect all ``results`` across pages.

        We increment the ``page`` query param using ``total_pages`` rather than
        following the response's ``next`` URL, because ``next`` can carry INSCYD's
        internal host (e.g. 127.0.0.1) which would not be reachable.
        """
        first = self._get(path, params={**params, "page": 1})
        results = list(first.get("results", []))
        total_pages = int(first.get("total_pages") or 1)
        last_page = min(total_pages, max_pages) if max_pages else total_pages
        for page in range(2, last_page + 1):
            data = self._get(path, params={**params, "page": page})
            results.extend(data.get("results", []))
        return results

    # ── public API ───────────────────────────────────────────────────
    def get_all_data(
        self,
        *,
        user_id: int | None = None,
        user_email: str | None = None,
        sport_id: int | None = None,
        athlete_display_id: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        max_pages: int | None = None,
    ) -> list[dict]:
        """All tests matching the filters (raw API result dicts). Paginated."""
        params = _clean_params(
            user_id=user_id,
            user_email=user_email,
            sport_id=sport_id,
            athlete_display_id=athlete_display_id,
            start_date=start_date,
            end_date=end_date,
        )
        return self._paginate("athlete/all_data/", params, max_pages=max_pages)

    def get_last_data(
        self,
        *,
        user_id: int | None = None,
        user_email: str | None = None,
        sport_id: int | None = None,
        athlete_display_id: int | None = None,
    ) -> list[dict]:
        """Latest test(s) only (raw API result dicts)."""
        params = _clean_params(
            user_id=user_id,
            user_email=user_email,
            sport_id=sport_id,
            athlete_display_id=athlete_display_id,
        )
        data = self._get("athlete/last_data/", params=params)
        return list(data.get("results", []))

    def verify_access(
        self,
        *,
        user_id: int | None = None,
        user_email: str | None = None,
        sport_id: int | None = None,
        athlete_display_id: int | None = None,
    ) -> dict:
        """Light auth/connectivity check (single ``last_data`` request, no paging).

        Returns a small status dict. Raises :class:`InscydAPIError` on auth or
        connectivity failure. Use this before any bulk fetch.
        """
        params = _clean_params(
            user_id=user_id,
            user_email=user_email,
            sport_id=sport_id,
            athlete_display_id=athlete_display_id,
        )
        data = self._get("athlete/last_data/", params=params)
        results = data.get("results", []) or []
        sample = results[0] if results else {}
        return {
            "ok": True,
            "base_url": self.base_url,
            "count": data.get("count"),
            "results_returned": len(results),
            "sample_athlete_display_id": sample.get("athlete_display_id"),
            "sample_created_at": sample.get("created_at"),
        }
