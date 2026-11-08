"""Live Massive API client for 1-min OHLCV bars (docs/agent-docs/live-prediction-agent-plan.md
section 1.1 / step 4; corrected 2026-11-03 against the real API -- see that plan's section 6).

Thin wrapper around the official `massive` SDK (`pip install massive`, formerly `polygon-api-client`
-- Massive is "formerly Polygon.io" per its own PyPI description) rather than hand-rolled HTTP
calls. The original hand-rolled client guessed at the endpoint path/response shape and got a 404
in practice; the SDK's `RESTClient.get_aggs` is the real, documented way to fetch aggregate bars.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.massive.com"  # massive.rest.BASE, confirmed against the SDK source
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.0

# Substrings seen in real Massive/Polygon error responses that mean "this will never succeed by
# retrying" (e.g. plan/entitlement limits) -- retrying these 3x just wastes time and API quota.
NON_RETRYABLE_ERROR_MARKERS = ("NOT_AUTHORIZED", "UNAUTHORIZED", "FORBIDDEN")


class MassiveClient:
    """Fetches split/dividend-adjusted 1-min OHLCV bars from Massive via the official SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        client: Any | None = None,
    ):
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        if not self._api_key:
            raise ValueError("MASSIVE_API_KEY is not set (pass api_key= or set the env var)")
        self._client = client if client is not None else self._default_client(base_url)

    def _default_client(self, base_url: str) -> Any:
        from massive import RESTClient  # lazy import: keep this module importable without the dep present

        return RESTClient(api_key=self._api_key, base=base_url)

    def fetch_bars(self, ticker: str, start_ms: int, end_ms: int) -> list[dict]:
        """Fetch adjusted 1-min bars for `ticker` in `[start_ms, end_ms]` as `BAR_FIELDS` dicts.

        Retries transient errors with exponential backoff; raises `RuntimeError` after
        `MAX_RETRIES` so the scheduler can mark the buffer stale rather than silently serve old
        data (plan.md section 1.1). Does **not** retry errors matching
        `NON_RETRYABLE_ERROR_MARKERS` (e.g. a Massive/Polygon plan not entitled to this data
        timeframe) -- those fail on the first attempt instead of wasting 3x the time/quota on a
        request that can never succeed.
        """
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(MAX_RETRIES):
            attempts_made = attempt + 1
            try:
                aggs = self._client.get_aggs(
                    ticker, 1, "minute", from_=start_ms, to=end_ms,
                    # split+dividend adjusted, matching the training data convention (spec 2.5)
                    adjusted=True, sort="asc",
                )
                return self._parse_bars(ticker, aggs)
            except Exception as exc:  # noqa: BLE001 -- network/HTTP/parse errors all handled the same way
                last_error = exc
                logger.warning("massive fetch_bars(%s) attempt %d failed: %s", ticker, attempt, exc)
                if any(marker in str(exc) for marker in NON_RETRYABLE_ERROR_MARKERS):
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
        raise RuntimeError(
            f"failed to fetch bars for {ticker!r} after {attempts_made} attempt(s)"
        ) from last_error

    def fetch_bars_for_many(self, tickers: list[str], start_ms: int, end_ms: int) -> list[dict]:
        """Fetch bars for multiple tickers (e.g. the regime basket), concatenated.

        A single ticker's failure (e.g. one regime ETF not covered by the current Massive plan)
        is logged and skipped rather than aborting the whole basket -- the caller still gets
        bars for every ticker that succeeded, matching `Scheduler.poll_once`'s per-tracked-ticker
        isolation. A missing ticker's feature columns simply zero out downstream
        (data_source.get_features_for_ticker), which is a graceful degradation, not a crash.
        """
        combined: list[dict] = []
        for ticker in tickers:
            try:
                combined.extend(self.fetch_bars(ticker, start_ms, end_ms))
            except Exception:
                logger.exception("fetch_bars_for_many: skipping %s after repeated failure", ticker)
        return combined

    @staticmethod
    def _parse_bars(ticker: str, aggs) -> list[dict]:
        """Shape the SDK's `list[Agg]` into `data_source.BAR_FIELDS`-compatible dicts.

        `Agg` has no ticker field of its own (the request was already scoped to one ticker), so
        it's added back in here.
        """
        return [
            {
                "ticker": ticker,
                "timestamp": int(a.timestamp),
                "open": float(a.open),
                "high": float(a.high),
                "low": float(a.low),
                "close": float(a.close),
                "volume": int(a.volume) if a.volume is not None else 0,
                "vwap": float(a.vwap) if a.vwap is not None else float(a.close),
                "transactions": int(a.transactions) if a.transactions is not None else 0,
            }
            for a in aggs
        ]
