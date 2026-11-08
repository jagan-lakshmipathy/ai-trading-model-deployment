"""Live FRED API client + cache (docs/agent-docs/live-prediction-agent-plan.md section 1.2 /
step 3).

`FredClient` wraps `fredapi.Fred` to fetch the macro series documented in
docs/server-docs/data-readme.md. `FredCache` decouples the (slow, daily/monthly/quarterly at
the source) FRED refresh cadence from the much faster bars-polling loop: callers ask for the
current forward-filled macro frame on every predict() call without triggering a network request
each time -- it's only refetched once `refresh_interval_s` has elapsed since the last refresh.
"""
from __future__ import annotations

import os
import time
from typing import Protocol

import pandas as pd

# Same 10 series as docs/server-docs/data-readme.md's `data/fred_econ_data_6yrs.csv` table.
FRED_SERIES = [
    "FEDFUNDS", "CPIAUCSL", "PPIACO", "GDPC1", "UNRATE",
    "PAYEMS", "CPILFESL", "DGS10", "DGS2", "PCE",
]

# These series are daily/monthly/quarterly at the source -- no need to poll more than once a day.
DEFAULT_REFRESH_INTERVAL_S = 24 * 60 * 60


class FredClient:
    """Thin wrapper around `fredapi.Fred` for the fixed `FRED_SERIES` set."""

    def __init__(self, api_key: str | None = None):
        from fredapi import Fred  # lazy import: keep this module importable without the dep

        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self._api_key:
            raise ValueError("FRED_API_KEY is not set (pass api_key= or set the env var)")
        self._fred = Fred(api_key=self._api_key)

    def fetch_all(self) -> pd.DataFrame:
        """Fetch every configured series and merge onto a common daily (YYYY-MM-DD) index."""
        columns = {series_id: self._fred.get_series(series_id) for series_id in FRED_SERIES}
        df = pd.DataFrame(columns)
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        return df.sort_index()


class _FredDataSource(Protocol):
    def fetch_all(self) -> pd.DataFrame: ...


class FredCache:
    """Caches a `fetch_all()`-shaped source, refreshing at most once per `refresh_interval_s`."""

    def __init__(
        self,
        client: _FredDataSource,
        refresh_interval_s: int = DEFAULT_REFRESH_INTERVAL_S,
    ):
        self._client = client
        self._refresh_interval_s = refresh_interval_s
        self._cached: pd.DataFrame | None = None
        self._last_refresh: float | None = None

    def is_stale(self, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self._last_refresh is None or (now - self._last_refresh) >= self._refresh_interval_s

    def get(self, *, now: float | None = None) -> pd.DataFrame:
        """Return the cached, forward-filled macro frame, refreshing first if stale."""
        now = time.monotonic() if now is None else now
        if self.is_stale(now=now):
            self._cached = self._client.fetch_all().ffill()
            self._last_refresh = now
        return self._cached
