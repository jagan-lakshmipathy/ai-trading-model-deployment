"""Dynamic on-demand ticker tracking: registration, warmup backfill, idle eviction
(docs/agent-docs/live-prediction-agent-plan.md section 1.5 / step 5).

Wraps a `Scheduler` (buffers) + `MassiveClient` (immediate backfill fetch) + `HiddenStateManager`
(reset on eviction) so `predict(ticker)` can register unseen tickers on demand, wait for a
warmup window before serving a first real prediction, and release idle tickers' server-side
state after a period of inactivity.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .scheduler import DEFAULT_DATA_DELAY_MS

logger = logging.getLogger(__name__)

# >= longest feature lookback (past_peak_dev_60m) plus headroom -- matches buffer.DEFAULT_MAX_AGE_MS.
DEFAULT_WARMUP_MS = 90 * 60 * 1000
DEFAULT_IDLE_TIMEOUT_S = 30 * 60  # evict a ticker after 30 min with no predict() calls


class WarmingUp(Exception):
    """Raised by `TickerTracker.ensure_ready` while a newly tracked ticker's buffer is too short.

    Callers (ultimately the `predict` tool) should catch this and return a "warming up, retry
    shortly" response rather than a prediction (plan.md section 1.5) -- never silently compute a
    prediction from a too-short rolling window.
    """

    def __init__(self, ticker: str, have_ms: int, need_ms: int):
        super().__init__(f"{ticker} is warming up: have {have_ms}ms of history, need {need_ms}ms")
        self.ticker = ticker
        self.have_ms = have_ms
        self.need_ms = need_ms


class TickerTracker:
    """Manages dynamic ticker registration, warmup, and idle eviction on top of a `Scheduler`."""

    def __init__(
        self,
        scheduler,
        massive_client,
        hidden_state_manager,
        warmup_ms: int = DEFAULT_WARMUP_MS,
        idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        now_s: Callable[[], float] = time.monotonic,
        data_delay_ms: int = DEFAULT_DATA_DELAY_MS,
    ):
        self._scheduler = scheduler
        self._massive = massive_client
        self._hidden_state = hidden_state_manager
        self._warmup_ms = warmup_ms
        self._idle_timeout_s = idle_timeout_s
        self._now_ms = now_ms
        self._now_s = now_s
        self._data_delay_ms = data_delay_ms
        self._last_access_s: dict[str, float] = {}
        self._lock = threading.Lock()

    def ensure_ready(self, ticker: str) -> None:
        """Register `ticker` if unseen (triggering an immediate warmup backfill), record this
        access for idle-eviction purposes, and raise `WarmingUp` if there isn't yet enough
        buffered history to compute the rolling-window features.
        """
        is_new = self._scheduler.buffer_for(ticker) is None
        buffer = self._scheduler.track(ticker)
        with self._lock:
            self._last_access_s[ticker] = self._now_s()

        if is_new:
            self._backfill(ticker, buffer)

        timestamps = buffer.timestamps()
        have_ms = 0 if len(timestamps) < 2 else timestamps[-1] - timestamps[0]
        if not timestamps or have_ms < self._warmup_ms:
            raise WarmingUp(ticker, have_ms, self._warmup_ms)

    def _backfill(self, ticker: str, buffer) -> None:
        """Immediate one-off fetch of the warmup window, instead of waiting for the next poll tick."""
        end_ms = self._now_ms() - self._data_delay_ms
        start_ms = end_ms - self._warmup_ms
        try:
            bars = self._massive.fetch_bars(ticker, start_ms, end_ms)
            buffer.add_bars(bars)
        except Exception:
            logger.exception("warmup backfill failed for %s; will retry on next poll tick", ticker)

    def evict_idle(self) -> list[str]:
        """Drop tracking/hidden-state for any ticker with no `ensure_ready` call in
        `idle_timeout_s`. Returns the list of evicted tickers.

        Note: `HiddenStateManager.reset` zeroes the ticker's cached state rather than removing
        its dict entry -- functionally equivalent to a fresh start, but doesn't free that small
        per-ticker entry's memory. Acceptable at this scale; revisit if idle-ticker churn grows.
        """
        now = self._now_s()
        with self._lock:
            idle = [t for t, last in self._last_access_s.items() if now - last >= self._idle_timeout_s]
            for ticker in idle:
                del self._last_access_s[ticker]
        for ticker in idle:
            self._scheduler.untrack(ticker)
            self._hidden_state.reset(ticker)
            logger.info("evicted idle ticker %s (no predict() calls for %ds)", ticker, self._idle_timeout_s)
        return idle
