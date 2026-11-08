"""Background poll loop wiring MassiveClient/FredCache into the live buffers
(docs/agent-docs/live-prediction-agent-plan.md section 1.4 / step 4).

Timer-based polling only (no streaming subscription for v1, per the approved design).
Trading-calendar-aware gating (market hours/holidays) is applied to the background loop only
(`_run`, via an optional `market_calendar`) -- `poll_once()` itself stays ungated so manual,
caller-triggered fetches (e.g. `TickerTracker`'s warmup backfill) always run immediately.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .buffer import RegimeBuffer, TickerBuffer
from .calendar import MarketCalendar
from .fred_client import FredCache
from .massive_client import MassiveClient

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S = 60  # matches bar granularity (plan.md "Agent polling cadence")
DEFAULT_LOOKBACK_MS = 5 * 60 * 1000  # short trailing window per tick; buffers dedup overlap

# Plans below real-time (e.g. Polygon/Massive "Stocks Starter") only serve data older than a
# fixed delay -- requesting bars up to the literal current instant gets a NOT_AUTHORIZED error
# on every ticker, not just specific ones. 16 minutes gives Starter's 15-min delay a 1-min buffer.
DEFAULT_DATA_DELAY_MS = 16 * 60 * 1000


class Scheduler:
    """Polls Massive for new bars per tracked ticker + the shared regime basket, on a timer."""

    def __init__(
        self,
        massive_client: MassiveClient,
        regime_buffer: RegimeBuffer,
        fred_cache: FredCache | None = None,
        poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
        lookback_ms: int = DEFAULT_LOOKBACK_MS,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        market_calendar: MarketCalendar | None = None,
        data_delay_ms: int = DEFAULT_DATA_DELAY_MS,
    ):
        self._massive = massive_client
        self._regime_buffer = regime_buffer
        self._fred_cache = fred_cache
        self._poll_interval_s = poll_interval_s
        self._lookback_ms = lookback_ms
        self._now_ms = now_ms
        self._market_calendar = market_calendar
        self._data_delay_ms = data_delay_ms
        self._ticker_buffers: dict[str, TickerBuffer] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def track(self, ticker: str) -> TickerBuffer:
        """Register `ticker` for polling (idempotent); returns its buffer."""
        with self._lock:
            if ticker not in self._ticker_buffers:
                self._ticker_buffers[ticker] = TickerBuffer(ticker)
            return self._ticker_buffers[ticker]

    def untrack(self, ticker: str) -> None:
        with self._lock:
            self._ticker_buffers.pop(ticker, None)

    def tracked_tickers(self) -> list[str]:
        with self._lock:
            return list(self._ticker_buffers)

    def buffer_for(self, ticker: str) -> TickerBuffer | None:
        with self._lock:
            return self._ticker_buffers.get(ticker)

    def poll_once(self) -> None:
        """Fetch one tick's worth of new bars for the regime basket and every tracked ticker.

        A single ticker's fetch failure is logged and skipped rather than aborting the whole
        tick -- one bad ticker shouldn't stall the regime basket or every other tracked ticker.
        """
        end_ms = self._now_ms() - self._data_delay_ms
        start_ms = end_ms - self._lookback_ms

        regime_bars = self._massive.fetch_bars_for_many(self._regime_buffer.tickers, start_ms, end_ms)
        self._regime_buffer.add_bars(regime_bars)

        for ticker in self.tracked_tickers():
            try:
                bars = self._massive.fetch_bars(ticker, start_ms, end_ms)
            except Exception:
                logger.exception("poll failed for ticker %s; buffer left as-is", ticker)
                continue
            buffer = self.buffer_for(ticker)
            if buffer is not None:
                buffer.add_bars(bars)

        if self._fred_cache is not None:
            self._fred_cache.get()

    def start(self) -> None:
        """Start the background poll loop (idempotent; no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_s + 5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._market_calendar is None or self._market_calendar.is_open(self._now_ms()):
                try:
                    self.poll_once()
                except Exception:
                    logger.exception("poll_once failed")
            self._stop_event.wait(self._poll_interval_s)
