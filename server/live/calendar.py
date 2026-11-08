"""Trading-calendar-aware market-hours gating for the scheduler's poll loop
(docs/agent-docs/live-prediction-agent-plan.md section 1.4 / step 8).

Wraps `pandas_market_calendars`'s NYSE calendar so the background poll loop can skip polling
outside market hours/holidays, rather than hitting Massive/FRED on a fixed timer regardless of
whether the market is even open. Manual, caller-triggered fetches (e.g. `TickerTracker`'s warmup
backfill) intentionally bypass this gating -- a trader registering a new ticker should get its
warmup data immediately, not wait for the next gated tick.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_CALENDAR_NAME = "NYSE"


class MarketCalendar:
    """Answers "is the market open at this instant?" for a given exchange calendar.

    Caches each calendar date's open/close bounds after the first lookup -- the scheduler calls
    `is_open` on every poll tick, and a given date's schedule never changes within a run.
    """

    def __init__(self, calendar_name: str = DEFAULT_CALENDAR_NAME):
        import pandas_market_calendars as mcal  # lazy import: keep this module importable without the dep present

        self._calendar = mcal.get_calendar(calendar_name)
        self._schedule_cache: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}

    def is_open(self, now_ms: int) -> bool:
        """Whether the market is open at `now_ms` (epoch ms, UTC)."""
        now = pd.Timestamp(now_ms, unit="ms", tz="UTC")
        date_str = now.strftime("%Y-%m-%d")
        if date_str not in self._schedule_cache:
            schedule = self._calendar.schedule(start_date=date_str, end_date=date_str)
            if schedule.empty:
                self._schedule_cache[date_str] = None  # weekend/holiday -- no session that day
            else:
                self._schedule_cache[date_str] = (
                    schedule.iloc[0]["market_open"],
                    schedule.iloc[0]["market_close"],
                )

        bounds = self._schedule_cache[date_str]
        if bounds is None:
            return False
        market_open, market_close = bounds
        return market_open <= now <= market_close
