"""Unit tests for MarketCalendar (docs/agent-docs/live-prediction-agent-plan.md step 8).

Uses the real `pandas_market_calendars` NYSE calendar -- no network calls needed, the library's
schedule data is bundled/computed locally.
"""
import unittest

from server.live.calendar import MarketCalendar

# 2026-01-05 is a Monday (regular trading day), NYSE hours 09:30-16:00 America/New_York
# == 14:30-21:00 UTC.
MARKET_OPEN_MS = 1767623400000  # 2026-01-05 14:30:00 UTC (market open)
MID_SESSION_MS = 1767636600000  # 2026-01-05 18:10:00 UTC (mid-session)
MARKET_CLOSE_MS = 1767646800000  # 2026-01-05 21:00:00 UTC (market close)
AFTER_CLOSE_MS = 1767646860000  # one minute after close
BEFORE_OPEN_MS = 1767623340000  # one minute before open

SATURDAY_MS = 1768055400000  # 2026-01-10 (Saturday) 14:30 UTC
NEW_YEARS_DAY_MS = 1767277800000  # 2026-01-01 (holiday) 14:30 UTC


class MarketCalendarTests(unittest.TestCase):
    def test_open_during_regular_hours(self):
        cal = MarketCalendar()
        self.assertTrue(cal.is_open(MARKET_OPEN_MS))
        self.assertTrue(cal.is_open(MID_SESSION_MS))
        self.assertTrue(cal.is_open(MARKET_CLOSE_MS))

    def test_closed_outside_regular_hours_same_day(self):
        cal = MarketCalendar()
        self.assertFalse(cal.is_open(BEFORE_OPEN_MS))
        self.assertFalse(cal.is_open(AFTER_CLOSE_MS))

    def test_closed_on_weekend(self):
        cal = MarketCalendar()
        self.assertFalse(cal.is_open(SATURDAY_MS))

    def test_closed_on_holiday(self):
        cal = MarketCalendar()
        self.assertFalse(cal.is_open(NEW_YEARS_DAY_MS))

    def test_schedule_is_cached_per_date(self):
        cal = MarketCalendar()
        cal.is_open(MARKET_OPEN_MS)
        self.assertIn("2026-01-05", cal._schedule_cache)  # noqa: SLF001 -- test-only introspection
        cal.is_open(MID_SESSION_MS)  # same date -- must reuse the cached bounds, not re-lookup
        self.assertEqual(len(cal._schedule_cache), 1)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
