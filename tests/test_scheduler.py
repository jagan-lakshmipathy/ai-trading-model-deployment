"""Unit tests for Scheduler (docs/agent-docs/live-prediction-agent-plan.md step 4).

Uses stub MassiveClient/FredCache (duck-typed) -- no real network/httpx/fredapi calls.
"""
import unittest

from server.live.buffer import RegimeBuffer
from server.live.scheduler import Scheduler

BASE_TS = 1767614400000  # 2026-01-05 09:30:00 America/New_York, in epoch ms
MIN_MS = 60_000


def _bar(ticker: str, ts: int, close: float = 100.0) -> dict:
    return {
        "ticker": ticker,
        "timestamp": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000,
        "vwap": close,
        "transactions": 10,
    }


class _StubMassiveClient:
    def __init__(self, bars_by_ticker: dict[str, list[dict]], fail_tickers: set[str] | None = None):
        self._bars_by_ticker = bars_by_ticker
        self._fail_tickers = fail_tickers or set()
        self.fetch_calls: list[str] = []

    def fetch_bars(self, ticker: str, start_ms: int, end_ms: int) -> list[dict]:
        self.fetch_calls.append(ticker)
        if ticker in self._fail_tickers:
            raise RuntimeError(f"simulated failure for {ticker}")
        return self._bars_by_ticker.get(ticker, [])

    def fetch_bars_for_many(self, tickers: list[str], start_ms: int, end_ms: int) -> list[dict]:
        combined = []
        for ticker in tickers:
            combined.extend(self.fetch_bars(ticker, start_ms, end_ms))
        return combined


class _StubFredCache:
    def __init__(self):
        self.get_calls = 0

    def get(self):
        self.get_calls += 1


class SchedulerTests(unittest.TestCase):
    def test_track_untrack_idempotent(self):
        scheduler = Scheduler(_StubMassiveClient({}), RegimeBuffer(tickers=["SPY"]))
        buf1 = scheduler.track("AAPL")
        buf2 = scheduler.track("AAPL")
        self.assertIs(buf1, buf2)
        self.assertEqual(scheduler.tracked_tickers(), ["AAPL"])
        scheduler.untrack("AAPL")
        self.assertEqual(scheduler.tracked_tickers(), [])

    def test_poll_once_populates_regime_and_ticker_buffers(self):
        massive = _StubMassiveClient({
            "SPY": [_bar("SPY", BASE_TS)],
            "AAPL": [_bar("AAPL", BASE_TS), _bar("AAPL", BASE_TS + MIN_MS)],
        })
        regime_buffer = RegimeBuffer(tickers=["SPY"])
        fred_cache = _StubFredCache()
        scheduler = Scheduler(massive, regime_buffer, fred_cache=fred_cache, now_ms=lambda: BASE_TS + MIN_MS)
        scheduler.track("AAPL")

        scheduler.poll_once()

        self.assertEqual(len(regime_buffer.bars()), 1)
        self.assertEqual(len(scheduler.buffer_for("AAPL").bars()), 2)
        self.assertEqual(fred_cache.get_calls, 1)

    def test_poll_once_skips_failed_ticker_without_aborting(self):
        massive = _StubMassiveClient(
            {"SPY": [_bar("SPY", BASE_TS)], "GOOD": [_bar("GOOD", BASE_TS)]},
            fail_tickers={"BAD"},
        )
        regime_buffer = RegimeBuffer(tickers=["SPY"])
        scheduler = Scheduler(massive, regime_buffer, now_ms=lambda: BASE_TS)
        scheduler.track("BAD")
        scheduler.track("GOOD")

        scheduler.poll_once()  # must not raise despite BAD failing

        self.assertEqual(scheduler.buffer_for("BAD").bars(), [])
        self.assertEqual(len(scheduler.buffer_for("GOOD").bars()), 1)

    def test_poll_once_without_fred_cache(self):
        massive = _StubMassiveClient({"SPY": [_bar("SPY", BASE_TS)]})
        regime_buffer = RegimeBuffer(tickers=["SPY"])
        scheduler = Scheduler(massive, regime_buffer, now_ms=lambda: BASE_TS)
        scheduler.poll_once()  # no fred_cache configured -- should just skip it

    def test_background_loop_skips_ticks_when_market_closed(self):
        import time

        class _AlwaysClosed:
            def is_open(self, now_ms):
                return False

        massive = _StubMassiveClient({"SPY": [_bar("SPY", BASE_TS)]})
        regime_buffer = RegimeBuffer(tickers=["SPY"])
        scheduler = Scheduler(
            massive, regime_buffer, poll_interval_s=0.01,
            now_ms=lambda: BASE_TS, market_calendar=_AlwaysClosed(),
        )
        scheduler.start()
        time.sleep(0.1)  # several poll_interval_s ticks' worth
        scheduler.stop()

        self.assertEqual(massive.fetch_calls, [])  # gated out every tick -- never actually polled

    def test_background_loop_polls_when_market_open(self):
        import time

        class _AlwaysOpen:
            def is_open(self, now_ms):
                return True

        massive = _StubMassiveClient({"SPY": [_bar("SPY", BASE_TS)]})
        regime_buffer = RegimeBuffer(tickers=["SPY"])
        scheduler = Scheduler(
            massive, regime_buffer, poll_interval_s=0.01,
            now_ms=lambda: BASE_TS, market_calendar=_AlwaysOpen(),
        )
        scheduler.start()
        time.sleep(0.1)
        scheduler.stop()

        self.assertIn("SPY", massive.fetch_calls)


if __name__ == "__main__":
    unittest.main()
