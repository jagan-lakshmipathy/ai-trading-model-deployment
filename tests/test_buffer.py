"""Unit tests for TickerBuffer/RegimeBuffer (docs/agent-docs/live-prediction-agent-plan.md step 2)."""
import unittest

from server.live.buffer import RegimeBuffer, TickerBuffer

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


class TickerBufferTests(unittest.TestCase):
    def test_empty_buffer(self):
        buf = TickerBuffer("AAPL")
        self.assertTrue(buf.is_empty())
        self.assertIsNone(buf.latest_bar_ts())
        self.assertEqual(buf.bars(), [])

    def test_add_and_sort(self):
        buf = TickerBuffer("AAPL")
        buf.add_bars([_bar("AAPL", BASE_TS + 2 * MIN_MS), _bar("AAPL", BASE_TS)])
        self.assertEqual(buf.timestamps(), [BASE_TS, BASE_TS + 2 * MIN_MS])

    def test_dedup_last_write_wins(self):
        buf = TickerBuffer("AAPL")
        buf.add_bars([_bar("AAPL", BASE_TS, close=100.0)])
        buf.add_bars([_bar("AAPL", BASE_TS, close=101.0)])
        self.assertEqual(len(buf.bars()), 1)
        self.assertEqual(buf.bars()[0]["close"], 101.0)

    def test_wrong_ticker_rejected(self):
        buf = TickerBuffer("AAPL")
        with self.assertRaises(ValueError):
            buf.add_bars([_bar("MSFT", BASE_TS)])

    def test_missing_field_rejected(self):
        buf = TickerBuffer("AAPL")
        bar = _bar("AAPL", BASE_TS)
        del bar["volume"]
        with self.assertRaises(ValueError):
            buf.add_bars([bar])

    def test_eviction_past_max_age(self):
        buf = TickerBuffer("AAPL", max_age_ms=5 * MIN_MS)
        buf.add_bars([_bar("AAPL", BASE_TS)])
        # Ten minutes later: the first bar (now 10 min stale) should be evicted on next add.
        buf.add_bars([_bar("AAPL", BASE_TS + 10 * MIN_MS)])
        self.assertEqual(buf.timestamps(), [BASE_TS + 10 * MIN_MS])
        self.assertEqual(buf.latest_bar_ts(), BASE_TS + 10 * MIN_MS)


class RegimeBufferTests(unittest.TestCase):
    def test_default_basket(self):
        rb = RegimeBuffer()
        self.assertEqual(rb.tickers, ["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD", "USD"])
        self.assertTrue(rb.is_empty())

    def test_routes_bars_by_ticker(self):
        rb = RegimeBuffer(tickers=["SPY", "QQQ"])
        rb.add_bars([_bar("SPY", BASE_TS), _bar("QQQ", BASE_TS), _bar("SPY", BASE_TS + MIN_MS)])
        combined = rb.bars()
        self.assertEqual([b["ticker"] for b in combined], ["SPY", "SPY", "QQQ"])

    def test_unknown_ticker_rejected(self):
        rb = RegimeBuffer(tickers=["SPY"])
        with self.assertRaises(ValueError):
            rb.add_bars([_bar("TSLA", BASE_TS)])

    def test_latest_bar_ts_is_most_behind_member(self):
        rb = RegimeBuffer(tickers=["SPY", "QQQ"])
        rb.add_bars([_bar("SPY", BASE_TS + 5 * MIN_MS), _bar("QQQ", BASE_TS)])
        self.assertEqual(rb.latest_bar_ts(), BASE_TS)

    def test_latest_bar_ts_none_when_empty(self):
        rb = RegimeBuffer(tickers=["SPY"])
        self.assertIsNone(rb.latest_bar_ts())


if __name__ == "__main__":
    unittest.main()
