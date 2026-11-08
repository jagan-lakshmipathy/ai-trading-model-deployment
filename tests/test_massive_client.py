"""Unit tests for MassiveClient (docs/agent-docs/live-prediction-agent-plan.md step 4;
corrected 2026-11-03 for the real `massive` SDK -- see plan section 6).

Uses a stub SDK client instead of real network calls -- MassiveClient only depends on a
duck-typed `.get_aggs(ticker, multiplier, timespan, from_, to, adjusted, sort) -> list[Agg]`.
"""
import unittest
from dataclasses import dataclass
from unittest import mock

from server.live.massive_client import MassiveClient

BASE_TS = 1767614400000  # 2026-01-05 09:30:00 America/New_York, in epoch ms
MIN_MS = 60_000


@dataclass
class _FakeAgg:
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    timestamp: int
    transactions: int


class _StubSDKClient:
    def __init__(self, aggs_by_ticker: dict[str, list], fail_tickers: set[str] | None = None):
        self._aggs_by_ticker = aggs_by_ticker
        self._fail_tickers = fail_tickers or set()
        self.calls: list[dict] = []

    def get_aggs(self, ticker, multiplier, timespan, from_, to, adjusted=None, sort=None):
        self.calls.append({
            "ticker": ticker, "multiplier": multiplier, "timespan": timespan,
            "from_": from_, "to": to, "adjusted": adjusted, "sort": sort,
        })
        if ticker in self._fail_tickers:
            raise RuntimeError(f"simulated failure for {ticker}")
        return self._aggs_by_ticker.get(ticker, [])


def _aggs(count: int) -> list[_FakeAgg]:
    return [
        _FakeAgg(
            open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i,
            volume=1000 + i, vwap=100.4 + i, timestamp=BASE_TS + i * MIN_MS, transactions=10,
        )
        for i in range(count)
    ]


class MassiveClientTests(unittest.TestCase):
    def test_requires_api_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                MassiveClient(api_key=None, client=_StubSDKClient({}))

    def test_fetch_bars_parses_response(self):
        client_sdk = _StubSDKClient({"SPY": _aggs(2)})
        client = MassiveClient(api_key="k", client=client_sdk)
        bars = client.fetch_bars("SPY", BASE_TS, BASE_TS + MIN_MS)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0]["ticker"], "SPY")
        self.assertEqual(bars[0]["timestamp"], BASE_TS)
        self.assertEqual(bars[1]["timestamp"], BASE_TS + MIN_MS)
        self.assertEqual(client_sdk.calls[0]["adjusted"], True)
        self.assertEqual(client_sdk.calls[0]["timespan"], "minute")

    def test_retries_then_succeeds(self):
        class _FlakySDKClient:
            def __init__(self):
                self.calls = 0

            def get_aggs(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient")
                return _aggs(1)

        client_sdk = _FlakySDKClient()
        client = MassiveClient(api_key="k", client=client_sdk)
        with mock.patch("server.live.massive_client.time.sleep"):
            bars = client.fetch_bars("AAPL", BASE_TS, BASE_TS + MIN_MS)
        self.assertEqual(len(bars), 1)
        self.assertEqual(client_sdk.calls, 2)

    def test_raises_after_max_retries(self):
        client_sdk = _StubSDKClient({}, fail_tickers={"AAPL"})
        client = MassiveClient(api_key="k", client=client_sdk)
        with mock.patch("server.live.massive_client.time.sleep"):
            with self.assertRaises(RuntimeError):
                client.fetch_bars("AAPL", BASE_TS, BASE_TS + MIN_MS)
        self.assertEqual(len(client_sdk.calls), 3)

    def test_non_retryable_error_fails_fast(self):
        class _AlwaysUnauthorized:
            def __init__(self):
                self.calls = 0

            def get_aggs(self, *args, **kwargs):
                self.calls += 1
                raise RuntimeError('{"status":"NOT_AUTHORIZED","message":"plan doesn\'t include this"}')

        client_sdk = _AlwaysUnauthorized()
        client = MassiveClient(api_key="k", client=client_sdk)
        with mock.patch("server.live.massive_client.time.sleep") as sleep_mock:
            with self.assertRaises(RuntimeError):
                client.fetch_bars("SPY", BASE_TS, BASE_TS + MIN_MS)
        self.assertEqual(client_sdk.calls, 1)  # one attempt only, no wasted retries
        sleep_mock.assert_not_called()

    def test_fetch_bars_for_many_concatenates(self):
        client_sdk = _StubSDKClient({"SPY": _aggs(1), "QQQ": _aggs(1)})
        client = MassiveClient(api_key="k", client=client_sdk)
        bars = client.fetch_bars_for_many(["SPY", "QQQ"], BASE_TS, BASE_TS + MIN_MS)
        self.assertEqual([b["ticker"] for b in bars], ["SPY", "QQQ"])

    def test_fetch_bars_for_many_isolates_per_ticker_failure(self):
        client_sdk = _StubSDKClient({"QQQ": _aggs(1)}, fail_tickers={"SPY"})
        client = MassiveClient(api_key="k", client=client_sdk)
        with mock.patch("server.live.massive_client.time.sleep"):
            bars = client.fetch_bars_for_many(["SPY", "QQQ"], BASE_TS, BASE_TS + MIN_MS)
        # SPY failed (and was retried 3x internally), but QQQ still comes through.
        self.assertEqual([b["ticker"] for b in bars], ["QQQ"])


if __name__ == "__main__":
    unittest.main()
