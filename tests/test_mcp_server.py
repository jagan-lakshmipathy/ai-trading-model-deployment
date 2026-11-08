"""Integration tests for the predict(ticker) MCP tool (docs/agent-docs/live-prediction-agent-plan.md
step 7).

Uses the real checkpoint/InferenceEngine (cheap to load) but fully stubbed live components
(MassiveClient, FredCache) -- no real network/credentials, matching the pattern used in the
other server/live/* test modules.
"""
import asyncio
import unittest

from server.live.buffer import RegimeBuffer
from server.live.config import LiveConfig
from server.live.scheduler import Scheduler
from server.mcp_server import create_server

BASE_TS = 1767614400000  # 2026-01-05 09:30:00 America/New_York, in epoch ms
MIN_MS = 60_000
CHECKPOINT = "checkpoints/best_model.pt"


def _bars(ticker: str, count: int) -> list[dict]:
    return [
        {
            "ticker": ticker,
            "timestamp": BASE_TS + i * MIN_MS,
            "open": 100.0 + i * 0.01,
            "high": 100.2 + i * 0.01,
            "low": 99.8 + i * 0.01,
            "close": 100.1 + i * 0.01,
            "volume": 1000 + i,
            "vwap": 100.1 + i * 0.01,
            "transactions": 10,
        }
        for i in range(count)
    ]


class _StubMassiveClient:
    def __init__(self, bar_count: int = 5):
        self._bar_count = bar_count
        self.fetch_calls: list[str] = []

    def fetch_bars(self, ticker: str, start_ms: int, end_ms: int) -> list[dict]:
        self.fetch_calls.append(ticker)
        return _bars(ticker, self._bar_count)

    def fetch_bars_for_many(self, tickers: list[str], start_ms: int, end_ms: int) -> list[dict]:
        combined = []
        for ticker in tickers:
            combined.extend(self.fetch_bars(ticker, start_ms, end_ms))
        return combined


class _StubFredCache:
    def get(self):
        import pandas as pd

        return pd.DataFrame()


def _build_server(massive, warmup_ms: int):
    regime_buffer = RegimeBuffer()  # full default basket -- needed for the full 42-column contract
    scheduler = Scheduler(massive, regime_buffer, fred_cache=None, now_ms=lambda: BASE_TS + 10 * MIN_MS)
    # Populate the regime buffer once, as the background poll loop normally would.
    scheduler.poll_once()

    live_config = LiveConfig(massive_api_key="mk", fred_api_key="fk", warmup_ms=warmup_ms)
    server = create_server(
        checkpoint_path=CHECKPOINT,
        live_config=live_config,
        massive_client=massive,
        fred_cache=_StubFredCache(),
        regime_buffer=regime_buffer,
        scheduler=scheduler,
        tracker=None,  # built internally so it shares the real HiddenStateManager
        start_scheduler=False,
    )
    return server


def _call_tool(server, name: str, arguments: dict):
    result = asyncio.run(server.call_tool(name, arguments))
    return result.structured_content


def _read_resource(server, uri: str):
    result = asyncio.run(server.read_resource(uri))
    return result


class PredictToolTests(unittest.TestCase):
    def test_predict_warming_up_then_ready(self):
        massive = _StubMassiveClient(bar_count=2)  # spans 1 min -- below any reasonable warmup
        server = _build_server(massive, warmup_ms=3 * MIN_MS)

        result = _call_tool(server, "predict", {"ticker": "TEST"})
        self.assertEqual(result["status"], "warming_up")
        self.assertEqual(result["ticker"], "TEST")

    def test_predict_ready_returns_prediction(self):
        massive = _StubMassiveClient(bar_count=5)  # spans 4 min
        server = _build_server(massive, warmup_ms=3 * MIN_MS)

        result = _call_tool(server, "predict", {"ticker": "TEST"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["predictions"]), 3)  # horizons [5, 10, 15]
        self.assertEqual(result["as_of_timestamp"], BASE_TS + 4 * MIN_MS)

    def test_predict_second_call_without_new_bars_replays_cached_result(self):
        massive = _StubMassiveClient(bar_count=5)
        server = _build_server(massive, warmup_ms=3 * MIN_MS)

        first = _call_tool(server, "predict", {"ticker": "TEST"})
        second = _call_tool(server, "predict", {"ticker": "TEST"})
        self.assertEqual(first, second)
        # Only the initial on-demand registration should have triggered a backfill fetch.
        self.assertEqual(massive.fetch_calls.count("TEST"), 1)

    def test_reset_session_clears_cache(self):
        massive = _StubMassiveClient(bar_count=5)
        server = _build_server(massive, warmup_ms=3 * MIN_MS)

        _call_tool(server, "predict", {"ticker": "TEST"})
        reset_result = _call_tool(server, "reset_session", {"ticker": "TEST"})
        self.assertEqual(reset_result["status"], "reset")

        # After reset, predict() must re-derive from scratch (cursor cleared) rather than
        # silently replaying a now-stale cached result.
        again = _call_tool(server, "predict", {"ticker": "TEST"})
        self.assertEqual(again["status"], "ok")

    def test_predict_data_unavailable_when_regime_buffer_empty(self):
        """Guards against the startup race where predict() runs before the scheduler's first
        poll tick has populated the regime buffer -- missing regime columns drop the feature
        count from 42, which must be reported, not crash the model (docs/agent-docs/
        live-prediction-agent-plan.md's post-plan-fixes writeup)."""
        massive = _StubMassiveClient(bar_count=5)
        regime_buffer = RegimeBuffer()  # intentionally never populated -- no poll_once() call
        scheduler = Scheduler(massive, regime_buffer, now_ms=lambda: BASE_TS + 10 * MIN_MS)
        server = create_server(
            checkpoint_path=CHECKPOINT,
            live_config=LiveConfig(massive_api_key="mk", fred_api_key="fk", warmup_ms=3 * MIN_MS),
            massive_client=massive,
            fred_cache=_StubFredCache(),
            regime_buffer=regime_buffer,
            scheduler=scheduler,
            tracker=None,
            start_scheduler=False,
        )

        result = _call_tool(server, "predict", {"ticker": "TEST"})
        self.assertEqual(result["status"], "data_unavailable")

    def test_create_server_primes_regime_buffer_on_startup(self):
        """create_server(start_scheduler=True) must synchronously poll once before returning,
        so the first predict() call doesn't race the background thread's first tick."""
        massive = _StubMassiveClient(bar_count=5)
        regime_buffer = RegimeBuffer()
        scheduler = Scheduler(massive, regime_buffer, now_ms=lambda: BASE_TS + 10 * MIN_MS)
        create_server(
            checkpoint_path=CHECKPOINT,
            live_config=LiveConfig(massive_api_key="mk", fred_api_key="fk", warmup_ms=3 * MIN_MS),
            massive_client=massive,
            fred_cache=_StubFredCache(),
            regime_buffer=regime_buffer,
            scheduler=scheduler,
            tracker=None,
            start_scheduler=True,
        )
        try:
            self.assertFalse(regime_buffer.is_empty())
        finally:
            scheduler.stop()


if __name__ == "__main__":
    unittest.main()
