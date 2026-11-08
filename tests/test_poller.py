"""Unit tests for Poller (docs/agent-docs/live-prediction-agent-plan.md step 9).

Uses a stub async server/market-calendar -- no real MCP server, network, or model loading.
"""
import asyncio
import unittest

from agent.poller import Poller

BASE_MS = 1767614400000


class _StubCallToolResult:
    def __init__(self, structured_content):
        self.structured_content = structured_content


class _StubServer:
    def __init__(self, results_by_ticker: dict[str, dict], fail_tickers: set[str] | None = None):
        self._results = results_by_ticker
        self._fail_tickers = fail_tickers or set()
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict):
        ticker = arguments["ticker"]
        self.calls.append(ticker)
        if ticker in self._fail_tickers:
            raise RuntimeError(f"simulated failure for {ticker}")
        return _StubCallToolResult(self._results[ticker])


class PollerTests(unittest.TestCase):
    def test_poll_once_calls_predict_for_each_ticker_and_relays_results(self):
        server = _StubServer({
            "AAPL": {"status": "ok", "predictions": [{"horizon_minutes": 5}]},
            "MSFT": {"status": "warming_up", "have_ms": 0, "need_ms": 1000},
        })
        received: dict[str, dict] = {}
        poller = Poller(server, ["AAPL", "MSFT"], on_result=lambda t, r: received.update({t: r}))

        results = asyncio.run(poller.poll_once())

        self.assertEqual(server.calls, ["AAPL", "MSFT"])
        self.assertEqual(results["AAPL"]["status"], "ok")
        self.assertEqual(received["MSFT"]["status"], "warming_up")

    def test_poll_once_skips_failed_ticker_without_aborting(self):
        server = _StubServer(
            {"GOOD": {"status": "ok", "predictions": []}},
            fail_tickers={"BAD"},
        )
        poller = Poller(server, ["BAD", "GOOD"])

        results = asyncio.run(poller.poll_once())  # must not raise despite BAD failing

        self.assertNotIn("BAD", results)
        self.assertIn("GOOD", results)

    def test_default_log_result_does_not_raise_for_any_status(self):
        Poller._log_result("AAPL", {"status": "ok", "predictions": []})
        Poller._log_result("AAPL", {"status": "warming_up", "have_ms": 0, "need_ms": 1})
        Poller._log_result("AAPL", {"status": "no_new_bars"})

    def test_run_skips_ticks_when_market_closed(self):
        class _AlwaysClosed:
            def is_open(self, now_ms):
                return False

        server = _StubServer({"AAPL": {"status": "ok", "predictions": []}})
        poller = Poller(
            server, ["AAPL"], poll_interval_s=0.01, market_calendar=_AlwaysClosed(), now_ms=lambda: BASE_MS
        )

        async def _run_briefly():
            task = asyncio.ensure_future(poller.run())
            await asyncio.sleep(0.1)
            poller.stop()
            await task

        asyncio.run(_run_briefly())
        self.assertEqual(server.calls, [])

    def test_run_polls_when_market_open(self):
        class _AlwaysOpen:
            def is_open(self, now_ms):
                return True

        server = _StubServer({"AAPL": {"status": "ok", "predictions": []}})
        poller = Poller(
            server, ["AAPL"], poll_interval_s=0.01, market_calendar=_AlwaysOpen(), now_ms=lambda: BASE_MS
        )

        async def _run_briefly():
            task = asyncio.ensure_future(poller.run())
            await asyncio.sleep(0.1)
            poller.stop()
            await task

        asyncio.run(_run_briefly())
        self.assertIn("AAPL", server.calls)


if __name__ == "__main__":
    unittest.main()
