"""Thin predict(ticker) poller (docs/agent-docs/live-prediction-agent-plan.md section 2 / step 9).

Calls the MCP server's `predict` tool on a timer for each configured ticker, during market hours
only, and logs/relays whatever comes back. No local bar-fetching, hidden-state tracking, or
"fixture" concept -- the server owns all of that (see the plan's "Rejected approach"). Handles
the server's `"warming_up"` and stale-cache responses by just logging them, not treating them as
errors (per docs/agent-docs/live-prediction-agent-plan.md's discussion of warmup/staleness).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from .config import DEFAULT_POLL_INTERVAL_S

logger = logging.getLogger(__name__)


class Poller:
    """Calls `predict(ticker)` on `server` for each configured ticker, on a timer."""

    def __init__(
        self,
        server: Any,
        tickers: list[str],
        poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
        market_calendar: Any | None = None,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        on_result: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self._server = server
        self._tickers = list(tickers)
        self._poll_interval_s = poll_interval_s
        self._market_calendar = market_calendar
        self._now_ms = now_ms
        self._on_result = on_result or self._log_result
        self._stop_event = asyncio.Event()

    @staticmethod
    def _log_result(ticker: str, result: dict[str, Any]) -> None:
        status = result.get("status")
        if status == "ok":
            logger.info("predict(%s) -> %s", ticker, result.get("predictions"))
        elif status == "warming_up":
            logger.info(
                "predict(%s) warming up: have=%sms need=%sms",
                ticker, result.get("have_ms"), result.get("need_ms"),
            )
        else:
            logger.info("predict(%s) -> %s", ticker, result)

    async def poll_once(self) -> dict[str, dict[str, Any]]:
        """Call predict() for every configured ticker once. Returns `{ticker: result}`.

        A single ticker's failure is logged and skipped, not fatal to the whole tick -- mirrors
        `Scheduler.poll_once`'s per-ticker failure isolation.
        """
        results: dict[str, dict[str, Any]] = {}
        for ticker in self._tickers:
            try:
                call_result = await self._server.call_tool("predict", {"ticker": ticker})
                result = call_result.structured_content
            except Exception:
                logger.exception("predict(%s) failed", ticker)
                continue
            self._on_result(ticker, result)
            results[ticker] = result
        return results

    async def run(self) -> None:
        """Run the poll loop until `stop()` is called.

        Gated on market hours if `market_calendar` is configured (ticks are skipped, not
        errored, outside trading hours -- per the "reasonable window to predict" discussion in
        docs/agent-docs/live-prediction-agent-plan.md).
        """
        was_open = None  # force a log line the first time through, whichever state it's in
        while not self._stop_event.is_set():
            is_open = self._market_calendar is None or self._market_calendar.is_open(self._now_ms())
            if is_open != was_open:
                logger.info("market is %s -- %s", "open" if is_open else "closed",
                            "polling" if is_open else "waiting for next open")
                was_open = is_open
            if is_open:
                await self.poll_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval_s)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()


def main() -> None:
    """Entry point: build a real server + NYSE calendar, poll a fixed ticker list."""
    import argparse

    from server.live.calendar import MarketCalendar
    from server.mcp_server import create_server

    parser = argparse.ArgumentParser(description="Poll predict(ticker) on a timer.")
    parser.add_argument("tickers", nargs="+", help="Tickers to poll, e.g. AAPL MSFT")
    parser.add_argument("--poll-interval-s", type=int, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument(
        "--log-file", default=None,
        help="Append each predict() result as a JSON line here, for scripts/plot_predictions.py",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = create_server()

    on_result = None
    if args.log_file:
        from .recorder import JsonlRecorder

        recorder = JsonlRecorder(args.log_file)

        def on_result(ticker: str, result: dict) -> None:
            Poller._log_result(ticker, result)
            recorder(ticker, result)

    poller = Poller(
        server, args.tickers, poll_interval_s=args.poll_interval_s,
        market_calendar=MarketCalendar(), on_result=on_result,
    )
    asyncio.run(poller.run())


if __name__ == "__main__":
    main()
