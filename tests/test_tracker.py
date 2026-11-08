"""Unit tests for TickerTracker (docs/agent-docs/live-prediction-agent-plan.md step 5).

Uses stub Scheduler/MassiveClient/HiddenStateManager -- no real buffers/network/torch needed,
since TickerTracker only depends on their duck-typed interfaces.
"""
import unittest

from server.live.tracker import DEFAULT_WARMUP_MS, TickerTracker, WarmingUp

BASE_TS = 1767614400000  # 2026-01-05 09:30:00 America/New_York, in epoch ms
MIN_MS = 60_000


class _StubBuffer:
    def __init__(self):
        self._ts: list[int] = []

    def add_bars(self, bars: list[dict]) -> None:
        self._ts.extend(b["timestamp"] for b in bars)
        self._ts.sort()

    def timestamps(self) -> list[int]:
        return list(self._ts)


class _StubScheduler:
    def __init__(self):
        self._buffers: dict[str, _StubBuffer] = {}
        self.untrack_calls: list[str] = []

    def buffer_for(self, ticker: str):
        return self._buffers.get(ticker)

    def track(self, ticker: str) -> _StubBuffer:
        if ticker not in self._buffers:
            self._buffers[ticker] = _StubBuffer()
        return self._buffers[ticker]

    def untrack(self, ticker: str) -> None:
        self.untrack_calls.append(ticker)
        self._buffers.pop(ticker, None)


class _StubMassiveClient:
    def __init__(self, bars_by_ticker: dict[str, list[dict]]):
        self._bars_by_ticker = bars_by_ticker
        self.fetch_calls: list[str] = []

    def fetch_bars(self, ticker: str, start_ms: int, end_ms: int) -> list[dict]:
        self.fetch_calls.append(ticker)
        return self._bars_by_ticker.get(ticker, [])


class _StubHiddenStateManager:
    def __init__(self):
        self.reset_calls: list[str] = []

    def reset(self, ticker: str) -> None:
        self.reset_calls.append(ticker)


def _bar(ticker: str, ts: int) -> dict:
    return {"ticker": ticker, "timestamp": ts}


class TickerTrackerTests(unittest.TestCase):
    def test_unseen_ticker_triggers_backfill(self):
        massive = _StubMassiveClient({"AAPL": [_bar("AAPL", BASE_TS), _bar("AAPL", BASE_TS + MIN_MS)]})
        tracker = TickerTracker(_StubScheduler(), massive, _StubHiddenStateManager())
        with self.assertRaises(WarmingUp):
            tracker.ensure_ready("AAPL")
        self.assertEqual(massive.fetch_calls, ["AAPL"])

    def test_already_tracked_ticker_does_not_refetch_backfill(self):
        massive = _StubMassiveClient({"AAPL": [_bar("AAPL", BASE_TS)]})
        tracker = TickerTracker(_StubScheduler(), massive, _StubHiddenStateManager())
        with self.assertRaises(WarmingUp):
            tracker.ensure_ready("AAPL")
        with self.assertRaises(WarmingUp):
            tracker.ensure_ready("AAPL")
        self.assertEqual(massive.fetch_calls, ["AAPL"])  # only the first call backfilled

    def test_ready_once_warmup_window_satisfied(self):
        bars = [_bar("AAPL", BASE_TS), _bar("AAPL", BASE_TS + DEFAULT_WARMUP_MS)]
        massive = _StubMassiveClient({"AAPL": bars})
        tracker = TickerTracker(_StubScheduler(), massive, _StubHiddenStateManager())
        tracker.ensure_ready("AAPL")  # no exception: enough history spanning the warmup window

    def test_warming_up_exception_carries_have_need_ms(self):
        massive = _StubMassiveClient({"AAPL": [_bar("AAPL", BASE_TS)]})
        tracker = TickerTracker(_StubScheduler(), massive, _StubHiddenStateManager())
        with self.assertRaises(WarmingUp) as ctx:
            tracker.ensure_ready("AAPL")
        self.assertEqual(ctx.exception.ticker, "AAPL")
        self.assertEqual(ctx.exception.have_ms, 0)
        self.assertEqual(ctx.exception.need_ms, DEFAULT_WARMUP_MS)

    def test_backfill_failure_does_not_raise_from_ensure_ready_path(self):
        class _FailingMassiveClient:
            def fetch_bars(self, ticker, start_ms, end_ms):
                raise RuntimeError("network down")

        tracker = TickerTracker(_StubScheduler(), _FailingMassiveClient(), _StubHiddenStateManager())
        with self.assertRaises(WarmingUp):  # not RuntimeError -- backfill failure is logged, not fatal
            tracker.ensure_ready("AAPL")

    def test_evict_idle_removes_and_resets(self):
        massive = _StubMassiveClient({"AAPL": []})
        hidden_state = _StubHiddenStateManager()
        scheduler = _StubScheduler()
        now = [0.0]
        tracker = TickerTracker(
            scheduler, massive, hidden_state, idle_timeout_s=60, now_s=lambda: now[0]
        )
        with self.assertRaises(WarmingUp):
            tracker.ensure_ready("AAPL")

        now[0] = 30.0
        self.assertEqual(tracker.evict_idle(), [])  # not idle yet

        now[0] = 61.0
        self.assertEqual(tracker.evict_idle(), ["AAPL"])
        self.assertEqual(scheduler.untrack_calls, ["AAPL"])
        self.assertEqual(hidden_state.reset_calls, ["AAPL"])

    def test_evict_idle_resets_timer_on_reaccess(self):
        massive = _StubMassiveClient({"AAPL": []})
        scheduler = _StubScheduler()
        now = [0.0]
        tracker = TickerTracker(
            scheduler, massive, _StubHiddenStateManager(), idle_timeout_s=60, now_s=lambda: now[0]
        )
        with self.assertRaises(WarmingUp):
            tracker.ensure_ready("AAPL")

        now[0] = 50.0
        with self.assertRaises(WarmingUp):
            tracker.ensure_ready("AAPL")  # re-access before idle timeout resets the clock

        now[0] = 90.0  # 40s since re-access, still under the 60s idle timeout
        self.assertEqual(tracker.evict_idle(), [])


if __name__ == "__main__":
    unittest.main()
