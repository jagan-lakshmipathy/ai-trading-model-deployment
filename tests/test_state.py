"""Unit tests for HiddenStateManager's processed-bar cursor (docs/agent-docs/live-prediction-agent-plan.md step 1)."""
import unittest

import torch

from server.encoder import MultiHorizonGRUModel
from server.state import HiddenStateManager

# 2026-01-05 is a Monday; timestamps below stay within that single market-tz session.
BASE_TS = 1767614400000  # 2026-01-05 09:30:00 America/New_York, in epoch ms


def _make_manager() -> HiddenStateManager:
    model = MultiHorizonGRUModel(n_features=4, horizons=[5], hidden_size=8, num_layers=1)
    model.eval()
    return HiddenStateManager(model)


class NewTimestampsTests(unittest.TestCase):
    def test_unseen_ticker_returns_all_timestamps(self):
        mgr = _make_manager()
        ts = [BASE_TS, BASE_TS + 60_000, BASE_TS + 120_000]
        self.assertEqual(mgr.new_timestamps("AAPL", ts), ts)

    def test_rejects_non_ascending_timestamps(self):
        mgr = _make_manager()
        with self.assertRaises(ValueError):
            mgr.new_timestamps("AAPL", [BASE_TS, BASE_TS, BASE_TS + 60_000])
        with self.assertRaises(ValueError):
            mgr.new_timestamps("AAPL", [BASE_TS + 60_000, BASE_TS])

    def test_overlapping_window_only_returns_bars_after_cursor(self):
        mgr = _make_manager()
        hidden = mgr._get_or_create("AAPL").hidden  # noqa: SLF001 -- test-only introspection

        with mgr.step("AAPL", BASE_TS) as handle:
            self.assertTrue(torch.equal(handle.hidden, hidden))
            handle.commit(torch.ones_like(hidden))

        # First call processed BASE_TS; a resubmitted window containing it plus two new bars
        # should only report the two new ones as needing a forward pass.
        window = [BASE_TS, BASE_TS + 60_000, BASE_TS + 120_000]
        self.assertEqual(mgr.new_timestamps("AAPL", window), [BASE_TS + 60_000, BASE_TS + 120_000])

    def test_uncommitted_step_does_not_advance_cursor(self):
        mgr = _make_manager()
        with mgr.step("AAPL", BASE_TS):
            pass  # simulate a failed forward pass: never call handle.commit()

        self.assertEqual(mgr.new_timestamps("AAPL", [BASE_TS, BASE_TS + 60_000]), [BASE_TS, BASE_TS + 60_000])

    def test_reset_clears_cursor(self):
        mgr = _make_manager()
        with mgr.step("AAPL", BASE_TS) as handle:
            handle.commit(torch.zeros_like(handle.hidden))

        mgr.reset("AAPL")
        self.assertEqual(mgr.new_timestamps("AAPL", [BASE_TS]), [BASE_TS])


if __name__ == "__main__":
    unittest.main()
