"""Unit tests for FredCache (docs/agent-docs/live-prediction-agent-plan.md step 3).

Uses a stub data source instead of real `fredapi`/network calls -- FredCache only depends on
duck-typed `fetch_all() -> pd.DataFrame`, which is all these tests exercise.
"""
import unittest
from unittest import mock

import pandas as pd

from server.live.fred_client import FredCache


class _StubFredSource:
    def __init__(self, frames: list[pd.DataFrame]):
        self._frames = list(frames)
        self.call_count = 0

    def fetch_all(self) -> pd.DataFrame:
        self.call_count += 1
        # Return the next canned frame, repeating the last one once exhausted.
        idx = min(self.call_count - 1, len(self._frames) - 1)
        return self._frames[idx]


def _frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values)


class FredCacheTests(unittest.TestCase):
    def test_first_get_triggers_fetch(self):
        source = _StubFredSource([_frame({"FEDFUNDS": [5.0, None]})])
        cache = FredCache(source, refresh_interval_s=60)
        df = cache.get(now=0.0)
        self.assertEqual(source.call_count, 1)
        self.assertEqual(df["FEDFUNDS"].tolist(), [5.0, 5.0])  # forward-filled

    def test_second_get_before_interval_does_not_refetch(self):
        source = _StubFredSource([_frame({"FEDFUNDS": [5.0]})])
        cache = FredCache(source, refresh_interval_s=60)
        cache.get(now=0.0)
        cache.get(now=30.0)
        self.assertEqual(source.call_count, 1)

    def test_refetches_after_interval_elapses(self):
        source = _StubFredSource([_frame({"FEDFUNDS": [5.0]}), _frame({"FEDFUNDS": [5.25]})])
        cache = FredCache(source, refresh_interval_s=60)
        first = cache.get(now=0.0)
        second = cache.get(now=61.0)
        self.assertEqual(source.call_count, 2)
        self.assertEqual(first["FEDFUNDS"].tolist(), [5.0])
        self.assertEqual(second["FEDFUNDS"].tolist(), [5.25])

    def test_is_stale_before_first_fetch(self):
        source = _StubFredSource([_frame({"FEDFUNDS": [5.0]})])
        cache = FredCache(source, refresh_interval_s=60)
        self.assertTrue(cache.is_stale(now=0.0))
        cache.get(now=0.0)
        self.assertFalse(cache.is_stale(now=10.0))
        self.assertTrue(cache.is_stale(now=61.0))


class FredClientRequiresApiKeyTests(unittest.TestCase):
    def test_raises_without_api_key(self):
        from server.live.fred_client import FredClient

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                FredClient(api_key=None)


if __name__ == "__main__":
    unittest.main()
