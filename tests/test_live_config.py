"""Unit tests for load_live_config (docs/agent-docs/live-prediction-agent-plan.md step 6)."""
import unittest

from server.live.config import LiveConfig, load_live_config


class LoadLiveConfigTests(unittest.TestCase):
    def test_raises_when_both_vars_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            load_live_config(env={})
        self.assertIn("MASSIVE_API_KEY", str(ctx.exception))
        self.assertIn("FRED_API_KEY", str(ctx.exception))

    def test_raises_when_one_var_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            load_live_config(env={"MASSIVE_API_KEY": "mk"})
        self.assertIn("FRED_API_KEY", str(ctx.exception))
        self.assertNotIn("MASSIVE_API_KEY", str(ctx.exception))

    def test_raises_on_empty_string_value(self):
        with self.assertRaises(RuntimeError):
            load_live_config(env={"MASSIVE_API_KEY": "", "FRED_API_KEY": "fk"})

    def test_builds_config_with_defaults_when_both_present(self):
        config = load_live_config(env={"MASSIVE_API_KEY": "mk", "FRED_API_KEY": "fk"})
        self.assertIsInstance(config, LiveConfig)
        self.assertEqual(config.massive_api_key, "mk")
        self.assertEqual(config.fred_api_key, "fk")
        self.assertEqual(config.poll_interval_s, 60)
        self.assertEqual(config.regime_tickers, ["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD", "USD"])


if __name__ == "__main__":
    unittest.main()
