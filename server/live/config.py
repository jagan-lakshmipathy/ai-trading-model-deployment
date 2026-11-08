"""Live-serving configuration: env-var secrets + tunable poll/warmup/eviction settings
(docs/agent-docs/live-prediction-agent-plan.md section 1.7 / step 6).

`load_live_config` fails fast with a single, complete error listing every missing required
secret, rather than failing confusingly later on the first poll tick deep inside
massive_client.py/fred_client.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..data_source import REGIME_TICKERS
from .buffer import DEFAULT_MAX_AGE_MS
from .calendar import DEFAULT_CALENDAR_NAME
from .fred_client import DEFAULT_REFRESH_INTERVAL_S as DEFAULT_FRED_REFRESH_INTERVAL_S
from .massive_client import DEFAULT_BASE_URL
from .scheduler import DEFAULT_DATA_DELAY_MS, DEFAULT_LOOKBACK_MS, DEFAULT_POLL_INTERVAL_S
from .tracker import DEFAULT_IDLE_TIMEOUT_S, DEFAULT_WARMUP_MS

REQUIRED_ENV_VARS = ("MASSIVE_API_KEY", "FRED_API_KEY")


@dataclass(frozen=True)
class LiveConfig:
    massive_api_key: str
    fred_api_key: str
    massive_base_url: str = DEFAULT_BASE_URL
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S
    lookback_ms: int = DEFAULT_LOOKBACK_MS
    buffer_max_age_ms: int = DEFAULT_MAX_AGE_MS
    fred_refresh_interval_s: int = DEFAULT_FRED_REFRESH_INTERVAL_S
    warmup_ms: int = DEFAULT_WARMUP_MS
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S
    regime_tickers: list[str] = field(default_factory=lambda: list(REGIME_TICKERS))
    calendar_name: str = DEFAULT_CALENDAR_NAME
    # How far behind "now" to request data from -- must cover the Massive/Polygon account's
    # delayed-data entitlement (e.g. Stocks Starter's 15-min delay) or every fetch gets a
    # NOT_AUTHORIZED error regardless of ticker. Override via MASSIVE_DATA_DELAY_MS (ms) if your
    # plan's delay differs from the 16-min default (15-min delay + 1-min safety buffer).
    data_delay_ms: int = DEFAULT_DATA_DELAY_MS


def load_live_config(env: dict[str, str] | None = None) -> LiveConfig:
    """Build a `LiveConfig` from environment variables.

    Raises `RuntimeError` listing every missing required secret at once (not just the first one
    hit), so a misconfigured deployment gets one complete error up front instead of a chain of
    unset-env-var exceptions surfacing one at a time across different live modules.
    """
    source = env if env is not None else os.environ
    missing = [name for name in REQUIRED_ENV_VARS if not source.get(name)]
    if missing:
        raise RuntimeError(f"missing required environment variable(s): {', '.join(missing)}")
    kwargs: dict[str, object] = {
        "massive_api_key": source["MASSIVE_API_KEY"],
        "fred_api_key": source["FRED_API_KEY"],
    }
    delay_override = source.get("MASSIVE_DATA_DELAY_MS")
    if delay_override:
        kwargs["data_delay_ms"] = int(delay_override)
    return LiveConfig(**kwargs)
