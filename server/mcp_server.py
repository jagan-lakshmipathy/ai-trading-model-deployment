"""MCP server exposing the GRU price-prediction model (docs/agent-docs/live-prediction-agent-plan.md
section 1.6 / step 7; see docs/server-docs/plan.md section 4 for the original tool design).

Defaults to the baseline checkpoint (`checkpoints/best_model.pt`) -- the only variant validated
per docs/server-docs/model-deployment-spec.md section 2.2. Pass `normalized=True` to `create_server` to
serve the experimental z-scored variant instead; the two are never mixed within one server
instance, per the plan's "not silently mixed" requirement.

`predict(ticker)` no longer takes bars as input (that was the pre-live-data-fetching design,
see docs/agent-docs/live-prediction-agent-plan.md's "Rejected approach"). The server now owns
live data fetching/buffering (`live/buffer.py`, `live/scheduler.py`), dynamic ticker tracking
with warmup (`live/tracker.py`), and feeds `HiddenStateManager`'s processed-timestamp cursor
(`state.py`) so each bar only ever advances the model once, regardless of how often or how many
callers call `predict` for the same ticker.
"""
from __future__ import annotations

import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from .config import ModelConfig
from .data_source import get_features_for_ticker
from .inference import InferenceEngine
from .live.buffer import RegimeBuffer
from .live.calendar import MarketCalendar
from .live.config import LiveConfig, load_live_config
from .live.fred_client import FredCache, FredClient
from .live.massive_client import MassiveClient
from .live.scheduler import Scheduler
from .live.tracker import TickerTracker, WarmingUp
from .state import HiddenStateManager

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "checkpoints/best_model.pt"
DEFAULT_CHECKPOINT_NORMALIZED = "checkpoints/best_model_normalized.pt"


def create_server(
    checkpoint_path: str | None = None,
    normalized: bool = False,
    model_config: ModelConfig | None = None,
    live_config: LiveConfig | None = None,
    massive_client: MassiveClient | None = None,
    fred_cache: FredCache | None = None,
    regime_buffer: RegimeBuffer | None = None,
    scheduler: Scheduler | None = None,
    tracker: TickerTracker | None = None,
    market_calendar: MarketCalendar | None = None,
    start_scheduler: bool = True,
) -> MCPServer:
    """Build the MCP server with a freshly-loaded model, a fresh hidden-state cache, and the
    live data-fetching stack (buffers/scheduler/tracker).

    Every live component is independently injectable (used by tests to avoid real network
    calls/credentials) -- any left as `None` is built from `live_config` (or
    `load_live_config()`'s real environment-variable lookup if `live_config` is also `None`).
    """
    cfg = model_config or ModelConfig()
    path = checkpoint_path or (DEFAULT_CHECKPOINT_NORMALIZED if normalized else DEFAULT_CHECKPOINT)
    engine = InferenceEngine(path, cfg, n_features=42, normalized=normalized)
    state_mgr = HiddenStateManager(engine.model)

    live_cfg = live_config or load_live_config()
    massive = massive_client or MassiveClient(
        api_key=live_cfg.massive_api_key, base_url=live_cfg.massive_base_url
    )
    regime_buf = regime_buffer or RegimeBuffer(
        tickers=live_cfg.regime_tickers, max_age_ms=live_cfg.buffer_max_age_ms
    )
    fred = fred_cache or FredCache(
        FredClient(api_key=live_cfg.fred_api_key), refresh_interval_s=live_cfg.fred_refresh_interval_s
    )
    sched = scheduler or Scheduler(
        massive, regime_buf, fred_cache=fred,
        poll_interval_s=live_cfg.poll_interval_s, lookback_ms=live_cfg.lookback_ms,
        market_calendar=market_calendar or MarketCalendar(live_cfg.calendar_name),
        data_delay_ms=live_cfg.data_delay_ms,
    )
    track = tracker or TickerTracker(
        sched, massive, state_mgr, warmup_ms=live_cfg.warmup_ms, idle_timeout_s=live_cfg.idle_timeout_s,
        data_delay_ms=live_cfg.data_delay_ms,
    )

    if start_scheduler:
        # Prime the shared regime buffer synchronously before returning -- otherwise the first
        # predict() call can race the background poll thread's first tick and see an empty
        # regime buffer, which drops all 10 regime-derived feature columns entirely (not just
        # zero-fills them) and crashes the model's fixed-size input projection.
        try:
            sched.poll_once()
        except Exception:
            logger.exception("initial regime-buffer priming poll failed; will retry on next tick")
        sched.start()

    # Cache of the most recent prediction per ticker, returned as-is when predict() is called
    # again before any new bar has arrived -- avoids re-running inference on unchanged data.
    last_prediction: dict[str, dict[str, Any]] = {}

    server = MCPServer(
        name="ai-trading-model",
        description="Multi-horizon GRU peak/dip price prediction for 1-min equity bars.",
        version="0.2.0",
    )

    @server.tool()
    def predict(ticker: str) -> dict[str, Any]:
        """Predict per-horizon peak/dip prices for `ticker` from the server's own live buffers.

        Registers `ticker` for live polling on first use (triggering an immediate warmup
        backfill) and returns `{"status": "warming_up", ...}` instead of a prediction until
        enough history has accumulated -- never computes from a too-short rolling window. Once
        ready, only bars newer than this ticker's processed-bar cursor are forward-passed
        through the model (docs/agent-docs/live-prediction-agent-plan.md "Rejected approach"),
        so calling `predict` more often than new bars arrive just replays the last result.
        """
        try:
            track.ensure_ready(ticker)
        except WarmingUp as warming:
            return {
                "ticker": ticker,
                "status": "warming_up",
                "have_ms": warming.have_ms,
                "need_ms": warming.need_ms,
            }

        ticker_buffer = sched.buffer_for(ticker)
        ticker_bars = ticker_buffer.bars()
        regime_bars = regime_buf.bars()
        fred_rows = fred.get().to_dict(orient="index")

        meta, features = get_features_for_ticker(ticker_bars, regime_bars, fred_rows)
        ticker_meta = meta[meta["ticker"] == ticker].reset_index(drop=True)
        ticker_features = features[meta["ticker"] == ticker]
        if len(ticker_meta) == 0:
            raise ValueError(f"no bars available for ticker {ticker!r}")
        if ticker_features.shape[1] != 42:
            # Regime buffer was empty/incomplete when features were built (e.g. a startup race
            # with the scheduler's first poll tick) -- missing regime columns are dropped
            # entirely rather than zero-filled, so the feature matrix is the wrong shape for the
            # model. Report this distinctly rather than letting the model crash on a shape
            # mismatch; the next call should succeed once the regime buffer has caught up.
            return {
                "ticker": ticker,
                "status": "data_unavailable",
                "reason": f"expected 42 feature columns, got {ticker_features.shape[1]} "
                          "(regime buffer likely still empty/incomplete)",
            }

        timestamps = [int(ts) for ts in ticker_meta["timestamp"]]
        new_timestamps = state_mgr.new_timestamps(ticker, timestamps)
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        preds = None
        close = realized_vol = last_ts = None
        for ts in new_timestamps:
            idx = ts_to_idx[ts]
            with state_mgr.step(ticker, ts) as handle:
                step_preds, next_hidden = engine.forward(ticker_features[idx : idx + 1], handle.hidden)
                handle.commit(next_hidden)
            preds = step_preds[-1]
            close = float(ticker_meta["close"].iloc[idx])
            realized_vol = float(ticker_meta["realized_vol_15m"].iloc[idx])
            last_ts = ts

        if preds is None:
            # No bar newer than the cursor -- nothing to forward-pass; replay the last result.
            cached = last_prediction.get(ticker)
            if cached is None:
                return {"ticker": ticker, "status": "no_new_bars", "as_of_timestamp": None}
            return cached

        predictions = engine.denormalize(preds, close=close, realized_vol_15m=realized_vol)
        result = {
            "ticker": ticker,
            "status": "ok",
            "as_of_timestamp": last_ts,
            "close": close,
            "predictions": [
                {"horizon_minutes": p.horizon, "peak_price": p.peak_price, "dip_price": p.dip_price}
                for p in predictions
            ],
        }
        last_prediction[ticker] = result
        return result

    @server.tool()
    def reset_session(ticker: str) -> dict[str, str]:
        """Manually zero the cached hidden state/cursor for `ticker` (e.g. suspected state drift)."""
        state_mgr.reset(ticker)
        last_prediction.pop(ticker, None)
        return {"status": "reset", "ticker": ticker}

    @server.resource("model-info://contract")
    def model_info() -> dict[str, Any]:
        """Static summary of the model contract (docs/server-docs/model-deployment-spec.md) plus
        live buffer/tracking status (docs/agent-docs/live-prediction-agent-plan.md section 1.6)."""
        return {
            "checkpoint": path,
            "normalized": normalized,
            "horizons_minutes": cfg.horizons,
            "n_features": 42,
            "regime_tickers": regime_buf.tickers,
            "regime_latest_bar_ts": regime_buf.latest_bar_ts(),
            "tracked_tickers": [
                {"ticker": t, "latest_bar_ts": sched.buffer_for(t).latest_bar_ts()}
                for t in sched.tracked_tickers()
            ],
            "known_risks": [
                "No ticker_id -- model has no explicit per-ticker identity signal.",
                "No peak>=close>=dip constraint -- sanity-check/clip predictions downstream.",
                "Validated only on the 23 training tickers and the historical test-split window.",
                "Requires split/dividend-adjusted input prices matching the training data source.",
                "model.eval() is enforced server-side; do not re-enable dropout.",
                "Regime ETFs not covered by the Massive account's current plan tier silently "
                "drop out of the regime buffer (fetch_bars_for_many isolates per-ticker "
                "failures) -- see docs/agent-docs/live-prediction-agent-plan.md section 4.1.",
            ],
        }

    return server


def main() -> None:
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
