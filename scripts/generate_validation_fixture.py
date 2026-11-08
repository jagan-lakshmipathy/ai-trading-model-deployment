"""Generate tests/fixtures/validation_reference.npz -- the frozen ground-truth reference used by
server/validate_against_eval.py (docs/plan.md section 7).

Must be run with the SOURCE `ai-trading-model` repo's own interpreter/venv (it imports
`ai_trading_model.*`, which only exists there), with CWD set to that repo so its relative
`config/train_config.yaml` / `data/*.db` paths resolve. Example:

    cd /home/jagan/projects/ai-systems/ai-trading-model
    .venv/bin/python /home/jagan/projects/ai-systems/ai-trading-model-deployment/scripts/generate_validation_fixture.py

Picks the first two full trading sessions for `TICKER` starting `DAYS_OFFSET` days into the
dataset (skipping ahead so there's non-trivial history/volatility rather than the very first,
possibly-thin day of data), and captures everything server/validate_against_eval.py needs to
replay those two sessions through this repo's serving path and diff against them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # so `ai_trading_model` resolves when this script isn't run via `-m`

from ai_trading_model.config import load_config
from ai_trading_model.input import (
    build_features_and_targets,
    get_earliest_timestamp,
    load_bars,
    load_fred,
    load_regime_wide,
)
from ai_trading_model.input.sessions import build_sessions
from ai_trading_model.model.encoder import MultiHorizonGRUModel

TICKER = "AAPL"
DAYS_OFFSET = 400  # calendar days into the dataset before picking sessions
WINDOW_DAYS = 5  # span of bars to load, must be enough to contain 2 full sessions
OUTPUT_PATH = (
    "/home/jagan/projects/ai-systems/ai-trading-model-deployment/tests/fixtures/validation_reference.npz"
)
CHECKPOINT_PATH = "checkpoints/best_model.pt"  # baseline, unnormalized -- see spec section 2.2
CONFIG_PATH = "config/train_config.yaml"
EPS = 1e-8  # must match training's target-normalization eps (docs/model-design.md section 2)
MS_PER_DAY = 24 * 60 * 60 * 1000


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    earliest_ts = get_earliest_timestamp(cfg.data.trading_db, [TICKER])
    start_ts = earliest_ts + DAYS_OFFSET * MS_PER_DAY
    end_ts = start_ts + WINDOW_DAYS * MS_PER_DAY

    trading_df = load_bars(cfg.data.trading_db, [TICKER], start_ts, end_ts)
    regime_df_raw = load_bars(cfg.data.regime_db, cfg.data.regime_tickers, start_ts, end_ts)
    regime_wide = load_regime_wide(cfg.data.regime_db, cfg.data.regime_tickers, start_ts, end_ts)
    fred_df_full = load_fred(cfg.data.fred_csv)

    meta, features, _targets = build_features_and_targets(
        trading_df, regime_wide, fred_df_full, cfg.model.horizons
    )
    sessions = build_sessions(meta)
    if len(sessions) < 2:
        raise RuntimeError(f"need at least 2 sessions, found {len(sessions)} -- widen WINDOW_DAYS")
    sess0, sess1 = sessions[0], sessions[1]
    print(f"session0: {sess0}\nsession1: {sess1}")

    model = MultiHorizonGRUModel(
        n_features=features.shape[1],
        horizons=cfg.model.horizons,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        head_hidden_size=cfg.model.head_hidden_size,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    # Each session run as one batched GRU call, matching how train.py/evaluate.py would process
    # a full session in one TBPTT chunk -- NOT how serving replays it (bar-by-bar, see
    # server/validate_against_eval.py's note on the resulting floating-point tolerance).
    preds_by_session = []
    for sess in (sess0, sess1):
        hidden = model.init_hidden(1)
        sess_feats = features[sess.start_idx : sess.end_idx]
        x = torch.from_numpy(sess_feats).unsqueeze(0)
        with torch.no_grad():
            preds, _ = model(x, hidden)
        preds_by_session.append(preds.squeeze(0).numpy())
    preds0, preds1 = preds_by_session

    def denorm(preds: np.ndarray, close: np.ndarray, vol: np.ndarray) -> np.ndarray:
        return close + preds * (close * vol + EPS)

    close0 = meta["close"].to_numpy()[sess0.start_idx : sess0.end_idx]
    vol0 = meta["realized_vol_15m"].to_numpy()[sess0.start_idx : sess0.end_idx]
    close1 = meta["close"].to_numpy()[sess1.start_idx : sess1.end_idx]
    vol1 = meta["realized_vol_15m"].to_numpy()[sess1.start_idx : sess1.end_idx]
    priced0 = denorm(preds0, close0[:, None, None], vol0[:, None, None])
    priced1 = denorm(preds1, close1[:, None, None], vol1[:, None, None])

    # Trim the raw bars handed to the serving path to the two sessions plus 60 min of lookback
    # before session0 (longest rolling feature window -- past_peak_dev_60m), so the serving path
    # gets exactly the warmup it needs without unrelated extra history.
    ts0 = trading_df["timestamp"].iloc[sess0.start_idx]
    ts1 = trading_df["timestamp"].iloc[sess1.end_idx - 1]
    lookback_start = ts0 - 60 * 60_000
    ticker_bars_slice = trading_df[
        (trading_df["timestamp"] >= lookback_start) & (trading_df["timestamp"] <= ts1)
    ]
    regime_bars_slice = regime_df_raw[
        (regime_df_raw["timestamp"] >= lookback_start) & (regime_df_raw["timestamp"] <= ts1)
    ]

    date_strs = sorted(
        set(
            __import__("pandas")
            .to_datetime(ticker_bars_slice["timestamp"], unit="ms", utc=True)
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
        )
    )
    fred_rows = {
        d: (fred_df_full.loc[d].dropna().to_dict() if d in fred_df_full.index else {}) for d in date_strs
    }

    np.savez(
        OUTPUT_PATH,
        ticker=TICKER,
        ticker_bars=ticker_bars_slice.to_records(index=False),
        regime_bars=regime_bars_slice.to_records(index=False),
        fred_rows_json=json.dumps(fred_rows),
        sess0_start=sess0.start_idx,
        sess0_end=sess0.end_idx,
        sess0_date=sess0.session_date,
        sess1_start=sess1.start_idx,
        sess1_end=sess1.end_idx,
        sess1_date=sess1.session_date,
        features_sess0=features[sess0.start_idx : sess0.end_idx],
        features_sess1=features[sess1.start_idx : sess1.end_idx],
        preds_sess0=preds0,
        preds_sess1=preds1,
        priced_sess0=priced0,
        priced_sess1=priced1,
        close_sess0=close0,
        vol_sess0=vol0,
        close_sess1=close1,
        vol_sess1=vol1,
        horizons=np.array(cfg.model.horizons),
    )
    print(f"saved reference fixture to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
