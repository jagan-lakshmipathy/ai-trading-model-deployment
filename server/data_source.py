"""Serving-time data assembly -- "accept pre-fetched bars" mode (docs/server-docs/plan.md sections 5, 6, 8).

This module does not fetch anything itself yet: callers (ultimately the MCP `predict` tool)
supply the trailing window of target-ticker bars, regime ETF bars, and macro rows directly, and
this module just shapes them into the `(trading_df, regime_wide, fred_df)` triple
`features.build_features` expects. A live Massive/IBKR + FRED API-backed implementation is
deferred (see docs/server-docs/plan.md section 8, step 7) and should slot in behind the same
`get_features_for_ticker` signature.
"""
from __future__ import annotations

import pandas as pd

from .features import BAR_COLUMNS, build_features

REGIME_TICKERS = ["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD", "USD"]
BAR_FIELDS = [c.strip() for c in BAR_COLUMNS.split(",")]


def bars_to_df(bars: list[dict]) -> pd.DataFrame:
    """Validate and shape a list of caller-supplied bar dicts into the `bars` schema.

    Each dict must have exactly the fields in `BAR_FIELDS` (docs/server-docs/data-readme.md `bars` table).
    Raises ValueError on missing fields or non-monotonic/duplicate timestamps per ticker, checked
    on the as-given row order (spec section 4, "reject bars with non-monotonic timestamps per
    ticker") before any sorting -- a caller streaming bars out of order is a bug to surface, not
    silently paper over by sorting first.
    """
    if not bars:
        raise ValueError("bars must be non-empty")
    missing = [f for f in BAR_FIELDS if any(f not in b for b in bars)]
    if missing:
        raise ValueError(f"bar(s) missing required field(s): {missing}")

    raw = pd.DataFrame(bars)[BAR_FIELDS]
    for ticker, group in raw.groupby("ticker"):
        if not group["timestamp"].is_monotonic_increasing:
            raise ValueError(f"non-monotonic timestamps for ticker {ticker!r}")
        if group["timestamp"].duplicated().any():
            raise ValueError(f"duplicate timestamps for ticker {ticker!r}")

    df = raw.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return df


def fred_rows_to_df(fred_rows: dict[str, dict]) -> pd.DataFrame:
    """Shape caller-supplied `{date_str: {series_id: value, ...}, ...}` into `load_fred`'s format.

    Mirrors `features.load_fred`'s output: forward-filled, indexed by calendar date (YYYY-MM-DD).
    """
    df = pd.DataFrame.from_dict(fred_rows, orient="index").sort_index()
    return df.ffill()


def get_features_for_ticker(
    ticker_bars: list[dict],
    regime_bars: list[dict],
    fred_rows: dict[str, dict],
) -> tuple[pd.DataFrame, "pd.Series"]:
    """Build the 42-column feature matrix for one ticker from pre-fetched inputs.

    `ticker_bars`: trailing window of 1-min bars for the target ticker (needs ~60 minutes of
    history for the longest rolling window, `past_peak_dev_60m` -- see docs/server-docs/plan.md section 5).
    `regime_bars`: trailing window of bars for `REGIME_TICKERS` (any subset missing drops those
    feature columns to 0 after `fillna`, which silently diverges from the 42-column training
    contract -- caller should supply the full basket per spec section 2.4).
    `fred_rows`: macro rows as `{date_str: {series_id: value}}`, same series IDs as
    docs/server-docs/data-readme.md.

    Returns (meta, features) exactly as `features.build_features` does, for one or more tickers
    -- callers feeding a single ticker get all rows for that ticker only, in timestamp order.
    """
    trading_df = bars_to_df(ticker_bars)
    regime_df = bars_to_df(regime_bars) if regime_bars else pd.DataFrame(columns=BAR_FIELDS)
    fred_df = fred_rows_to_df(fred_rows) if fred_rows else pd.DataFrame()

    if not regime_df.empty:
        close = regime_df.pivot(index="timestamp", columns="ticker", values="close")
        volume = regime_df.pivot(index="timestamp", columns="ticker", values="volume")
        close.columns = [f"{t}_close" for t in close.columns]
        volume.columns = [f"{t}_volume" for t in volume.columns]
        regime_wide = pd.concat([close, volume], axis=1).sort_index().ffill()
    else:
        regime_wide = pd.DataFrame()

    return build_features(trading_df, regime_wide, fred_df)
