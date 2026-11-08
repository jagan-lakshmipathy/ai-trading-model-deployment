"""Derive per-(ticker, trading-day) session boundaries from 1-min bars.

See "Batching for recurrent state" in docs/server-docs/model-design.md: a session is one
ticker's contiguous run of bars within a single market-timezone trading day,
and is the unit that TBPTT batching streams and resets hidden state on.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MARKET_TZ = "America/New_York"


@dataclass(frozen=True)
class Session:
    ticker: str
    session_date: str  # YYYY-MM-DD in market timezone
    start_idx: int  # row index into the owning DataFrame/array, inclusive
    end_idx: int  # row index into the owning DataFrame/array, exclusive


def add_session_date(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.Series:
    """Market-timezone calendar date (YYYY-MM-DD) for each row's timestamp (epoch ms, UTC)."""
    ts = pd.to_datetime(df[timestamp_col], unit="ms", utc=True).dt.tz_convert(MARKET_TZ)
    return ts.dt.strftime("%Y-%m-%d")


def build_sessions(
    df: pd.DataFrame, ticker_col: str = "ticker", timestamp_col: str = "timestamp"
) -> list[Session]:
    """Split a multi-ticker, multi-day bars DataFrame into contiguous (ticker, session_date) blocks.

    Requires df sorted by (ticker, timestamp) -- session boundaries are derived from row
    contiguity, so an unsorted frame would silently produce bogus sessions.
    """
    sort_cols = [ticker_col, timestamp_col]
    if not df[sort_cols].equals(df[sort_cols].sort_values(sort_cols)):
        raise ValueError("df must be sorted by (ticker, timestamp) before building sessions")

    session_date = add_session_date(df, timestamp_col)
    group_key = list(zip(df[ticker_col], session_date))

    sessions: list[Session] = []
    start_idx = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or group_key[i] != group_key[start_idx]:
            ticker, date = group_key[start_idx]
            sessions.append(Session(ticker=ticker, session_date=date, start_idx=start_idx, end_idx=i))
            start_idx = i
    return sessions
