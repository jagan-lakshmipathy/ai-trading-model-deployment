"""Join trading/regime/macro data sources into feature + target arrays.

Implements a first-cut subset of the source repo's price-prediction-feature-design.md (Feature Groups
1-3) -- enough signal to validate the TBPTT pipeline end-to-end. Rolling features are
computed per-ticker over the full series rather than reset at session boundaries (the
TBPTT hidden-state reset in `batching.py` already prevents state leakage across sessions
at the model level); richer indicators (RSI, MACD, Bollinger, ATR, beta-vs-SPY) are left
as a follow-up once this baseline is validated.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .sessions import MARKET_TZ

BAR_COLUMNS = "ticker, timestamp, open, high, low, close, volume, vwap, transactions"


def load_bars(db_path: str, tickers: list[str], start_ms: int, end_ms: int) -> pd.DataFrame:
    """Load `bars` rows for `tickers` in `[start_ms, end_ms]`, sorted by (ticker, timestamp)."""
    placeholders = ",".join("?" * len(tickers))
    query = (
        f"SELECT {BAR_COLUMNS} FROM bars "
        f"WHERE ticker IN ({placeholders}) AND timestamp BETWEEN ? AND ? "
        f"ORDER BY ticker, timestamp"
    )
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=[*tickers, start_ms, end_ms])


def get_earliest_timestamp(db_path: str, tickers: list[str]) -> int:
    placeholders = ",".join("?" * len(tickers))
    query = f"SELECT MIN(timestamp) FROM bars WHERE ticker IN ({placeholders})"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(query, tickers).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"No bars found for tickers {tickers} in {db_path}")
    return int(row[0])


def load_regime_wide(db_path: str, regime_tickers: list[str], start_ms: int, end_ms: int) -> pd.DataFrame:
    """Load regime ETF bars and pivot to one `close`/`volume` column pair per ticker, indexed by timestamp."""
    bars = load_bars(db_path, regime_tickers, start_ms, end_ms)
    close = bars.pivot(index="timestamp", columns="ticker", values="close")
    volume = bars.pivot(index="timestamp", columns="ticker", values="volume")
    close.columns = [f"{t}_close" for t in close.columns]
    volume.columns = [f"{t}_volume" for t in volume.columns]
    return pd.concat([close, volume], axis=1).sort_index().ffill()


def load_fred(csv_path: str) -> pd.DataFrame:
    """Load FRED macro CSV, forward-filled, indexed by calendar date (YYYY-MM-DD)."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = df.index.strftime("%Y-%m-%d")
    return df.ffill()


def _ret(close: pd.Series, k: int) -> pd.Series:
    return close / close.shift(k) - 1.0


def _macro_features(session_date: pd.Series, fred_df: pd.DataFrame) -> pd.DataFrame:
    aligned = fred_df.reindex(session_date).reset_index(drop=True)
    aligned.index = session_date.index

    last_update = pd.Series(pd.NaT, index=session_date.index)
    non_null = aligned.notna().any(axis=1)
    last_update[non_null] = pd.to_datetime(session_date[non_null])
    last_update = last_update.ffill()
    days_since_macro_update = (pd.to_datetime(session_date) - last_update).dt.days.fillna(0)

    out = pd.DataFrame(index=session_date.index)
    out["fed_funds_rate"] = aligned.get("FEDFUNDS")
    out["cpi_yoy"] = aligned.get("CPIAUCSL").pct_change(252) if "CPIAUCSL" in aligned else np.nan
    out["core_cpi_yoy"] = aligned.get("CPILFESL").pct_change(252) if "CPILFESL" in aligned else np.nan
    out["ppi_yoy"] = aligned.get("PPIACO").pct_change(252) if "PPIACO" in aligned else np.nan
    out["gdp_qoq"] = aligned.get("GDPC1").pct_change(1) if "GDPC1" in aligned else np.nan
    out["unemployment_rate"] = aligned.get("UNRATE")
    out["payrolls_mom_change"] = aligned.get("PAYEMS").diff() if "PAYEMS" in aligned else np.nan
    out["pce_mom_change"] = aligned.get("PCE").pct_change(1) if "PCE" in aligned else np.nan
    out["yield_10y"] = aligned.get("DGS10")
    out["yield_2y"] = aligned.get("DGS2")
    out["yield_curve_slope"] = out["yield_10y"] - out["yield_2y"]
    out["days_since_macro_update"] = days_since_macro_update.to_numpy()
    return out.ffill().fillna(0.0)


def _compute_feature_columns(
    df: pd.DataFrame, g, regime_wide: pd.DataFrame, fred_df: pd.DataFrame
) -> pd.DataFrame:
    """Feature Groups 1-3 only (docs/server-docs/model-deployment-spec.md section 2.1) -- no targets.

    Shared by both `build_features_and_targets` (training, adds forward-looking targets) and
    `build_features` (serving, which never has future bars to compute targets from). Must stay
    the single source of this math so the two entry points can't silently diverge -- see
    docs/server-docs/plan.md section 6.
    """
    close = df["close"]

    feats = pd.DataFrame(index=df.index)

    # Group 1 -- target ticker price/volume
    for k in (5, 15, 30, 60):
        feats[f"past_peak_dev_{k}m"] = (
            g["high"].transform(lambda s, k=k: s.rolling(k, min_periods=1).max()) - close
        ) / close
        feats[f"past_dip_dev_{k}m"] = (
            g["low"].transform(lambda s, k=k: s.rolling(k, min_periods=1).min()) - close
        ) / close
    feats["ret_1m"] = g["close"].transform(lambda s: _ret(s, 1))
    feats["log_volume"] = np.log1p(df["volume"])
    vol_mean_20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean().shift(1))
    feats["rel_volume_20m"] = df["volume"] / vol_mean_20.replace(0, np.nan)
    feats["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    feats["bar_range"] = (df["high"] - df["low"]) / close
    log_ret = g["close"].transform(lambda s: np.log(s / s.shift(1)))
    feats["realized_vol_15m"] = log_ret.groupby(df["ticker"]).transform(
        lambda s: s.rolling(15, min_periods=2).std()
    )

    session_date = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(MARKET_TZ)
    minutes_since_open = (session_date.dt.hour - 9) * 60 + (session_date.dt.minute - 30)
    feats["minutes_since_open"] = minutes_since_open.clip(lower=0)
    feats["minutes_to_close"] = (16 * 60 - (session_date.dt.hour * 60 + session_date.dt.minute)).clip(lower=0)
    tod = session_date.dt.hour * 60 + session_date.dt.minute
    feats["tod_sin"] = np.sin(2 * np.pi * tod / (24 * 60))
    feats["tod_cos"] = np.cos(2 * np.pi * tod / (24 * 60))
    dow = session_date.dt.dayofweek
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Group 2 -- cross-asset regime signals
    regime = regime_wide.reindex(df["timestamp"]).reset_index(drop=True)
    regime.index = df.index
    if "SPY_close" in regime:
        feats["spy_ret_1m"] = _ret(regime["SPY_close"], 1)
        feats["spy_ret_5m"] = _ret(regime["SPY_close"], 5)
        feats["spy_ret_15m"] = _ret(regime["SPY_close"], 15)
        feats["market_realized_vol_15m"] = (
            np.log(regime["SPY_close"] / regime["SPY_close"].shift(1)).rolling(15, min_periods=2).std()
        )
    if "QQQ_close" in regime:
        feats["qqq_ret_5m"] = _ret(regime["QQQ_close"], 5)
    if "IWM_close" in regime:
        feats["iwm_ret_5m"] = _ret(regime["IWM_close"], 5)
    if "HYG_close" in regime and "TLT_close" in regime:
        feats["risk_on_off_spread"] = _ret(regime["HYG_close"], 5) - _ret(regime["TLT_close"], 5)
    if "GLD_close" in regime:
        feats["gold_ret_5m"] = _ret(regime["GLD_close"], 5)
    if "USD_close" in regime:
        feats["usd_ret_5m"] = _ret(regime["USD_close"], 5)
    if "SPY_volume" in regime:
        spy_vol_mean_20 = regime["SPY_volume"].rolling(20, min_periods=1).mean().shift(1)
        feats["rel_volume_spy_5m"] = regime["SPY_volume"] / spy_vol_mean_20.replace(0, np.nan)

    # Group 3 -- macro context, forward-filled per calendar date
    date_str = session_date.dt.strftime("%Y-%m-%d")
    feats = pd.concat([feats, _macro_features(date_str, fred_df)], axis=1)

    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(
    trading_df: pd.DataFrame, regime_wide: pd.DataFrame, fred_df: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Serving-time path: same Feature Groups 1-3 math as training, no target computation.

    Unlike `build_features_and_targets`, this never looks forward in time (no
    `s.shift(-1)`/future bars needed), so it's safe to call with only data available up to
    `now` -- the live-serving case where tomorrow's bars don't exist yet.
    Returns (meta_df, features): meta_df has `ticker`/`timestamp`/`close`/`realized_vol_15m`
    row-aligned with `trading_df` (sorted by ticker, timestamp), needed to denormalize
    predictions back to price (see `inference.denormalize`).
    """
    df = trading_df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False)
    feats = _compute_feature_columns(df, g, regime_wide, fred_df)

    meta = df[["ticker", "timestamp", "close"]].reset_index(drop=True)
    meta["realized_vol_15m"] = feats["realized_vol_15m"].to_numpy()
    return meta, feats.to_numpy(dtype=np.float32)


def build_features_and_targets(
    trading_df: pd.DataFrame,
    regime_wide: pd.DataFrame,
    fred_df: pd.DataFrame,
    horizons: list[int],
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Join the three sources and compute Feature Groups 1-3 plus volatility-scaled deltas.

    Returns (meta_df, features, targets): meta_df has `ticker`/`timestamp` row-aligned with
    `trading_df` (sorted by ticker, timestamp) for use with `build_sessions`; `targets`
    columns are ordered [delta_peak_h0, delta_dip_h0, delta_peak_h1, ...] per `horizons`.
    """
    df = trading_df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False)
    close = df["close"]
    feats = _compute_feature_columns(df, g, regime_wide, fred_df)

    # Targets: volatility-scaled deltas (docs/server-docs/model-design.md section 2)
    realized_vol = feats["realized_vol_15m"].replace(0, np.nan)
    target_cols = []
    for w in horizons:
        future_high = g["high"].transform(lambda s, w=w: s.shift(-1).rolling(w, min_periods=1).max())
        future_low = g["low"].transform(lambda s, w=w: s.shift(-1).rolling(w, min_periods=1).min())
        delta_peak = (future_high - close) / (close * realized_vol + eps)
        delta_dip = (future_low - close) / (close * realized_vol + eps)
        # Near-zero realized_vol (illiquid bars) blows up the ratio; clip to a sane range.
        target_cols.append(delta_peak.fillna(0.0).clip(-20.0, 20.0))
        target_cols.append(delta_dip.fillna(0.0).clip(-20.0, 20.0))
    targets = np.stack([c.to_numpy() for c in target_cols], axis=1).astype(np.float32)

    # close/realized_vol_15m kept in meta so eval code can denormalize delta targets back to price.
    meta = df[["ticker", "timestamp", "close"]].reset_index(drop=True)
    meta["realized_vol_15m"] = feats["realized_vol_15m"].to_numpy()
    return meta, feats.to_numpy(dtype=np.float32), targets
