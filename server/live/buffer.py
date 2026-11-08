"""In-memory rolling bar buffers for live serving (docs/agent-docs/live-prediction-agent-plan.md
section 1.3 / step 2).

`TickerBuffer` holds one ticker's trailing raw bars, deduped by timestamp and evicted past a
max age. `RegimeBuffer` holds the shared regime-ETF basket as a set of per-ticker `TickerBuffer`s
fetched once per poll tick and reused across every tracked ticker's feature computation. Both
store bars as plain dicts matching `data_source.BAR_FIELDS` so they feed straight into
`data_source.bars_to_df` / `get_features_for_ticker` without reshaping.
"""
from __future__ import annotations

from ..data_source import BAR_FIELDS, REGIME_TICKERS

# 60 min is the longest rolling feature window (past_peak_dev_60m); keep extra headroom for the
# 20-bar mean in rel_volume_20m and for gaps in fetch cadence (plan.md section 1.3).
DEFAULT_MAX_AGE_MS = 90 * 60 * 1000


class TickerBuffer:
    """Rolling raw-bar buffer for one ticker: deduped by timestamp, evicted past `max_age_ms`."""

    def __init__(self, ticker: str, max_age_ms: int = DEFAULT_MAX_AGE_MS):
        self.ticker = ticker
        self.max_age_ms = max_age_ms
        self._bars: dict[int, dict] = {}

    def add_bars(self, bars: list[dict]) -> None:
        """Merge in new/updated bars (by timestamp, last-write-wins), then evict stale ones."""
        for bar in bars:
            if bar.get("ticker") != self.ticker:
                raise ValueError(f"bar for {bar.get('ticker')!r} does not belong in {self.ticker!r} buffer")
            missing = [f for f in BAR_FIELDS if f not in bar]
            if missing:
                raise ValueError(f"bar missing required field(s): {missing}")
            self._bars[bar["timestamp"]] = bar
        if self._bars:
            self._evict(max(self._bars))

    def _evict(self, latest_ts: int) -> None:
        cutoff = latest_ts - self.max_age_ms
        for ts in [ts for ts in self._bars if ts < cutoff]:
            del self._bars[ts]

    def bars(self) -> list[dict]:
        """All buffered bars, sorted by timestamp ascending."""
        return [self._bars[ts] for ts in sorted(self._bars)]

    def timestamps(self) -> list[int]:
        """Buffered bar timestamps, sorted ascending (feed to `HiddenStateManager.new_timestamps`)."""
        return sorted(self._bars)

    def latest_bar_ts(self) -> int | None:
        return max(self._bars) if self._bars else None

    def is_empty(self) -> bool:
        return not self._bars


class RegimeBuffer:
    """Shared rolling raw-bar buffer across the regime ETF basket (plan.md section 1.3)."""

    def __init__(self, tickers: list[str] | None = None, max_age_ms: int = DEFAULT_MAX_AGE_MS):
        self.tickers = list(tickers) if tickers is not None else list(REGIME_TICKERS)
        self._buffers = {t: TickerBuffer(t, max_age_ms=max_age_ms) for t in self.tickers}

    def add_bars(self, bars: list[dict]) -> None:
        by_ticker: dict[str, list[dict]] = {}
        for bar in bars:
            ticker = bar.get("ticker")
            if ticker not in self._buffers:
                raise ValueError(f"{ticker!r} is not part of the regime basket {self.tickers}")
            by_ticker.setdefault(ticker, []).append(bar)
        for ticker, ticker_bars in by_ticker.items():
            self._buffers[ticker].add_bars(ticker_bars)

    def bars(self) -> list[dict]:
        """All buffered bars across the whole basket, grouped by ticker then timestamp order."""
        combined: list[dict] = []
        for ticker in self.tickers:
            combined.extend(self._buffers[ticker].bars())
        return combined

    def latest_bar_ts(self) -> int | None:
        """Oldest of each ticker's latest timestamp -- the whole basket is only as fresh as its
        most-behind member, which is what should drive staleness detection."""
        timestamps = [ts for t in self.tickers if (ts := self._buffers[t].latest_bar_ts()) is not None]
        return min(timestamps) if timestamps else None

    def is_empty(self) -> bool:
        return all(self._buffers[t].is_empty() for t in self.tickers)
