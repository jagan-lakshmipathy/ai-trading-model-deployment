"""Per-ticker GRU hidden-state cache for streaming inference (docs/server-docs/model-deployment-spec.md section 2.6).

Maintains one hidden state per ticker, zeroing it whenever a new bar belongs to a different
market-tz trading session than the last one seen for that ticker (new day, or first bar for a
ticker). Thread-safe via a per-ticker lock so concurrent MCP requests for different tickers
don't block each other, while requests for the *same* ticker are serialized (required anyway,
since they must be fed to the model in order).

Also tracks a per-ticker processed-bar cursor (last timestamp forward-passed through the
model). This lets a caller holding a trailing window of bars (e.g. a live predict() path that
re-fetches the last ~60 minutes every call) filter down to only the bars this ticker hasn't
seen yet via `new_timestamps`, so repeated/overlapping window submissions advance the hidden
state exactly once per bar instead of replaying already-processed bars on top of already-
advanced state (docs/agent-docs/live-prediction-agent-plan.md, "Rejected approach").
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import pandas as pd
import torch

from .encoder import MultiHorizonGRUModel
from .sessions import MARKET_TZ


def _session_date(timestamp_ms: int) -> str:
    """Market-timezone calendar date (YYYY-MM-DD) for a single epoch-ms timestamp."""
    return pd.Timestamp(timestamp_ms, unit="ms", tz="UTC").tz_convert(MARKET_TZ).strftime("%Y-%m-%d")


@dataclass
class _TickerState:
    hidden: torch.Tensor
    session_date: str
    last_processed_ts: int | None
    lock: threading.Lock


class StepHandle:
    """Yielded by `HiddenStateManager.step`; carries the hidden state to feed the model.

    Call `commit(next_hidden)` after a successful `model.forward` to persist the new state.
    If the `with` block exits without a `commit` call (including via an exception), the cached
    state is left untouched — the next `step()` for this ticker simply reuses the same state
    it would have before this call, rather than silently advancing on a failed/aborted step.
    """

    def __init__(self, state: _TickerState, timestamp_ms: int):
        self._state = state
        self._timestamp_ms = timestamp_ms
        self.hidden = state.hidden
        self._committed = False

    def commit(self, next_hidden: torch.Tensor) -> None:
        self._state.hidden = next_hidden
        self._state.last_processed_ts = self._timestamp_ms
        self._committed = True


class HiddenStateManager:
    """Keyed by ticker: caches the GRU hidden state and the last-seen session date."""

    def __init__(self, model: MultiHorizonGRUModel, device: torch.device | None = None):
        self._model = model
        self._device = device or torch.device("cpu")
        self._states: dict[str, _TickerState] = {}
        self._registry_lock = threading.Lock()  # guards creation of new per-ticker entries only

    def _get_or_create(self, ticker: str) -> _TickerState:
        with self._registry_lock:
            state = self._states.get(ticker)
            if state is None:
                state = _TickerState(
                    hidden=self._model.init_hidden(1, device=self._device),
                    session_date="",
                    last_processed_ts=None,
                    lock=threading.Lock(),
                )
                self._states[ticker] = state
            return state

    def new_timestamps(self, ticker: str, timestamps: Sequence[int]) -> list[int]:
        """Filter a trailing window's bar timestamps down to ones not yet forward-passed.

        `timestamps` must be sorted strictly ascending with no duplicates (raises ValueError
        otherwise). Returns the subsequence strictly after this ticker's processed-bar cursor,
        or all of them if the ticker has no cursor yet (never stepped, or reset since).
        """
        ts_list = list(timestamps)
        if any(a >= b for a, b in zip(ts_list, ts_list[1:])):
            raise ValueError("timestamps must be sorted strictly ascending with no duplicates")
        state = self._get_or_create(ticker)
        with state.lock:
            cursor = state.last_processed_ts
        if cursor is None:
            return ts_list
        return [ts for ts in ts_list if ts > cursor]

    @contextmanager
    def step(self, ticker: str, timestamp_ms: int) -> Iterator[StepHandle]:
        """Context manager yielding a `StepHandle` with the hidden state for this ticker's next bar.

        Zeroes the cached state first if `timestamp_ms` falls on a new market-tz session (new
        day, or first bar ever seen for this ticker). Usage:

            with state_mgr.step(ticker, bar.timestamp) as handle:
                preds, next_hidden = model(x, handle.hidden)
                handle.commit(next_hidden)
        """
        state = self._get_or_create(ticker)
        with state.lock:
            session_date = _session_date(timestamp_ms)
            if session_date != state.session_date:
                state.hidden = self._model.init_hidden(1, device=self._device)
                state.session_date = session_date
            yield StepHandle(state, timestamp_ms)

    def reset(self, ticker: str) -> None:
        """Manually force the next `step(ticker, ...)` call to zero the hidden state.

        Also clears the processed-bar cursor, so the next `new_timestamps` call treats every
        bar as new (a full replay is exactly what you want after a suspected state drift).
        """
        state = self._get_or_create(ticker)
        with state.lock:
            state.hidden = self._model.init_hidden(1, device=self._device)
            state.session_date = ""
            state.last_processed_ts = None

    def reset_all(self) -> None:
        with self._registry_lock:
            tickers = list(self._states.keys())
        for ticker in tickers:
            self.reset(ticker)

