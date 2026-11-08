"""JSONL recorder for predict() results, feeding scripts/plot_predictions.py.

The poller only logs to stdout by default -- this persists one JSON line per `predict()` result
so prediction history survives the process and can be visualized later.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlRecorder:
    """Callable compatible with `Poller(on_result=...)`; appends one JSON line per call."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, ticker: str, result: dict[str, Any]) -> None:
        record = {"recorded_at_ms": int(time.time() * 1000), "ticker": ticker, **result}
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")
