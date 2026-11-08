"""Plot predict(ticker) history against actual price (docs/agent-docs/live-prediction-agent-plan.md
addendum: visualization tool).

Reads a JSONL file produced by `agent.poller --log-file ...` (via `agent.recorder.JsonlRecorder`)
and plots:
  - the actual realized close price over time, as recorded at each poll tick, and
  - for each requested horizon (default 5, 10, 15 -- the model's trained horizons), the
    predicted peak/dip price, plotted at the *target* time (as_of_timestamp + horizon minutes)
    so it lines up against where the actual price series will be once that time arrives.

Usage:
    .venv/bin/python -m scripts.plot_predictions predictions.jsonl --ticker AAPL
    .venv/bin/python -m scripts.plot_predictions predictions.jsonl --ticker AAPL --output chart.png
    .venv/bin/python -m scripts.plot_predictions predictions.jsonl --ticker AAPL --non-overlapping
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt

from scripts._prediction_utils import non_overlapping_subset

DEFAULT_HORIZONS = (5, 10, 15)  # must match server/train_config.yaml's horizons


def load_records(path: str, ticker: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("ticker") == ticker and record.get("status") == "ok":
                records.append(record)
    records.sort(key=lambda r: r["as_of_timestamp"])
    return records


def _to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def plot(
    records: list[dict], ticker: str, horizons: tuple[int, ...], output: str | None,
    non_overlapping: bool = False,
) -> None:
    if not records:
        raise ValueError(f"no 'ok' predict() records found for ticker {ticker!r}")

    actual_times = [_to_datetime(r["as_of_timestamp"]) for r in records]
    actual_prices = [r["close"] for r in records]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_times, actual_prices, label="actual close", color="black", linewidth=1.5)

    colors = plt.get_cmap("tab10")
    for i, horizon in enumerate(horizons):
        # Thinning only affects which predictions get plotted as markers -- the actual-price
        # line above always uses every recorded tick, full resolution, regardless of this flag.
        series = non_overlapping_subset(records, horizon * 60_000) if non_overlapping else records
        peak_times, peak_prices, dip_times, dip_prices = [], [], [], []
        for r in series:
            match = next((p for p in r["predictions"] if p["horizon_minutes"] == horizon), None)
            if match is None:
                continue
            target_time = _to_datetime(r["as_of_timestamp"] + horizon * 60_000)
            peak_times.append(target_time)
            peak_prices.append(match["peak_price"])
            dip_times.append(target_time)
            dip_prices.append(match["dip_price"])

        color = colors(i)
        ax.scatter(peak_times, peak_prices, color=color, marker="^", s=18,
                   label=f"predicted peak @{horizon}m", alpha=0.7)
        ax.scatter(dip_times, dip_prices, color=color, marker="v", s=18,
                   label=f"predicted dip @{horizon}m", alpha=0.7)

    ax.set_title(f"{ticker}: predicted peak/dip vs. actual close")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("price")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"saved plot to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", help="JSONL file produced by agent.poller --log-file")
    parser.add_argument("--ticker", required=True, help="Ticker to plot, e.g. AAPL")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS),
        help="Horizons (minutes) to plot, default: 5 10 15",
    )
    parser.add_argument("--output", default=None, help="Save to this file instead of showing interactively")
    parser.add_argument(
        "--non-overlapping", action="store_true",
        help="Thin each horizon's markers to non-overlapping windows -- readable on long/dense runs "
             "(the actual-price line is unaffected, always full resolution)",
    )
    args = parser.parse_args()

    records = load_records(args.log_file, args.ticker)
    plot(records, args.ticker, tuple(args.horizons), args.output, non_overlapping=args.non_overlapping)


if __name__ == "__main__":
    main()
