"""Plot the signed peak/dip prediction error distribution across tickers, broken out per horizon
(docs/agent-docs/model_analysis.md section 4).

For each ticker *and horizon separately*, computes `(predicted - actual) / actual * 100` for peak
and dip predictions (non-overlapping windows, see scripts/evaluate_live_predictions.py),
normalizes each ticker/horizon's errors by that ticker/horizon's own max |error| so all tickers
share a [-100%, +100%] x-axis, buckets into 20%-wide bins, and plots one line per ticker per
horizon panel -- making the peak-over/dip-under calibration bias from
docs/agent-docs/model_analysis.md visible as an off-center distribution. Horizons are kept
separate (not pooled) because error magnitude grows with horizon -- pooling would let the
largest-horizon tail dominate each ticker's normalization and distort where shorter-horizon
errors land on the shared axis.

Usage:
    .venv/bin/python -m scripts.plot_error_distribution predictions.jsonl
    .venv/bin/python -m scripts.plot_error_distribution predictions.jsonl --output chart.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts._prediction_utils import non_overlapping_subset
from scripts.evaluate_live_predictions import DEFAULT_HORIZONS, MATCH_TOLERANCE_MS, _find_actual_at

N_SIDE_BUCKETS = 5  # 5 buckets per side -> 20%-wide bins over [-100, 100]
SIDE_SIZE = 100 / N_SIDE_BUCKETS


def load_by_ticker(path: str) -> dict[str, list[dict]]:
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                by_ticker[record["ticker"]].append(record)
    for records in by_ticker.values():
        records.sort(key=lambda r: r["as_of_timestamp"])
    return dict(by_ticker)


def _bucket_signed(signed_fracs: list[float]) -> list[int]:
    """Buckets values in [-100, 100] into 2*N_SIDE_BUCKETS bins, most-negative first."""
    counts = [0] * (2 * N_SIDE_BUCKETS)
    for frac in signed_fracs:
        frac = max(-100.0, min(100.0, frac))
        if frac < 0:
            idx = N_SIDE_BUCKETS - 1 - min(int((-frac) // SIDE_SIZE), N_SIDE_BUCKETS - 1)
        else:
            idx = N_SIDE_BUCKETS + min(int(frac // SIDE_SIZE), N_SIDE_BUCKETS - 1)
        counts[idx] += 1
    return counts


def signed_pct_errors(records: list[dict], price_key: str, horizon: int) -> list[float]:
    """Non-overlapping signed % errors `(pred - actual) / actual * 100` for one price field/horizon."""
    by_ts = {r["as_of_timestamp"]: r for r in records}
    horizon_ms = horizon * 60_000
    subset = non_overlapping_subset(records, horizon_ms)
    errors = []
    for r in subset:
        match = _find_actual_at(by_ts, r["as_of_timestamp"] + horizon_ms)
        if match is None:
            continue
        pred = next((p for p in r["predictions"] if p["horizon_minutes"] == horizon), None)
        if pred is None:
            continue
        actual = match["close"]
        errors.append((pred[price_key] - actual) / actual * 100)
    return errors


def plot(by_ticker: dict[str, list[dict]], horizons: tuple[int, ...], output: str) -> None:
    centers = [-100 + (i + 0.5) * SIDE_SIZE for i in range(2 * N_SIDE_BUCKETS)]
    fig, axes = plt.subplots(2, len(horizons), figsize=(6 * len(horizons), 10), sharey=True)
    colors = plt.cm.tab10.colors

    for row, (kind, price_key) in enumerate((("Peak", "peak_price"), ("Dip", "dip_price"))):
        for col, horizon in enumerate(horizons):
            ax = axes[row][col]
            for i, ticker in enumerate(sorted(by_ticker)):
                errors = signed_pct_errors(by_ticker[ticker], price_key, horizon)
                if not errors:
                    continue
                max_abs = max(abs(e) for e in errors)
                signed_fracs = [e / max_abs * 100 for e in errors]
                counts = _bucket_signed(signed_fracs)
                pct = [c / len(errors) * 100 for c in counts]
                ax.plot(centers, pct, marker="o", label=ticker, color=colors[i % len(colors)])
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{kind} error, {horizon}m horizon")
            ax.set_xlabel(
                "Signed error bucket center (% of own max |error|)\n"
                "(negative=under-predicted, positive=over-predicted)"
            )
            if col == 0:
                ax.set_ylabel("% of predictions")
            ax.grid(alpha=0.3)

    axes[0][-1].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", help="JSONL file produced by agent.poller --log-file")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--output", default="error_distribution.png")
    args = parser.parse_args()

    by_ticker = load_by_ticker(args.log_file)
    plot(by_ticker, tuple(args.horizons), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
