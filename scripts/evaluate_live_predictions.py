"""Evaluate predict(ticker) accuracy from a predictions.jsonl history file
(docs/agent-docs/model-improvement-roadmap.md item 1).

Reads the same JSONL format as scripts/plot_predictions.py (produced by
`agent.poller --log-file ...`), but computes accuracy statistics instead of a chart --
using *non-overlapping* windows per horizon (see docs/agent-docs/model-improvement-roadmap.md
"Why the first observation isn't meaningful yet" for why overlapping 1-min-apart windows give
misleadingly noisy/overconfident stats).

Usage:
    .venv/bin/python -m scripts.evaluate_live_predictions predictions.jsonl
    .venv/bin/python -m scripts.evaluate_live_predictions predictions.jsonl --ticker AAPL
    .venv/bin/python -m scripts.evaluate_live_predictions predictions.jsonl --horizons 5 10
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from scripts._prediction_utils import non_overlapping_subset

DEFAULT_HORIZONS = (5, 10, 15)  # must match server/train_config.yaml's horizons
MATCH_TOLERANCE_MS = 90_000  # how close an actual bar's timestamp must be to the exact target


def load_records(path: str) -> dict[str, list[dict]]:
    """Load all 'ok' records, grouped by ticker and sorted by as_of_timestamp."""
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


def _find_actual_at(by_ts: dict[int, dict], target_ts: int) -> dict | None:
    """Nearest record to `target_ts` within `MATCH_TOLERANCE_MS`, preferring an exact hit."""
    if target_ts in by_ts:
        return by_ts[target_ts]
    best = None
    best_dt = MATCH_TOLERANCE_MS + 1
    for ts, record in by_ts.items():
        dt = abs(ts - target_ts)
        if dt < best_dt:
            best, best_dt = record, dt
    return best if best_dt <= MATCH_TOLERANCE_MS else None


def _proportion_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Normal-approximation confidence interval for a proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = successes / n
    margin = z * math.sqrt(p * (1 - p) / n) if n > 1 else float("nan")
    return (p, max(0.0, p - margin), min(1.0, p + margin))


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """Normal-approximation confidence interval for a mean. Returns (mean, lo, hi)."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / n
    if n < 2:
        return (mean, float("nan"), float("nan"))
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    margin = z * math.sqrt(variance / n)
    return (mean, mean - margin, mean + margin)


def evaluate_ticker(records: list[dict], horizons: tuple[int, ...]) -> list[dict]:
    """Returns one summary row per horizon for this ticker."""
    by_ts = {r["as_of_timestamp"]: r for r in records}
    rows = []
    for horizon in horizons:
        horizon_ms = horizon * 60_000
        subset = non_overlapping_subset(records, horizon_ms)

        peak_errors, dip_errors, in_band, dir_correct = [], [], [], []
        for r in subset:
            match = _find_actual_at(by_ts, r["as_of_timestamp"] + horizon_ms)
            if match is None:
                continue
            pred = next((p for p in r["predictions"] if p["horizon_minutes"] == horizon), None)
            if pred is None:
                continue
            actual = match["close"]
            peak, dip, close0 = pred["peak_price"], pred["dip_price"], r["close"]
            peak_errors.append(peak - actual)
            dip_errors.append(dip - actual)
            in_band.append(dip <= actual <= peak)
            actual_dir = actual - close0
            pred_dir = (peak + dip) / 2 - close0
            if actual_dir != 0:
                dir_correct.append((actual_dir > 0) == (pred_dir > 0))

        n_matched = len(peak_errors)
        peak_mean, peak_lo, peak_hi = _mean_ci(peak_errors)
        dip_mean, dip_lo, dip_hi = _mean_ci(dip_errors)
        band_p, band_lo, band_hi = _proportion_ci(sum(in_band), len(in_band))
        dir_p, dir_lo, dir_hi = _proportion_ci(sum(dir_correct), len(dir_correct))

        rows.append({
            "horizon": horizon,
            "n_candidates": len(subset),
            "n_matched": n_matched,
            "peak_error_mean": peak_mean, "peak_error_ci": (peak_lo, peak_hi),
            "dip_error_mean": dip_mean, "dip_error_ci": (dip_lo, dip_hi),
            "band_coverage": band_p, "band_coverage_ci": (band_lo, band_hi),
            "dir_accuracy": dir_p, "dir_accuracy_ci": (dir_lo, dir_hi),
            "dir_n": len(dir_correct),
        })
    return rows


def _fmt_ci(lo: float, hi: float) -> str:
    if math.isnan(lo) or math.isnan(hi):
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def print_report(ticker: str, rows: list[dict]) -> None:
    print(f"=== {ticker} ===")
    for row in rows:
        h = row["horizon"]
        print(f"  horizon={h:2d}m  non-overlapping candidates={row['n_candidates']:3d}  matched={row['n_matched']:3d}")
        if row["n_matched"] < 5:
            print("    (too few independent samples for a meaningful estimate -- treat as noise)")
            continue
        print(f"    peak error:  mean={row['peak_error_mean']:+.4f}  95% CI={_fmt_ci(*row['peak_error_ci'])}")
        print(f"    dip  error:  mean={row['dip_error_mean']:+.4f}  95% CI={_fmt_ci(*row['dip_error_ci'])}")
        print(f"    band coverage: {row['band_coverage']*100:.1f}%  95% CI={_fmt_ci(*[v*100 for v in row['band_coverage_ci']])}")
        print(f"    directional accuracy (n={row['dir_n']}): {row['dir_accuracy']*100:.1f}%  "
              f"95% CI={_fmt_ci(*[v*100 for v in row['dir_accuracy_ci']])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", help="JSONL file produced by agent.poller --log-file")
    parser.add_argument("--ticker", action="append", default=None,
                        help="Ticker(s) to evaluate (repeatable); default: all tickers in the file")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    args = parser.parse_args()

    by_ticker = load_records(args.log_file)
    tickers = args.ticker or sorted(by_ticker)
    for ticker in tickers:
        records = by_ticker.get(ticker, [])
        if not records:
            print(f"=== {ticker} ===\n  no 'ok' records found")
            continue
        rows = evaluate_ticker(records, tuple(args.horizons))
        print_report(ticker, rows)


if __name__ == "__main__":
    main()
