"""Shared helpers for scripts/plot_predictions.py and scripts/evaluate_live_predictions.py."""
from __future__ import annotations


def non_overlapping_subset(records: list[dict], horizon_ms: int) -> list[dict]:
    """Greedily keep records at least `horizon_ms` apart (by `as_of_timestamp`), so each kept
    record's target window doesn't overlap the previous one's -- used to avoid plotting/scoring
    the same stretch of time multiple times (see docs/agent-docs/model-improvement-roadmap.md).
    """
    kept = []
    next_allowed_ts = float("-inf")
    for r in records:
        if r["as_of_timestamp"] >= next_allowed_ts:
            kept.append(r)
            next_allowed_ts = r["as_of_timestamp"] + horizon_ms
    return kept
