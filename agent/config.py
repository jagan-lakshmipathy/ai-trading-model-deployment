"""Agent configuration: which tickers to poll and how often
(docs/agent-docs/live-prediction-agent-plan.md section 2 / step 9).
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_POLL_INTERVAL_S = 60  # matches bar granularity (plan.md "Agent polling cadence")


@dataclass(frozen=True)
class AgentConfig:
    tickers: list[str]
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S
