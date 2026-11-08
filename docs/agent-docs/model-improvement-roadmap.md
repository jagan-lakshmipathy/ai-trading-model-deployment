# Model Improvement Roadmap

Captures the discussion from the first live `agent.poller` run (2026-11-03, AAPL/MSFT, ~95
minutes, via Massive Stocks Starter with a 15-min data delay) and the agreed next steps. See
[docs/agent-docs/live-prediction-agent-plan.md](./live-prediction-agent-plan.md) for the serving
infrastructure this builds on.

## Why the first observation isn't meaningful yet

A quick ad-hoc analysis of the first `predictions.jsonl` run showed AAPL/MSFT band coverage of
41-59% and directional accuracy of 41-62% across the 5/10/15-min horizons, with a consistent
"peak over-predicted, dip under-predicted" bias on both tickers. These numbers are **not**
trustworthy evidence of model quality, for two compounding reasons:

1. **Overlapping, autocorrelated samples.** Polling every 1 minute means a 5-min-horizon window
   shares 4 of 5 minutes with the previous poll's window (14 of 15 for the 15-min horizon). The
   effective number of *independent* samples in a 95-minute run is roughly `duration / horizon`
   — about 19 for the 5-min horizon, 9 for 10-min, and only ~6 for 15-min. At `n=6`, a true 50%
   rate can easily show up as 20% or 80% by chance.
2. **One short, quiet, single-regime window.** 95 minutes of one mostly-flat afternoon on 2
   tickers (of the 23 in the training set) says little about behavior across volatile opens,
   closes, varied volume, or the other 21 tickers — especially the volatile/illiquid ones
   (`BMNR`/`IREN`/`ONDS`) already flagged as unverified in
   [docs/server-docs/model-deployment-spec.md §5](../server-docs/model-deployment-spec.md).

The pipeline itself is validated correct (bit-identical to training-time code per
`server.validate_against_eval`) — what's unproven is whether *this specific live data source/
regime* behaves like the documented historical test-split metrics. That's what the steps below
are designed to actually answer.

## Roadmap

### 1. Build a proper evaluation harness (do this before collecting more data) — **DONE 2026-11-03**

A repeatable script, not another one-off terminal command, that:
- Uses **non-overlapping** windows per horizon (sample every 5 min for the 5-min horizon, every
  10 min for the 10-min horizon, etc.) to remove the autocorrelation problem.
- Reports **confidence intervals**, not just point estimates.
- Breaks results down **per ticker, per horizon**.
- Compares against the **documented test-split baseline** metrics in
  [docs/server-docs/plan.md](../server-docs/plan.md) (MAE, directional accuracy, coverage) — the
  real question is "does live match what training already proved," not "is live good in
  isolation."
- Location: [scripts/evaluate_live_predictions.py](../../scripts/evaluate_live_predictions.py),
  reading the same `predictions.jsonl` format as
  [scripts/plot_predictions.py](../../scripts/plot_predictions.py). Usage:
  `.venv/bin/python -m scripts.evaluate_live_predictions predictions.jsonl`.
- Confirmed working against the first real AAPL/MSFT run: the confidence intervals are
  appropriately huge given the small non-overlapping sample (e.g. AAPL 15-min directional
  accuracy showed 16.7% with a 95% CI of `[0.0%, 46.5%]`) — this is the autocorrelation problem
  made visible as honest numbers, confirming the first run's point estimates alone
  weren't trustworthy, exactly as expected.
- **Not yet done**: automated comparison against the documented test-split baseline numbers
  (currently a manual cross-check against [docs/server-docs/plan.md](../server-docs/plan.md)).
- `scripts/plot_predictions.py` also gained an opt-in `--non-overlapping` flag (shares the
  thinning logic with the evaluation script via `scripts/_prediction_utils.py`), for readability
  on long/dense runs where every-1-minute markers would otherwise overplot into an unreadable
  swarm — the actual-price line is unaffected, always full resolution regardless of this flag.

### 2. Extend duration *and* ticker/regime coverage

- Run for **48+ hours minimum**, spanning multiple full trading sessions (opens, closes, lunch
  lulls) — one quiet afternoon isn't representative.
- Add **more than 2 tickers** — ideally several of the 23 training tickers including at least one
  volatile/illiquid name (`BMNR`, `IREN`, or `ONDS`), since those are the ones most likely to hit
  edge cases (near-zero-volatility clipping, volume-division guards) that AAPL/MSFT won't.
- Consider **historical backtesting** (replaying Massive's historical bars, not just the live
  tail) to accumulate a meaningful sample much faster than waiting in real time.

### 3. Investigate the observed bias once there's a real sample

Both tickers showed the same pattern: peak consistently over-predicted, dip consistently
under-predicted. If this persists with a statistically meaningful (non-overlapping, multi-day,
multi-ticker) sample, compare it against the model's already-documented "small conservative
bias" ([docs/server-docs/model-deployment-spec.md §5](../server-docs/model-deployment-spec.md))
to determine if it's the same known effect or something new/larger — which would point at a
calibration issue or a live-vs-training data mismatch (see #4).

### 4. Check for a live-data vs. training-data mismatch

- **Split/dividend adjustment convention** must still match what the model was trained on
  ([docs/server-docs/model-deployment-spec.md §2.5](../server-docs/model-deployment-spec.md)) —
  worth explicitly re-verifying now that the data source is Massive's live Stocks Starter API
  rather than the original local DB ingestion.
- **The 15-minute delay is a product/latency limitation, not a modeling one** — even a perfectly
  accurate model has limited usefulness for live trading decisions if its freshest input is
  always 15 minutes stale. Worth deciding whether this requires a plan upgrade depending on the
  intended use case.

### 5. Model-level improvements (further out; from [docs/server-docs/model-design.md §6](../server-docs/model-design.md))

Not yet implemented, called out as future work there:
- `ticker_id` embedding (model currently has no per-ticker identity signal).
- News sentiment features.
- Explicit `peak ≥ close ≥ dip` constraint (currently unconstrained, can produce nonsensical
  bands).
- Probabilistic/quantile outputs instead of point estimates.
- Head-to-head comparison of the experimental normalized-feature checkpoint vs. the baseline,
  on live data too (only validated historically so far, per
  [docs/server-docs/model-deployment-spec.md §2.2](../server-docs/model-deployment-spec.md)).

### 6. (Under consideration) Trading strategy directions, informed by the error/bias analysis

See [model_analysis.md](./model_analysis.md) for the full data behind these — captured here as
candidate directions, not yet decided or scoped:

- **Expected-profit/return reframing.** Instead of predicting raw peak/dip price levels, train
  on a profit/return-oriented target (e.g. `(actual - close0)` or realized return, possibly a
  quantile/probabilistic output per item 5 above). Price-level regression rewards the model for
  being numerically close to `close0`, which is consistent with the observed pattern of small
  errors but ~50% directional accuracy (model_analysis.md §2) — a profit-oriented loss would
  force the model to commit to a direction instead. Requires new labels/loss, an explicit
  transaction-cost model, and walk-forward (not just more live) validation against a naive
  baseline.
- **Bias-corrected scale-in/scale-out (grid/rebalancing) strategy.** Rather than betting on
  direction, sell a portion of a position as price approaches the (bias-corrected) predicted
  peak and rebuy a portion as it approaches the (bias-corrected) predicted dip, profiting from
  oscillation rather than correct direction calls. This sidesteps the ~50% directional-accuracy
  problem (it doesn't need direction to be right), but:
  - Must correct for the known bias first (model_analysis.md §1-3): raw peak is over-predicted
    and raw dip is under-predicted, so uncorrected triggers are too conservative (fewer completed
    cycles) rather than wrong-directioned.
  - Needs explicit position-sizing logic (e.g. proportional to predicted move size or a
    volatility-based/Kelly-style rule) — not something the current `predict()` output provides.
  - Needs a realized-P&L backtest (not just error/coverage stats) against the 4-day history,
    since grid strategies can bleed steadily in a sustained one-directional trend even when
    per-prediction error/coverage stats look fine.
  - Must account for transaction costs and the 15-min data delay, since this strategy trades
    more frequently than a one-shot directional bet.

## Immediate next action

Item 1 (evaluation harness) is built. Next: extend duration/ticker coverage (item 2) — run
`agent.poller` with more tickers (including a volatile one) over 48+ hours, then run
`scripts.evaluate_live_predictions` on the resulting log.
