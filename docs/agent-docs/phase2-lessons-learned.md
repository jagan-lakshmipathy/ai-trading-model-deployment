# Phase II Model Development — Lessons Learned from Phase I

Captures key takeaways from the first model's design, deployment, and live evaluation
([live-prediction-agent-plan.md](./live-prediction-agent-plan.md),
[model-improvement-roadmap.md](./model-improvement-roadmap.md),
[model_analysis.md](./model_analysis.md),
[docs/server-docs/model-design.md](../server-docs/model-design.md),
[docs/server-docs/model-deployment-spec.md](../server-docs/model-deployment-spec.md)), to inform
new model development as identified in `AI_Trading_Architecture.pdf`.

## Modeling lessons

1. **Price-level regression masks poor directional accuracy.** The current model predicts raw
   peak/dip price levels, which rewards being numerically close to `close0` — this produced small
   absolute errors but only ~50% directional accuracy (see
   [model_analysis.md](./model_analysis.md)). Consider training on return/profit-oriented targets
   or explicitly penalizing direction errors, not just magnitude.
2. **No per-ticker identity signal.** The model has no `ticker_id` embedding, so it can't learn
   ticker-specific behavior (volatility regime, liquidity profile, etc.). Flagged in
   [model-design.md §6](../server-docs/model-design.md) as unimplemented future work — worth
   prioritizing for the new model.
3. **Unconstrained outputs produce nonsensical bands.** Peak/dip/close are predicted
   independently with no `peak ≥ close ≥ dip` constraint. Add this as an explicit architectural or
   loss constraint.
4. **Point estimates hide uncertainty.** Move toward probabilistic/quantile outputs instead of
   single-point predictions — this also enables proper confidence-interval-based evaluation and
   bias correction.
5. **Known conservative bias.** Peaks are consistently over-predicted and dips under-predicted (a
   "small conservative bias" documented in
   [model-deployment-spec.md §5](../server-docs/model-deployment-spec.md)). Benchmark any new
   model against this known bias to see if it's inherent to the problem or an artifact of
   training/labeling choices.
6. **Untested feature ideas.** News sentiment features were identified but never added — worth
   evaluating for the new model.

## Evaluation/validation lessons (process, not model)

7. **Don't trust short/overlapping-window live samples.** Polling at high frequency creates
   heavily autocorrelated windows — effective independent sample size can be tiny (e.g. ~6 samples
   for a 15-min horizon over a 95-minute run). Build the non-overlapping, CI-reporting evaluation
   harness (already done —
   [scripts/evaluate_live_predictions.py](../../scripts/evaluate_live_predictions.py)) **before**
   drawing conclusions from new model live runs.
8. **One ticker/regime isn't representative.** Validate across volatile/illiquid tickers, not just
   liquid ones like AAPL/MSFT — edge cases (near-zero volatility, volume-division guards) only
   show up there.
9. **Run long enough to span real regime diversity** (opens, closes, multiple sessions) — 48+
   hours minimum, ideally historical backtesting to accumulate samples faster than real-time
   waiting.
10. **Live data ≠ training data silently.** Re-verify split/dividend adjustment conventions match
    training whenever the data source changes, and be aware of data-delay (the Massive feed had a
    15-min delay) — a model can be accurate yet still unusable for latency-sensitive strategies.

## Engineering/process lessons (useful if Phase II reuses the serving stack)

11. Install new dependencies into the venv immediately when adding to `pyproject.toml` — don't
    just edit the TOML.
12. Verify third-party API/SDK shapes against real docs early — assumptions about REST endpoints
    turned out wrong (had to switch to the official `massive` SDK).
13. Isolate per-ticker/per-item failures in any fetch loop so one bad ticker doesn't abort a whole
    batch.
14. Add defensive feature-count/shape checks before model inference, since silent schema drift
    (e.g., missing regime columns) can crash a fixed-size input layer.
