# Model Deployment Spec — Inputs/Outputs Contract

What a caller must supply to, and can expect back from, the current best checkpoint
(`checkpoints/best_model.pt`, all-23-ticker / 300-epoch / target-clipping run — see
[docs/plan.md](./plan.md) for training history and the "Evaluation" section of
[docs/plan.md](./plan.md#evaluation-per-horizon-test-split-metrics-current-best-checkpoint) for
accuracy numbers). This is the practical contract for real-world/serving use, as opposed to
[docs/model-design.md](./model-design.md) (architecture rationale). The full intended feature
design document (`price-prediction-feature-design.md`) lives only in the source `ai-trading-model`
repo, not here — note the **implemented** feature set is a subset, flagged below.

## 1. Model identity

- Architecture: `MultiHorizonGRUModel` ([server/encoder.py](../../server/encoder.py)) —
  `Linear(n_features -> hidden_size)` projection, 2-layer GRU (`hidden_size=128`), 3 independent
  per-horizon MLP heads (`head_hidden_size=64`), one per `horizons: [5, 10, 15]`.
- `n_features = 42` (fixed by the feature set below — the checkpoint's `input_proj` weight shape
  is `(128, 42)`; a different feature set/count requires retraining, not just reloading weights).
- **Must call `model.eval()` before inference** — `dropout=0.1` is active in `.train()` mode and
  will make outputs stochastic/non-reproducible if left on.

## 2. Inputs

### 2.1 Feature vector — exact column order (42 columns)

Built by `build_features_and_targets` ([server/features.py](../../server/features.py)).
**Order matters** — the model has no column names at inference time, only positional indices.

| # | Column | Formula | Source | Typical range |
| --- | --- | --- | --- | --- |
| 0 | `past_peak_dev_5m` | `(max(high[t-4..t]) - close[t]) / close[t]` | target ticker bars | ~0 to 0.01+ |
| 1 | `past_dip_dev_5m` | `(min(low[t-4..t]) - close[t]) / close[t]` | target ticker bars | ~-0.01 to 0 |
| 2 | `past_peak_dev_15m` | same, 15-bar window | target ticker bars | wider than 5m |
| 3 | `past_dip_dev_15m` | same, 15-bar window | target ticker bars | wider than 5m |
| 4 | `past_peak_dev_30m` | same, 30-bar window | target ticker bars | wider still |
| 5 | `past_dip_dev_30m` | same, 30-bar window | target ticker bars | wider still |
| 6 | `past_peak_dev_60m` | same, 60-bar window | target ticker bars | widest |
| 7 | `past_dip_dev_60m` | same, 60-bar window | target ticker bars | widest |
| 8 | `ret_1m` | `close[t]/close[t-1] - 1` | target ticker bars | ~-0.02 to 0.02 |
| 9 | `log_volume` | `log1p(volume[t])` | target ticker bars | depends on ticker liquidity, unbounded above |
| 10 | `rel_volume_20m` | `volume[t] / mean(volume[t-20..t-1])` | target ticker bars | ~0.1-5, spikes higher |
| 11 | `vwap_dev` | `(close[t] - vwap[t]) / vwap[t]` | target ticker bars | small, near 0 |
| 12 | `bar_range` | `(high[t] - low[t]) / close[t]` | target ticker bars | small positive |
| 13 | `realized_vol_15m` | rolling 15-bar std of 1-min log returns | target ticker bars | small positive, **also the divisor for target denormalization — see §3** |
| 14 | `minutes_since_open` | minutes since 09:30 ET, clipped ≥0 | timestamp (market tz) | 0-390 |
| 15 | `minutes_to_close` | minutes to 16:00 ET, clipped ≥0 | timestamp (market tz) | 0-390 |
| 16 | `tod_sin` | `sin(2π·tod/1440)` | timestamp | [-1, 1] |
| 17 | `tod_cos` | `cos(2π·tod/1440)` | timestamp | [-1, 1] |
| 18 | `dow_sin` | `sin(2π·dow/7)` | timestamp | [-1, 1] |
| 19 | `dow_cos` | `cos(2π·dow/7)` | timestamp | [-1, 1] |
| 20 | `spy_ret_1m` | `SPY.close[t]/SPY.close[t-1] - 1` | regime DB (`SPY`) | small, near 0 |
| 21 | `spy_ret_5m` | same, 5-bar | regime DB (`SPY`) | small, near 0 |
| 22 | `spy_ret_15m` | same, 15-bar | regime DB (`SPY`) | small, near 0 |
| 23 | `market_realized_vol_15m` | rolling 15-bar std of `SPY` log returns | regime DB (`SPY`) | small positive |
| 24 | `qqq_ret_5m` | 5-bar return | regime DB (`QQQ`) | small, near 0 |
| 25 | `iwm_ret_5m` | 5-bar return | regime DB (`IWM`) | small, near 0 |
| 26 | `risk_on_off_spread` | `ret_5m(HYG) - ret_5m(TLT)` | regime DB (`HYG`,`TLT`) | small, near 0 |
| 27 | `gold_ret_5m` | 5-bar return | regime DB (`GLD`) | small, near 0 |
| 28 | `usd_ret_5m` | 5-bar return | regime DB (`USD`) | small, near 0 |
| 29 | `rel_volume_spy_5m` | `SPY.volume[t] / mean(SPY.volume[t-20..t-1])` | regime DB (`SPY`) | ~0.1-5 |
| 30 | `fed_funds_rate` | last known level | FRED `FEDFUNDS` | ~0-6 (%) |
| 31 | `cpi_yoy` | 252-trading-day % change | FRED `CPIAUCSL` | ~0-0.1 |
| 32 | `core_cpi_yoy` | 252-trading-day % change | FRED `CPILFESL` | ~0-0.1 |
| 33 | `ppi_yoy` | 252-trading-day % change | FRED `PPIACO` | ~-0.1-0.2 |
| 34 | `gdp_qoq` | quarter-over-quarter % change | FRED `GDPC1` | ~-0.05-0.05 |
| 35 | `unemployment_rate` | last known level | FRED `UNRATE` | ~3-10 (%) |
| 36 | `payrolls_mom_change` | month-over-month diff | FRED `PAYEMS` | thousands, can be negative |
| 37 | `pce_mom_change` | month-over-month % change | FRED `PCE` | small % |
| 38 | `yield_10y` | last known level | FRED `DGS10` | ~0-6 (%) |
| 39 | `yield_2y` | last known level | FRED `DGS2` | ~0-6 (%) |
| 40 | `yield_curve_slope` | `yield_10y - yield_2y` | derived | can be negative (inversion) |
| 41 | `days_since_macro_update` | days since last non-null FRED row | derived | 0-~90 |

Tensor shape at inference: `(1, 1, 42)` for single-step streaming (`batch=1`, `chunk_len=1`), or
`(1, T, 42)` if feeding a known historical run. `dtype=float32`.

### 2.2 Normalization — **none is currently applied in the baseline checkpoint**

This is the most important gap to know before deploying `checkpoints/best_model.pt`.
[docs/model-design.md](./model-design.md#11-input-handling) calls for z-scoring features (fit
on train only) before the encoder, and the source `ai-trading-model` repo's `input/splits.py`
even has a working `fit_scaler`/`FittedScaler` for this — **but it is never called from its
`train.py`**. The baseline model was trained on,
and therefore expects, the **raw engineered feature values** exactly as computed by
`build_features_and_targets` (ratios/returns/rolling stats as-is, not z-scored). Do not
normalize features yourself before calling this checkpoint — that would feed it
out-of-distribution inputs relative to training.

**Experimental normalized variant:** the source `ai-trading-model` repo's `train_normalized.py` /
`evaluate_normalized.py`, trained
via [server/train_config_normalized.yaml](../../server/train_config_normalized.yaml), z-scores
every one of the 42 feature columns (mean/std fit on the train split only, applied unchanged to
val/test/serving) before the input projection. It is kept as a fully separate
checkpoint/config/module pair precisely because the accuracy win is unproven — the baseline
`checkpoints/best_model.pt` is never touched by this experiment. **This checkpoint has a
different on-disk format**: instead of a bare `state_dict`, it's a dict
`{"model_state": ..., "feat_mean": np.ndarray(42,), "feat_std": np.ndarray(42,)}` — callers must
load it accordingly and apply `(features - feat_mean) / feat_std` to every input row before
calling the model, using these exact saved values (not freshly refit ones). Until this variant
is evaluated against the baseline and promoted, treat `checkpoints/best_model.pt` (bare state
dict, unnormalized) as the only one validated for deployment.

### 2.3 `ticker_id` — not implemented

[docs/model-design.md](./model-design.md#12-input-handling) proposed an optional learned
`ticker_id` embedding so a shared model can specialize per ticker. **This was never
implemented** — the feature vector above contains no ticker identity at all. The model
generalizes across the 23 training tickers purely from price/volume-derived features; it has
no explicit way to "know" which ticker it's looking at. Keep this in mind if deploying on a
ticker very different in character from the training set (e.g. much more/less volatile).

### 2.4 Data source / freshness requirements

- **Target ticker bars**: 1-min OHLCV, must match the schema/columns in
  [docs/data-readme.md](./data-readme.md) (`bars` table) — `open, high, low, close, volume, vwap, transactions`.
- **Regime ETF bars**: same schema, for `SPY, QQQ, IWM, HYG, TLT, GLD, USD` (the model needs all
  of these to populate columns 20-29; missing any one drops those columns and misaligns the
  42-column contract — don't deploy with a reduced regime basket without retraining).
- **Macro (FRED)**: daily/monthly/quarterly series forward-filled onto the 1-min grid, per
  [docs/data-readme.md](./data-readme.md#datafred_econ_data_6yrscsv). Feed the **last known
  value as of `date(t)`**, never a same-day-or-later release — look-ahead here directly leaks
  future information into a feature.
- **No look-ahead anywhere**: every feature at time `t` must be computable using only data
  timestamped ≤ `t`. This includes regime ETF bars and macro data, not just the target ticker.

**Update (2026-11-01): the MCP tool itself no longer takes any of this as input.** The server
now fetches and buffers all three sources itself (live-polled from Massive/FRED on a background
schedule) — `predict(ticker)` takes just a ticker symbol. The requirements above still describe
what the server's own data pipeline must satisfy internally; they're no longer something a
caller needs to assemble. See
[docs/agent-docs/live-prediction-agent-plan.md](../agent-docs/live-prediction-agent-plan.md) for
the live data-fetching design (buffers, scheduler, warmup/dynamic ticker tracking, market-hours
gating) and its known gaps (the Massive API shape is unverified against real docs/responses).

### 2.5 Split/dividend adjustment — **verified: source data is already split-adjusted**

[docs/data-readme.md](./data-readme.md) documents `splits`/`dividends` tables in the source DBs,
and the source `ai-trading-model` repo's `price-prediction-feature-design.md` (data hygiene
notes section) calls for adjusting prices for splits before computing returns, which led to an
initial concern that `build_features_and_targets`/`load_bars` **never join or apply the `splits`
table**.
Empirically checking `data/trading_data.db` resolved this: for the 4 verifiable split events in
the `splits` table within the data window (`AMZN` 20-for-1 on 2022-06-06, `GOOGL` 20-for-1 on
2022-07-18, `NVDA` 10-for-1 on 2024-06-10, `TSLA` 3-for-1 on 2022-08-25 — `BMNR`'s 2025-05-16
split has no bars before it in this DB to check), the `close` price shows **no discontinuity**
across the split date (ratio of last-bar-before to first-bar-after is ~0.98-1.01 in every case,
vs. the 3x-20x jump a raw/unadjusted series would show). Conclusion: **`bars` data from
Massive/Polygon is already split-adjusted** — the `splits` table is reference/bookkeeping data
only, not something the pipeline needs to apply itself. No change needed to
`build_features_and_targets`/`load_bars`; feed raw `bars.close`/`high`/`low` directly, as the
pipeline already does.

**Deployment requirement: the model expects split/dividend-adjusted input.** It was trained
exclusively on Massive/Polygon `bars` data, which is pre-adjusted for both splits (verified
above) and dividends (not independently re-verified the same way — see caveat below). Any
serving-time data source **must** supply split/dividend-adjusted prices in the same convention —
feeding raw/unadjusted prices from a different vendor, or prices adjusted only for splits but
not dividends (or vice versa), is an out-of-distribution input relative to training and will
silently degrade predictions rather than error out.

**Standard vendor adjustment conventions** (for context when sourcing serving-time data from a
different provider):

1. **Unadjusted/raw** — exactly what traded; discontinuities at both split and ex-dividend dates.
2. **Split-adjusted only** — corrects share-count jumps from splits, but leaves ex-dividend price
   drops in place.
3. **Split + dividend adjusted ("fully adjusted" / "adjusted close")** — also backward-adjusts
   every ex-dividend price drop, so the series is internally consistent on a total-return basis.
   This is what Polygon's (and therefore Massive's) `adjusted=true` bars parameter applies —
   **splits and dividends are bundled into a single adjustment flag**, not offered as independent
   toggles, which is the standard convention across most vendors (Polygon, Yahoo Finance's
   "Adj Close", etc.).

**Verification caveat:** only split-adjustment was empirically checked (§ above, via the 4
testable `splits` table events showing no price discontinuity). Dividend-adjustment was
*inferred*, not independently verified the same way, from the bundled-flag vendor convention
above. To close this gap rigorously, cross-check the `dividends` table's `ex_dividend_date`/
`cash_amount` against `bars.close` around each ex-dividend date for a ticker with a dividend
large enough to produce a detectable price drop (same method as the split check) — not yet done.

### 2.6 Session boundaries / hidden state (streaming inference)

- Hidden state shape: `(num_layers=2, batch=1, hidden_size=128)`.
- **Zero the hidden state at the start of each new trading session** (new ticker, or new
  market-tz calendar day — see [server/sessions.py](../../server/sessions.py),
  `MARKET_TZ = "America/New_York"`). Carrying state across an overnight/weekend gap or across a
  ticker switch is out-of-distribution relative to training (the source `ai-trading-model` repo's
  `training-loop.md` documents why, not present in this repo).
- Within a session, feed bars **in order, one per step**, carrying hidden state forward
  (`hidden = next_hidden`, no `.detach()` needed at inference — that's a training-only TBPTT
  concern).
- Only market-hours 1-min bars count as "steps" — do not synthesize bars for non-trading
  minutes/gaps.

**Update (2026-11-01):** `server/state.py`'s `HiddenStateManager` additionally tracks a
per-ticker processed-bar cursor (`new_timestamps`), so the same bar is never forward-passed more
than once no matter how many times `predict(ticker)` is called — this matters now that the
server (not the caller) owns the bar buffer and `predict` may be called more often than new bars
arrive. See [docs/agent-docs/live-prediction-agent-plan.md](../agent-docs/live-prediction-agent-plan.md)
for the full rationale (the rejected caller-supplies-bars design hit exactly this problem).

## 3. Outputs

- Raw model output `preds`: shape `(batch, chunk_len, 3, 2)` — for each of the 3 horizons
  (`[5, 10, 15]`, in that order) a `[delta_peak, delta_dip]` pair, in **normalized delta space**,
  not raw price.
- **Denormalization** (invert the volatility scaling from
  [docs/model-design.md](./model-design.md#2-target-normalization-the-delta)), using the *same
  timestep's* `close[t]` and `realized_vol_15m[t]` (feature column 13, same value fed as input):

  ```
  peak_price_W = close[t] + delta_peak_W * (close[t] * realized_vol_15m[t] + eps)
  dip_price_W  = close[t] + delta_dip_W  * (close[t] * realized_vol_15m[t] + eps)
  ```
  with `eps = 1e-8` (must match training exactly).
- Targets (and therefore predictions, by construction of training) are **clipped to
  `[-20.0, 20.0]`** in delta space ([docs/plan.md](./plan.md#bug-found-unclipped-targets-caused-traintestval-scale-mismatch)) —
  extreme predictions near that boundary should be treated with suspicion (possible
  near-zero-volatility degenerate input) rather than taken at face value.
- No explicit `peak >= close >= dip` constraint is enforced by the architecture (see
  [docs/model-design.md](./model-design.md#13-per-horizon-output-heads), optional softplus/exp
  constraint was never added) — in principle a predicted `dip_price_W > close[t]` or
  `peak_price_W < close[t]` is possible and should be handled (clip/flag) downstream rather than
  assumed impossible.
- Known **small conservative bias** ([docs/plan.md](./plan.md#evaluation-per-horizon-test-split-metrics-current-best-checkpoint)):
  the model slightly under-predicts both peak and dip magnitude (bias ~0.004-0.006% of `close`
  at 5-min, shrinking at longer horizons) — consider a small safety margin if using predictions
  directly as take-profit/stop-loss levels.

## 4. Checkpoint loading

```python
model = MultiHorizonGRUModel(
    n_features=42, horizons=[5, 10, 15], hidden_size=128,
    num_layers=2, head_hidden_size=64, dropout=0.1,
)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()
```

For the experimental normalized checkpoint, load the dict form and apply the saved scaler to
every input row before calling the model:

```python
checkpoint = torch.load("checkpoints/best_model_normalized.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
features = (features - checkpoint["feat_mean"]) / checkpoint["feat_std"]
```

`hidden_size`/`num_layers`/`head_hidden_size`/`horizons` must exactly match
[server/train_config.yaml](../../server/train_config.yaml) at the time this checkpoint was
produced — the state dict has no shape-checking beyond what `load_state_dict` enforces, and a
silently-mismatched `dropout` value (e.g. forgetting `model.eval()`) will not error but will
degrade/randomize output.

## 5. Summary of deployment risks / things to double check before going live

1. **No feature normalization in the baseline checkpoint** — raw engineered features only; the
   experimental z-scored variant (§2.2) uses a different checkpoint format and requires applying
   the saved `feat_mean`/`feat_std`, not freshly computed ones.
2. **Model expects split/dividend-adjusted input** — verified the training source data already
   is (§2.5); any serving-time data source must match this convention, or predictions will
   silently degrade.
3. **No `ticker_id`** — model has no explicit per-ticker identity signal.
4. **No peak≥close≥dip constraint** — sanity-check/clip predictions downstream.
5. **Target/behavior only validated on the 23 training tickers** and the test-split time window
   (`2026-04-14` onward per [docs/plan.md](./plan.md)) — accuracy on new tickers or materially
   different market regimes is unverified.
6. **`model.eval()` required** — dropout is on by default in `.train()` mode.
7. Full regime ETF basket (`SPY, QQQ, IWM, HYG, TLT, GLD, USD`) and FRED macro series must all be
   available and current at serving time, or the 42-column feature contract breaks.
8. **Two checkpoint formats in play** — `checkpoints/best_model.pt` (bare `state_dict`,
   unnormalized, the only one validated so far) vs. `checkpoints/best_model_normalized.pt` (dict
   with `model_state`/`feat_mean`/`feat_std`, experimental) — loading one with the other's
   assumptions will fail or silently misbehave.
