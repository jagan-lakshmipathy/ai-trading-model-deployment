# Validation Fixture — `generate_validation_fixture.py`

Documents [scripts/generate_validation_fixture.py](../../scripts/generate_validation_fixture.py):
what it does, what it needs as input, and the exact schema of the `.npz` file it produces. See
[docs/plan.md](./plan.md) section 7 for how this fits into the overall validation plan, and
[server/validate_against_eval.py](../../server/validate_against_eval.py) for how the fixture is
consumed.

## Purpose

Produces a frozen, ground-truth reference (`tests/fixtures/validation_reference.npz`) by running
the **source `ai-trading-model` repo's own training-time code** (`build_features_and_targets`,
`build_sessions`, `MultiHorizonGRUModel` + a real checkpoint) over a small, real slice of
historical data. `server/validate_against_eval.py` then replays the same slice through this
repo's serving path (`data_source` → `state` → `inference`) and diffs the two, to catch any
divergence between the training-time and serving-time feature/model code.

## Where it must run

- **Interpreter**: the source repo's own venv (`/home/jagan/projects/ai-systems/ai-trading-model/.venv/bin/python`) —
  it imports `ai_trading_model.*`, which only exists there, not in this repo.
- **Working directory**: the source repo's root — all paths in `config/train_config.yaml`
  (`data/trading_data.db`, etc.) are relative to that directory.

```bash
cd /home/jagan/projects/ai-systems/ai-trading-model
.venv/bin/python /home/jagan/projects/ai-systems/ai-trading-model-deployment/scripts/generate_validation_fixture.py
```

## Required inputs

### Module-level constants (edit these to target a different slice)

| Constant | Meaning | Notes |
| --- | --- | --- |
| `TICKER` | Which ticker's sessions to capture | Must be one of the 23 tickers in `data/trading_data.db` (`AAL, AAPL, AMD, AMZN, BAC, BMNR, F, GOOGL, INTC, IREN, META, MSFT, MU, NOK, NU, NVDA, ONDS, PATH, PCG, PLTR, PLUG, SPCX, TSLA` — see [docs/data-readme.md](./data-readme.md)) |
| `DAYS_OFFSET` | Calendar days to skip past the ticker's earliest bar before looking for sessions | Default `400` — arbitrary, just far enough past the start to avoid a possibly-thin first day; doesn't need to land on an actual trading day |
| `WINDOW_DAYS` | Span of bars to load starting at `start_ts` | Must contain **at least 2 full trading sessions** after weekends/holidays are excluded; default `5` reliably satisfies this. The script raises `RuntimeError` if fewer than 2 sessions are found in the window |
| `OUTPUT_PATH` | Absolute path to write the `.npz` fixture to | Defaults to this repo's `tests/fixtures/validation_reference.npz` |
| `CHECKPOINT_PATH` | Which checkpoint to run inference with | Relative to source-repo CWD. Only the **baseline** (bare `state_dict`, unnormalized) checkpoint path is implemented — using the normalized variant would additionally require applying its saved `feat_mean`/`feat_std` before the forward pass, which this script does not currently do |
| `CONFIG_PATH` | YAML providing `hidden_size`/`horizons`/etc. and the three data-source paths below | Relative to source-repo CWD; default `config/train_config.yaml` |

### External data files (not constants — read indirectly via `CONFIG_PATH`'s YAML)

| File (relative to source repo root) | Used for | Referenced via |
| --- | --- | --- |
| `data/trading_data.db` | `TICKER`'s raw 1-min OHLCV bars | `cfg.data.trading_db` |
| `data/hf_regime_data.db` | Regime ETF bars (`SPY, QQQ, IWM, HYG, TLT, GLD, USD`, per `cfg.data.regime_tickers`) | `cfg.data.regime_db` |
| `data/fred_econ_data_6yrs.csv` | Macro series | `cfg.data.fred_csv` |
| `checkpoints/best_model.pt` | Trained model weights | `CHECKPOINT_PATH` |
| `config/train_config.yaml` | Hyperparameters + the three paths above | `CONFIG_PATH` |

These `.db`/`.csv` files already contain all 23 tickers and the full regime/macro history — to
target a different ticker you only change `TICKER` (and possibly `DAYS_OFFSET`/`WINDOW_DAYS` if
that ticker's available date range differs); you don't supply new data files.

## What it does, step by step

1. Loads `config/train_config.yaml` for hyperparameters and data source paths.
2. Finds `TICKER`'s earliest available bar timestamp, then computes a `[start_ts, end_ts]`
   window `WINDOW_DAYS` long, starting `DAYS_OFFSET` days after that earliest bar.
3. Loads `TICKER`'s bars, the regime ETF bars, and the full FRED macro CSV for that window.
4. Runs the source repo's real `build_features_and_targets` to get the ground-truth 42-column
   feature matrix (`features`) and row metadata (`meta`, with `ticker`/`timestamp`/`close`/
   `realized_vol_15m`).
5. Runs `build_sessions(meta)` to find trading-day session boundaries; keeps only the **first
   two** sessions found (`sess0`, `sess1`) and discards the rest, even if the window spans more
   than two (e.g. a 5-day window can yield 4 sessions — the extra two are computed but unused).
   Two is the minimum needed to validate the hidden-state-reset-at-session-boundary behavior.
6. Loads the real checkpoint into `MultiHorizonGRUModel`, runs each session through it as **one
   batched GRU call** (`model(x, hidden)` with the whole session's bars as one sequence) —
   this mirrors how `train.py`/`evaluate.py` process a session as a single TBPTT chunk, which is
   *not* how the serving path replays it (bar-by-bar, one model call per bar). This difference is
   the source of the small floating-point tolerance documented in
   `server/validate_against_eval.py` (`PRED_TOL`).
7. Denormalizes the raw predictions to `(peak_price, dip_price)` using each row's `close`/
   `realized_vol_15m`, per [docs/model-design.md](./model-design.md) section 2's formula.
8. Trims the raw bars down to exactly session0 + session1, plus 60 minutes of lookback before
   session0 (the longest rolling feature window, `past_peak_dev_60m`) — this is what gets handed
   to the serving path later, so it gets exactly the warmup it needs and nothing extraneous.
9. Builds a `{date_str: {series_id: value}}` FRED-rows dict covering every calendar date in that
   trimmed window.
10. Saves everything to `OUTPUT_PATH` via `np.savez`.

## `.npz` key schema

| Key | Type / shape | Description |
| --- | --- | --- |
| `ticker` | `str` | The single ticker this fixture covers (e.g. `"AAPL"`) |
| `ticker_bars` | structured array, `bars` schema | Raw 1-min bars for `ticker`: 60 min lookback through end of session1 |
| `regime_bars` | structured array, `bars` schema | Same window, for the 7 regime ETFs |
| `fred_rows_json` | `str` (JSON) | `{date_str: {series_id: value}}` for every calendar date in `ticker_bars` |
| `sess0_start`, `sess0_end` | `int` | Row indices into the source repo's `features`/`meta` arrays (not into `ticker_bars`) bounding session 0 |
| `sess0_date` | `str` `YYYY-MM-DD` | Market-tz calendar date of session 0 |
| `sess1_start`, `sess1_end`, `sess1_date` | same, for session 1 | The trading day immediately following session 0 |
| `features_sess0`, `features_sess1` | `float32 (n_rows, 42)` | Ground-truth feature matrix per session, from `build_features_and_targets` |
| `preds_sess0`, `preds_sess1` | `float32 (n_rows, 3, 2)` | Raw model output: `[delta_peak, delta_dip]` per horizon (`[5, 10, 15]` min, in that order), from one batched GRU call per session |
| `priced_sess0`, `priced_sess1` | `float32 (n_rows, 3, 2)` | `preds_sessN` denormalized to `(peak_price, dip_price)` per horizon |
| `close_sess0`, `close_sess1` | `float32 (n_rows,)` | `close[t]` used for denormalization |
| `vol_sess0`, `vol_sess1` | `float32 (n_rows,)` | `realized_vol_15m[t]` used for denormalization |
| `horizons` | `int array` | e.g. `[5, 10, 15]` |

`n_rows` is the number of bars in each session (not fixed — varies by how many 1-min bars
(including any extended-hours bars present in the source data) fall on that trading day; the
generated fixture had 792 rows for session0 and 850 for session1).

## Known limitations of the current fixture / script

- Captures only **one ticker, two sessions** — doesn't exercise ticker-specific edge cases
  (e.g. low-liquidity/volatile tickers like `BMNR`/`IREN`/`ONDS`).
- Only supports the **baseline checkpoint**; the normalized variant's `feat_mean`/`feat_std`
  scaling step is not implemented here.
- `build_sessions` may find more than 2 sessions in the window (e.g. 4, for a 5-day window) —
  sessions beyond the first two are silently discarded, not an error.
- See [docs/plan.md](./plan.md) section 8, step 6, for the full list of validation
  coverage limitations.

## Regenerating

Re-running the script is expected to reproduce the exact same fixture (same session dates, same
feature values, same predictions) given the same constants and unchanged source data/checkpoint —
this was confirmed when the script was extracted from an earlier inline version. If a rerun
produces different session dates or values unexpectedly, that's a sign something changed
upstream (data, checkpoint, or source repo code) and worth investigating before trusting
`server/validate_against_eval.py`'s results.
