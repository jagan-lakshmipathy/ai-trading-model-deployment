# Deployment Plan

Status check of what's been lifted from `ai-trading-model` per [README.md](../../README.md), gaps
found, and the implementation plan for session/hidden-state management, the MCP server layer,
and validation.

## 1. What's present vs. what the README called for

| README item | Status | Notes |
| --- | --- | --- |
| `checkpoints/best_model.pt`, `best_model_normalized.pt` | Present under [checkpoints/](../../checkpoints) | Directory name typo (`checpoints/`) fixed — renamed to `checkpoints/`. |
| `encoder.py` (`MultiHorizonGRUModel`) | Present in [server/encoder.py](../../server/encoder.py) | Flat module, not `ai_trading_model/model/encoder.py` package path. |
| `features.py` (`build_features_and_targets`) | Present in [server/features.py](../../server/features.py) | Still computes targets (forward-looking labels), not stripped down to inference-only as the README suggested. Needs a serving-time variant that skips target computation (no future bars available live) but keeps identical feature math. |
| `sessions.py` | Present in [server/sessions.py](../../server/sessions.py) | `build_sessions`/`add_session_date`, matches spec §2.6. |
| `config.py` | Present in [server/config.py](../../server/config.py) | Trimmed to just `ModelConfig` (horizons, hidden_size, num_layers, head_hidden_size, dropout) — the only config actually needed at serving time; the training-only `DataConfig`/`TrainingConfig` were removed. |
| `train_config.yaml` / `train_config_normalized.yaml` | Present | Confirms `hidden_size=128`, `num_layers=2`, `head_hidden_size=64`, `horizons=[5,10,15]`, `dropout=0.1`. |
| Docs (`model-deployment-spec.md`, `data-readme.md`, `model-design.md`) | Present | Read and summarized above; this is the primary contract reference. |
| `evaluate.py` / `evaluate_normalized.py` | **Removed** | Imported a package layout that doesn't exist here (`.input`, `.model.encoder`, `.train`) and were never executable in this repo. Per §7 (Validation plan), the intended workflow runs `evaluate.py` in the source `ai-trading-model` repo (where it works unmodified) to produce reference numbers — these broken copies added no value here. |
| Source `bars`/regime/FRED data (`data/*.db`, `data/*.csv`) | **Not present, and not needed here** | Serving will fetch live bars from Massive (or an Interactive Brokers feed) and live macro series from the FRED API directly — no local DB/CSV copies in this repo. The historical-row validation step (§6) still relies on the source repo's existing local DBs, but only to produce reference numbers there, not to populate anything in this repo. |
| MCP server layer | **Not started** | New code, this repo's main deliverable. |
| Session/hidden-state manager (serving-time) | **Not started** | `sessions.py` gives session *boundaries* for historical batches; serving needs a live, per-(ticker) stateful hidden-state cache, which is a different (streaming) problem. |

### Key gaps to resolve before coding

1. ~~**Missing runtime scaffolding**~~ — resolved: added [pyproject.toml](../../pyproject.toml)
   (`torch`, `pandas`, `numpy`, `pyyaml` deps) and [server/\_\_init\_\_.py](../../server/__init__.py)
   so `server/` is now a proper installable package (`pip install -e .`); relative imports
   (`from .sessions import MARKET_TZ` in `features.py`) resolve correctly under this layout —
   verified via `python -c "from server.encoder import MultiHorizonGRUModel"` in a fresh venv.
2. **`evaluate.py`/`evaluate_normalized.py` depend on training-only modules** (`batching.py`, `train.py`) not lifted (and have since been removed from this repo, see above). Neither is actually needed here:
   - `batching.py` (`iter_tbptt_batches`) implements TBPTT chunking/padding purely for training efficiency — irrelevant to serving, which streams one bar at a time per session with no chunking. Not porting it.
   - `train.py`'s `_date_cutoffs`/`_split_by_date` only reproduce which rows fell in the chronological test split. We don't need to reproduce that split in this repo — instead, run `evaluate.py` unmodified in the source `ai-trading-model` repo (where `train.py`/`batching.py` still exist) to produce reference predictions for a chosen set of historical rows, and diff this repo's serving path against those reference numbers. See §7.
3. **No live data source wired up yet** — `data_source.py` needs a live Massive (or Interactive Brokers) bars feed for target-ticker + regime-ETF bars, and a live FRED API client for macro series. No local SQLite/CSV copies are used at serving time; this replaces the earlier `load_bars`/`load_fred` SQLite/CSV readers in [server/features.py](../../server/features.py) with live-feed equivalents that produce the same shape of data.
4. ~~Checkpoint directory name typo (`checpoints/`)~~ — fixed, renamed to `checkpoints/`.

## 2. Target module layout

```
server/
  config.py            # existing — trim to ModelConfig + a new ServingConfig
  encoder.py            # existing — no changes needed (pure nn.Module)
  features.py           # existing — add a serving-time single-row/incremental path
  sessions.py           # existing — reused as-is for session-boundary/day-reset logic
  inference.py          # NEW — load checkpoint(s), run model.forward(), denormalize outputs
  state.py              # NEW — per-ticker hidden-state cache + session reset logic (serving)
  mcp_server.py          # NEW — MCP server exposing tools/resources around inference.py
  data_source.py        # NEW — pre-fetched-bars shaping helpers (bars_to_df, get_features_for_ticker)
  live/                 # NEW — live data fetching/buffering/scheduling (see below)
tests/ or scripts/
  validate_against_eval.py  # NEW — replays known historical rows and diffs vs evaluate.py output
```

**Update (2026-11-01):** the live data-fetching/buffering/scheduling work originally scoped for
`data_source.py` here was implemented as a separate `server/live/` subpackage instead (see
[docs/agent-docs/live-prediction-agent-plan.md](../agent-docs/live-prediction-agent-plan.md) for
the full design and step-by-step build log): `live/buffer.py` (`TickerBuffer`/`RegimeBuffer`),
`live/massive_client.py`, `live/fred_client.py` (+ `FredCache`), `live/scheduler.py`,
`live/tracker.py` (dynamic ticker registration/warmup/eviction), `live/calendar.py`
(market-hours gating), `live/config.py` (`LiveConfig`/env-var secrets). `data_source.py` itself
was kept as-is (the pre-fetched-bars shaping helpers `bars_to_df`/`get_features_for_ticker` are
still the shared code path, now fed from `live/buffer.py`'s buffers instead of caller-supplied
lists). `mcp_server.py`'s `predict` tool signature changed from `predict(ticker, ticker_bars,
regime_bars, fred_rows)` to `predict(ticker)` accordingly — see §5 below.

## 3. Session / hidden-state management (`server/state.py`)

Per spec §2.6:

- Maintain a `dict[str, torch.Tensor]` (or small LRU) keyed by `ticker`, each holding the current
  GRU hidden state `(num_layers=2, batch=1, hidden_size=128)` plus the last-seen market-tz
  session date for that ticker.
- On each new bar for a ticker:
  - Compute `session_date` via `sessions.add_session_date` (or an equivalent single-row helper).
  - If the ticker has no cached state, or the cached session date differs from the new bar's
    session date (new trading day, or ticker switch — trivially true since state is keyed by
    ticker), zero the hidden state (`model.init_hidden(1)`) before this step.
  - Otherwise carry the cached hidden state forward.
  - After `model.forward`, cache the returned `next_hidden` and the session date.
- No `.detach()` needed (inference-only, no backward pass) — confirmed by spec.
- Thread-safety: if the MCP server can receive concurrent requests per ticker, guard the cache
  with a per-ticker lock (simple `threading.Lock` is enough at this scale).
- Expose an explicit `reset(ticker)` / `reset_all()` for manual intervention (e.g. operator
  suspects state drift) and for test setup.

## 4. MCP server layer (`server/mcp_server.py`)

- Use the official `mcp` Python SDK (stdio or streamable-HTTP transport — TBD based on how
  traders will connect; default to stdio for a first cut, matching typical MCP client setups).
- Tools to expose:
  - `predict(ticker: str, bar: BarInput) -> PredictionOutput` — feed one new 1-min bar for a
    ticker, return per-horizon `(peak_price, dip_price)` for all 3 horizons, denormalized to
    raw price per spec §3. Internally: build the 42-feature row (needs recent rolling-window
    context — see §5 open question), advance hidden state via `state.py`, run `inference.py`.
  - `reset_session(ticker: str)` — manual hidden-state reset.
  - `model_info()` — static resource/tool returning the contract summary (features list,
    horizons, checkpoint variant in use, known risks from spec §5) so traders/clients can
    introspect without reading docs.
- Choose one checkpoint variant to serve by default (baseline `best_model.pt`, since it's the
  only one validated per spec §2.2) with the normalized variant available behind a config flag/
  second tool, not silently mixed.
- Input validation at the MCP boundary: reject bars with non-monotonic timestamps per ticker,
  missing OHLCV fields, or a timestamp inside a known future (look-ahead guard) — cheap and
  catches the most likely caller mistakes.

## 5. Rolling-window features need history, not just the latest bar — **resolved, see agent-docs plan**

Columns like `past_peak_dev_60m`, `rel_volume_20m`, `realized_vol_15m`, `spy_ret_15m`, etc.
require a trailing window of bars (up to 60 minutes), not just the single incoming bar. The
MCP `predict` tool originally had two options:

- **(a)** the caller supplies the last ~60 minutes of target-ticker + regime-ETF bars each call
  (simplest, but pushes the data-fetch burden to the client), or
- **(b)** this server maintains its own rolling buffer per ticker (and for the shared regime
  ETFs) fed by a live data source, recomputing the feature row from the buffer each step.

**Decision: (b) was implemented** — see
[docs/agent-docs/live-prediction-agent-plan.md](../agent-docs/live-prediction-agent-plan.md) for
the full design/build log. The server now polls live bars from Massive (same vendor as training,
so the split/dividend-adjustment convention stays consistent per spec §2.5) and pulls macro
series from the live FRED API (same series IDs as [docs/data-readme.md](./data-readme.md)) via a
background scheduler, maintaining per-ticker/shared-regime rolling buffers. `predict(ticker)` no
longer accepts bars at all — callers only supply a ticker symbol; a brand-new ticker triggers an
immediate warmup backfill and returns a `"warming_up"` status until enough history has
accumulated.

## 6. Avoiding cross-repo feature/model-code duplication

The feature math (`build_features_and_targets`) and architecture (`MultiHorizonGRUModel`) exist
in two places: the source `ai-trading-model` repo (training) and this repo (serving, currently a
forked/adapted copy in [server/features.py](../../server/features.py) and
[server/encoder.py](../../server/encoder.py)). Nothing currently prevents the two copies from
silently drifting apart — a real risk per the README ("this is the part most likely to silently
diverge from training if not ported carefully"). Options considered, in order of preference:

1. **Extract a shared installable package** — move the feature-computation core, model class,
   and session-boundary logic into their own package (e.g. `ai_trading_model_core`) with a
   `pyproject.toml` and semantic version. Both repos declare an explicit version pin
   (`ai-trading-model-core==1.2.0`) in `requirements.txt`/`pyproject.toml`. Delivery mechanism can
   be a private PyPI/Artifactory index, or `pip install git+<repo>@<tag>` (git-as-transport, but
   still a real versioned, installed package — `pip list`/`pip show` reflect it, CI can pin and
   enforce the version). **Recommended.** Single source of truth, enforceable version pin,
   testable in isolation; upfront refactor cost and need for a lightweight release process.
2. **Git submodule** — embed the source repo (or an extracted subset) as a submodule pinned to a
   commit SHA. Lighter to set up than full packaging, but weaker guarantees: no install isolation
   or version metadata (`pip show` sees nothing), Python needs a path hack to import it, and nothing
   stops the two repos' submodule pointers from silently diverging to different commits. In
   practice this is closer to option 3 (vendored copy) with better change-tracking than a true
   shared package.
3. **Vendored copy + automated drift check** — keep `server/features.py`/`server/encoder.py` as
   adapted copies (current state), but make the §6 validation plan below (row-by-row feature/
   prediction diff against the source repo's `evaluate.py` output) a **required, automated** gate
   (e.g. CI job) rather than a manual one-off run. Zero refactor cost, but only catches drift
   *after* it happens and requires discipline to keep running; not a fix for the duplication
   itself, only a safety net.

**Decision for this repo:** not yet made — tracked here as an open item. Short-term, unblock
implementation by keeping the vendored copy (option 3) and treating the §7 validation suite as
the drift safety net; revisit option 1 (shared package) once the serving path is proven correct,
since extracting the package is easier to do well with a working, tested serving implementation
to validate the extraction against.

## 7. Validation plan

1. In the **source `ai-trading-model` repo** (unmodified, `train.py`/`batching.py` present there),
   run `evaluate.py`/`evaluate_normalized.py` against a small slice of `data/trading_data.db` and
   capture the per-row predictions (and, with a small instrumentation tweak, the 42-column
   feature vectors) for a handful of known (ticker, timestamp) rows from the test split. Treat
   this as the frozen ground-truth reference — nothing from `batching.py`/`train.py` needs to be
   ported into this repo.
2. Build `validate_against_eval.py` in this repo: replay the same historical rows one bar at a
   time through the new `state.py` + `inference.py` serving path (simulating streaming) and diff:
   - the 42-column feature vector against the captured reference feature vectors for the same rows,
   - the raw model output (`preds` tensor) and denormalized `(peak_price, dip_price)` against the
     reference predictions captured in step 1.
   - Tolerance: exact match expected (same deterministic math) modulo floating-point order-of-
     operations; flag any difference above a tight epsilon (e.g. `1e-5`) for investigation.
3. Confirm hidden-state reset behavior specifically: construct a session that spans a day
   boundary and verify the serving path's hidden state zeroes at the boundary exactly where
   `sessions.build_sessions` would split it in the source repo's batch/eval path.
4. Only after (2) and (3) pass, exercise the MCP tool interface end-to-end (`predict` calls) on
   the same historical rows to catch any boundary-layer (serialization, input validation) bugs
   introduced by the MCP wrapping itself.

## 8. Sequencing

1. ~~Resolve scaffolding gaps: package structure (`server/` → import-safe), `requirements.txt`.~~ — done.
2. ~~Implement `state.py` (hidden-state manager) + unit tests against synthetic sessions.~~ — done:
   [server/state.py](../../server/state.py)'s `HiddenStateManager` keys cached hidden state by
   ticker, zeroes it on a new market-tz session date (via `server/sessions.MARKET_TZ`) or first
   bar for a ticker, and exposes a `step(ticker, timestamp_ms)` context manager (yielding a
   `StepHandle` with `.hidden`/`.commit(next_hidden)`) plus `reset`/`reset_all`. Using a context
   manager instead of a bare acquire/release pair ensures the per-ticker lock is always released
   even if `model.forward` raises mid-step, and an uncommitted step leaves cached state
   untouched rather than silently advancing. Verified via an inline smoke test covering
   session-reset, within-session carry-forward, per-ticker isolation, and manual reset.
3. ~~Implement `inference.py` (checkpoint loading for both variants, forward pass, denormalization,
   per spec §2.2/§3).~~ — done: [server/inference.py](../../server/inference.py)'s `InferenceEngine`
   loads either the bare-`state_dict` baseline or the `{"model_state","feat_mean","feat_std"}`
   normalized checkpoint (applying the saved, not freshly-fit, scaler per spec §2.2), always on
   CPU, with `model.eval()` enforced. `forward()` runs the model over one or more feature rows
   given a hidden state (pairs with `state.py`'s `StepHandle`); `denormalize()` inverts the
   volatility-scaled delta transform (`close + delta * (close * realized_vol_15m + eps)`,
   `eps=1e-8`) into per-horizon `(peak_price, dip_price)`. Verified end-to-end against both real
   checkpoints in [checkpoints/](../../checkpoints).
4. ~~Implement `data_source.py` in "accept pre-fetched bars" mode first (defer live Massive/IBKR +
   FRED API polling); wire a serving-time `features.py` path that reuses
   `build_features_and_targets` math without target computation (per §6, this keeps the current
   vendored-copy approach — revisit shared-package extraction after validation per §6's
   decision).~~ — done: [server/features.py](../../server/features.py) now splits a shared
   `_compute_feature_columns` helper (Groups 1-3 only, no look-ahead) out of
   `build_features_and_targets`, and adds `build_features` — the serving-time entry point that
   reuses the identical math with no target/label computation. [server/data_source.py](../../server/data_source.py)
   implements pre-fetched-bars mode: `bars_to_df` validates/shapes caller-supplied bar dicts
   (rejects missing fields, non-monotonic or duplicate timestamps per ticker, checked before any
   sorting so out-of-order input is surfaced rather than silently tolerated), and
   `get_features_for_ticker` assembles `(trading_df, regime_wide, fred_df)` from caller-supplied
   ticker/regime bars + FRED rows and calls `build_features`. Verified end-to-end: synthetic bars
   → `get_features_for_ticker` → 42-column feature matrix → `InferenceEngine.forward` →
   `denormalize` → sane `(peak_price, dip_price)` output.
5. ~~Implement `mcp_server.py` wrapping the above as MCP tools.~~ — done:
   [server/mcp_server.py](../../server/mcp_server.py) uses the official `mcp` SDK (v2.x API,
   `MCPServer`/`@server.tool()`/`@server.resource()`; note the SDK renamed `FastMCP` →
   `MCPServer` between v1 and v2). `create_server(checkpoint_path=None, normalized=False, ...)`
   wires one `InferenceEngine` + one `HiddenStateManager` and exposes:
   - `predict(ticker, ticker_bars, regime_bars, fred_rows)` — builds features via
     `data_source.get_features_for_ticker`, steps the cached hidden state, runs inference, and
     returns denormalized `(peak_price, dip_price)` per horizon.
   - `reset_session(ticker)` — manual hidden-state reset.
   - `model-info://contract` resource — static contract summary (checkpoint in use, horizons,
     regime basket, known risks from spec §5).
   Defaults to the baseline checkpoint (the only validated variant per spec §2.2); the normalized
   variant is available via `normalized=True`, never mixed within one server instance. `main()`
   runs over stdio transport. Verified in-process end-to-end (`list_tools`, `call_tool` for both
   tools, `read_resource` for `model-info://contract`) with synthetic bars. Added `mcp>=2.0` to
   [pyproject.toml](../../pyproject.toml) and regenerated [requirements.txt](../../requirements.txt).
6. ~~Implement and run the validation suite (§7); fix any divergence before considering this
   done.~~ — done, using the source repo at `/home/jagan/projects/ai-systems/ai-trading-model`
   (confirmed present with its `data/trading_data.db` etc.). Generated a frozen reference fixture
   ([tests/fixtures/validation_reference.npz](../../tests/fixtures/validation_reference.npz)) by
   running the source repo's actual `build_features_and_targets`/`build_sessions`/
   `MultiHorizonGRUModel` + `checkpoints/best_model.pt` (byte-identical, `md5sum`-verified, to
   this repo's copy) over two consecutive AAPL trading sessions, including the regime-ETF bars
   and FRED rows needed to reproduce all 42 columns. [server/validate_against_eval.py](../../server/validate_against_eval.py)
   replays that fixture through the actual serving path
   (`data_source.get_features_for_ticker` → `sessions.build_sessions` → `HiddenStateManager.step`
   bar-by-bar → `InferenceEngine.forward`/`denormalize`) and diffs against the reference. Results:
   - **Feature vectors: exact match (0.0 max diff)** across both sessions — confirms the serving
     feature pipeline is bit-for-bit identical to the source's training-time code.
   - **Session-boundary hidden-state reset: confirmed** — the replay loop asserts a zeroed hidden
     state at each session's first bar; no failures.
   - **Raw model output**: small diff (~1.8e-5 to 2.7e-5), isolated via a same-weights/same-input
     batched-vs-streamed GRU comparison to be floating-point reduction-order noise (running N
     steps as one batched call vs. N individual single-step calls — cuDNN/BLAS use different
     matmul reduction order per sequence length), **not a serving-path bug**. Tolerance set to
     `2e-4` accordingly, with the root cause documented inline in `validate_against_eval.py`.
   - **Denormalized `(peak_price, dip_price)`: match within ~3e-6** — this noise washes out
     almost entirely once converted back to price space.
   All checks pass (`python -m server.validate_against_eval`). The fixture is reproducible, not a
   one-off artifact: [scripts/generate_validation_fixture.py](../../scripts/generate_validation_fixture.py)
   is checked into this repo (run it with the source repo's own interpreter/venv, CWD = source
   repo root — re-running it regenerates the fixture from scratch and was confirmed to reproduce
   identical session boundaries/results). Full documentation of required inputs, the step-by-step
   generation process, and the `.npz` key schema lives in [docs/validation-fixture.md](./validation-fixture.md)
   (also summarized more briefly in the module docstring of
   [server/validate_against_eval.py](../../server/validate_against_eval.py)).

   **Coverage limitations (not yet addressed):**
   - Only **one ticker (AAPL)**, **two sessions**, **~1,642 bars** — doesn't exercise
     ticker-specific edge cases (e.g. low-liquidity/volatile tickers like `BMNR`/`IREN`/`ONDS`
     hitting the `realized_vol`-near-zero clipping or `rel_volume`/`vwap_dev` division guards
     differently).
   - Only the **baseline checkpoint** (`best_model.pt`) was run through this rigorous
     fixture-diff; the normalized checkpoint's loading/denormalization was smoke-tested earlier
     (step 3) but not validated against a frozen reference the same way.
   - Only the regime-ETF/macro values that happened to apply to this one historical slice —
     doesn't test missing-regime-ticker degradation, stale/missing FRED data, or
     `days_since_macro_update` edge cases.
   - No adversarial/malformed-input testing at the MCP tool boundary (step 4 of the original
     §7 plan) folded into this script — only lightly covered by `mcp_server.py`'s own smoke test.
   - `PRED_TOL`'s value was set from one isolated empirical experiment (batched-vs-streamed GRU
     diff on random inputs), not a formal error-propagation bound — reasonable at this model
     scale, but worth re-checking if the architecture or sequence lengths change materially.
7. ~~Build the live Massive/IBKR bars feed and live FRED API client in `data_source.py` (§5 option
   b) once the above is proven correct on historical replay.~~ — done, as `server/live/` (not
   `data_source.py` itself): see
   [docs/agent-docs/live-prediction-agent-plan.md](../agent-docs/live-prediction-agent-plan.md)
   for the full design, step-by-step build log, and known gaps (notably: the Massive API
   endpoint/response shape is unverified against real docs/responses — flagged in that plan's
   §6). `predict(ticker)` now reads from live-polled buffers instead of caller-supplied bars.
8. Revisit §6 (shared package extraction) once the serving path is validated.
