# ai-trading-model-deployment

MCP-based deployment of the price-prediction model trained in the `ai-trading-model` repo.

## Files/artifacts to lift from `ai-trading-model`

1. **Checkpoints**:
   - `checkpoints/best_model.pt` (baseline, bare state_dict)
   - `checkpoints/best_model_normalized.pt` (dict with `model_state`/`feat_mean`/`feat_std`) — if deploying the normalized variant instead/also.

2. **Model architecture code** — needed to reconstruct the class before `load_state_dict`:
   - `ai_trading_model/model/encoder.py` (`MultiHorizonGRUModel`)
   - `train_loop.py`/`loss.py` are training-time only and not needed for inference-only serving.

3. **Feature-building code** — needed to reproduce the exact 42-column feature vector at serving time:
   - `ai_trading_model/input/features.py` (`build_features_and_targets` — adapt the feature-computation logic minus the target/label computation, since serving doesn't need forward-looking labels)
   - `ai_trading_model/input/sessions.py` (session boundary / market-tz logic, needed to know when to reset hidden state)
   - Possibly `ai_trading_model/config.py` if it defines shared constants (horizons, hidden_size, etc.) rather than hardcoding them here.

4. **Config values** — not necessarily the YAML files, but the exact hyperparameters baked into the checkpoint:
   - From `config/train_config.yaml`: `hidden_size=128`, `num_layers=2`, `head_hidden_size=64`, `horizons=[5,10,15]`, `dropout=0.1`, `n_features=42` — must match exactly or `load_state_dict` will fail/mismatch.

5. **Documentation** (copy, don't modify in the source repo):
   - `docs/server-docs/model-deployment-spec.md` — the inputs/outputs contract; should become the primary reference doc here.
   - `docs/server-docs/data-readme.md` — the exact source schema (`bars` table columns) this repo must replicate/query for live data (regime ETFs `SPY,QQQ,IWM,HYG,TLT,GLD,USD` + FRED macro series).
   - `docs/server-docs/model-design.md` — background for the target normalization formula (delta de-normalization math) needed to invert predictions back to prices.

## Steps

1. Copy `ai_trading_model/model/encoder.py` and the feature/session-building logic from `ai_trading_model/input/` into this repo (likely refactored into inference-only modules — strip out training-only code paths like TBPTT batching).
2. Copy the checkpoint(s) into a `checkpoints/` (or similar) folder here.
3. Copy the three docs listed above as reference material.
4. Implement a live data-fetch layer that replicates the exact feature pipeline (same 42-column order, same regime ETFs, same FRED series, forward-filled correctly, no look-ahead) — this is the part most likely to silently diverge from training if not ported carefully.
5. Implement session/hidden-state management per §2.6 of the deployment spec (zero hidden state per new session/ticker/day).
6. Wire up the MCP server layer around this inference core (tool/resource definitions, etc.) — new code specific to this repo.
7. Validate by running a few known historical rows through both the old `evaluate.py`/`evaluate_normalized.py` scripts (source repo) and this new MCP server, confirming identical outputs — a good regression check before going live.

## Running the live prediction agent

The live data-fetching/buffering/scheduling stack and the `agent/` poller are built — see
[docs/agent-docs/live-prediction-agent-plan.md](docs/agent-docs/live-prediction-agent-plan.md)
for the full design.

```bash
export MASSIVE_API_KEY=...
export FRED_API_KEY=...
.venv/bin/python -m agent.poller AAPL MSFT --poll-interval-s 60
```

**Important: this is a single process, not two.** `agent/poller.py`'s `main()` builds the MCP
server (`server.mcp_server.create_server()`) directly in-process and calls its `predict` tool
in-memory — there is no separate server process to start first, and no stdio/network transport
between the agent and the server today. The agent *is* the server's host process in this design.

[server/mcp_server.py](server/mcp_server.py)'s own `main()` (`server.run(transport="stdio")`) is
a genuinely standalone MCP server, meant for an external MCP client (e.g. Claude Desktop) to
connect to over stdio — that's a different, separate way to run this repo, not used by the
`agent/` poller.

If you later want the agent and server to be truly separate processes (e.g. one server shared by
multiple consumers), `agent/poller.py` would need to hold an actual MCP client connection
instead of a direct in-process `server` reference — not implemented yet.

## Plotting predicted vs. actual price

Record prediction history to a file, then plot it:

```bash
.venv/bin/pip install -e ".[plotting]"  # matplotlib, only needed for this tool

.venv/bin/python -m agent.poller AAPL --poll-interval-s 60 --log-file predictions.jsonl

# in another terminal, once some history has accumulated:
.venv/bin/python -m scripts.plot_predictions predictions.jsonl --ticker AAPL --output chart.png
```

[scripts/plot_predictions.py](scripts/plot_predictions.py) plots the actual realized close price
as a line, and each horizon's (5/10/15-min) predicted peak/dip as markers positioned at their
*target* time (`as_of_timestamp + horizon`), so you can visually compare a prediction against
where the actual price line ends up once that time arrives. Omit `--output` to open an
interactive window instead of saving to a file. Add `--non-overlapping` to thin each horizon's
markers to non-overlapping windows (the actual-price line is unaffected, always full
resolution) — recommended once you have more than an hour or two of history, since dense
minute-by-minute markers overplot into an unreadable swarm on longer runs.

## Evaluating live prediction accuracy

The same `predictions.jsonl` log can be scored for accuracy, not just plotted:

```bash
.venv/bin/python -m scripts.evaluate_live_predictions predictions.jsonl
.venv/bin/python -m scripts.evaluate_live_predictions predictions.jsonl --ticker AAPL
```

[scripts/evaluate_live_predictions.py](scripts/evaluate_live_predictions.py) reports, per ticker
per horizon: mean peak/dip error, band coverage (actual price fell within the predicted
peak/dip range), and directional accuracy — each with a 95% confidence interval. Unlike a naive
check, it only uses **non-overlapping** windows per horizon (see
[docs/agent-docs/model-improvement-roadmap.md](docs/agent-docs/model-improvement-roadmap.md) for
why overlapping 1-min-apart windows give misleadingly overconfident stats) — expect wide
confidence intervals until you've collected a multi-day, multi-ticker sample.

## Tomorrow's workflow (collecting a meaningful sample)

Per the current step in
[docs/agent-docs/model-improvement-roadmap.md](docs/agent-docs/model-improvement-roadmap.md)
("extend duration/ticker coverage"):

1. Start the poller with **more tickers, including at least one volatile/illiquid one** (e.g.
   `BMNR`, `IREN`, or `ONDS` — flagged as unverified in
   [docs/server-docs/model-deployment-spec.md §5](docs/server-docs/model-deployment-spec.md)),
   and let it run **48+ hours** so it spans multiple full trading sessions, not just one
   afternoon:
   ```bash
   .venv/bin/python -m agent.poller AAPL MSFT BMNR --poll-interval-s 60 --log-file predictions.jsonl
   ```
2. Once it's accumulated enough history, check accuracy with confidence intervals:
   ```bash
   .venv/bin/python -m scripts.evaluate_live_predictions predictions.jsonl
   ```
3. Visually inspect per ticker (use `--non-overlapping` now that there's more history to plot):
   ```bash
   .venv/bin/python -m scripts.plot_predictions predictions.jsonl --ticker AAPL --non-overlapping --output aapl.png
   ```
4. Compare the resulting confidence intervals/bias against today's first-run numbers and the
   documented test-split baseline in [docs/server-docs/plan.md](docs/server-docs/plan.md) — are
   they narrowing/converging, or still too wide to draw conclusions? See
   [docs/agent-docs/model-improvement-roadmap.md](docs/agent-docs/model-improvement-roadmap.md)
   for the full reasoning and next steps after that.
