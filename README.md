# UpScale

A simple, chat-first desktop AI crypto trading assistant.

> **Status: prototype.** The chat UI and a local API run end-to-end. The Vision agent reads
> chart screenshots with Claude; the Market and Technical Analysis agents use live,
> read-only CoinGecko data and deterministic calculations. News, Opportunity and Risk are
> still mocks that return clearly labeled placeholder output. There is no trading of any kind.

UpScale is meant to explain evidence, scenarios, risks and uncertainty. It is not
financial advice and does not guarantee outcomes.

## Architecture

```
User → Chat UI (Tauri + React) → POST /chat → Orchestrator
                                                  ├── Vision Agent            (live: Claude reads screenshots)
                                                  ├── Technical Analysis Agent (live: SMA/EMA/RSI/MACD, trend, levels)
                                                  ├── Market Agent            (live: CoinGecko)
                                                  ├── News & Sentiment Agent
                                                  ├── Opportunity Agent       (scenarios, never buy/sell calls)
                                                  └── Risk Agent              (reviews all of the above)
```

```
UpScale/
├── app/                    Desktop app
│   ├── src/                React + TypeScript UI (Vite)
│   │   ├── components/     Header, MessageList, Composer
│   │   ├── hooks/useChat   Conversation state + calls to the backend
│   │   ├── lib/            API client, image helpers
│   │   └── types/          Shared chat types (mirror backend schemas)
│   └── src-tauri/          Tauri 2 (Rust) desktop shell
└── backend/                Python FastAPI, runs locally
    ├── upscale/
    │   ├── main.py         GET /health, POST /chat
    │   ├── orchestrator.py Runs selected agents, merges results into one Analysis
    │   ├── routing.py      Picks agents for a request (keyword-based for now)
    │   ├── agents/         base.py (Agent interface) + one module per agent
    │   ├── services/       market_data.py (cache/rate limit), coingecko.py,
    │   │                   indicators.py (pure math), technical_analysis.py,
    │   │                   vision.py (image checks, Claude call, guardrails)
    │   ├── schemas.py      Request/response models, AgentResult, Analysis
    │   └── config.py       Env config (reads repo-root .env)
    └── tests/
```

The backend is stateless: the UI sends the whole conversation on each `/chat` call.
Screenshots are sent as base64 (PNG/JPEG/WebP/GIF, ≤5 MB each, ≤4 per message).

`/chat` returns `message` (plain text the UI shows) plus `analysis`: summary, evidence,
scenarios, risks, uncertainty, routing reasons, and each agent's raw result. To replace a
mock agent, subclass `upscale.agents.Agent` with the same `name` and swap it into
`default_agents()`.

Market data is fetched by `upscale.services` (never by agents directly), cached for 60 s per
asset and limited to 20 provider calls per minute. It works without an API key; set
`UPSCALE_COINGECKO_API_KEY` to a free CoinGecko demo key for higher limits. If data can't be
retrieved, the Market Agent reports an error rather than any value, and the rest of the
analysis still runs.

The service also returns historical OHLCV candles (`get_candles(symbol, timeframe, limit)`,
≤500 per request, cached up to 5 min). UpScale's timeframes are 1m/5m/15m/1h/4h/1d, but
CoinGecko's public API only offers **4h** (up to 180 candles / 30 days) and no per-candle
volume. Other timeframes raise `UnsupportedTimeframeError` until a provider that supports
them is added to `candle_providers`; candles are never synthesized.

Technical analysis (`services/technical_analysis.py`) runs on those candles: SMA 20/50,
EMA 20/50, RSI 14 (Wilder), MACD 12/26/9, 50-candle high/low, a close-vs-SMA trend label and
approximate swing-point support/resistance. Periods live in `TechnicalAnalysisConfig`.
Anything without enough candles is reported as unavailable rather than estimated.

Screenshots go to the Vision agent (`services/vision.py`), which asks Claude
(`UPSCALE_VISION_MODEL`, default `claude-opus-5`) to transcribe the chart into a strict JSON
schema, then drops anything not explicitly visible. Its detected coin and timeframe feed the
Market and Technical agents unless your message names them; screenshot values are shown as
visual readings, never as live data. It needs `ANTHROPIC_API_KEY` (or an `ant auth login`
profile); without one, screenshot analysis reports "not configured" and everything else runs.

### Outcome tracking (measurement only)

`services/outcomes/` records what happened after Growth Scout surfaced a token or Analyze made
a decision. It never changes Scout scoring, stages, Risk, Opportunity or BUY / SELL / WAIT,
and it reports no win rates, expected returns or simulated trades.

* **Observations** (immutable, enforced by SQLite triggers): each Scout ranking anchors a
  candidate when it is first ranked, re-enters, changes stage, moves 10+ points, or its last
  anchor is 6 h old (`ObservationPolicy`). Every Analyze with a real decision on a
  measurable market is stored with its triggers, invalidation, reference price and risk.
* **Horizons** 5m / 15m / 1h / 4h / 24h (configurable) are PENDING until their window has
  passed, then COMPLETE, PARTIAL or UNAVAILABLE, each missing part with its reason. Prices
  come only from the exact pool (chain + token + pool) or exact exchange market: never a
  ticker match, never stitched across pools. Terminal states (pool gone, liquidity
  collapse, no trades) stay in the dataset.
* **Collector**: a background task that sleeps until the next horizon is due. It reuses
  Scout's stored snapshots first, then batched pool lookups and candles in its own
  "outcomes" request lane: it never uses capacity reserved for Analyze, outranks Scout on
  the shared GeckoTerminal quota (it may use Scout refresh's idle reservation), makes no
  requests while (or just after) Scout or Analyze runs or for 2 minutes after a real HTTP
  429, and serves the longest-deferred work first so nothing due starves. Observations
  without a reference price make no candle requests (their price outcome is not
  measurable). `UPSCALE_OUTCOMES=0` switches it off; `UPSCALE_OUTCOME_CONFIG` overrides
  thresholds.
* **Background Scout**: the same Scout scan a manual refresh runs, every 30 minutes while
  the backend is up (first run 5 minutes after startup), so anchors accumulate without the
  UI open. It is the lowest priority on every provider quota: it defers (retrying in 5
  minutes) while Analyze runs or just ran, while due outcome work is waiting for provider
  quota, while a Scout refresh runs, or while a discovery provider is busy or recently
  answered "rate limited". `UPSCALE_BACKGROUND_SCOUT=0` switches it off;
  `UPSCALE_BACKGROUND_SCOUT_INTERVAL_MINUTES` sets the interval (at least 5).
  `GET /scout/background/status` shows its state and last result.
* **API** (developer-oriented): `GET /outcomes/scout[/{id}]`, `/outcomes/decisions[/{id}]`,
  `/outcomes/summary?group_by=stage&horizon=1h` (cohorts under 20 measured outcomes report
  INSUFFICIENT_SAMPLE instead of statistics), `/outcomes/status` (the last cycle's
  deferrals split into UpScale's own quota vs. provider 429s, and the live GeckoTerminal
  quota), and `/outcomes/replay`, a
  read-only historical replay from stored snapshots (not a backtest; also
  `python -m upscale.services.outcomes.replay`).

### Replay Lab (calibration research, no lookahead)

`services/replay_lab/` replays historical decision times T through the **production**
Scout, Technical, Risk and Opportunity code, stores the frozen decision, and only then reveals
the known 5m / 15m / 1h / 4h / 24h outcomes (live outcome definitions). It is research
evidence, not a backtest: no fills, fees, slippage, size or latency are assumed, and replay
alone never shows a setup is profitable. It never trades, never touches keys or wallets, and
never changes production configuration.

* **Separate data**: `UPSCALE_REPLAY_DB` (default `~/.upscale/replay.sqlite3`), every row
  `HISTORICAL_REPLAY`. The live Scout database is only read (`mode=ro`) as an archive of what
  Scout recorded; the replay store refuses to open a live database.
* **No lookahead**: a `HistoricalClock` per sample. While deciding, only evidence at or
  before T is readable (candles must have *closed* by T); future data needs the receipt of
  the committed decision. Decisions are immutable (SQLite triggers + SHA-256). Domain code
  that ages things against "now" uses `services/clock.py`, frozen at T during replay.
* **Evidence**: `RECORDED` samples are Scout's stored snapshots (full market evidence);
  `CANDLES` samples have pool candles only (Scout can't be reconstructed: no historical
  trade counts, liquidity or market cap). On-chain safety and news have no historical source
  and are reported unavailable (production Opportunity then treats them as missing).
* **Splits**: time-based CALIBRATION / VALIDATION / HOLDOUT (70/15/15), persisted; HOLDOUT
  windows are sticky across jobs and only readable with `--include-holdout
  --final-evaluation` (logged). Findings are EXPERIMENTAL, CALIBRATION-only, stored in the
  replay database.
* **Lowest provider priority**: its own `replay` lane on the production GeckoTerminal quota
  (no reservation), pausing while a local UpScale backend runs or after a provider 429.

```bash
python -m upscale.services.replay_lab run --start 2026-09-28 --end 2026-09-29 \
  --chains solana --max-samples 20 --min-spacing-minutes 60 --mode MARKET_ONLY
python -m upscale.services.replay_lab status
python -m upscale.services.replay_lab summary --horizon 1h --group-by stage [--baselines] [--findings]
python -m upscale.services.replay_lab resume <job_id>
```

### Point-in-Time Evidence Archive

`services/evidence_archive/` records, append-only, the evidence production UpScale already
obtained, and when it was true (`observed_at`): Scout market observations, Analyze's DEX
snapshots, on-chain safety snapshots (authorities, token program, supply, holder
concentration, excluded pools, and failures with their reason), social momentum (metrics
only, never posts), every Growth Scout evaluation and every Analyze reply linked to its
decision observation. Each record carries provenance (provider, production component,
availability) and version fingerprints (`services/versions.py`). It adds no provider
requests: production hands over objects it already has (`hooks.emit`), a background thread
writes them, and it never raises into or slows production.

* Storage: `UPSCALE_EVIDENCE_DB` (default `evidence.sqlite3` next to `UPSCALE_SCOUT_DB`),
  append-only (triggers), payloads redacted with the log-safety rules and compressed.
* Replay Lab reads it read-only: archived on-chain safety observed within 60 minutes
  before T (`--safety-max-age-minutes`) is served to the production OnchainSafety agent and
  Growth Scout; later evidence is invisible; current chain state is never used.
* Optional enrichment (`UPSCALE_EVIDENCE_SAFETY_ENRICHMENT=1`): after a scan, at most
  `UPSCALE_EVIDENCE_SAFETY_MAX_PER_REFRESH` ranked Solana candidates without recent safety
  evidence get a lookup, below every other production workload.
* Status: `GET /evidence/status`, `python -m upscale.services.evidence_archive status`
  (also `readiness`, and `show --asset solana:<mint> --kind safety --at <time>`).

### Calibration Engine (experimental research only)

`services/calibration/` analyzes LIVE_FORWARD outcomes and HISTORICAL_REPLAY samples (never
merged: every table is per origin, plus a combined view showing its composition) and
produces EXPERIMENTAL findings and single-change candidate configurations. It never changes
production scoring, thresholds, confidence or decisions, and never promotes a candidate.

* Splits: replay keeps its own; live observations are split by UTC day in 20-day cycles
  (14 CALIBRATION, 3 VALIDATION, 3 HOLDOUT), sticky and chronological. HOLDOUT is only read by
  `final-evaluate <id> --confirm-final-evaluation` (logged; the candidate is frozen first).
* Evidence rules: 20 measured / 5 assets for descriptive stats, 50 / 10 for candidates,
  100 / 20 for stronger evidence; per-asset caps and spacing against correlated samples.
* Storage: `UPSCALE_CALIBRATION_DB` (default `calibration.sqlite3` next to the Scout
  database), append-only; candidate status only moves forward.
* CLI: `python -m upscale.services.calibration status | readiness | analyze --horizon 1h |
  findings | create-candidates | candidates | validate <id> | compare <id> | final-evaluate <id>
  --confirm-final-evaluation`. API (read-only): `GET /calibration/status`, `/readiness`,
  `/findings`, `/candidates`.

## Prerequisites

- Node.js ≥ 20.19 (22 LTS recommended)
- Python ≥ 3.11
- Rust (stable), only needed for the desktop window
- Tauri system dependencies, see "Desktop window" below

Optional: `cp .env.example .env` to override defaults.

## Run the backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn upscale.main:app --reload --port 8000
```

Check it: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`.
Interactive API docs: http://127.0.0.1:8000/docs

Tests and checks (after installing `.[dev]`):

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
```

## Run the frontend

With the backend running, in a second terminal:

```bash
cd app
npm install
npm run dev          # UI in the browser at http://localhost:1420
```

### Desktop window (Tauri)

```bash
cd app
npm run tauri dev
```

**On WSL2 (Ubuntu)** this opens a Linux window via WSLg and first needs:

```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev
```

**Windows build (the target platform)** must be built on Windows, not in WSL:
install Node, Rust (MSVC toolchain) and the Microsoft C++ Build Tools, clone the repo
on the Windows side, then run `npm install` and `npm run tauri build` in `app/`.
The backend can keep running in WSL, since WSL2 forwards `127.0.0.1:8000` to Windows.

## Other commands

```bash
cd app && npm run build    # typecheck + production bundle
cd app && npm run lint     # oxlint
```
