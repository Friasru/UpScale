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
