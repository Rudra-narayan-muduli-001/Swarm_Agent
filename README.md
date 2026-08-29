# Swarm Agent

A multi-agent "swarm" in plain Python with a live-diagram browser UI. Three agents — **Planner (triage)**, **Researcher**, **Writer** — pass control to each other via handoffs (an OpenAI-Swarm-style pattern), all implemented from scratch on top of LiteLLM (provider-agnostic: OpenAI, Anthropic, Gemini, Groq, local Ollama, ...).

```
Planner (triage) ──handoff──▶ Researcher ──handoff──▶ Writer ──handoff──▶ Planner
                                │ tools: web_search, read_url
```

The UI is a dark, two-pane app: **Process rail (left)** shows every handoff, tool call and token stream diagrammatically; **Conversation (right)** is the chat. See `design.md` for the full design system and `ARCHITECTURE.md` for the backend map.

---

## Run Instructions

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ (3.11 recommended) | Check with `python --version` |
| pip | latest | `python -m pip install --upgrade pip` |
| Git | any | to clone |
| LLM API key | — | At least one of OpenAI / Anthropic / Gemini / Groq, or a local Ollama instance |

No Node.js, no build step — the frontend is zero-build vanilla JS served from `static/` by FastAPI.

### 1) Clone & enter

```bash
git clone <your-repo-url>
cd "Swarm Agent"
```

### 2) Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> If activation is blocked on Windows, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry.

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

This installs `fastapi`, `uvicorn`, `litellm`, `httpx`, `beautifulsoup4`, `python-dotenv` (`requirements.txt:1`).

### 4) Configure environment

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Open `.env` and set **at minimum**:

```ini
LLM_MODEL=gpt-4o-mini              # or claude-3-5-sonnet-latest, gemini/gemini-1.5-flash, groq/llama-3.3-70b-versatile, ollama/llama3.1
OPENAI_API_KEY=sk-...              # key matching your LLM_MODEL provider
# TAVILY_API_KEY=tvly-...          # optional — much better search; without it DuckDuckGo is used
# PORT=8000                        # optional — server port
# LLM_TEMPERATURE=0.7              # optional
```

All variables live in `.env` and are loaded by `app/core/llm.py:15` (`load_dotenv`). Each agent can override the model with its own `model` field — see `app/agents/*.py`.

| Variable | Purpose |
|---|---|
| `LLM_MODEL` | Model string; LiteLLM picks the provider by prefix (`gpt-4o-mini`, `claude-...`, `gemini/...`, `ollama/llama3.1`, `groq/...`) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` | Key for your chosen provider |
| `OLLAMA_API_BASE` | Only for local models (default `http://localhost:11434`) |
| `TAVILY_API_KEY` | Optional — when set, `app/tools/web_search.py:27` uses Tavily; otherwise DuckDuckGo HTML scrape |
| `PORT` / `HOST` | Server bind (`run.py:8` reads `PORT` default `8000`, `HOST` default `127.0.0.1`) |
| `LLM_TEMPERATURE` | Global temperature (`app/core/llm.py:20` default `0.7`) |

### 5) Run the server

```bash
python run.py
# equivalent: uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Open **http://127.0.0.1:8000** in a browser. You should see the dark two-pane UI (header with `● Swarm Agent` + `Pipeline: Planner → Researcher → Writer`, left **Process** rail with the muted architecture diagram, right **Conversation** with the greeting bubble).

Try a research prompt:

> *What are the latest developments in solid-state batteries? Write me a summary.*

The left rail will light up: `triage` → `handoff_to_researcher` → `researcher` (`web_search`, `read_url`) → `handoff_to_writer` → `writer` — all over a single SSE stream.

### 6) Verify it’s working

```bash
# agents roster (powers the header legend)
curl http://127.0.0.1:8000/api/agents
# -> {"agents":[{"name":"triage",...},{"name":"researcher",...},{"name":"writer",...}]}

# static assets
curl -I http://127.0.0.1:8000/static/style.css
curl -I http://127.0.0.1:8000/static/app.js
```

In the browser devtools, the **Network** tab should show a single `GET /api/chat?message=...` with `Content-Type: text/event-stream` and events `session`, `agent`, `token`, `tool_call`, `tool_result`, `done` (`app/main.py:70`).

### 7) Stop & restart

- Stop: `Ctrl+C` in the server terminal.
- Restart: `python run.py` again. Sessions are in-memory (`app/sessions.py:13` TTL 3600s) and are lost on restart — that’s expected.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | You forgot `pip install -r requirements.txt` or venv isn’t activated (`which python` should point to `.venv`). |
| `port already in use` / `Address already in use` | Change `PORT` in `.env` or kill the old process (`lsof -i :8000` → `kill -9 <pid>`). |
| `AuthenticationError` / `401` | Wrong API key for the chosen `LLM_MODEL`. Key prefix matters: `gpt-*` needs `OPENAI_API_KEY`, `claude-*` needs `ANTHROPIC_API_KEY`. |
| `web_search returned no results` | DuckDuckGo rate-limited — set `TAVILY_API_KEY` in `.env` and restart. |
| Page is blank / `404 /static/*` | Check `static/` contains `index.html`, `style.css`, `app.js` (rebuilt per `design.md`). Re-pull if missing. |
| No tokens stream, only `error` event | Read the error bubble — usually an LLM provider error. Check `LLM_MODEL` spelling and key. |

---

## Configuration — Full Reference

Everything lives in `.env` (`app/core/llm.py:15`). See `.env.example:1` for the template. Per-agent model override example in `app/agents/triage.py:44`:

```python
triage_agent = Agent(name="triage", instructions=..., tools=[...], model="gpt-4o-mini")
```

---

## How It Works

- **`app/core/agent.py:22`** — `Agent` is `{name, instructions, tools, model}`. `function_to_schema()` turns a plain Python function’s signature + docstring into an OpenAI-style tool schema.
- **`app/core/runtime.py:43`** — `run_swarm()` loop: stream an LLM completion, execute tool calls, repeat until no more tool calls. A return value `{"handoff": "researcher"}` swaps the active agent mid-run; the loop persists so one user message flows through several agents on one SSE connection. Emits `agent`/`token`/`tool_call`/`tool_result`/`done`.
- **`app/core/llm.py:34`** — `llm_stream()` LiteLLM wrapper; `stream=True`, accumulates `tool_calls` while forwarding `on_token` fragments.
- **`app/main.py:28`** — FastAPI: `GET /` serves `static/index.html`, `GET /api/agents` lists the roster, `GET /api/chat` is the SSE stream. Sessions via `app/sessions.py:27`.
- **`static/`** — zero-build UI: `index.html` is the two-pane shell, `style.css` is the warm-graphite token system (`design.md:3`), `app.js` is the `EventSource` client that renders the live rail + chat.

Full diagrams and token tables: **`design.md`** (design system) and **`ARCHITECTURE.md`** (backend architecture).

---

## Extending

**Add a tool** — write a plain function with a docstring and type hints, add it to an agent’s `tools` list:

```python
def weather(city: str) -> str:
    """Get the current weather for `city`."""
    return fetch_weather(city)

researcher_agent.tools.append(weather)
```

**Add an agent** — create `app/agents/x.py`, give it instructions, tools, and a `handoff_to_<x>` tool on the agents that should be able to reach it, then register it in `app/agents/__init__.py:10` (`REGISTRY`).

---

## Project Structure

```
Swarm Agent/
├── app/
│   ├── main.py              # FastAPI + SSE
│   ├── sessions.py          # in-mem sessions (TTL 3600s)
│   ├── core/
│   │   ├── agent.py         # Agent + schema helpers
│   │   ├── runtime.py       # run_swarm loop
│   │   └── llm.py           # LiteLLM streaming
│   ├── agents/
│   │   ├── triage.py        # Planner
│   │   ├── researcher.py    # web_search + read_url
│   │   └── writer.py        # report synthesis
│   └── tools/
│       ├── web_search.py    # Tavily or DuckDuckGo
│       └── read_url.py      # fetch + strip HTML
├── static/
│   ├── index.html           # two-pane shell
│   ├── style.css            # warm-graphite design tokens
│   └── app.js               # SSE → rail+chat renderer
├── design.md                # design system & colour plan
├── ARCHITECTURE.md          # backend architecture
├── run.py                   # uvicorn entry
├── requirements.txt
└── .env.example
```

---

## Limitations / Next Steps

- Sessions are in-memory only — swap `app/sessions.py:13` for a DB when needed.
- No parallel fan-out; handoffs are strictly sequential (`app/core/runtime.py:110`).
- DuckDuckGo search is flaky under rate limits; set `TAVILY_API_KEY` for production.
- Messages are plain text in storage; the UI streams them as `pre-wrap` — add a markdown renderer if desired.
