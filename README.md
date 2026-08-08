# Swarm Agent

A multi-agent "swarm" in plain Python with a simple browser UI. Three agents
— **Planner**, **Researcher**, **Writer** — pass control to each other via
handoffs (an OpenAI-Swarm-style pattern), all implemented from scratch on top
of LiteLLM (provider-agnostic: OpenAI, Anthropic, Gemini, local Ollama, ...).

```
Planner (triage) ──handoff──▶ Researcher ──handoff──▶ Writer ──handoff──▶ Planner
                                │ tools: web_search, read_url
```

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt     # Windows
# source .venv/bin/pip install -r requirements.txt  # macOS/Linux

cp .env.example .env        # then set LLM_MODEL + your API key
python run.py
```

Open http://127.0.0.1:8000 and ask something research-y, e.g.
*"What are the latest developments in solid-state batteries? Write me a summary."*

## Configuration

Everything lives in `.env`:

| Variable            | Purpose                                                  |
|---------------------|----------------------------------------------------------|
| `LLM_MODEL`         | Model string; LiteLLM picks the provider by prefix (`gpt-4o-mini`, `claude-...`, `gemini/...`, `ollama/llama3.1`) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Key for your chosen provider |
| `OLLAMA_API_BASE`   | Only for local models                                    |
| `TAVILY_API_KEY`    | Optional: better web search than the built-in DuckDuckGo |
| `PORT`              | Server port (default 8000)                               |

Each agent can override the model with its own `model` field (see
`app/agents/*.py`).

## How it works

- **`app/core/agent.py`** — an `Agent` is just `{name, instructions, tools, model}`.
  A tool is a plain Python function; its signature + docstring are converted
  into an OpenAI-style function schema.
- **`app/core/runtime.py`** — the run loop: stream an LLM completion, execute
  any tool calls, append results, repeat until the agent answers without tool
  calls. A tool returning `{"handoff": "researcher"}` switches the active
  agent mid-run, so one user message can flow through several agents over a
  single connection. Token chunks, tool calls, and handoffs are pushed to the
  UI as events (`token`, `tool_call`, `tool_result`, `agent`, `done`).
- **`app/core/llm.py`** — thin LiteLLM wrapper; streams and collects tool
  calls from any provider.
- **`app/main.py`** — FastAPI server. `GET /api/chat` is a Server-Sent
  Events stream; `GET /api/agents` lists the roster. Sessions are held
  in memory (`app/sessions.py`).
- **`static/`** — zero-build chat UI: streaming message bubbles, color-coded
  agent handoff pills, and collapsible tool-call chips.

## Extending

**Add a tool** — write a plain function with a docstring and type hints, add
it to an agent's `tools` list:

```python
def weather(city: str) -> str:
    """Get the current weather for `city`."""
    return fetch_weather(city)

researcher_agent.tools.append(weather)
```

**Add an agent** — create `app/agents/x.py`, give it instructions, tools, and
a `handoff_to_<x>` tool on the agents that should be able to reach it, then
register it in `app/agents/__init__.py`.

## Limitations / next steps

- Sessions are in-memory only (lost on restart) — swap `app/sessions.py`
  for a DB when needed.
- No parallel fan-out yet; handoffs are strictly sequential.
- DuckDuckGo search is free but flaky under rate limits; set `TAVILY_API_KEY`
  for production-quality research.
- Messages are plain text; add a markdown renderer to the UI if desired.
