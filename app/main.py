"""Swarm Agent — FastAPI server.

Serves the browser UI from /static and exposes two endpoints:
- GET /api/agents   -> JSON list of agents
- GET /api/chat     -> SSE stream of one swarm run (tokens, tool calls,
                       agent handoffs). GET keeps the client-side
                       EventSource API usable (no fetch streaming needed).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agents import REGISTRY, get_agent
from app.core.runtime import run_swarm
from app.sessions import get_session, touch

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Swarm Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/agents")
async def list_agents() -> dict:
    return {
        "agents": [
            {"name": agent.name, "description": agent.description}
            for agent in REGISTRY.values()
        ]
    }


@app.get("/api/chat")
async def chat(session_id: str | None = None, message: str = "", agent: str | None = None):
    message = message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    start_agent = get_agent(agent) if agent else None
    if agent and start_agent is None:
        raise HTTPException(status_code=404, detail=f"unknown agent '{agent}'")

    session = get_session(session_id)
    session.messages.append({"role": "user", "content": message})

    return StreamingResponse(
        _sse_stream(session, start_agent or next(iter(REGISTRY.values()))),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_stream(session, start_agent):
    """Bridge the synchronous run loop (worker thread) to the async response."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event: str, data: dict) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))
        except RuntimeError:  # loop already closed (client disconnected)
            pass

    task = asyncio.create_task(
        asyncio.to_thread(run_swarm, session, start_agent, REGISTRY, on_event)
    )

    def fmt(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        while True:
            try:
                event, data = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield fmt(event, data)
            except asyncio.TimeoutError:
                if task.done():
                    break
    finally:
        if not task.done():
            task.cancel()
        touch(session)
