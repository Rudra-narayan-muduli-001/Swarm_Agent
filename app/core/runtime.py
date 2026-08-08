"""The swarm run loop.

An OpenAI-Swarm-style loop: call the active agent's LLM, execute any tool
calls it makes, and keep going until it responds without tool calls. A tool
result of ``{"handoff": "<agent_name>"}`` swaps the active agent mid-run —
the loop persists across handoffs so a single user request can flow through
several agents, all streamed over one connection.

Events pushed via ``on_event(event, data)`` (thread-safe; the UI layer
forwards them to an async queue):

- ``agent``      {"name": ...}          the active agent changed
- ``token``      {"content": ...}       streamed text fragment
- ``tool_call``  {"tool": ..., "arguments": {...}, "agent": ...}
- ``tool_result`` {"tool": ..., "result": ...}   truncated for display
- ``error``      {"message": ...}
- ``done``       {}                      always the last event
"""

from __future__ import annotations

import json
from typing import Callable

from app.core.agent import Agent, function_to_schema, is_handoff, to_text
from app.core.llm import ToolCall, llm_stream

MAX_HISTORY = 40
MAX_TOOL_RESULT_CHARS = 12_000


def _tools_by_name(agent: Agent) -> dict[str, Callable]:
    return {tool.__name__: tool for tool in agent.tools}


def _execute(fn: Callable, args: dict) -> object:
    try:
        return fn(**args)
    except Exception as exc:  # surface tool failures to the model
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_swarm(session, start_agent: Agent, registry: dict[str, Agent], on_event: Callable) -> None:
    """Run the loop for one user message. Mutates ``session.messages``."""
    messages = session.messages
    agent = start_agent
    try:
        on_event("agent", {"name": agent.name})
        while True:
            history = [{"role": "system", "content": agent.instructions}] + messages[-MAX_HISTORY:]
            schemas = [function_to_schema(tool) for tool in agent.tools]
            content, calls = llm_stream(
                agent,
                history,
                schemas,
                on_token=lambda frag: on_event("token", {"content": frag}),
            )

            if not calls:
                if content:
                    messages.append({"role": "assistant", "content": content})
                return

            caller = agent
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in calls
                    ],
                }
            )

            next_agent: Agent | None = None
            for call in calls:
                fn = _tools_by_name(caller).get(call.name)
                if fn is None:
                    payload = {"error": f"unknown tool '{call.name}'"}
                else:
                    on_event("tool_call", {"tool": call.name, "arguments": call.arguments, "agent": caller.name})
                    payload = _execute(fn, call.arguments)

                if is_handoff(payload):
                    target = registry.get(payload["handoff"])
                    if target is None:
                        payload = {"error": f"unknown agent '{payload['handoff']}'"}
                    else:
                        messages.append(
                            {"role": "tool", "tool_call_id": call.id, "content": to_text(payload)}
                        )
                        on_event("tool_result", {"tool": call.name, "result": f"handoff to {target.name}"})
                        next_agent = target
                        continue

                content_str = to_text(payload)
                if len(content_str) > MAX_TOOL_RESULT_CHARS:
                    content_str = content_str[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content_str})
                on_event("tool_result", {"tool": call.name, "result": content_str[:500]})

            if next_agent is not None:
                agent = next_agent
                on_event("agent", {"name": agent.name})
    except Exception as exc:
        on_event("error", {"message": f"{type(exc).__name__}: {exc}"})
    finally:
        on_event("done", {})
