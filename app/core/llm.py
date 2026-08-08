"""LiteLLM-backed streaming chat. Provider is chosen purely by the model
string (e.g. ``gpt-4o-mini`` vs ``ollama/llama3.1``) plus the matching API
key in the environment."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

from litellm import completion  # noqa: E402

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


def resolve_model(agent) -> str:
    return agent.model or DEFAULT_MODEL


def llm_stream(agent, messages: list[dict], tools: list[dict], on_token: Callable[[str], None]):
    """Stream one completion. Returns (content, [ToolCall]).

    ``on_token`` is invoked for every text fragment as it arrives, so the UI
    can stream tokens live while tool calls are collected in the background.
    """
    kwargs = {
        "model": resolve_model(agent),
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    model = kwargs["model"]
    if model.startswith("groq/") and os.getenv("GROQ_API_BASE"):
        kwargs["api_base"] = os.getenv("GROQ_API_BASE")

    response = completion(**kwargs)

    content = ""
    calls: dict[int, dict] = {}
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if delta.content:
            content += delta.content
            on_token(delta.content)
        if delta.tool_calls:
            for call in delta.tool_calls:
                index = call.index or 0
                entry = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    entry["id"] = call.id
                if call.function:
                    if call.function.name:
                        entry["name"] += call.function.name
                    if call.function.arguments:
                        entry["arguments"] += call.function.arguments

    tool_calls: list[ToolCall] = []
    for entry in calls.values():
        if not entry["id"]:
            entry["id"] = f"call_{uuid.uuid4().hex[:12]}"
        try:
            arguments = json.loads(entry["arguments"] or "{}")
            if not isinstance(arguments, dict):
                arguments = {"_raw": entry["arguments"]}
        except json.JSONDecodeError:
            arguments = {"_raw": entry["arguments"]}
        tool_calls.append(ToolCall(id=entry["id"], name=entry["name"], arguments=arguments))

    return content, tool_calls
