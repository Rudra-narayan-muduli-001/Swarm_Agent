"""Agent definition and tool-schema helpers.

An agent is just a name, a system prompt, an optional model override, and a
list of Python functions it may call. Handoffs are ordinary tools whose
return value is a dict like ``{"handoff": "agent_name"}`` — the runtime
detects that and switches the active agent.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

Tool = Callable[..., Any]

HANDOFF_KEY = "handoff"


@dataclass
class Agent:
    name: str
    instructions: str
    tools: list[Tool] = field(default_factory=list)
    model: str | None = None
    description: str = ""


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def function_to_schema(func: Tool) -> dict:
    """Build an OpenAI-style function schema from a Python function.

    Relies on type hints and the docstring. Parameters without a recognized
    annotation default to ``string``.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = inspect.signature(func.__call__)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = {"type": _TYPE_MAP.get(param.annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def is_handoff(value: Any) -> bool:
    return isinstance(value, dict) and HANDOFF_KEY in value


def handoff(target: str, context: str | None = None) -> dict:
    """Return value for handoff tools."""
    payload = {HANDOFF_KEY: target}
    if context:
        payload["context"] = context
    return payload


def to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
