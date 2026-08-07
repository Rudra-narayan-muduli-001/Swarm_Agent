"""The three research agents. Handoffs are tools that return
``{"handoff": "<agent name>"}`` — the runtime switches agents on that."""

from __future__ import annotations

from app.agents.researcher import researcher_agent
from app.agents.triage import triage_agent
from app.agents.writer import writer_agent

ALL_AGENTS = [triage_agent, researcher_agent, writer_agent]
REGISTRY = {agent.name: agent for agent in ALL_AGENTS}


def get_agent(name: str) -> "Agent | None":
    return REGISTRY.get(name)
