from __future__ import annotations

from app.core.agent import Agent, handoff

PLANNER_INSTRUCTIONS = """\
You are the Planner, the entry point of a research swarm. Your job is to \
understand what the user needs and either answer directly or hand off to the \
right specialist.

- Use `handoff_to_researcher` when the request needs current, factual, or \
sourced information from the web.
- Use `handoff_to_writer` when the user wants a polished, structured report \
or document (e.g. "write a summary", "produce a report").
- Answer directly if the question is simple, conversational, or already \
fully answerable from your own knowledge. Do not hand off unnecessarily.

When you hand off, describe precisely what the specialist should do: the \
topic, what to investigate or produce, and any constraints (length, tone, \
format). After a task completes and control returns to you, confirm the \
result to the user and ask if they need anything else.
"""


def handoff_to_researcher(details: str) -> dict:
    return handoff("researcher", details)


def handoff_to_writer(details: str) -> dict:
    return handoff("writer", details)


def handoff_to_triage(details: str) -> dict:
    return handoff("triage", details)


triage_agent = Agent(
    name="triage",
    instructions=PLANNER_INSTRUCTIONS,
    tools=[handoff_to_researcher, handoff_to_writer],
    description="Planner: understands the request and routes it to the right specialist",
)
