"""Planner / triage agent — the entry point of the swarm."""

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
    """Hand off to the Researcher agent, who will gather information from the web. \
Provide the full research task: topic, specific questions to answer, and any constraints."""
    return handoff("researcher", details)


def handoff_to_writer(details: str) -> dict:
    """Hand off to the Writer agent, who will produce a polished, structured report. \
Provide the topic, the material to base the report on, and any format requirements."""
    return handoff("writer", details)


def handoff_to_triage(details: str) -> dict:
    """Hand off back to the Planner agent. Use this when your task is complete, \
you need clarification, or the next step is unclear. Include a short note describing what happened."""
    return handoff("triage", details)


triage_agent = Agent(
    name="triage",
    instructions=PLANNER_INSTRUCTIONS,
    tools=[handoff_to_researcher, handoff_to_writer],
    description="Planner: understands the request and routes it to the right specialist",
)
