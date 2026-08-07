"""Researcher agent — gathers facts from the web."""

from __future__ import annotations

from app.agents.triage import handoff_to_triage
from app.core.agent import Agent, handoff
from app.tools.read_url import read_url
from app.tools.web_search import web_search

RESEARCHER_INSTRUCTIONS = """\
You are the Researcher. Your job is to gather accurate, up-to-date \
information on the assigned topic using `web_search` and `read_url`.

Guidelines:
- Run 1-3 targeted searches; only dig deeper when the first results are thin.
- Read the most promising pages with `read_url` to get real content, not just snippets.
- Note the source URL for every fact you plan to use.
- Never invent facts, figures, or URLs.

When you have enough material, hand off to the Writer
(`handoff_to_writer`) with a compact but complete summary of your findings:
key facts, conflicting points, and the source URLs. If you need clarification
or a different direction, hand off to the Planner (`handoff_to_triage`).
"""


def handoff_to_writer(details: str) -> dict:
    """Hand off to the Writer agent to turn the research into a polished report. \
Include your research summary, source URLs, and any format requirements."""
    return handoff("writer", details)


researcher_agent = Agent(
    name="researcher",
    instructions=RESEARCHER_INSTRUCTIONS,
    tools=[web_search, read_url, handoff_to_writer, handoff_to_triage],
    description="Researcher: gathers facts from the web with sources",
)
