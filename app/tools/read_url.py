"""Fetch a web page and strip it down to plain text for LLM context."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "form", "aside")


def read_url(url: str, max_chars: int = 8000) -> str:
    """Fetch the page at `url` and return its readable text content.

    Use this after web_search to read the actual content of promising pages.
    The text is truncated to `max_chars` characters.
    """
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
            follow_redirects=True,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(STRIP_TAGS):
            tag.decompose()
        text = " ".join(soup.get_text().split())
        if not text:
            return "No readable text content found at this URL."
        return text[: int(max_chars)]
    except Exception as exc:
        return f"Error reading {url}: {type(exc).__name__}: {exc}"
