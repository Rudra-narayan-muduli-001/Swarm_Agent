"""Web search: Tavily if an API key is configured, otherwise DuckDuckGo
(HTML endpoint scrape, no key required). Always fails soft — an error is
returned to the model instead of crashing the run."""

from __future__ import annotations

import json
import os
import urllib.parse

import httpx
from bs4 import BeautifulSoup

DDG_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
MAX_CHARS = 15_000


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for `query` and return a JSON list of {title, url, snippet} results.

    Use this to answer questions about current events, facts, or anything that
    changes over time. Always cite the returned URLs in your answer.
    """
    max_results = max(1, min(int(max_results), 10))
    try:
        if os.getenv("TAVILY_API_KEY"):
            results = _tavily(query, max_results)
        else:
            results = _duckduckgo(query, max_results)
    except Exception as exc:
        return json.dumps({"error": f"web_search failed: {type(exc).__name__}: {exc}"}, ensure_ascii=False)

    if not results:
        return json.dumps({"error": "web_search returned no results for this query"}, ensure_ascii=False)

    payload = json.dumps(results, ensure_ascii=False)
    return payload[:MAX_CHARS]


def _duckduckgo(query: str, max_results: int) -> list[dict]:
    response = httpx.get(
        DDG_URL,
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
        follow_redirects=True,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    results: list[dict] = []
    for item in soup.select("div.result"):
        link = item.select_one("a.result__a")
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        url = link["href"]
        if url.startswith("//duckduckgo.com/l/?uddg="):
            url = urllib.parse.unquote(url.split("uddg=", 1)[1].split("&rut=", 1)[0])
        snippet_node = item.select_one("a.result__snippet") or item.select_one("div.result__snippet")
        snippet = snippet_node.get_text(strip=True) if snippet_node else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _tavily(query: str, max_results: int) -> list[dict]:
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.getenv("TAVILY_API_KEY"),
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]
