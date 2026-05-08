"""
tools/search.py — DuckDuckGo web search implementation.

Returns a condensed JSON string of the top 3 results to keep the
LLM's context window lean.
"""

import json
from duckduckgo_search import DDGS


def web_search(query: str) -> str:
    """Execute a DuckDuckGo search and return the top 3 results as compact JSON.

    Args:
        query: The search query string.

    Returns:
        A JSON string containing a list of {title, snippet} dicts,
        or an error payload if the search fails.
    """
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=3))

        condensed = [
            {"title": r.get("title", ""), "snippet": r.get("body", "")}
            for r in raw_results
        ]

        return json.dumps(condensed, separators=(",", ":"))

    except Exception as exc:
        return json.dumps({"error": str(exc)}, separators=(",", ":"))
