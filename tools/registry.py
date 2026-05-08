"""
tools/registry.py — Tool JSON schemas for the Ollama API.

Each entry follows the Ollama tool-calling format (OpenAI-compatible
function schema). Add new tools here as the agent grows.
"""

from tools.search import web_search

# ── Schema definitions ────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Use this tool when you need "
                "up-to-date information, current events, package versions, "
                "documentation, or anything beyond your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# ── Dispatch map ──────────────────────────────────────────────────────
# Maps function names to their Python callables.

TOOL_DISPATCH: dict[str, callable] = {
    "web_search": web_search,
}
