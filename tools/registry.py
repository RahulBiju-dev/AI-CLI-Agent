"""
tools/registry.py — Tool JSON schemas for the Ollama API.

Each entry follows the Ollama tool-calling format (OpenAI-compatible
function schema). Add new tools here as the agent grows.
"""

from tools.search import web_search
from tools.document import read_document
from tools.file import read_file, create_file

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
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Extract and read text from a PDF or Word document (.docx). "
                "Use this tool when the user asks you to read, summarize, or analyze "
                "a local document file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the PDF or Word document file.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a text file from the local filesystem. "
                "Use this tool when the user asks you to view, read, or analyze "
                "code or text files on their computer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to read.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": (
                "Create a new .txt or .md file with the provided content at a specified directory. "
                "Use this tool ONLY when explicitly told to make or write a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path where the file should be created. Must end with .txt or .md.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text or markdown content to write into the file.",
                    }
                },
                "required": ["file_path", "content"],
            },
        },
    }
]

# ── Dispatch map ──────────────────────────────────────────────────────
# Maps function names to their Python callables.

TOOL_DISPATCH: dict[str, callable] = {
    "web_search": web_search,
    "read_document": read_document,
    "read_file": read_file,
    "create_file": create_file,
}
