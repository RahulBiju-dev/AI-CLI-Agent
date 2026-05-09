"""
tools/registry.py — Tool JSON schemas for the Ollama API.

Each entry follows the Ollama tool-calling format (OpenAI-compatible
function schema). Add new tools here as the agent grows.
"""

from tools.search import web_search
from tools.document import read_document
from tools.file import read_file, create_file
from tools.spotify import spotify_play

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
                "Create a new file with the provided content at a specified directory. "
                "Use this tool ONLY when explicitly told to make or write a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path where the file should be created.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write into the file.",
                    }
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play",
            "description": (
                "Open Spotify and play a specific song, album, playlist, or artist. "
                "Use this tool when the user asks you to play music on Spotify. "
                "You can provide a Spotify URI, a Spotify URL, or a natural language "
                "search query like 'Bohemian Rhapsody by Queen'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A Spotify URI (spotify:track:...), a Spotify URL "
                            "(https://open.spotify.com/...), or a search query "
                            "describing the song, album, playlist, or artist to play."
                        ),
                    },
                    "content_type": {
                        "type": "string",
                        "description": (
                            "The type of content to search for. "
                            "One of: track, album, playlist, artist. Defaults to track."
                        ),
                        "enum": ["track", "album", "playlist", "artist"],
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
    "read_document": read_document,
    "read_file": read_file,
    "create_file": create_file,
    "spotify_play": spotify_play,
}
