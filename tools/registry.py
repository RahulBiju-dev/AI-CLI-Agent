"""
tools/registry.py — Tool JSON schemas for the Ollama API.

Each entry follows the Ollama tool-calling format (OpenAI-compatible
function schema). Add new tools here as the agent grows.
"""

from tools.search import web_search
from tools.document import read_document
from tools.file import read_file, create_file
from tools.code import view_code
from tools.spotify import spotify_play
from tools.browser import open_browser
from tools.vault_indexer import index_vault
from tools.vault_search import search_vault

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
                "Extract and read text from a PDF or Word document (.docx) with page, chunk, and query controls. "
                "Use this tool when the user asks you to read, summarize, or analyze "
                "a local document file. For large documents, first call it with only file_path for a preview, "
                "then use pages, query, or chunk to retrieve the exact relevant parts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the PDF or Word document file.",
                    },
                    "pages": {
                        "type": "string",
                        "description": "Optional PDF page selection using 1-based pages/ranges, e.g. '1-3,8'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query to return relevant snippets instead of a full text preview.",
                    },
                    "chunk": {
                        "type": "integer",
                        "description": "Optional 0-based chunk number for large extracted document text.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Optional approximate characters per chunk. Defaults to 12000.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional maximum characters returned in text fields. Defaults to 14000.",
                    },
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
                "Read the contents of a text file from the local filesystem with line range, chunk, and query controls. "
                "Use this tool when the user asks you to view, read, or analyze "
                "text files on their computer. For large files, use query to find relevant lines or lines/chunk to read a bounded section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to read.",
                    },
                    "lines": {
                        "type": "string",
                        "description": "Optional line range to read, e.g. '20-80' or '42'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional text search query to return matching snippets and line numbers.",
                    },
                    "chunk": {
                        "type": "integer",
                        "description": "Optional 0-based chunk number for large files.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Optional approximate characters per chunk. Defaults to 12000.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional maximum characters returned in text fields. Defaults to 14000.",
                    },
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
                "Open Spotify and play a specific song. "
                "Use this tool when the user asks you to play a song on Spotify. "
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
                            "(https://open.spotify.com/track/...), or a search query "
                            "describing the song to play (e.g. 'Bohemian Rhapsody Queen')."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser",
            "description": (
                "Open the user's default web browser to a specific website or search query. "
                "Use this tool when the user asks to open a website, search for something in the browser, "
                "or watch a video."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The URL to open (e.g. 'youtube.com') or a search term (e.g. 'cute cats').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_code",
            "description": (
                "View and read source code files in all common programming languages (Python, JavaScript, C++, Rust, Go, Java, C#, Ruby, PHP, Swift, Kotlin, R, Scala, Lisp, Haskell, Erlang, Elixir, Julia, Perl, Lua, Shell, PowerShell, Groovy, Dart, Fortran, Pascal, Assembly, and many more). "
                "Displays code with line numbers and supports viewing specific line ranges. "
                "Can also scan folders for files with a specific extension. "
                "Use this tool when the user asks to view, read, analyze, or answer questions about source code files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to a source code file or folder.",
                    },
                    "lines": {
                        "type": "string",
                        "description": "Optional line range to view for a single file (e.g., '1-50' or '10-20'). If not provided, shows entire file.",
                    },
                    "extension": {
                        "type": "string",
                        "description": "When file_path is a folder, scan for files with this extension (e.g., '.py', '.js', '.cpp'). Without this, a folder path will return an error.",
                    }
                },
                "required": ["file_path"],
            },
        },
    }
]

# Add RAG tooling: index the vault and search it
TOOL_SCHEMAS.extend([
    {
        "type": "function",
        "function": {
            "name": "index_vault",
            "description": "Index a folder or a single local file into the persistent ChromaDB vault using Ollama embeddings. Use this before vault_search for large files or document collections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "ChromaDB collection name (optional)."},
                    "file_path": {"type": "string", "description": "Optional: specific file to index."},
                    "chunk_size": {"type": "integer", "description": "Optional chunk size for indexing. Defaults to 1800 characters."},
                    "chunk_overlap": {"type": "integer", "description": "Optional overlap between chunks. Defaults to 250 characters."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vault_search",
            "description": "Search the indexed vault for relevant chunks and return compact snippets, source paths, chunk indexes, and character offsets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection": {"type": "string"},
                    "top_k": {"type": "integer", "description": "Optional number of chunks to return. Defaults to 6."},
                    "max_chars": {"type": "integer", "description": "Optional maximum characters in the combined context. Defaults to 7000."},
                    "source": {"type": "string", "description": "Optional source or source_path value from a previous search result to restrict search."}
                },
                "required": ["query"]
            }
        }
    },
])

# ── Dispatch map ──────────────────────────────────────────────────────
# Maps function names to their Python callables.

TOOL_DISPATCH: dict[str, callable] = {
    "web_search": web_search,
    "read_document": read_document,
    "read_file": read_file,
    "create_file": create_file,
    "view_code": view_code,
    "spotify_play": spotify_play,
    "open_browser": open_browser,
}

# Dispatch RAG tools
TOOL_DISPATCH.update({
    "index_vault": index_vault,
    "vault_search": search_vault,
})
