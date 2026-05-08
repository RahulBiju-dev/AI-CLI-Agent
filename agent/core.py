"""
agent/core.py — Core chat loop with tool-call interception.

Manages conversation history, sends requests to the custom Ollama model,
intercepts any tool calls, executes them, feeds results back, and
streams the final synthesized answer.
"""

import json
import sys

import ollama

from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS

# ── Configuration ─────────────────────────────────────────────────────

MODEL_NAME = "gemma-agent"

# ANSI escape helpers
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────

def _print_status(icon: str, message: str, color: str = _CYAN) -> None:
    """Print a formatted status line to stderr so it doesn't mix with piped output."""
    print(f"{color}{_BOLD}{icon}  {message}{_RESET}", file=sys.stderr)


def _process_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Execute each tool call and return the corresponding tool-role messages."""
    tool_messages: list[dict] = []

    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]

        handler = TOOL_DISPATCH.get(fn_name)
        if handler is None:
            _print_status("⚠", f"Unknown tool: {fn_name}", _RED)
            result = json.dumps({"error": f"Unknown tool '{fn_name}'"})
        else:
            _print_status("🔍", f"Searching the web: {_DIM}{fn_args.get('query', '')}{_RESET}", _YELLOW)
            result = handler(**fn_args)
            _print_status("✓", "Search complete — synthesizing answer…", _GREEN)

        tool_messages.append({"role": "tool", "content": result})

    return tool_messages


# ── Main loop ─────────────────────────────────────────────────────────

def run() -> None:
    """Run the interactive agent loop."""
    history: list[dict] = []

    print(f"\n{_CYAN}{_BOLD}╭───────────────────────────────────────╮{_RESET}")
    print(f"{_CYAN}{_BOLD}│   Gemma CLI Agent  ·  type /quit      │{_RESET}")
    print(f"{_CYAN}{_BOLD}╰───────────────────────────────────────╯{_RESET}\n")

    while True:
        # ── User input ────────────────────────────────────────────────
        try:
            user_input = input(f"{_GREEN}{_BOLD}>>> {_RESET}").strip()
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            break

        history.append({"role": "user", "content": user_input})

        # ── Initial LLM call (may return tool calls) ──────────────────
        response = ollama.chat(
            model=MODEL_NAME,
            messages=history,
            tools=TOOL_SCHEMAS,
        )

        assistant_msg = response["message"]
        history.append(assistant_msg)

        # ── Tool-call loop (iterative, in case of chained calls) ──────
        while assistant_msg.get("tool_calls"):
            tool_results = _process_tool_calls(assistant_msg["tool_calls"])
            history.extend(tool_results)

            response = ollama.chat(
                model=MODEL_NAME,
                messages=history,
                tools=TOOL_SCHEMAS,
            )
            assistant_msg = response["message"]
            history.append(assistant_msg)

        # ── Print final answer ────────────────────────────────────────
        answer = assistant_msg.get("content", "")
        if answer:
            print(f"\n{_BOLD}{answer}{_RESET}\n")
