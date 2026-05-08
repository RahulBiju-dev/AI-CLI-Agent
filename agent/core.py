"""
agent/core.py — Core chat loop with tool-call interception.

Manages conversation history, sends requests to the custom Ollama model,
intercepts any tool calls, executes them, feeds results back, and
streams the final synthesized answer with visible thinking status.
"""

import json
import sys
import threading
import time
import itertools

import ollama

from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS

# ── Configuration ─────────────────────────────────────────────────────

MODEL_NAME = "gemma-agent"

# ANSI escape helpers
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"
_CLEAR_LINE = "\033[2K\r"


# ── Helpers ───────────────────────────────────────────────────────────

def _print_status(icon: str, message: str, color: str = _CYAN) -> None:
    """Print a formatted status line to stderr so it doesn't mix with piped output."""
    print(f"{color}{_BOLD}{icon}  {message}{_RESET}", file=sys.stderr)


class _Spinner:
    """Animated spinner that shows a message while the model is working."""

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Thinking", color: str = _MAGENTA) -> None:
        self._message = message
        self._color = color
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _animate(self) -> None:
        for frame in itertools.cycle(self._FRAMES):
            if self._stop_event.is_set():
                break
            print(
                f"{_CLEAR_LINE}{self._color}{_BOLD}{frame}  {self._message}…{_RESET}",
                end="",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.08)
        # Clear the spinner line when done
        print(_CLEAR_LINE, end="", file=sys.stderr, flush=True)

    def start(self) -> "_Spinner":
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def update(self, message: str) -> None:
        """Update the spinner message in-place."""
        self._message = message

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()


def _stream_thinking_response(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Stream a response, showing thinking progress and the final answer.

    Returns the full assistant message dict (with thinking + content)
    for appending to history.
    """
    spinner = _Spinner("Thinking").start()

    thinking_buf = ""
    content_buf = ""
    in_thinking = False
    thinking_displayed = False

    stream = ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        think=True,
    )

    for chunk in stream:
        msg = chunk.message

        # ── Tool calls come through as non-streamed chunks ────────
        if msg.tool_calls:
            spinner.stop()
            # Build the assistant message with any accumulated content
            assistant_msg = {"role": "assistant", "content": content_buf}
            if thinking_buf:
                assistant_msg["thinking"] = thinking_buf
            assistant_msg["tool_calls"] = [
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
            return assistant_msg

        # ── Thinking tokens ───────────────────────────────────────
        thinking_chunk = getattr(msg, "thinking", None) or ""
        if thinking_chunk:
            if not in_thinking:
                in_thinking = True
                spinner.stop()
                # Print thinking header
                print(
                    f"\n{_MAGENTA}{_DIM}┌─ thinking ─────────────────────────────{_RESET}",
                    file=sys.stderr,
                )
                thinking_displayed = True

            thinking_buf += thinking_chunk
            # Print thinking content in dim magenta
            print(
                f"{_MAGENTA}{_DIM}{thinking_chunk}{_RESET}",
                end="",
                file=sys.stderr,
                flush=True,
            )
            continue

        # ── Content tokens ────────────────────────────────────────
        content_chunk = msg.content or ""
        if content_chunk:
            if in_thinking:
                # Transition from thinking to answering
                in_thinking = False
                print(
                    f"\n{_MAGENTA}{_DIM}└────────────────────────────────────────{_RESET}\n",
                    file=sys.stderr,
                )
                spinner.stop()
            elif spinner._thread and not spinner._stop_event.is_set():
                spinner.stop()
                if not thinking_displayed:
                    print()  # newline before answer

            content_buf += content_chunk
            print(f"{_BOLD}{content_chunk}{_RESET}", end="", flush=True)

    # End of stream
    spinner.stop()

    if in_thinking:
        # Stream ended while still in thinking (no content followed)
        print(
            f"\n{_MAGENTA}{_DIM}└────────────────────────────────────────{_RESET}\n",
            file=sys.stderr,
        )

    if content_buf:
        print("\n")  # final newlines after streamed answer

    # Build the full message for history
    assistant_msg = {"role": "assistant", "content": content_buf}
    if thinking_buf:
        assistant_msg["thinking"] = thinking_buf
    return assistant_msg


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


# ── Slash commands ────────────────────────────────────────────────────

_COMMANDS_HELP = f"""
{_CYAN}{_BOLD}Available commands:{_RESET}
  {_GREEN}/help{_RESET}    — Show this help message
  {_GREEN}/clear{_RESET}   — Clear conversation history
  {_GREEN}/model{_RESET}   — Show current model info
  {_GREEN}/quit{_RESET}    — Exit the agent  {_DIM}(also /exit, /q){_RESET}
"""


def _handle_command(cmd: str, history: list[dict]) -> bool | None:
    """Handle a slash command. Returns True if handled, None to quit."""
    cmd_lower = cmd.lower().strip()

    if cmd_lower in ("/quit", "/exit", "/q"):
        return None  # Signal to quit

    if cmd_lower == "/help":
        print(_COMMANDS_HELP)
        return True

    if cmd_lower == "/clear":
        history.clear()
        print(f"{_CYAN}{_BOLD}✓  Conversation history cleared.{_RESET}\n")
        return True

    if cmd_lower == "/model":
        try:
            info = ollama.show(MODEL_NAME)
            model_info = getattr(info, "modelinfo", None) or {}
            family = model_info.get("general.architecture", "unknown")
            params = model_info.get("general.parameter_count", "unknown")
            print(f"\n{_CYAN}{_BOLD}Model:{_RESET}  {MODEL_NAME}")
            print(f"{_CYAN}{_BOLD}Family:{_RESET} {family}")
            print(f"{_CYAN}{_BOLD}Params:{_RESET} {params}\n")
        except Exception:
            print(f"\n{_CYAN}{_BOLD}Model:{_RESET}  {MODEL_NAME}\n")
        return True

    # Unknown command
    print(f"{_RED}Unknown command: {cmd}{_RESET}  {_DIM}(type /help for available commands){_RESET}\n")
    return True


# ── Main loop ─────────────────────────────────────────────────────────

def run() -> None:
    """Run the interactive agent loop."""
    history: list[dict] = []

    print(f"\n{_CYAN}{_BOLD}╭───────────────────────────────────────╮{_RESET}")
    print(f"{_CYAN}{_BOLD}│   Gemma CLI Agent  ·  type /help      │{_RESET}")
    print(f"{_CYAN}{_BOLD}╰───────────────────────────────────────╯{_RESET}\n")

    while True:
        # ── User input ────────────────────────────────────────────────
        try:
            user_input = input(f"{_GREEN}{_BOLD}>>> {_RESET}").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # ── Handle slash commands ─────────────────────────────────────
        if user_input.startswith("/"):
            result = _handle_command(user_input, history)
            if result is None:
                break  # /quit
            continue  # Command was handled, skip LLM call

        history.append({"role": "user", "content": user_input})

        # ── LLM call with streaming + thinking ────────────────────────
        assistant_msg = _stream_thinking_response(
            model=MODEL_NAME,
            messages=history,
            tools=TOOL_SCHEMAS,
        )
        history.append(assistant_msg)

        # ── Tool-call loop (iterative, in case of chained calls) ──────
        while assistant_msg.get("tool_calls"):
            tool_results = _process_tool_calls(assistant_msg["tool_calls"])
            history.extend(tool_results)

            # Follow-up call after tool results — also streamed
            assistant_msg = _stream_thinking_response(
                model=MODEL_NAME,
                messages=history,
                tools=TOOL_SCHEMAS,
            )
            history.append(assistant_msg)
