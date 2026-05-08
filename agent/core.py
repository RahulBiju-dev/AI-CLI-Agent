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
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS

# ── Configuration ─────────────────────────────────────────────────────

MODEL_NAME = "gemma-agent"

# Parameters that accept float values via /set parameter
_FLOAT_PARAMS = {"temperature", "top_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty", "min_p", "tfs_z"}
# Parameters that accept integer values via /set parameter
_INT_PARAMS = {"num_ctx", "num_predict", "repeat_last_n", "seed", "num_gpu", "num_thread", "num_keep"}
_ALL_PARAMS = _FLOAT_PARAMS | _INT_PARAMS

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

# Initialize Rich console
_console = Console()


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
    options: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Stream a response, showing thinking progress and the final answer.

    Returns the full assistant message dict (with thinking + content)
    for appending to history.
    """
    spinner = _Spinner("Thinking").start()
    t_start = time.monotonic()

    thinking_buf = ""
    content_buf = ""
    in_thinking = False
    thinking_displayed = False

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": True,
    }
    if tools:
        kwargs["tools"] = tools
    if options:
        kwargs["options"] = options

    stream = ollama.chat(**kwargs)

    live = None
    try:
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

                # Initialize Live display on the first content chunk
                if live is None:
                    live = Live(
                        Markdown(content_buf),
                        console=_console,
                        auto_refresh=False,
                        vertical_overflow="visible",
                    )
                    live.start()

                # Update Markdown rendering in real-time
                live.update(Markdown(content_buf), refresh=True)

    finally:
        if live:
            live.stop()

    # End of stream
    spinner.stop()

    if in_thinking:
        # Stream ended while still in thinking (no content followed)
        print(
            f"\n{_MAGENTA}{_DIM}└────────────────────────────────────────{_RESET}\n",
            file=sys.stderr,
        )

    if content_buf:
        print()  # final newline after streamed answer

    # Verbose stats
    if verbose:
        elapsed = time.monotonic() - t_start
        t_tokens = len(thinking_buf.split()) if thinking_buf else 0
        c_tokens = len(content_buf.split()) if content_buf else 0
        total = t_tokens + c_tokens
        tps = total / elapsed if elapsed > 0 else 0
        print(
            f"{_DIM}  ⏱  {elapsed:.1f}s  ·  ~{total} tokens  ·  ~{tps:.1f} tok/s{_RESET}\n",
            file=sys.stderr,
        )

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
            if fn_name == "web_search":
                _print_status("🔍", f"Searching the web: {_DIM}{fn_args.get('query', '')}{_RESET}", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "Search complete — synthesizing answer…", _GREEN)
            elif fn_name == "read_document":
                _print_status("📄", f"Reading document: {_DIM}{fn_args.get('file_path', '')}{_RESET}", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "Document read — synthesizing answer…", _GREEN)
            elif fn_name == "read_file":
                _print_status("📂", f"Reading file: {_DIM}{fn_args.get('file_path', '')}{_RESET}", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "File read — synthesizing answer…", _GREEN)
            else:
                _print_status("⚙️", f"Executing {fn_name}…", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "Tool execution complete — synthesizing answer…", _GREEN)

        tool_messages.append({"role": "tool", "content": result})

    return tool_messages


# ── Slash commands ────────────────────────────────────────────────────

_COMMANDS_HELP = f"""
{_CYAN}{_BOLD}Available commands:{_RESET}
  {_GREEN}/help{_RESET}                          — Show this help message
  {_GREEN}/clear{_RESET}                         — Clear conversation history
  {_GREEN}/set parameter <name> <val>{_RESET}    — Set a model parameter  {_DIM}(e.g. temperature 0.7){_RESET}
  {_GREEN}/set system "<prompt>"{_RESET}         — Set the system prompt for this session
  {_GREEN}/set verbose{_RESET}                   — Show generation stats after each response
  {_GREEN}/set quiet{_RESET}                     — Hide generation stats  {_DIM}(default){_RESET}
  {_GREEN}/set wordwrap{_RESET}                  — Enable word wrapping  {_DIM}(default){_RESET}
  {_GREEN}/set nowordwrap{_RESET}                — Disable word wrapping
  {_GREEN}/show parameters{_RESET}               — Show current session parameters
  {_GREEN}/show system{_RESET}                   — Show the active system prompt
  {_GREEN}/show model{_RESET}                    — Show model info
  {_GREEN}/quit{_RESET}                          — Exit the agent  {_DIM}(also /exit, /q){_RESET}
"""


def _handle_set(args: str, session: dict, history: list[dict]) -> None:
    """Handle /set sub-commands."""
    parts = args.strip().split(None, 1)
    if not parts:
        print(f"{_RED}Usage: /set <subcommand> [args]{_RESET}  {_DIM}(type /help for details){_RESET}\n")
        return

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # ── /set verbose / /set quiet ─────────────────────────────────────
    if sub == "verbose":
        session["verbose"] = True
        print(f"{_CYAN}{_BOLD}✓  Verbose mode enabled — stats shown after each response.{_RESET}\n")
        return
    if sub == "quiet":
        session["verbose"] = False
        print(f"{_CYAN}{_BOLD}✓  Quiet mode enabled.{_RESET}\n")
        return

    # ── /set wordwrap / /set nowordwrap ───────────────────────────────
    if sub == "wordwrap":
        session["wordwrap"] = True
        print(f"{_CYAN}{_BOLD}✓  Word wrapping enabled.{_RESET}\n")
        return
    if sub == "nowordwrap":
        session["wordwrap"] = False
        print(f"{_CYAN}{_BOLD}✓  Word wrapping disabled.{_RESET}\n")
        return

    # ── /set system "<prompt>" ────────────────────────────────────────
    if sub == "system":
        # Strip surrounding quotes if present
        prompt = rest.strip().strip('"').strip("'")
        if not prompt:
            print(f"{_RED}Usage: /set system \"Your system prompt here\"{_RESET}\n")
            return
        # Update or insert system message at the start of history
        if history and history[0].get("role") == "system":
            history[0]["content"] = prompt
        else:
            history.insert(0, {"role": "system", "content": prompt})
        session["system"] = prompt
        # Truncate display for confirmation
        display = prompt if len(prompt) <= 80 else prompt[:77] + "…"
        print(f"{_CYAN}{_BOLD}✓  System prompt set:{_RESET} {_DIM}{display}{_RESET}\n")
        return

    # ── /set parameter <name> <value> ─────────────────────────────────
    if sub == "parameter":
        param_parts = rest.strip().split(None, 1)
        if len(param_parts) != 2:
            print(f"{_RED}Usage: /set parameter <name> <value>{_RESET}")
            print(f"{_DIM}  Available: {', '.join(sorted(_ALL_PARAMS))}{_RESET}\n")
            return

        name, raw_val = param_parts[0].lower(), param_parts[1]

        if name not in _ALL_PARAMS:
            print(f"{_RED}Unknown parameter: {name}{_RESET}")
            print(f"{_DIM}  Available: {', '.join(sorted(_ALL_PARAMS))}{_RESET}\n")
            return

        try:
            value = float(raw_val) if name in _FLOAT_PARAMS else int(raw_val)
        except ValueError:
            expected = "float" if name in _FLOAT_PARAMS else "integer"
            print(f"{_RED}Invalid value for {name}: expected {expected}, got '{raw_val}'{_RESET}\n")
            return

        session["options"][name] = value
        print(f"{_CYAN}{_BOLD}✓  {name} = {value}{_RESET}\n")
        return

    print(f"{_RED}Unknown /set subcommand: {sub}{_RESET}  {_DIM}(try: parameter, system, verbose, quiet, wordwrap, nowordwrap){_RESET}\n")


def _handle_show(args: str, session: dict, history: list[dict]) -> None:
    """Handle /show sub-commands."""
    sub = args.strip().lower() or "parameters"

    if sub == "parameters":
        opts = session.get("options", {})
        if not opts:
            print(f"{_DIM}  No custom parameters set (using model defaults).{_RESET}\n")
        else:
            print(f"\n{_CYAN}{_BOLD}Session parameters:{_RESET}")
            for k, v in sorted(opts.items()):
                print(f"  {_GREEN}{k}{_RESET} = {v}")
            print()
        # Also show flags
        flags = []
        if session.get("verbose"):
            flags.append("verbose")
        if not session.get("wordwrap", True):
            flags.append("nowordwrap")
        if flags:
            print(f"{_DIM}  Flags: {', '.join(flags)}{_RESET}\n")
        return

    if sub == "system":
        prompt = session.get("system", "")
        if not prompt:
            # Check if history has one from the Modelfile
            if history and history[0].get("role") == "system":
                prompt = history[0]["content"]
        if prompt:
            print(f"\n{_CYAN}{_BOLD}System prompt:{_RESET}\n{_DIM}{prompt}{_RESET}\n")
        else:
            print(f"{_DIM}  No system prompt set (using Modelfile default).{_RESET}\n")
        return

    if sub in ("model", "info"):
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
        return

    print(f"{_RED}Unknown /show subcommand: {sub}{_RESET}  {_DIM}(try: parameters, system, model){_RESET}\n")


def _handle_command(cmd: str, session: dict, history: list[dict]) -> bool | None:
    """Handle a slash command. Returns True if handled, None to quit."""
    parts = cmd.strip().split(None, 1)
    base = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if base in ("/quit", "/exit", "/q"):
        return None  # Signal to quit

    if base in ("/help", "/?"):
        print(_COMMANDS_HELP)
        return True

    if base == "/clear":
        history.clear()
        print(f"{_CYAN}{_BOLD}✓  Conversation history cleared.{_RESET}\n")
        return True

    if base == "/set":
        _handle_set(rest, session, history)
        return True

    if base == "/show":
        _handle_show(rest, session, history)
        return True

    # Unknown command
    print(f"{_RED}Unknown command: {base}{_RESET}  {_DIM}(type /help for available commands){_RESET}\n")
    return True


# ── Main loop ─────────────────────────────────────────────────────────

def run() -> None:
    """Run the interactive agent loop."""
    history: list[dict] = []
    session: dict = {
        "options": {},       # Runtime model parameters (temperature, etc.)
        "verbose": False,    # Show generation stats
        "wordwrap": True,    # Word wrapping (reserved for future use)
        "system": "",        # Custom system prompt override
    }

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
            result = _handle_command(user_input, session, history)
            if result is None:
                break  # /quit
            continue  # Command was handled, skip LLM call

        history.append({"role": "user", "content": user_input})

        # ── LLM call with streaming + thinking ────────────────────────
        assistant_msg = _stream_thinking_response(
            model=MODEL_NAME,
            messages=history,
            tools=TOOL_SCHEMAS,
            options=session["options"] or None,
            verbose=session["verbose"],
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
                options=session["options"] or None,
                verbose=session["verbose"],
            )
            history.append(assistant_msg)
