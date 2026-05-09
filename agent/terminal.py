"""
agent/terminal.py — Terminal helpers and lightweight LaTeX math renderer

Contains ANSI helpers, a spinner, and a compact LaTeX-to-terminal
renderer used by the streaming output in `agent.core`.
"""
from __future__ import annotations

import re
import sys
import threading
import itertools
import time
from rich.console import Console

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

# Shared console
_console = Console()


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


# Small helpers for rendering common LaTeX math to terminal-friendly text
_UNICODE_FRACTIONS = {
    ("1", "2"): "½",
    ("1", "3"): "⅓",
    ("2", "3"): "⅔",
    ("1", "4"): "¼",
    ("3", "4"): "¾",
    ("1", "5"): "⅕",
    ("2", "5"): "⅖",
    ("3", "5"): "⅗",
    ("4", "5"): "⅘",
    ("1", "6"): "⅙",
    ("5", "6"): "⅚",
    ("1", "8"): "⅛",
    ("3", "8"): "⅜",
    ("5", "8"): "⅝",
    ("7", "8"): "⅞",
}

_LATEX_SYMBOLS = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\pi": "π",
    r"\sigma": "σ",
    r"\phi": "φ",
    r"\omega": "ω",
    r"\partial": "∂",
    r"\nabla": "∇",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
    r"\iint": "∬",
    r"\iiint": "∭",
    r"\oint": "∮",
    r"\times": "×",
    r"\cdot": "·",
    r"\div": "÷",
    r"\pm": "±",
    r"\mp": "∓",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\lt": "<",
    r"\gt": ">",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\equiv": "≡",
    r"\sim": "∼",
    r"\simeq": "≃",
    r"\propto": "∝",
    r"\infty": "∞",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\mapsto": "↦",
    r"\uparrow": "↑",
    r"\downarrow": "↓",
    r"\updownarrow": "↕",
    r"\implies": "⇒",
    r"\iff": "⇔",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\emptyset": "∅",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subseteq": "⊆",
    r"\subset": "⊂",
    r"\supseteq": "⊇",
    r"\supset": "⊃",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\setminus": "∖",
    r"\mathbb": "",
    r"\mathrm": "",
    r"\mathbf": "",
    r"\mathcal": "",
    r"\quad": " ",
    r"\qquad": " ",
    r"\left": "",
    r"\right": "",
}

_LATEX_SPACING = {
    r"\,": " ",
    r"\!": "",
    r"\;": " ",
    r"\:": " ",
}

# Unicode superscript / subscript mapping for common characters
_SUP_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "n": "ⁿ",
    "i": "ⁱ",
}

_SUB_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "o": "ₒ",
    "x": "ₓ",
    "i": "ᵢ",
    "r": "ᵣ",
    "u": "ᵤ",
    "v": "ᵥ",
    "p": "ᵨ",
}


def _to_superscript(text: str) -> str:
    out = []
    for ch in text:
        out.append(_SUP_MAP.get(ch, ch))
    return "".join(out)


def _to_subscript(text: str) -> str:
    out = []
    for ch in text:
        out.append(_SUB_MAP.get(ch, ch))
    return "".join(out)


def _extract_braced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _render_latex_math(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return ""

    expr = expr.replace(r"\displaystyle", "")
    expr = expr.replace(r"\textstyle", "")

    # Handle \frac{a}{b}
    def replace_frac(text: str) -> str:
        output = []
        index = 0
        while index < len(text):
            if text.startswith(r"\frac", index):
                numerator = _extract_braced(text, index + 5)
                if numerator is not None:
                    denominator = _extract_braced(text, numerator[1])
                    if denominator is not None:
                        rendered_num = _render_latex_math(numerator[0])
                        rendered_den = _render_latex_math(denominator[0])
                        if rendered_num.isdigit() and rendered_den.isdigit():
                            output.append(_UNICODE_FRACTIONS.get((rendered_num, rendered_den), f"{rendered_num}/{rendered_den}"))
                        else:
                            output.append(f"({rendered_num})/({rendered_den})")
                        index = denominator[1]
                        continue
            output.append(text[index])
            index += 1
        return "".join(output)

    # Handle \sqrt{...}
    def replace_sqrt(text: str) -> str:
        output = []
        index = 0
        while index < len(text):
            if text.startswith(r"\sqrt", index):
                radicand = _extract_braced(text, index + 5)
                if radicand is not None:
                    rendered = _render_latex_math(radicand[0])
                    output.append(f"√({rendered})")
                    index = radicand[1]
                    continue
            output.append(text[index])
            index += 1
        return "".join(output)

    expr = replace_frac(expr)
    expr = replace_sqrt(expr)

    # Replace common LaTeX symbols
    # Longer keys are placed first by sorting to avoid partial matches
    for latex in sorted(_LATEX_SYMBOLS.keys(), key=lambda s: -len(s)):
        expr = expr.replace(latex, _LATEX_SYMBOLS[latex])

    for latex, replacement in _LATEX_SPACING.items():
        expr = expr.replace(latex, replacement)

    # Superscript/subscript handling: ^{...}, _{...}, ^x, _x
    def replace_scripts(text: str) -> str:
        # caret superscript
        def sup_repl(m: re.Match[str]) -> str:
            token = m.group(1)
            if token.startswith("{") and token.endswith("}"):
                inner = token[1:-1]
            else:
                inner = token
            inner_rendered = _render_latex_math(inner)
            mapped = _to_superscript(inner_rendered)
            if mapped != inner_rendered:
                return mapped
            return f"^{inner_rendered}"

        # subscript
        def sub_repl(m: re.Match[str]) -> str:
            token = m.group(1)
            if token.startswith("{") and token.endswith("}"):
                inner = token[1:-1]
            else:
                inner = token
            inner_rendered = _render_latex_math(inner)
            mapped = _to_subscript(inner_rendered)
            if mapped != inner_rendered:
                return mapped
            return f"_{inner_rendered}"

        text = re.sub(r"\^(\{.*?\}|.)", sup_repl, text)
        text = re.sub(r"_(\{.*?\}|.)", sub_repl, text)
        return text

    expr = replace_scripts(expr)

    # Strip remaining braces and collapse whitespace
    expr = expr.replace("{", "").replace("}", "")
    expr = re.sub(r"\s+", " ", expr)
    return expr.strip()


def _render_terminal_markdown(text: str) -> str:
    def replace_block(match: re.Match[str]) -> str:
        rendered = _render_latex_math(match.group(1))
        return f"\n{rendered}\n"

    def replace_inline(match: re.Match[str]) -> str:
        return _render_latex_math(match.group(1))

    text = re.sub(r"\$\$(.+?)\$\$", replace_block, text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", replace_inline, text)
    return text.replace(r"\$", "$")
