#!/usr/bin/env python3
"""
main.py — Entry point for the Gemma CLI Agent.

Creates the custom Ollama model from the Modelfile (if needed)
and launches the interactive chat loop.
"""

import sys

import ollama

from agent.core import MODEL_NAME, run

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _ensure_model() -> None:
    """Create the custom model from the Modelfile if it doesn't already exist."""
    try:
        ollama.show(MODEL_NAME)
    except ollama.ResponseError:
        print(f"{_CYAN}{_BOLD}⟳  Building model '{MODEL_NAME}' from Modelfile…{_RESET}")
        ollama.create(model=MODEL_NAME, from_="./Modelfile")
        print(f"{_CYAN}{_BOLD}✓  Model ready.{_RESET}\n")


def main() -> None:
    try:
        _ensure_model()
        run()
    except KeyboardInterrupt:
        print(f"\n{_DIM}Interrupted — goodbye.{_RESET}")
        sys.exit(0)
    except ollama.ResponseError as exc:
        print(f"\n{_RED}{_BOLD}Ollama error:{_RESET} {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
