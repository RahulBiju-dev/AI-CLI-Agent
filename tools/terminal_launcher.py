"""Safely open a terminal window at an existing directory."""

from __future__ import annotations

import json
from pathlib import Path

from agent.platform_runtime import select_terminal_command, spawn_detached


def _resolve_directory(path: object) -> Path:
    """Return a normalized existing directory without interpreting shell syntax."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("A directory path is required.")
    value = path.strip()
    if len(value) > 4096 or "\0" in value:
        raise ValueError("The directory path is invalid.")

    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    try:
        directory = directory.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"Directory does not exist: {value}") from None
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {value}")
    return directory


def _linux_terminal_command(directory: Path) -> tuple[list[str], str] | None:
    """Compatibility wrapper around the Fedora-native terminal selector."""
    selected = select_terminal_command(directory, platform_name="linux")
    if selected is None:
        return None
    return list(selected.argv), selected.backend


def open_terminal_at_path(path: str, confirmed: bool = False) -> str:
    """Open a terminal at ``path`` after an explicit user or routine approval."""
    if confirmed is not True:
        return json.dumps({
            "error": "Opening a terminal requires explicit user approval.",
            "required": "Call again with confirmed=true only when the user requested it.",
        })

    try:
        directory = _resolve_directory(path)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    try:
        selected = select_terminal_command(directory)
        if selected is None:
            return json.dumps({
                "error": "No supported native terminal is installed.",
                "supported_backends": {
                    "fedora_linux": ["GNOME Console", "GNOME Terminal", "Konsole", "Xfce Terminal", "Ptyxis"],
                    "windows": ["Windows Terminal", "PowerShell 7", "Windows PowerShell", "Command Prompt"],
                },
            })
        spawn_detached(
            selected.argv,
            cwd=directory,
            new_console=selected.new_console,
        )
        return json.dumps({
            "success": True,
            "backend": selected.backend,
            "terminal": selected.backend,
            "path": str(directory),
            "message": f"Sent a request to open {selected.backend} at '{directory}'; no command was executed.",
        }, ensure_ascii=False)
    except OSError as exc:
        return json.dumps({
            "error": f"Failed to open a terminal at '{directory}': {exc}"
        }, ensure_ascii=False)
