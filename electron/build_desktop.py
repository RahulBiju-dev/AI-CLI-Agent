"""Run electron-builder with Electron's bundled Node on Linux or Windows."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def _electron_executable() -> Path:
    base = ROOT / "node_modules" / "electron" / "dist"
    candidates = [base / "electron.exe", base / "electron"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Electron is not installed under node_modules. Install the desktop dev dependencies first."
    )


def main() -> int:
    cli = ROOT / "node_modules" / "electron-builder" / "out" / "cli" / "cli.js"
    if not cli.is_file():
        raise FileNotFoundError(
            "electron-builder is not installed under node_modules. Install the desktop dev dependencies first."
        )
    environment = dict(os.environ)
    environment["ELECTRON_RUN_AS_NODE"] = "1"
    environment.setdefault("CI", "true")
    loader = f"process.noAsar=true;require({str(cli)!r})"
    command = [
        str(_electron_executable()),
        "-e",
        loader,
        "build",
        *sys.argv[1:],
        "--publish",
        "never",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, shell=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
