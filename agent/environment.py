"""Small, dependency-free loader for Selene's server-side ``.env`` file."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path | None:
    """Load simple KEY=VALUE entries without replacing process environment.

    This intentionally supports only the subset needed by this project: blank
    lines, comments, optional ``export``, and single/double quoted values.  It
    never logs values because this file commonly contains provider secrets.
    """
    destination = os.environ if environ is None else environ
    if path is not None:
        candidates = [Path(path).expanduser()]
    else:
        explicit = str(destination.get("SELENE_ENV_FILE") or "").strip()
        candidates = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        candidates.extend([
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ])
    target = next((candidate for candidate in candidates if candidate.is_file()), None)
    if target is None:
        return None

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _ENV_NAME.fullmatch(name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        destination.setdefault(name, value)
    return target
