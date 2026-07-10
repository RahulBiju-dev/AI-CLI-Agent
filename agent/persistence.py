"""Crash-safe persistence primitives for Selene's critical local state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class PersistenceError(RuntimeError):
    """Raised when existing state cannot be read without risking data loss."""


def _sync_parent(directory: Path) -> None:
    """Best-effort directory fsync after replacement (supported on POSIX)."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    private: bool = False,
    durable: bool = True,
) -> Path:
    """Write bytes to a same-directory temporary file and atomically replace."""
    destination = Path(path)
    if not isinstance(data, bytes):
        raise TypeError("atomic_write_bytes requires bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        if private and hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if private:
            try:
                os.chmod(destination, 0o600)
            except OSError:
                # Windows ACLs, not POSIX mode bits, protect the per-user store.
                if os.name != "nt":
                    raise
        if durable:
            _sync_parent(destination.parent)
        return destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    private: bool = False,
    durable: bool = True,
) -> Path:
    if not isinstance(text, str):
        raise TypeError("atomic_write_text requires str")
    return atomic_write_bytes(path, text.encode(encoding), private=private, durable=durable)


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    private: bool = False,
    durable: bool = True,
    indent: int | None = 2,
) -> Path:
    serialized = json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False)
    if indent is not None:
        serialized += "\n"
    return atomic_write_text(path, serialized, private=private, durable=durable)


def read_json_preserved(path: str | Path, *, expected_type: type | tuple[type, ...] | None = None) -> Any:
    """Read JSON and raise without modifying malformed or unexpected existing data."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PersistenceError(
            f"Existing state at '{source}' is unreadable or malformed and was preserved: {exc}"
        ) from exc
    if expected_type is not None and not isinstance(value, expected_type):
        expected_name = (
            ", ".join(item.__name__ for item in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise PersistenceError(
            f"Existing state at '{source}' has an unexpected format (expected {expected_name}) and was preserved."
        )
    return value
