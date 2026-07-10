"""Small cooperative-cancellation primitive shared by runtime operations."""

from __future__ import annotations

import threading


class OperationCancelled(RuntimeError):
    """Raised when an operation observes a requested cancellation."""


class CancellationToken:
    """Thread-safe cancellation state with an optional human-readable reason."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "Operation cancelled"

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "Operation cancelled") -> bool:
        """Request cancellation and return whether this call changed the state."""
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = str(reason or "Operation cancelled")
            self._event.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled(self.reason)
