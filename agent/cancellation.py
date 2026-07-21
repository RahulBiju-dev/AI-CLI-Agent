"""Small cooperative-cancellation primitive shared by runtime operations."""

from __future__ import annotations

import threading
from collections.abc import Callable


class OperationCancelled(RuntimeError):
    """Raised when an operation observes a requested cancellation."""


class CancellationToken:
    """Thread-safe cancellation state with an optional human-readable reason."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "Operation cancelled"
        self._callbacks: list[Callable[[], None]] = []

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
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # A cleanup hook must not prevent other owned resources from
                # receiving the cancellation request.
                pass
        return True

    def add_cancel_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Invoke ``callback`` on cancellation and return a remover for it."""
        if not callable(callback):
            raise TypeError("Cancellation callbacks must be callable")
        run_now = False
        with self._lock:
            if self._event.is_set():
                run_now = True
            else:
                self._callbacks.append(callback)

        if run_now:
            try:
                callback()
            except Exception:
                pass

        def remove() -> None:
            with self._lock:
                try:
                    self._callbacks.remove(callback)
                except ValueError:
                    pass

        return remove

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled(self.reason)
