"""Terminal voice input for Selene (TUI / classic CLI).

Mirrors the Web UI voice flow in ``agent/static/app.js``:

* toggle start / stop listening
* keep a *base* composer draft and append live transcripts
* continuous capture with progressive results
* cooperative stop (abort vs clean stop)

ALSA / Pulse / JACK drivers often spam stderr when opening devices. That noise
destroys full-screen TUI painting, so native device open/close calls use a
short stderr guard. Missing optional packages never block Selene startup.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal


ErrorCode = Literal[
    "not-allowed",
    "service-not-allowed",
    "audio-capture",
    "no-speech",
    "network",
    "busy",
    "unsupported",
    "unknown",
]

# Same user-facing messages as agent/static/app.js (adapted for terminal).
ERROR_MESSAGES: dict[str, str] = {
    "not-allowed": "Microphone access was denied. Allow it in system settings to use voice input.",
    "service-not-allowed": "Voice recognition is blocked by your environment.",
    "audio-capture": "No working microphone was found.",
    "no-speech": "I didn't hear anything. Try speaking a little closer to the microphone.",
    "network": "Voice recognition is unavailable right now. You can still type your message.",
    "busy": "Voice input is already busy. Wait a moment and try again.",
    "unsupported": (
        "Voice input needs optional packages: SpeechRecognition and a microphone backend "
        "(PyAudio). Install them, then retry /speech."
    ),
    "unknown": "Voice input stopped unexpectedly. You can still type your message.",
}


@dataclass(frozen=True)
class SpeechCapability:
    available: bool
    detail: str = ""


_capability_cache: SpeechCapability | None = None
_capability_lock = threading.Lock()
_silence_lock = threading.RLock()
_silence_depth = 0
_silence_saved_fd: int | None = None
_silence_devnull_fd: int | None = None


@contextmanager
def audio_stderr_silence() -> Iterator[None]:
    """Temporarily redirect fd 2 to /dev/null (nested, process-wide).

    PortAudio / ALSA print device-enumeration failures to stderr. On a
    full-screen TUI those lines paint over the alternate screen. Silence only
    around noisy native setup/teardown — never hold it during recognition.
    """
    global _silence_depth, _silence_saved_fd, _silence_devnull_fd

    redirected = True
    with _silence_lock:
        if _silence_depth == 0:
            devnull = None
            saved = None
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                saved = os.dup(2)
                os.dup2(devnull, 2)
            except OSError:
                for fd in (saved, devnull):
                    if fd is not None:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                redirected = False
            else:
                _silence_devnull_fd = devnull
                _silence_saved_fd = saved
        if redirected:
            _silence_depth += 1

    if not redirected:
        yield
        return

    try:
        yield
    finally:
        with _silence_lock:
            _silence_depth = max(0, _silence_depth - 1)
            if _silence_depth == 0 and _silence_saved_fd is not None:
                try:
                    os.dup2(_silence_saved_fd, 2)
                except OSError:
                    pass
                try:
                    os.close(_silence_saved_fd)
                except OSError:
                    pass
                if _silence_devnull_fd is not None:
                    try:
                        os.close(_silence_devnull_fd)
                    except OSError:
                        pass
                _silence_saved_fd = None
                _silence_devnull_fd = None


def clear_speech_capability_cache() -> None:
    """Drop cached probe result (tests / device hot-plug)."""
    global _capability_cache
    with _capability_lock:
        _capability_cache = None


def speech_capability(*, force: bool = False) -> SpeechCapability:
    """Return whether voice input can be started in this environment.

    Only the optional Python stack is checked here. Opening an audio device can
    block for seconds on some PipeWire/ALSA setups, so device validation belongs
    to the background capture worker and never to the TUI event thread.
    """
    global _capability_cache
    with _capability_lock:
        if _capability_cache is not None and not force:
            return _capability_cache

    try:
        import speech_recognition as sr  # noqa: F401
    except Exception as exc:
        result = SpeechCapability(
            False,
            f"SpeechRecognition is not installed ({exc}). "
            "Install with: pip install SpeechRecognition PyAudio",
        )
        with _capability_lock:
            _capability_cache = result
        return result

    result = SpeechCapability(True, "SpeechRecognition audio stack available")
    with _capability_lock:
        _capability_cache = result
    return result


def _map_exception(exc: BaseException) -> ErrorCode:
    name = type(exc).__name__
    message = str(exc).casefold()
    if "request" in name.casefold() or "network" in message or "connection" in message:
        return "network"
    if "unknownvalue" in name.casefold() or "no speech" in message:
        return "no-speech"
    if "permission" in message or "denied" in message or "not allowed" in message:
        return "not-allowed"
    if "device" in message or "microphone" in message or "input" in message or "pcm" in message:
        return "audio-capture"
    return "unknown"


class VoiceInputController:
    """Continuous voice capture with Web-UI-compatible toggle semantics."""

    def __init__(
        self,
        *,
        on_transcript: Callable[[str], None] | None = None,
        on_active: Callable[[bool], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        language: str | None = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_active = on_active
        self._on_error = on_error
        self._on_finished = on_finished
        self._language = language or "en-US"

        self._lock = threading.RLock()
        self._active = False
        self._suppress_error = False
        self._base_text = ""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._generation = 0
        self._recognizer = None
        self._microphone = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def set_base_text(self, text: str) -> None:
        with self._lock:
            self._base_text = str(text or "")

    def get_base_text(self) -> str:
        with self._lock:
            return self._base_text

    def toggle(self, *, base_text: str | None = None) -> bool:
        """Start if idle, stop if active. Returns the new active state."""
        if self.active:
            self.stop()
            return False
        self.start(base_text=base_text)
        return self.active

    def start(self, *, base_text: str | None = None) -> bool:
        """Begin continuous listening without probing audio on the caller thread."""
        capability = speech_capability()
        if not capability.available:
            self._emit_error("unsupported", capability.detail or ERROR_MESSAGES["unsupported"])
            return False

        with self._lock:
            if self._active:
                self._emit_error("busy", ERROR_MESSAGES["busy"])
                return False
            if base_text is not None:
                self._base_text = str(base_text)
            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            self._stop = stop_event
            self._suppress_error = False
            self._active = True
            thread = threading.Thread(
                target=self._listen_loop,
                args=(generation, stop_event),
                name="selene-speech",
                daemon=True,
            )
            self._thread = thread

        self._set_active(True)
        thread.start()
        return True

    def stop(self, *, abort: bool = False, silent: bool = False) -> None:
        """Stop listening (mirrors ``stopVoiceInput`` in app.js)."""
        with self._lock:
            if not self._active and self._thread is None:
                return
            self._suppress_error = bool(silent)
            self._stop.set()
            self._active = False
            if abort:
                self._generation += 1  # invalidate late results from this worker
            microphone = self._microphone
        # Mark inactive immediately for UI (onend equivalent).
        self._set_active(False)
        # Closing the active PortAudio stream wakes recognizer.listen promptly.
        # Never join here: stop() is called by the Textual event thread.
        stream = getattr(microphone, "stream", None)
        if abort and stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _set_active(self, active: bool) -> None:
        callback = self._on_active
        if callback is None:
            return
        try:
            callback(bool(active))
        except Exception:
            pass

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def _emit_error(
        self, code: str, message: str | None = None, *, generation: int | None = None
    ) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            if self._suppress_error and code != "unsupported":
                return
        text = message or ERROR_MESSAGES.get(code, ERROR_MESSAGES["unknown"])
        callback = self._on_error
        if callback is None:
            return
        try:
            callback(code, text)
        except Exception:
            pass

    def _emit_transcript(self, transcript: str) -> None:
        transcript = " ".join(str(transcript or "").split()).strip()
        if not transcript:
            return
        with self._lock:
            base = self._base_text.rstrip()
            # Match web: base && transcript ? `${base} ${transcript}` : (base || transcript)
            if base and transcript:
                combined = f"{base} {transcript}"
            else:
                combined = base or transcript
            # Commit before notifying the UI. Programmatic Input.Changed events
            # may write this same value back; a later phrase then appends once.
            self._base_text = combined
        callback = self._on_transcript
        if callback is None:
            return
        try:
            callback(combined)
        except Exception:
            pass

    def _listen_loop(self, generation: int, stop_event: threading.Event) -> None:
        """Background capture loop — continuous phrase recognition."""
        try:
            import speech_recognition as sr
        except Exception as exc:
            self._emit_error("unsupported", str(exc), generation=generation)
            self._finish_loop(generation)
            return

        recognizer = sr.Recognizer()
        # Slightly more patient defaults for conversational dictation.
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.55
        recognizer.non_speaking_duration = 0.3
        recognizer.operation_timeout = 4.0

        microphone = None
        # A just-closed PortAudio stream may need a moment to release the
        # default device. Retry in this worker without delaying the UI.
        for attempt in range(3):
            if stop_event.is_set() or not self._is_current(generation):
                self._finish_loop(generation)
                return
            try:
                with audio_stderr_silence():
                    microphone = sr.Microphone()
                break
            except Exception as exc:
                if attempt == 2:
                    code = _map_exception(exc)
                    self._emit_error(
                        code, ERROR_MESSAGES.get(code, str(exc)), generation=generation
                    )
                    self._finish_loop(generation)
                    return
                stop_event.wait(0.08)

        try:
            # Silence only native open/close operations. Redirecting fd 2 for
            # the full recording would hide unrelated diagnostics process-wide.
            with audio_stderr_silence():
                source = microphone.__enter__()
            with self._lock:
                if generation == self._generation:
                    self._microphone = microphone
            try:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.18)
                except Exception:
                    pass

                while not stop_event.is_set() and self._is_current(generation):
                    try:
                        audio = recognizer.listen(
                            source,
                            timeout=0.65,
                            phrase_time_limit=8,
                        )
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as exc:
                        if stop_event.is_set() or not self._is_current(generation):
                            break
                        code = _map_exception(exc)
                        self._emit_error(code)
                        if code in {"not-allowed", "audio-capture", "service-not-allowed"}:
                            break
                        continue

                    if not self._is_current(generation):
                        break

                    try:
                        # Same service family as browser Web Speech (Google).
                        text = recognizer.recognize_google(audio, language=self._language)
                    except sr.UnknownValueError:
                        # Equivalent to browser "no-speech" for a single phrase —
                        # stay listening (continuous) instead of hard-stopping.
                        continue
                    except sr.RequestError as exc:
                        self._emit_error(
                            "network",
                            ERROR_MESSAGES["network"] + f" ({exc})",
                            generation=generation,
                        )
                        break
                    except Exception as exc:
                        code = _map_exception(exc)
                        self._emit_error(code, generation=generation)
                        if code in {"network", "not-allowed", "service-not-allowed"}:
                            break
                        continue

                    # Progressive "final" chunks: fold into base so the next phrase
                    # appends (mirrors continuous results joining in app.js).
                    if self._is_current(generation):
                        self._emit_transcript(text)
            finally:
                with audio_stderr_silence():
                    microphone.__exit__(None, None, None)
        except Exception as exc:
            if not stop_event.is_set() and self._is_current(generation):
                code = _map_exception(exc)
                self._emit_error(code, generation=generation)
        finally:
            self._finish_loop(generation)

    def _finish_loop(self, generation: int) -> None:
        finished_callback = None
        current_thread = threading.current_thread()
        with self._lock:
            if self._thread is current_thread:
                self._thread = None
                self._microphone = None
            if generation != self._generation:
                return
            self._active = False
            self._suppress_error = False
            finished_callback = self._on_finished
        # onend always clears active in the web UI.
        self._set_active(False)
        if finished_callback is not None:
            try:
                finished_callback()
            except Exception:
                pass


def capture_once(
    *,
    timeout: float = 6.0,
    phrase_time_limit: float = 12.0,
    language: str | None = None,
) -> tuple[str | None, str | None]:
    """One-shot capture for classic CLI. Returns ``(text, error_message)``."""
    capability = speech_capability()
    if not capability.available:
        return None, capability.detail or ERROR_MESSAGES["unsupported"]
    try:
        import speech_recognition as sr
    except Exception as exc:
        return None, str(exc)

    recognizer = sr.Recognizer()
    try:
        with audio_stderr_silence():
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.35)
                except Exception:
                    pass
                audio = recognizer.listen(
                    source,
                    timeout=max(1.0, float(timeout)),
                    phrase_time_limit=max(1.0, float(phrase_time_limit)),
                )
            text = recognizer.recognize_google(audio, language=language or "en-US")
        cleaned = " ".join(str(text).split()).strip()
        return (cleaned or None), None
    except sr.WaitTimeoutError:
        return None, ERROR_MESSAGES["no-speech"]
    except sr.UnknownValueError:
        return None, ERROR_MESSAGES["no-speech"]
    except sr.RequestError as exc:
        return None, ERROR_MESSAGES["network"] + f" ({exc})"
    except Exception as exc:
        code = _map_exception(exc)
        return None, ERROR_MESSAGES.get(code, str(exc))
