"""Native platform contracts shared by Selene's CLI, tools, and packagers.

Fedora/Linux is the reference backend.  Windows paths and process flags are
kept native so callers never need a POSIX compatibility layer.  This module is
deliberately small: it owns OS mechanics, while tools retain their own safety
and confirmation policies.
"""

from __future__ import annotations

import importlib.util
import ntpath
import os
import shutil
import signal
import subprocess
import sys
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Collection, Mapping, Sequence


APP_NAME = "Selene"
APP_DIR_NAME = "selene"
LEGACY_DATA_DIR_NAME = ".selene-agent"
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def platform_family(value: str | None = None) -> str:
    """Return a stable ``linux``, ``windows``, or ``macos`` platform name."""
    current = (value or sys.platform).casefold()
    if current.startswith(("win", "cygwin", "msys")):
        # Cygwin/MSYS are recognized only to select native Windows mechanics;
        # Selene itself never requires those environments.
        return "windows"
    if current.startswith(("darwin", "mac")):
        return "macos"
    return "linux"


def _expanded_path(value: str, home: Path) -> Path:
    """Expand a leading tilde using an injected home path for deterministic tests."""
    if value == "~":
        return home
    if value.startswith(("~/", "~\\")):
        return home / value[2:]
    return Path(value)


def _absolute_environment_base(value: str, fallback: Path, home: Path) -> Path:
    """Use an environment directory only when it is absolute.

    XDG explicitly treats relative directory values as invalid. Windows
    per-user roots should be held to the same rule so installation or working
    directories can never accidentally become runtime stores.
    """
    raw = str(value or "").strip()
    if not raw:
        return fallback
    candidate = _expanded_path(raw, home)
    return candidate if candidate.is_absolute() else fallback


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved per-user locations and the reason the active store was chosen."""

    data_dir: Path
    state_dir: Path
    config_dir: Path
    cache_dir: Path
    legacy_data_dir: Path
    source: str

    def compatibility_candidates(self, relative_path: str | Path) -> tuple[Path, ...]:
        """Return active then legacy read candidates without moving either store."""
        relative = _safe_relative_path(relative_path)
        active = self.data_dir / relative
        legacy = self.legacy_data_dir / relative
        return (active,) if active == legacy else (active, legacy)

    def report(self) -> dict[str, str]:
        return {
            "data_dir": str(self.data_dir),
            "state_dir": str(self.state_dir),
            "config_dir": str(self.config_dir),
            "cache_dir": str(self.cache_dir),
            "legacy_data_dir": str(self.legacy_data_dir),
            "source": self.source,
        }


def resolve_runtime_paths(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
    legacy_exists: bool | None = None,
) -> RuntimePaths:
    """Resolve Selene's active per-user store without silently migrating data.

    Precedence is ``SELENE_DATA_DIR``, an existing legacy ``~/.selene-agent``
    store, then the native platform convention.  Explicit storage keeps all
    categories together for compatibility with existing Selene deployments.
    """
    environment = os.environ if environ is None else environ
    family = platform_family(platform_name)
    if home is None:
        home_value = environment.get("USERPROFILE") if family == "windows" else environment.get("HOME")
        home_path = Path(home_value) if home_value else Path.home()
    else:
        home_path = Path(home)
    home_path = home_path.expanduser().absolute()
    legacy = home_path / LEGACY_DATA_DIR_NAME

    explicit = str(environment.get("SELENE_DATA_DIR", "")).strip()
    if explicit:
        selected = _expanded_path(explicit, home_path).absolute()
        return RuntimePaths(selected, selected, selected, selected / "cache", legacy, "SELENE_DATA_DIR")

    existing_legacy = legacy.is_dir() if legacy_exists is None else bool(legacy_exists)
    if existing_legacy:
        # Compatibility is intentionally location selection, not migration.
        return RuntimePaths(legacy, legacy, legacy, legacy / "cache", legacy, "existing-legacy-store")

    if family == "windows":
        local_app_data = str(environment.get("LOCALAPPDATA", "")).strip()
        base = _absolute_environment_base(
            local_app_data,
            home_path / "AppData" / "Local",
            home_path,
        )
        selected = base / APP_NAME
        return RuntimePaths(selected, selected / "State", selected / "Config", selected / "Cache", legacy, "windows-localappdata")

    if family == "macos":
        data = home_path / "Library" / "Application Support" / APP_NAME
        state = home_path / "Library" / "Application Support" / APP_NAME / "State"
        config = home_path / "Library" / "Preferences" / APP_NAME
        cache = home_path / "Library" / "Caches" / APP_NAME
        return RuntimePaths(data, state, config, cache, legacy, "macos-user-library")

    data_base = _absolute_environment_base(
        environment.get("XDG_DATA_HOME", ""), home_path / ".local" / "share", home_path
    )
    state_base = _absolute_environment_base(
        environment.get("XDG_STATE_HOME", ""), home_path / ".local" / "state", home_path
    )
    config_base = _absolute_environment_base(
        environment.get("XDG_CONFIG_HOME", ""), home_path / ".config", home_path
    )
    cache_base = _absolute_environment_base(
        environment.get("XDG_CACHE_HOME", ""), home_path / ".cache", home_path
    )
    return RuntimePaths(
        data_base / APP_DIR_NAME,
        state_base / APP_DIR_NAME,
        config_base / APP_DIR_NAME,
        cache_base / APP_DIR_NAME,
        legacy,
        "linux-xdg",
    )


def get_runtime_paths(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
    legacy_exists: bool | None = None,
) -> RuntimePaths:
    """Resolve runtime paths, accepting the same testable inputs as the resolver.

    Keeping this convenience wrapper contract-identical prevents callers from
    having to switch functions merely because they need an injected environment
    during startup or a platform simulation in a focused test.
    """
    return resolve_runtime_paths(
        platform_name=platform_name,
        environ=environ,
        home=home,
        legacy_exists=legacy_exists,
    )


def _safe_relative_path(value: str | Path) -> Path:
    raw = str(value)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if not raw or posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
        raise ValueError("Resource paths must be non-empty paths relative to Selene's resource root.")
    if "\x00" in raw:
        raise ValueError("Resource path contains a null byte.")
    return Path(*posix.parts)


def resource_path(
    relative_path: str | Path,
    *,
    must_exist: bool = True,
    environ: Mapping[str, str] | None = None,
    meipass: str | Path | None = None,
) -> Path:
    """Locate a source or PyInstaller resource without depending on the cwd."""
    relative = _safe_relative_path(relative_path)
    environment = os.environ if environ is None else environ
    roots: list[Path] = []
    explicit = str(environment.get("SELENE_RESOURCE_DIR", "")).strip()
    if explicit:
        roots.append(Path(explicit).expanduser())
    bundle_root = meipass if meipass is not None else getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.append(Path(bundle_root))
    roots.append(Path(__file__).resolve().parent.parent)

    candidates: list[Path] = []
    for root in roots:
        candidate = root.resolve(strict=False) / relative
        if candidate not in candidates:
            candidates.append(candidate)
        if not must_exist or candidate.exists():
            return candidate
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Selene resource '{relative}' was not found (searched: {searched}).")


def find_executable(*names: str, path: str | None = None) -> str | None:
    """Return the first installed executable from an ordered native preference list."""
    for name in names:
        if not isinstance(name, str) or not name.strip() or "\x00" in name:
            continue
        discovered = shutil.which(name, path=path)
        if discovered:
            return discovered
    return None


def linux_application_dirs(
    *, environ: Mapping[str, str] | None = None, home: str | Path | None = None
) -> tuple[Path, ...]:
    """Return bounded XDG desktop-entry roots, with user overrides first."""
    environment = os.environ if environ is None else environ
    home_path = Path(home) if home is not None else Path(environment.get("HOME") or Path.home())
    data_home_value = str(environment.get("XDG_DATA_HOME", "")).strip() or "~/.local/share"
    data_home = _expanded_path(data_home_value, home_path)
    data_dirs_value = str(environment.get("XDG_DATA_DIRS", "")).strip() or "/usr/local/share:/usr/share"
    data_dirs = [part for part in data_dirs_value.split(os.pathsep) if part]
    roots = [
        data_home / "applications",
        data_home / "flatpak" / "exports" / "share" / "applications",
        *(_expanded_path(part, home_path) / "applications" for part in data_dirs),
    ]
    flatpak = Path("/var/lib/flatpak/exports/share/applications")
    if flatpak not in roots:
        roots.append(flatpak)
    snap = Path("/var/lib/snapd/desktop/applications")
    if snap not in roots:
        roots.append(snap)
    return tuple(dict.fromkeys(roots))


def windows_start_menu_dirs(*, environ: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Return the two native, bounded Windows Start Menu application roots."""
    environment = os.environ if environ is None else environ
    roots: list[Path] = []
    for variable in ("APPDATA", "PROGRAMDATA"):
        base = str(environment.get(variable, "")).strip()
        if base:
            roots.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return tuple(dict.fromkeys(roots))


@dataclass(frozen=True)
class NativeCommand:
    argv: tuple[str, ...]
    backend: str
    new_console: bool = False


def select_terminal_command(
    directory: str | Path,
    *,
    platform_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> NativeCommand | None:
    """Select a native terminal that inherits or explicitly receives ``directory``."""
    location = str(directory)
    family = platform_family(platform_name)
    if family == "windows":
        candidates = (
            (("wt.exe", "wt"), ("-d", location), "Windows Terminal"),
            (("pwsh.exe", "pwsh"), (), "PowerShell 7"),
            (("powershell.exe", "powershell"), (), "Windows PowerShell"),
            (("cmd.exe", "cmd"), (), "Command Prompt"),
        )
        for names, arguments, backend in candidates:
            executable = next((found for name in names if (found := which(name))), None)
            if executable:
                return NativeCommand((executable, *arguments), backend, new_console=True)
        return None
    if family == "macos":
        executable = which("open") or "/usr/bin/open"
        return NativeCommand((executable, "-a", "Terminal", location), "Terminal")

    # Fedora's GNOME Console is preferred, followed by common native desktop
    # terminals. Ptyxis is included as a modern Fedora fallback.
    candidates = (
        ("kgx", ("--working-directory", location), "GNOME Console"),
        ("gnome-terminal", ("--working-directory", location), "GNOME Terminal"),
        ("konsole", ("--workdir", location), "Konsole"),
        ("xfce4-terminal", ("--working-directory", location), "Xfce Terminal"),
        ("ptyxis", ("--new-window", "--working-directory", location), "Ptyxis"),
        ("kitty", ("--directory", location), "Kitty"),
        ("alacritty", ("--working-directory", location), "Alacritty"),
        # xterm safely inherits cwd; no command is passed.
        ("xterm", (), "XTerm"),
    )
    for name, arguments, backend in candidates:
        if executable := which(name):
            return NativeCommand((executable, *arguments), backend)
    return None


@dataclass
class OwnedProcess:
    """A child process created by Selene and therefore eligible for tree cleanup."""

    process: subprocess.Popen
    platform_name: str
    argv: tuple[str, ...]
    new_process_group: bool

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def poll(self):
        return self.process.poll()


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("A non-empty argument array is required.")
    normalized = tuple(str(item) for item in argv)
    if any(not item or "\x00" in item for item in normalized):
        raise ValueError("Process arguments must be non-empty and contain no null bytes.")
    return normalized


def spawn_detached(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    platform_name: str | None = None,
    new_console: bool = False,
    stdin=None,
    stdout=None,
    stderr=None,
) -> OwnedProcess:
    """Spawn an owned native process using an argument array and no shell."""
    normalized = _validate_argv(argv)
    family = platform_family(platform_name)
    kwargs: dict = {
        "cwd": str(cwd) if cwd is not None else None,
        "shell": False,
        "close_fds": True,
    }
    if family == "windows" and new_console:
        # Interactive PowerShell/cmd fallbacks need real standard handles for
        # their newly allocated console. DEVNULL would immediately deliver EOF
        # or create a terminal that cannot accept input.
        if stdin is not None:
            kwargs["stdin"] = stdin
        if stdout is not None:
            kwargs["stdout"] = stdout
        if stderr is not None:
            kwargs["stderr"] = stderr
    else:
        kwargs["stdin"] = subprocess.DEVNULL if stdin is None else stdin
        kwargs["stdout"] = subprocess.DEVNULL if stdout is None else stdout
        kwargs["stderr"] = subprocess.DEVNULL if stderr is None else stderr
    if family == "windows":
        create_new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        kwargs["creationflags"] = create_new_group | (create_new_console if new_console else detached)
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(list(normalized), **kwargs)
    return OwnedProcess(process, family, normalized, True)


def terminate_process_tree(handle: OwnedProcess, *, grace_seconds: float = 3.0) -> bool:
    """Terminate only a process tree represented by a Selene-owned handle."""
    if not isinstance(handle, OwnedProcess):
        raise ValueError("Process-tree termination requires a Selene-owned process handle.")
    process = handle.process
    if process.poll() is not None:
        return True
    grace = max(0.1, min(float(grace_seconds), 30.0))

    if handle.platform_name == "windows":
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(handle.pid), "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=grace,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            # A stripped-down Windows environment may not expose taskkill on
            # PATH. Fall back to the exact owned child; never enumerate or kill
            # processes by image name.
            try:
                process.terminate()
            except OSError:
                pass
    else:
        try:
            os.killpg(handle.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except OSError:
            process.terminate()

    try:
        process.wait(timeout=grace)
        return True
    except subprocess.TimeoutExpired:
        if handle.platform_name == "windows":
            try:
                subprocess.run(
                    ["taskkill.exe", "/PID", str(handle.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=grace,
                    shell=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
        else:
            try:
                os.killpg(handle.pid, signal.SIGKILL)
            except ProcessLookupError:
                return True
            except OSError:
                process.kill()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            return False
        return True


@dataclass(frozen=True)
class NativeOperationResult:
    ok: bool
    backend: str
    requested: bool
    message: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "ok": self.ok,
            "backend": self.backend,
            "requested": self.requested,
        }
        if self.message:
            result["message"] = self.message
        if self.error:
            result["error"] = self.error
        return result


def validate_http_url(value: object) -> str:
    """Validate a browser target without permitting shell or local-file schemes."""
    url = str(value or "").strip()
    if not url or len(url) > 4096 or any(ord(character) < 32 for character in url):
        raise ValueError("URL is empty or invalid.")
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"URL is invalid: {exc}") from None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ValueError("URL must use http or https and include a host.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing embedded credentials are not accepted.")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("URL port is outside the valid range.")
    return url


def open_url_native(value: object) -> NativeOperationResult:
    """Request exactly one launch in the user's native default browser."""
    try:
        url = validate_http_url(value)
    except ValueError as exc:
        return NativeOperationResult(False, "default-browser", False, error=str(exc))
    try:
        accepted = bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception as exc:
        return NativeOperationResult(False, "default-browser", False, error=f"Browser launch failed: {exc}")
    if not accepted:
        return NativeOperationResult(False, "default-browser", False, error="The operating system did not accept the browser launch request.")
    return NativeOperationResult(
        True,
        "default-browser",
        True,
        message=f"Sent one browser launch request for {url}; page loading was not verified.",
    )


def open_native_target(
    value: object,
    *,
    platform_name: str | None = None,
    allowed_uri_schemes: Collection[str] = (),
) -> NativeOperationResult:
    """Open a discovered local record or an explicitly allowed registered URI.

    Model-supplied paths and arbitrary URI schemes are intentionally not part of
    this contract. Callers must first discover a real local record, or opt into a
    narrow scheme such as ``spotify`` for an internally constructed URI.
    """
    target = str(value or "").strip()
    if not target or len(target) > 4096 or "\x00" in target:
        return NativeOperationResult(False, "native-shell", False, error="Native launch target is invalid.")
    family = platform_family(platform_name)
    normalized_schemes = {str(scheme).casefold() for scheme in allowed_uri_schemes}
    parsed = urllib.parse.urlsplit(target)
    windows_drive = family == "windows" and bool(PureWindowsPath(target).drive)
    if parsed.scheme and not windows_drive:
        if parsed.scheme.casefold() not in normalized_schemes:
            return NativeOperationResult(
                False,
                "native-shell",
                False,
                error=f"The registered URI scheme '{parsed.scheme}' is not allowed for this operation.",
            )
    else:
        local_target = Path(target).expanduser()
        if not local_target.exists():
            return NativeOperationResult(
                False,
                "native-shell",
                False,
                error="The discovered native launch target no longer exists.",
            )
        target = str(local_target.resolve(strict=True))
    try:
        if family == "windows":
            startfile = getattr(os, "startfile", None)
            if startfile is None:
                return NativeOperationResult(False, "windows-shell-execute", False, error="Windows ShellExecute is unavailable.")
            startfile(target)
            return NativeOperationResult(True, "windows-shell-execute", True, message="The Windows shell accepted the launch request; loading was not verified.")
        if family == "macos":
            spawn_detached([find_executable("open") or "/usr/bin/open", target], platform_name="macos")
            return NativeOperationResult(True, "macos-open", True, message="macOS accepted the launch request; loading was not verified.")
        executable = find_executable("gio")
        if not executable:
            return NativeOperationResult(False, "linux-gio", False, error="No supported native opener is installed.")
        spawn_detached([executable, "open", target], platform_name="linux")
        return NativeOperationResult(True, "linux-gio", True, message="The Linux desktop accepted the launch request; loading was not verified.")
    except OSError as exc:
        return NativeOperationResult(False, f"{family}-native-open", False, error=f"Native launch failed: {exc}")


def validate_filename_component(name: object, *, platform_name: str | None = None) -> str:
    """Validate one portable path component, including Windows reserved names."""
    value = str(name or "")
    if not value or value in {".", ".."} or len(value) > 255 or any(ord(char) < 32 for char in value):
        raise ValueError("Filename is empty or invalid.")
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError("Filename must be one path component.")
    if platform_family(platform_name) == "windows":
        if value.endswith((" ", ".")):
            raise ValueError("Windows filenames cannot end in a space or period.")
        if any(char in '<>:"|?*' for char in value):
            raise ValueError("Filename contains characters Windows does not allow.")
        device_name = value.split(".", 1)[0].rstrip(" .").upper()
        if device_name in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"'{value}' is a reserved Windows filename.")
    return value


def path_is_within(path_value: str | Path, root_value: str | Path, *, platform_name: str | None = None) -> bool:
    """Test path containment with native semantics, never raw string prefixes."""
    if platform_family(platform_name) == "windows":
        path_normalized = ntpath.normcase(ntpath.abspath(str(path_value)))
        root_normalized = ntpath.normcase(ntpath.abspath(str(root_value)))
        try:
            return ntpath.commonpath([path_normalized, root_normalized]) == root_normalized
        except ValueError:
            return False
    path = Path(path_value).expanduser().resolve(strict=False)
    root = Path(root_value).expanduser().resolve(strict=False)
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def capability_report(*, platform_name: str | None = None) -> dict[str, dict[str, object]]:
    """Report optional native capabilities without importing platform-only modules."""
    family = platform_family(platform_name)
    terminal = select_terminal_command(Path.home(), platform_name=family)
    dbus_available = family == "linux" and importlib.util.find_spec("dbus") is not None
    if family == "linux":
        spotify = {
            "backend": "linux-mpris-dbus",
            "available": dbus_available,
            "detail": "dbus-python is required for confirmed playback control.",
        }
    elif family == "windows":
        spotify = {
            "backend": "windows-uri-handler",
            "available": getattr(os, "startfile", None) is not None,
            "detail": "URI launch requests are supported; playback confirmation is unavailable.",
        }
    else:
        spotify = {
            "backend": "macos-automation",
            "available": bool(find_executable("osascript")),
            "detail": "Spotify automation depends on the installed desktop application.",
        }
    return {
        "platform": {"backend": family, "available": True},
        "terminal": {"backend": terminal.backend if terminal else "unavailable", "available": terminal is not None},
        "browser": {"backend": "default-browser", "available": True},
        "spotify": spotify,
    }
