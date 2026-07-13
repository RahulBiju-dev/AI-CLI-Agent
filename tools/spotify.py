"""
tools/spotify.py — Spotify desktop control.

Allows the agent to launch Spotify and play specific songs
on Windows, macOS, and Linux (via MPRIS2 D-Bus).

Requirements:
    - Spotify installed
    - dbus-python (Linux only, usually pre-installed on Fedora/GNOME)
"""

import json
import subprocess
import time
import shutil
import re
import urllib.parse

from agent.platform_runtime import open_native_target, platform_family, spawn_detached


_MPRIS_BUS_PREFIX = "org.mpris.MediaPlayer2."
_MPRIS_OBJECT_PATH = "/org/mpris/MediaPlayer2"
_MPRIS_PLAYER_INTERFACE = "org.mpris.MediaPlayer2.Player"
_MPRIS_PROPERTIES_INTERFACE = "org.freedesktop.DBus.Properties"
_KNOWN_SPOTIFY_MPRIS_NAMES = (
    "org.mpris.MediaPlayer2.spotify",
    "org.mpris.MediaPlayer2.spotify_player",
    "org.mpris.MediaPlayer2.com.spotify.Client",
)
_SPOTIFY_MPRIS_PLAYER_NAMES = (
    "spotify",
    "spotify_player",
    "com.spotify.client",
)


# ── Spotify launch helpers ────────────────────────────────────────────

def _find_spotify_command() -> list[str]:
    """Return the shell command list to launch Spotify."""
    # Check native install first
    if shutil.which("spotify"):
        return ["spotify"]

    # Flatpak
    try:
        result = subprocess.run(
            ["flatpak", "list", "--app", "--columns=application"],
            capture_output=True, text=True, timeout=5,
        )
        if "com.spotify.Client" in result.stdout:
            return ["flatpak", "run", "com.spotify.Client"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Snap
    try:
        result = subprocess.run(
            ["snap", "list", "spotify"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return ["snap", "run", "spotify"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return []


def _spotify_backend(platform_name: str | None = None) -> str:
    """Return the truthful native capability used on the current platform."""
    family = platform_family(platform_name)
    if family == "linux":
        return "linux-mpris-dbus"
    if family == "windows":
        return "windows-uri-handler"
    if family == "macos":
        return "macos-automation"
    return "unsupported"


def _is_spotify_running() -> bool:
    """Check if Spotify is currently running."""
    try:
        family = platform_family()
        if family == "windows":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Spotify.exe", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return "Spotify.exe" in result.stdout
        elif family == "macos":
            result = subprocess.run(
                ["pgrep", "-x", "Spotify"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        else:
            result = subprocess.run(
                ["pgrep", "-f", "spotify"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _launch_spotify() -> bool:
    """Launch Spotify if not already running. Returns True on success."""
    if _is_spotify_running():
        return True

    try:
        family = platform_family()
        if family == "windows":
            # ShellExecute handles registered URI schemes without cmd.exe or
            # an interpolated command string. Acceptance is not proof of load.
            return open_native_target(
                "spotify:",
                platform_name="windows",
                allowed_uri_schemes={"spotify"},
            ).ok
        elif family == "macos":
            spawn_detached(["open", "-a", "Spotify"], platform_name="macos")
        else:
            cmd = _find_spotify_command()
            if not cmd:
                return False

            spawn_detached(cmd, platform_name="linux")

        # Poll briefly instead of imposing a fixed startup sleep.
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if _is_spotify_running():
                time.sleep(1)
                return True
        return False
    except Exception:
        return False


# ── D-Bus helpers ─────────────────────────────────────────────────────

def _spotify_mpris_service_names(bus) -> tuple[str, ...]:
    """Return active Spotify MPRIS names before compatibility fallbacks.

    MPRIS permits instance-qualified names such as
    ``org.mpris.MediaPlayer2.spotify.instance123``. Flatpak applications may
    also use an application-ID-shaped player name. Enumerating the session bus
    avoids assuming that every Spotify package owns one fixed name.
    """
    try:
        active_names = {str(name) for name in bus.list_names()}
    except Exception:
        # Some constrained D-Bus proxies do not allow enumeration. The known
        # names below remain useful and preserve the old behavior.
        active_names = set()

    discovered = []
    for name in active_names:
        if not name.startswith(_MPRIS_BUS_PREFIX):
            continue
        suffix = name[len(_MPRIS_BUS_PREFIX):].casefold()
        if any(
            suffix == player_name or suffix.startswith(f"{player_name}.")
            for player_name in _SPOTIFY_MPRIS_PLAYER_NAMES
        ):
            discovered.append(name)

    ordered = []
    for name in (*_KNOWN_SPOTIFY_MPRIS_NAMES, *sorted(discovered)):
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def _get_spotify_interface(interface_name: str):
    """Return one interface from any active Spotify MPRIS service."""
    import dbus

    bus = dbus.SessionBus()
    for service in _spotify_mpris_service_names(bus):
        try:
            proxy = bus.get_object(service, _MPRIS_OBJECT_PATH)
            return dbus.Interface(proxy, interface_name)
        except dbus.exceptions.DBusException:
            continue
    return None


def _get_spotify_dbus():
    """Get the Spotify D-Bus MPRIS2 player interface."""
    return _get_spotify_interface(_MPRIS_PLAYER_INTERFACE)


def _get_spotify_properties():
    """Get the Spotify D-Bus properties interface."""
    return _get_spotify_interface(_MPRIS_PROPERTIES_INTERFACE)


def _wait_for_spotify_dbus(timeout_seconds: float = 6.0, poll_interval: float = 0.25):
    """Wait briefly for Spotify to publish its MPRIS service after launch."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        player = _get_spotify_dbus()
        if player is not None:
            return player
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(poll_interval, remaining))


def _linux_uri_fallback_result(uri: str, primary_backend: str, reason: str) -> dict:
    """Try Fedora's native URI handler without claiming playback succeeded."""
    fallback = open_native_target(
        uri,
        platform_name="linux",
        allowed_uri_schemes={"spotify"},
    )
    if fallback.ok:
        return {
            "success": True,
            "backend": fallback.backend,
            "primary_backend": primary_backend,
            "supported": True,
            "capability": "uri-launch-fallback",
            "launch_requested": True,
            "playback_confirmed": False,
            "message": (
                f"{reason} Linux accepted a native URI launch request for {uri}; "
                "playback was not verified."
            ),
            "uri": uri,
        }
    return {
        "error": (
            f"{reason} The Fedora/Linux native URI fallback was not accepted."
        ),
        "backend": primary_backend,
        "fallback_backend": fallback.backend,
        "supported": False,
    }


# ── URI helpers ───────────────────────────────────────────────────────

def _is_spotify_uri(text: str) -> bool:
    """Check if the text is a Spotify URI or URL."""
    return bool(
        re.fullmatch(r"spotify:(track|album|playlist|artist|show|episode):[a-zA-Z0-9]+", text)
        or re.fullmatch(
            r"https?://open\.spotify\.com/(track|album|playlist|artist|show|episode)/[a-zA-Z0-9]+(?:\?[^\s]*)?",
            text,
        )
    )


def _url_to_uri(url: str) -> str:
    """Convert a Spotify URL to a Spotify URI.

    Example:
        https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC
        → spotify:track:4uLU6hMCjMI75M1A2tKUQC
    """
    match = re.fullmatch(
        r"https?://open\.spotify\.com/(track|album|playlist|artist|show|episode)/([a-zA-Z0-9]+)",
        url.split("?", 1)[0],
    )
    if match:
        return f"spotify:{match.group(1)}:{match.group(2)}"
    return url


def _search_spotify_uri(query: str) -> str | None:
    """Search DuckDuckGo to find a Spotify track URI for a given query.

    Queries ddgs directly to access the href field from results,
    which contains the actual Spotify URL.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return None

    search_query = f"site:open.spotify.com/track {query}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
    except Exception:
        return None

    for r in results:
        href = r.get("href", "")
        match = re.match(
            r"https?://open\.spotify\.com/track/([a-zA-Z0-9]+)",
            href,
        )
        if match:
            return f"spotify:track:{match.group(1)}"

    return None


# ── Public tool function ──────────────────────────────────────────────

def spotify_play(query: str) -> str:
    """
    Open Spotify and play a specific song, album, or playlist.

    This function attempts to launch the Spotify application if it isn't running,
    resolves the provided query to a Spotify URI (either natively or via search),
    and initiates playback using OS-specific mechanisms (e.g., cmd on Windows,
    osascript on macOS, or D-Bus MPRIS2 on Linux).

    Args:
        query (str): A Spotify URI, Spotify URL, or a text search query
               (e.g., "Bohemian Rhapsody Queen").

    Returns:
        str: A JSON-encoded string indicating the success or failure of the
             playback attempt. On success, it may include currently playing
             metadata depending on the OS capabilities.
    """
    try:
        query = str(query or "").strip()
        if not query:
            return json.dumps({"error": "A Spotify URI, URL, or search query is required."})
        if len(query) > 500:
            return json.dumps({"error": "Spotify query exceeds the 500-character limit."})
        if any(ord(character) < 32 for character in query):
            return json.dumps({"error": "Spotify query contains invalid control characters."})
        # Resolve the URI before launching so Windows performs one native URI
        # request instead of opening Spotify twice.
        if _is_spotify_uri(query):
            uri = _url_to_uri(query) if query.startswith("http") else query
        else:
            uri = _search_spotify_uri(query)
            if not uri:
                # Fallback: open Spotify's own search
                uri = f"spotify:search:{urllib.parse.quote(query, safe='')}"

        backend = _spotify_backend()
        if backend == "windows-uri-handler":
            launch = open_native_target(
                uri,
                platform_name="windows",
                allowed_uri_schemes={"spotify"},
            )
            if not launch.ok:
                return json.dumps({
                    **launch.as_dict(),
                    "supported": False,
                    "capability": "uri-launch-only",
                    "error": launch.error or "No Windows application accepted the Spotify URI.",
                })
            return json.dumps({
                "success": True,
                "backend": backend,
                "supported": True,
                "capability": "uri-launch-only",
                "launch_requested": True,
                "playback_confirmed": False,
                "message": f"Windows accepted a Spotify URI launch request for {uri}; playback was not verified.",
                "uri": uri,
            })
        if backend == "macos-automation":
            script = 'on run argv\ntell application "Spotify" to play track (item 1 of argv)\nend run'
            completed = subprocess.run(["osascript", "-e", script, "--", uri], capture_output=True, timeout=10)
            if completed.returncode != 0:
                return json.dumps({"error": "Spotify rejected the playback request on macOS."})
            return json.dumps({
                "success": True,
                "backend": backend,
                "message": f"Opened Spotify with URI: {uri}",
                "uri": uri,
            })
        if backend == "linux-mpris-dbus":
            if not _launch_spotify():
                return json.dumps({
                    "error": "Could not launch Spotify through a native package, Flatpak, or Snap installation.",
                    "backend": backend,
                    "supported": False,
                })
            try:
                player = _wait_for_spotify_dbus()
            except (ImportError, ModuleNotFoundError):
                return json.dumps({
                    "error": "Spotify playback control requires dbus-python on Linux.",
                    "backend": backend,
                    "supported": False,
                    "missing_dependency": "dbus-python",
                })
            except Exception:
                return json.dumps(_linux_uri_fallback_result(
                    uri,
                    backend,
                    "The Fedora/Linux MPRIS session bus could not be reached.",
                ))
            if player:
                try:
                    player.OpenUri(uri)
                except Exception:
                    return json.dumps(_linux_uri_fallback_result(
                        uri,
                        backend,
                        "Spotify did not accept the MPRIS playback request.",
                    ))

                time.sleep(1)
                # Metadata confirms playback when available, but its absence
                # must not turn an accepted OpenUri request into false failure.
                try:
                    props = _get_spotify_properties()
                    if props is not None:
                        metadata = props.Get(
                            "org.mpris.MediaPlayer2.Player", "Metadata"
                        )
                        title = str(metadata.get("xesam:title", "Unknown"))
                        artists = metadata.get("xesam:artist", ["Unknown"])
                        artist = ", ".join(str(a) for a in artists)
                        album = str(metadata.get("xesam:album", "Unknown"))

                        return json.dumps({
                            "success": True,
                            "backend": backend,
                            "playback_confirmed": True,
                            "message": f"Now playing: {title} by {artist}",
                            "track": title,
                            "artist": artist,
                            "album": album,
                            "uri": uri,
                        })
                except Exception:
                    # The playback request was accepted; only confirmation
                    # metadata failed, so report that distinction truthfully.
                    pass

                return json.dumps({
                    "success": True,
                    "backend": backend,
                    "supported": True,
                    "playback_confirmed": False,
                    "message": f"Spotify accepted an MPRIS OpenUri request for {uri}; playback metadata was not available.",
                    "uri": uri,
                })

            # A Spotify package can have a working desktop URI handler while
            # its MPRIS service is unavailable (or hidden by a package
            # sandbox). Preserve Fedora's native desktop path as a truthful,
            # unconfirmed fallback instead of reporting a total failure.
            return json.dumps(_linux_uri_fallback_result(
                uri,
                backend,
                "Spotify did not publish an MPRIS service.",
            ))

        return json.dumps({
            "error": "Spotify control is not supported on this platform.",
            "backend": backend,
            "supported": False,
        })

    except Exception as exc:
        return json.dumps({"error": f"Spotify error: {str(exc)}"})
