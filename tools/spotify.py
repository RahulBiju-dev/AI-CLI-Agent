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

def _get_spotify_dbus():
    """Get the Spotify D-Bus MPRIS2 player interface."""
    import dbus

    bus = dbus.SessionBus()
    try:
        proxy = bus.get_object(
            "org.mpris.MediaPlayer2.spotify",
            "/org/mpris/MediaPlayer2",
        )
        player = dbus.Interface(proxy, "org.mpris.MediaPlayer2.Player")
        return player
    except dbus.exceptions.DBusException:
        # Try with flatpak instance name
        try:
            proxy = bus.get_object(
                "org.mpris.MediaPlayer2.spotify_player",
                "/org/mpris/MediaPlayer2",
            )
            player = dbus.Interface(proxy, "org.mpris.MediaPlayer2.Player")
            return player
        except dbus.exceptions.DBusException:
            return None


def _get_spotify_properties():
    """Get the Spotify D-Bus properties interface."""
    import dbus

    bus = dbus.SessionBus()
    for service in ("org.mpris.MediaPlayer2.spotify", "org.mpris.MediaPlayer2.spotify_player"):
        try:
            proxy = bus.get_object(service, "/org/mpris/MediaPlayer2")
            return dbus.Interface(proxy, "org.freedesktop.DBus.Properties")
        except dbus.exceptions.DBusException:
            continue
    return None


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
                player = _get_spotify_dbus()
            except (ImportError, ModuleNotFoundError):
                return json.dumps({
                    "error": "Spotify playback control requires dbus-python on Linux.",
                    "backend": backend,
                    "supported": False,
                    "missing_dependency": "dbus-python",
                })
            except Exception as exc:
                return json.dumps({
                    "error": f"The Fedora/Linux MPRIS session bus is unavailable: {exc}",
                    "backend": backend,
                    "supported": False,
                })
            if player:
                try:
                    player.OpenUri(uri)
                except Exception as exc:
                    return json.dumps({
                        "error": f"Spotify rejected the MPRIS playback request: {exc}",
                        "backend": backend,
                        "supported": True,
                    })

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

            return json.dumps({
                "error": "Could not connect to Spotify via the Fedora/Linux MPRIS D-Bus backend.",
                "backend": backend,
                "supported": False,
            })

        return json.dumps({
            "error": "Spotify control is not supported on this platform.",
            "backend": backend,
            "supported": False,
        })

    except Exception as exc:
        return json.dumps({"error": f"Spotify error: {str(exc)}"})
