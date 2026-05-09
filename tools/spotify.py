"""
tools/spotify.py — Spotify desktop control via D-Bus (MPRIS2).

Allows the agent to launch Spotify and play specific songs
on the local Linux desktop.

Requirements:
    - Spotify installed (Flatpak, Snap, or native)
    - dbus-python (system package, usually pre-installed on Fedora/GNOME)
"""

import json
import subprocess
import time
import shutil
import re


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


def _is_spotify_running() -> bool:
    """Check if Spotify is currently running."""
    try:
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

    cmd = _find_spotify_command()
    if not cmd:
        return False

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Wait for Spotify to start and register on D-Bus
        for _ in range(15):
            time.sleep(1)
            if _is_spotify_running():
                # Give it a moment to register the MPRIS interface
                time.sleep(2)
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
    try:
        proxy = bus.get_object(
            "org.mpris.MediaPlayer2.spotify",
            "/org/mpris/MediaPlayer2",
        )
        props = dbus.Interface(proxy, "org.freedesktop.DBus.Properties")
        return props
    except dbus.exceptions.DBusException:
        return None


# ── URI helpers ───────────────────────────────────────────────────────

def _is_spotify_uri(text: str) -> bool:
    """Check if the text is a Spotify URI or URL."""
    return bool(
        re.match(r"^spotify:(track|album|playlist|artist|show|episode):", text)
        or re.match(r"^https?://open\.spotify\.com/", text)
    )


def _url_to_uri(url: str) -> str:
    """Convert a Spotify URL to a Spotify URI.

    Example:
        https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC
        → spotify:track:4uLU6hMCjMI75M1A2tKUQC
    """
    match = re.match(
        r"https?://open\.spotify\.com/(track|album|playlist|artist|show|episode)/([a-zA-Z0-9]+)",
        url,
    )
    if match:
        return f"spotify:{match.group(1)}:{match.group(2)}"
    return url


def _search_spotify_uri(query: str) -> str | None:
    """Search DuckDuckGo to find a Spotify track URI for a given query.

    Queries ddgs directly to access the href field from results,
    which contains the actual Spotify URL.
    """
    from ddgs import DDGS

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
    """Open Spotify and play a specific song.

    Args:
        query: A Spotify URI, Spotify URL, or a search query
               (e.g. "Bohemian Rhapsody Queen").

    Returns:
        A JSON string indicating success or failure.
    """
    try:
        # Step 1: Ensure Spotify is running
        if not _launch_spotify():
            return json.dumps({
                "error": "Could not launch Spotify. Is it installed?"
            })

        # Step 2: Resolve the URI
        if _is_spotify_uri(query):
            uri = _url_to_uri(query) if query.startswith("http") else query
        else:
            uri = _search_spotify_uri(query)
            if not uri:
                # Fallback: open Spotify's own search
                uri = f"spotify:search:{query.replace(' ', '%20')}"

        # Step 3: Play via D-Bus OpenUri
        player = _get_spotify_dbus()
        if player:
            try:
                player.OpenUri(uri)
                time.sleep(1)

                # Get current track info for confirmation
                props = _get_spotify_properties()
                if props:
                    metadata = props.Get(
                        "org.mpris.MediaPlayer2.Player", "Metadata"
                    )
                    title = str(metadata.get("xesam:title", "Unknown"))
                    artists = metadata.get("xesam:artist", ["Unknown"])
                    artist = ", ".join(str(a) for a in artists)
                    album = str(metadata.get("xesam:album", "Unknown"))

                    return json.dumps({
                        "success": True,
                        "message": f"Now playing: {title} by {artist}",
                        "track": title,
                        "artist": artist,
                        "album": album,
                        "uri": uri,
                    })

                return json.dumps({
                    "success": True,
                    "message": f"Opened Spotify with URI: {uri}",
                    "uri": uri,
                })
            except Exception:
                pass

        return json.dumps({
            "error": "Could not connect to Spotify via D-Bus. Is Spotify running?"
        })

    except Exception as exc:
        return json.dumps({"error": f"Spotify error: {str(exc)}"})
