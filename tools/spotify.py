"""
tools/spotify.py — Spotify desktop control via D-Bus (MPRIS2).

Allows the agent to launch Spotify, play specific songs/albums/playlists,
and control playback on the local Linux desktop.

Requirements:
    - Spotify installed (Flatpak, Snap, or native)
    - dbus-python (system package, usually pre-installed on Fedora/GNOME)
    - xdg-open available (standard on Linux desktops)
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


# ── URI resolution ────────────────────────────────────────────────────

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
    """Use a web search to find a Spotify URI for a given query.

    Searches DuckDuckGo for the query on open.spotify.com and extracts
    the first Spotify URL found.
    """
    from tools.search import web_search

    search_query = f"site:open.spotify.com {query}"
    results_json = web_search(search_query)

    try:
        results = json.loads(results_json)
    except json.JSONDecodeError:
        return None

    if isinstance(results, dict) and "error" in results:
        return None

    # Look through results for a Spotify URL
    for result in results:
        snippet = result.get("snippet", "") + " " + result.get("title", "")
        urls = re.findall(
            r"https?://open\.spotify\.com/(track|album|playlist|artist)/[a-zA-Z0-9]+",
            snippet,
        )
        if urls:
            full_url = re.search(
                r"https?://open\.spotify\.com/(track|album|playlist|artist)/[a-zA-Z0-9]+",
                snippet,
            )
            if full_url:
                return _url_to_uri(full_url.group(0))

    return None


# ── Public tool function ──────────────────────────────────────────────

def spotify_play(query: str, content_type: str = "track") -> str:
    """Open Spotify and play a song, album, or playlist.

    Args:
        query: A Spotify URI, Spotify URL, or a search query
               (e.g. "Bohemian Rhapsody Queen", "Chill Vibes playlist").
        content_type: One of "track", "album", "playlist", or "artist".
                      Helps refine search when query is not a URI/URL.

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
            # Direct URI or URL provided
            uri = _url_to_uri(query) if query.startswith("http") else query
        else:
            # Search for the content
            search_suffix = {
                "track": "",
                "album": " album",
                "playlist": " playlist",
                "artist": " artist",
            }
            refined_query = query + search_suffix.get(content_type, "")
            uri = _search_spotify_uri(refined_query)

            if not uri:
                # Fallback: open Spotify's own search
                uri = f"spotify:search:{query.replace(' ', '%20')}"

        # Step 3: Play the content via D-Bus OpenUri
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
            except Exception as dbus_err:
                # D-Bus failed — fall back to xdg-open
                pass

        # Fallback: use xdg-open
        subprocess.Popen(
            ["xdg-open", uri],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return json.dumps({
            "success": True,
            "message": f"Opened Spotify with URI: {uri}",
            "uri": uri,
        })

    except Exception as exc:
        return json.dumps({"error": f"Spotify error: {str(exc)}"})
