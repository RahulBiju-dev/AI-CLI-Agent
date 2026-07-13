import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.spotify as spotify_tool
from agent.platform_runtime import NativeCommand, NativeOperationResult
from tools.app_launcher import (
    _find_windows_shortcut,
    _is_safe_desktop_entry,
    _validate_app_name,
)
from tools.browser import open_browser
from tools.spotify import spotify_play
from tools.terminal_launcher import open_terminal_at_path


class ApplicationLauncherSafetyTests(unittest.TestCase):
    def test_model_must_supply_display_name_not_path_or_command(self):
        for value in ("/usr/bin/editor", r"C:\Program Files\Editor.exe", "editor --unsafe"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_app_name(value)

    def test_shells_terminals_and_uninstallers_are_blocked_by_display_name(self):
        for value in ("Windows Terminal", "PowerShell ISE", "Git Bash", "Uninstall Editor"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_app_name(value)

    def test_linux_env_wrapper_cannot_hide_shell(self):
        self.assertFalse(_is_safe_desktop_entry({
            "name": "Suspicious",
            "exec": "env MODE=test bash -i",
            "terminal": False,
        }))
        self.assertTrue(_is_safe_desktop_entry({
            "name": "Editor",
            "exec": "env MODE=test /usr/bin/editor --new-window",
            "terminal": False,
        }))

    def test_windows_search_is_bounded_to_start_menu_and_filters_uninstaller(self):
        with tempfile.TemporaryDirectory() as temporary:
            start_menu = Path(temporary) / "Start Menu" / "Programs"
            start_menu.mkdir(parents=True)
            safe = start_menu / "Editor.lnk"
            safe.touch()
            (start_menu / "Uninstall Editor.lnk").touch()
            with patch("tools.app_launcher.windows_start_menu_dirs", return_value=(start_menu,)):
                match, suggestions = _find_windows_shortcut("Editor")
        self.assertEqual(match, safe)
        self.assertEqual(suggestions, [])

    def test_windows_search_filters_shell_shortcuts(self):
        with tempfile.TemporaryDirectory() as temporary:
            start_menu = Path(temporary)
            (start_menu / "Windows PowerShell.lnk").touch()
            with patch("tools.app_launcher.windows_start_menu_dirs", return_value=(start_menu,)):
                match, suggestions = _find_windows_shortcut("Windows PowerShell")
        self.assertIsNone(match)
        self.assertEqual(suggestions, [])


class TerminalAndBrowserTests(unittest.TestCase):
    def test_terminal_reports_backend_and_executes_no_user_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            selected = NativeCommand(("/usr/bin/kgx", "--working-directory", str(directory)), "GNOME Console")
            with (
                patch("tools.terminal_launcher.select_terminal_command", return_value=selected),
                patch("tools.terminal_launcher.spawn_detached") as spawn,
            ):
                result = json.loads(open_terminal_at_path(str(directory), confirmed=True))
        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "GNOME Console")
        spawn.assert_called_once_with(selected.argv, cwd=directory.resolve(), new_console=False)
        self.assertIn("no command was executed", result["message"])

    def test_browser_uses_one_native_request(self):
        native_result = NativeOperationResult(True, "default-browser", True, message="accepted; not verified")
        with patch("tools.browser.open_url_native", return_value=native_result) as native:
            result = json.loads(open_browser("example.com"))
        native.assert_called_once_with("https://example.com")
        self.assertTrue(result["ok"])
        self.assertEqual(result["url"], "https://example.com")


class SpotifyBackendTests(unittest.TestCase):
    def test_mpris_discovery_includes_instance_qualified_spotify_names(self):
        bus = MagicMock()
        bus.list_names.return_value = [
            ":1.42",
            "org.mpris.MediaPlayer2.vlc",
            "org.mpris.MediaPlayer2.vlc.instance_spotify",
            "org.mpris.MediaPlayer2.spotify.instance_2048",
            "org.mpris.MediaPlayer2.com.spotify.Client.instance_4096",
        ]

        services = spotify_tool._spotify_mpris_service_names(bus)

        self.assertIn("org.mpris.MediaPlayer2.spotify.instance_2048", services)
        self.assertIn("org.mpris.MediaPlayer2.com.spotify.Client.instance_4096", services)
        self.assertNotIn("org.mpris.MediaPlayer2.vlc", services)
        self.assertNotIn("org.mpris.MediaPlayer2.vlc.instance_spotify", services)

    def test_mpris_interface_uses_discovered_instance_name(self):
        class FakeDBusException(Exception):
            pass

        service = "org.mpris.MediaPlayer2.spotify.instance_2048"
        proxy = object()
        player = object()
        bus = MagicMock()
        bus.list_names.return_value = [service]

        def get_object(name, object_path):
            self.assertEqual(object_path, "/org/mpris/MediaPlayer2")
            if name != service:
                raise FakeDBusException(name)
            return proxy

        bus.get_object.side_effect = get_object
        fake_dbus = SimpleNamespace(
            SessionBus=MagicMock(return_value=bus),
            Interface=MagicMock(return_value=player),
            exceptions=SimpleNamespace(DBusException=FakeDBusException),
        )

        with patch.dict("sys.modules", {"dbus": fake_dbus}):
            result = spotify_tool._get_spotify_dbus()

        self.assertIs(result, player)
        fake_dbus.Interface.assert_called_once_with(
            proxy, "org.mpris.MediaPlayer2.Player"
        )

    def test_windows_uri_backend_never_claims_playback_confirmation(self):
        accepted = NativeOperationResult(True, "windows-shell-execute", True, message="accepted")
        with (
            patch("tools.spotify._spotify_backend", return_value="windows-uri-handler"),
            patch("tools.spotify.open_native_target", return_value=accepted) as native,
        ):
            result = json.loads(spotify_play("spotify:track:abc123"))
        self.assertTrue(result["success"])
        self.assertFalse(result["playback_confirmed"])
        self.assertEqual(result["capability"], "uri-launch-only")
        native.assert_called_once_with(
            "spotify:track:abc123",
            platform_name="windows",
            allowed_uri_schemes={"spotify"},
        )

    def test_missing_linux_dbus_is_an_explicit_capability_result(self):
        with (
            patch("tools.spotify._spotify_backend", return_value="linux-mpris-dbus"),
            patch("tools.spotify._launch_spotify", return_value=True),
            patch("tools.spotify._get_spotify_dbus", side_effect=ModuleNotFoundError("dbus")),
        ):
            result = json.loads(spotify_play("spotify:track:abc123"))
        self.assertFalse(result["supported"])
        self.assertEqual(result["missing_dependency"], "dbus-python")

    def test_fedora_open_uri_survives_metadata_failure_truthfully(self):
        player = MagicMock()
        with (
            patch("tools.spotify._spotify_backend", return_value="linux-mpris-dbus"),
            patch("tools.spotify._launch_spotify", return_value=True),
            patch("tools.spotify._get_spotify_dbus", return_value=player),
            patch("tools.spotify._get_spotify_properties", side_effect=RuntimeError("session changed")),
            patch("tools.spotify.time.sleep"),
        ):
            result = json.loads(spotify_play("spotify:track:abc123"))
        player.OpenUri.assert_called_once_with("spotify:track:abc123")
        self.assertTrue(result["success"])
        self.assertFalse(result["playback_confirmed"])
        self.assertIn("not available", result["message"])

    def test_fedora_waits_for_delayed_mpris_registration(self):
        player = MagicMock()
        with (
            patch("tools.spotify._spotify_backend", return_value="linux-mpris-dbus"),
            patch("tools.spotify._launch_spotify", return_value=True),
            patch("tools.spotify._get_spotify_dbus", side_effect=[None, player]) as get_player,
            patch("tools.spotify._get_spotify_properties", return_value=None),
            patch("tools.spotify.time.monotonic", side_effect=[10.0, 10.0]),
            patch("tools.spotify.time.sleep"),
        ):
            result = json.loads(spotify_play("spotify:track:abc123"))

        self.assertEqual(get_player.call_count, 2)
        player.OpenUri.assert_called_once_with("spotify:track:abc123")
        self.assertTrue(result["success"])

    def test_fedora_uses_native_uri_fallback_without_claiming_playback(self):
        accepted = NativeOperationResult(
            True, "linux-gio", True, message="accepted; not verified"
        )
        with (
            patch("tools.spotify._spotify_backend", return_value="linux-mpris-dbus"),
            patch("tools.spotify._launch_spotify", return_value=True),
            patch("tools.spotify._get_spotify_dbus", return_value=None),
            patch("tools.spotify.time.monotonic", side_effect=[10.0, 16.0]),
            patch("tools.spotify.open_native_target", return_value=accepted) as native,
        ):
            result = json.loads(spotify_play("spotify:track:abc123"))

        native.assert_called_once_with(
            "spotify:track:abc123",
            platform_name="linux",
            allowed_uri_schemes={"spotify"},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "linux-gio")
        self.assertEqual(result["primary_backend"], "linux-mpris-dbus")
        self.assertFalse(result["playback_confirmed"])

    def test_fedora_sanitizes_dbus_and_native_fallback_failures(self):
        rejected = NativeOperationResult(
            False,
            "linux-gio",
            False,
            error="secret socket and process detail",
        )
        with (
            patch("tools.spotify._spotify_backend", return_value="linux-mpris-dbus"),
            patch("tools.spotify._launch_spotify", return_value=True),
            patch(
                "tools.spotify._wait_for_spotify_dbus",
                side_effect=RuntimeError("secret D-Bus owner detail"),
            ),
            patch("tools.spotify.open_native_target", return_value=rejected),
        ):
            result = json.loads(spotify_play("spotify:track:abc123"))

        self.assertFalse(result["supported"])
        self.assertIn("MPRIS session bus could not be reached", result["error"])
        self.assertNotIn("secret", result["error"])


if __name__ == "__main__":
    unittest.main()
