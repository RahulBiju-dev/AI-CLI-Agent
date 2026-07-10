import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.platform_runtime import (
    NativeCommand,
    NativeOperationResult,
    OwnedProcess,
    get_runtime_paths,
    open_native_target,
    open_url_native,
    path_is_within,
    resolve_runtime_paths,
    select_terminal_command,
    spawn_detached,
    terminate_process_tree,
    validate_filename_component,
    validate_http_url,
)


class RuntimePathTests(unittest.TestCase):
    def test_explicit_data_directory_has_precedence_over_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            selected = Path(temporary) / "explicit data"
            paths = resolve_runtime_paths(
                platform_name="linux",
                environ={"HOME": str(home), "SELENE_DATA_DIR": str(selected)},
                home=home,
                legacy_exists=True,
            )
        self.assertEqual(paths.data_dir, selected.absolute())
        self.assertEqual(paths.source, "SELENE_DATA_DIR")

    def test_existing_legacy_store_is_used_without_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            paths = resolve_runtime_paths(
                platform_name="linux", environ={"HOME": str(home)}, home=home, legacy_exists=True
            )
        self.assertEqual(paths.data_dir, home / ".selene-agent")
        self.assertEqual(paths.source, "existing-legacy-store")

    def test_empty_xdg_values_use_fedora_compatible_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            paths = get_runtime_paths(
                platform_name="linux",
                environ={
                    "HOME": str(home),
                    "XDG_DATA_HOME": "",
                    "XDG_STATE_HOME": "",
                    "XDG_CONFIG_HOME": "",
                    "XDG_CACHE_HOME": "",
                },
                home=home,
                legacy_exists=False,
            )
        self.assertEqual(paths.data_dir, home / ".local" / "share" / "selene")
        self.assertEqual(paths.state_dir, home / ".local" / "state" / "selene")
        self.assertEqual(paths.config_dir, home / ".config" / "selene")
        self.assertEqual(paths.cache_dir, home / ".cache" / "selene")

    def test_relative_native_runtime_roots_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            linux = resolve_runtime_paths(
                platform_name="linux",
                environ={"HOME": str(home), "XDG_DATA_HOME": "relative-data"},
                home=home,
                legacy_exists=False,
            )
            windows = resolve_runtime_paths(
                platform_name="windows",
                environ={"USERPROFILE": str(home), "LOCALAPPDATA": "relative-data"},
                home=home,
                legacy_exists=False,
            )

        self.assertEqual(linux.data_dir, home / ".local" / "share" / "selene")
        self.assertEqual(windows.data_dir, home / "AppData" / "Local" / "Selene")

    def test_windows_uses_local_app_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = resolve_runtime_paths(
                platform_name="win32",
                environ={"USERPROFILE": str(root / "User"), "LOCALAPPDATA": str(root / "Local AppData")},
                home=root / "User",
                legacy_exists=False,
            )
        self.assertEqual(paths.data_dir, root / "Local AppData" / "Selene")
        self.assertEqual(paths.source, "windows-localappdata")

    def test_compatibility_candidates_do_not_copy_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = resolve_runtime_paths(
                platform_name="linux",
                environ={"HOME": str(root), "XDG_DATA_HOME": str(root / "xdg")},
                home=root,
                legacy_exists=False,
            )
            candidates = paths.compatibility_candidates("sessions/chat.json")
            self.assertEqual(candidates, (
                root / "xdg" / "selene" / "sessions" / "chat.json",
                root / ".selene-agent" / "sessions" / "chat.json",
            ))
            self.assertFalse(candidates[0].exists())


class FilesystemSafetyTests(unittest.TestCase):
    def test_windows_reserved_and_trailing_names_are_rejected(self):
        for name in ("CON", "con.txt", "COM9.log", "LPT1", "report. ", "report."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_filename_component(name, platform_name="windows")
        self.assertEqual(validate_filename_component("résumé.txt", platform_name="windows"), "résumé.txt")

    def test_windows_containment_is_case_insensitive_and_drive_aware(self):
        self.assertTrue(path_is_within(
            r"C:\Users\Rahul\Selene\Sessions\A.json",
            r"c:\users\rahul\selene",
            platform_name="windows",
        ))
        self.assertFalse(path_is_within(
            r"C:\Users\Rahul\Selene-Other\A.json",
            r"C:\Users\Rahul\Selene",
            platform_name="windows",
        ))
        self.assertFalse(path_is_within(
            r"D:\Selene\A.json", r"C:\Selene", platform_name="windows"
        ))


class NativeBackendTests(unittest.TestCase):
    @staticmethod
    def _which(mapping):
        return lambda name: mapping.get(name)

    def test_fedora_terminal_preference_starts_with_gnome_console(self):
        selected = select_terminal_command(
            "/tmp/Unicode Folder",
            platform_name="linux",
            which=self._which({"kgx": "/usr/bin/kgx", "gnome-terminal": "/usr/bin/gnome-terminal"}),
        )
        self.assertEqual(selected, NativeCommand(
            ("/usr/bin/kgx", "--working-directory", "/tmp/Unicode Folder"), "GNOME Console"
        ))

    def test_windows_terminal_falls_back_without_executing_a_command(self):
        selected = select_terminal_command(
            r"C:\Users\Rahul\Unicode Folder",
            platform_name="windows",
            which=self._which({"powershell.exe": r"C:\Windows\powershell.exe"}),
        )
        self.assertEqual(selected.argv, (r"C:\Windows\powershell.exe",))
        self.assertEqual(selected.backend, "Windows PowerShell")
        self.assertTrue(selected.new_console)

    @patch("agent.platform_runtime.subprocess.Popen")
    def test_spawn_detached_uses_argument_array_and_no_shell(self, popen):
        process = MagicMock(pid=123)
        popen.return_value = process
        handle = spawn_detached(["/usr/bin/example", "space value"], platform_name="linux")
        self.assertIsInstance(handle, OwnedProcess)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["/usr/bin/example", "space value"])
        self.assertIs(kwargs["shell"], False)
        self.assertIs(kwargs["start_new_session"], True)

    @patch("agent.platform_runtime.subprocess.Popen")
    def test_windows_new_console_keeps_interactive_standard_handles(self, popen):
        popen.return_value = MagicMock(pid=321)

        spawn_detached(["powershell.exe"], platform_name="windows", new_console=True)

        _, kwargs = popen.call_args
        self.assertNotIn("stdin", kwargs)
        self.assertNotIn("stdout", kwargs)
        self.assertNotIn("stderr", kwargs)
        self.assertTrue(kwargs["creationflags"])

    @patch("agent.platform_runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_windows_tree_kill_falls_back_to_exact_owned_child(self, _run):
        process = MagicMock(pid=456)
        process.poll.return_value = None
        process.wait.return_value = 0
        handle = OwnedProcess(process, "windows", ("owned.exe",), True)
        self.assertTrue(terminate_process_tree(handle, grace_seconds=0.2))
        process.terminate.assert_called_once_with()

    def test_tree_kill_rejects_unowned_processes(self):
        with self.assertRaises(ValueError):
            terminate_process_tree(MagicMock())

    def test_browser_url_validation_rejects_credentials(self):
        with self.assertRaises(ValueError):
            validate_http_url("https://user:secret@example.com/")

    @patch("agent.platform_runtime.webbrowser.open", return_value=True)
    def test_browser_launch_is_one_truthful_request(self, browser_open):
        result = open_url_native("https://example.com/")
        self.assertTrue(result.ok)
        self.assertTrue(result.requested)
        browser_open.assert_called_once_with("https://example.com/", new=2, autoraise=True)
        self.assertIn("not verified", result.message)

    def test_native_uri_requires_explicit_scheme_allowlist(self):
        result = open_native_target("spotify:track:abc", platform_name="windows")
        self.assertFalse(result.ok)
        self.assertIn("not allowed", result.error)

    def test_windows_native_open_accepts_only_existing_discovered_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            shortcut = Path(temporary) / "Editor.lnk"
            shortcut.touch()
            with patch.object(os, "startfile", create=True) as startfile:
                result = open_native_target(shortcut, platform_name="windows")
        self.assertTrue(result.ok)
        startfile.assert_called_once_with(str(shortcut.resolve()))


if __name__ == "__main__":
    unittest.main()
