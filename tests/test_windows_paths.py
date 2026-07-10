"""Windows path safety tests (run on any host via pure helpers)."""

from __future__ import annotations

import unittest
from pathlib import Path, PureWindowsPath

from agent.platform_runtime import path_is_within, validate_filename_component
from tools.file import create_file
import json


class WindowsFilenameTests(unittest.TestCase):
    def test_reserved_device_names_rejected(self):
        reserved = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM9", "LPT1", "LPT9",
            "con.txt", "nul.log", "COM1.data",
        ]
        for name in reserved:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_filename_component(name, platform_name="windows")

    def test_trailing_space_and_period_rejected(self):
        for name in ("report ", "report.", "notes ."):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_filename_component(name, platform_name="windows")

    def test_valid_windows_names_accepted(self):
        for name in ("Notes.md", "My File.txt", "data-2024.csv", "résumé.docx"):
            with self.subTest(name=name):
                self.assertEqual(
                    validate_filename_component(name, platform_name="windows"),
                    name,
                )

    def test_create_file_rejects_reserved_names(self):
        result = json.loads(create_file("CON.txt", "x"))
        self.assertIn("error", result)


class WindowsContainmentTests(unittest.TestCase):
    def test_case_insensitive_drive_aware_containment(self):
        self.assertTrue(
            path_is_within(
                r"C:\Users\Test\Docs\file.txt",
                r"c:\users\test",
                platform_name="windows",
            )
        )
        self.assertFalse(
            path_is_within(
                r"D:\Users\Test\Docs\file.txt",
                r"C:\Users\Test",
                platform_name="windows",
            )
        )
        self.assertFalse(
            path_is_within(
                r"C:\Users\TestExtra\file.txt",
                r"C:\Users\Test",
                platform_name="windows",
            )
        )

    def test_mixed_separators_normalize(self):
        # PureWindowsPath understanding of mixed separators.
        mixed = PureWindowsPath("C:/Users/Test\\Docs/file.txt")
        self.assertEqual(mixed.drive, "C:")
        self.assertTrue(
            path_is_within(
                str(mixed),
                r"C:\Users\Test",
                platform_name="windows",
            )
        )


class WindowsAppTargetSafetyTests(unittest.TestCase):
    def test_blocked_targets_are_rejected(self):
        from tools.app_launcher import _is_safe_windows_target

        self.assertFalse(_is_safe_windows_target(Path(r"C:\Windows\System32\cmd.exe")))
        self.assertFalse(_is_safe_windows_target(Path(r"C:\Tools\Uninstall.exe")))
        self.assertFalse(_is_safe_windows_target(Path(r"C:\Tools\helper.bat")))
        self.assertTrue(_is_safe_windows_target(Path(r"C:\Program Files\App\App.exe")))


if __name__ == "__main__":
    unittest.main()
