"""Unit tests for terminal directory resolution (stdlib unittest)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.terminal_launcher import _resolve_directory


class TestResolveDirectory(unittest.TestCase):
    def test_resolve_directory_invalid_types(self):
        for value in (None, 123, "", "   "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "A directory path is required."):
                    _resolve_directory(value)

    def test_resolve_directory_invalid_paths(self):
        long_path = "a" * 4097
        with self.assertRaisesRegex(ValueError, "The directory path is invalid."):
            _resolve_directory(long_path)
        with self.assertRaisesRegex(ValueError, "The directory path is invalid."):
            _resolve_directory("path/with/\0/null/byte")

    def test_resolve_directory_not_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "does_not_exist"
            with self.assertRaisesRegex(ValueError, "Directory does not exist:"):
                _resolve_directory(str(missing))

    def test_resolve_directory_not_a_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_file.txt"
            test_file.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Path is not a directory:"):
                _resolve_directory(str(test_file))

    def test_resolve_directory_absolute_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _resolve_directory(temp_dir)
            self.assertEqual(result, Path(temp_dir).resolve(strict=True))

    def test_resolve_directory_relative_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sub_dir = Path(temp_dir) / "sub_dir"
            sub_dir.mkdir()
            previous = os.getcwd()
            try:
                os.chdir(temp_dir)
                result = _resolve_directory("sub_dir")
            finally:
                os.chdir(previous)
            self.assertEqual(result, sub_dir.resolve(strict=True))

    def test_resolve_directory_expanduser(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir, "USERPROFILE": temp_dir}, clear=False):
                with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                    result = _resolve_directory("~")
            self.assertEqual(result, Path(temp_dir).resolve(strict=True))


if __name__ == "__main__":
    unittest.main()
