import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.persistence import (
    PersistenceError,
    atomic_write_bytes,
    atomic_write_json,
    read_json_preserved,
)


class AtomicPersistenceTests(unittest.TestCase):
    def test_atomic_json_round_trip_handles_unicode(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "nested" / "state.json"
            atomic_write_json(destination, {"name": "Selene 🌙", "items": [1, 2]})
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {
                "name": "Selene 🌙",
                "items": [1, 2],
            })
            self.assertEqual(list(destination.parent.glob(f".{destination.name}-*.tmp")), [])

    def test_failed_replace_preserves_original_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "critical.json"
            destination.write_text('{"original": true}\n', encoding="utf-8")
            with patch("agent.persistence.os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    atomic_write_json(destination, {"replacement": True})
            self.assertEqual(destination.read_text(encoding="utf-8"), '{"original": true}\n')
            self.assertEqual(list(destination.parent.glob(f".{destination.name}-*.tmp")), [])

    def test_malformed_json_is_preserved_for_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "state.json"
            malformed = '{"unfinished":'
            destination.write_text(malformed, encoding="utf-8")
            with self.assertRaises(PersistenceError) as raised:
                read_json_preserved(destination, expected_type=dict)
            self.assertIn("preserved", str(raised.exception))
            self.assertEqual(destination.read_text(encoding="utf-8"), malformed)

    @unittest.skipIf(os.name == "nt", "POSIX mode bits are not Windows ACLs")
    def test_private_atomic_write_uses_owner_only_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "credential.enc"
            atomic_write_bytes(destination, b"encrypted", private=True)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_non_finite_json_is_rejected_before_touching_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "state.json"
            destination.write_text('{"safe": true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                atomic_write_json(destination, {"bad": float("nan")})
            self.assertEqual(destination.read_text(encoding="utf-8"), '{"safe": true}\n')


if __name__ == "__main__":
    unittest.main()
