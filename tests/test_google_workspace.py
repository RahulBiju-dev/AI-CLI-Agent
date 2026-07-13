import base64
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # Optional integration dependency.
    AESGCM = None

import tools.google_workspace as google_workspace_tool
from tools.registry import TOOL_METADATA, get_tool_metadata


class _FakeKeyring(types.ModuleType):
    def __init__(self, value=None, *, unavailable=False):
        super().__init__("keyring")
        self.value = value
        self.unavailable = unavailable
        self.set_calls = []

    def get_password(self, service, user):
        if self.unavailable:
            raise RuntimeError("SecretService is unavailable")
        return self.value

    def set_password(self, service, user, value):
        if self.unavailable:
            raise RuntimeError("SecretService is unavailable")
        self.value = value
        self.set_calls.append((service, user, value))


def _payload():
    return {
        "client_config": {"installed": {"client_id": "test-client"}},
        "token": {
            "token": "access-value",
            "refresh_token": "refresh-value",
            "client_id": "test-client",
            "client_secret": "test-secret",
            "token_uri": "https://oauth2.googleapis.com/token",
        },
        "saved_at": "2026-07-13T00:00:00+00:00",
    }


def _write_envelope(path: Path, key: bytes, payload=None):
    nonce = b"n" * 12
    plaintext = json.dumps(payload or _payload(), separators=(",", ":")).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(
        nonce, plaintext, google_workspace_tool._CREDENTIAL_AAD
    )
    envelope = {
        "version": 1,
        "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
        "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
    }
    path.write_text(json.dumps(envelope), encoding="utf-8")


@unittest.skipIf(AESGCM is None, "cryptography is not installed")
class GoogleCredentialStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.credential_path = self.root / "google_oauth.enc"
        self.key_path = self.root / ".credential-key"
        self.path_patches = (
            patch.object(google_workspace_tool, "DATA_DIR", self.root),
            patch.object(google_workspace_tool, "CREDENTIAL_PATH", self.credential_path),
            patch.object(google_workspace_tool, "KEY_PATH", self.key_path),
        )
        for path_patch in self.path_patches:
            path_patch.start()

    def tearDown(self):
        for path_patch in reversed(self.path_patches):
            path_patch.stop()
        self.temporary.cleanup()

    def test_new_save_mirrors_key_locally_and_survives_keyring_outage(self):
        keyring = _FakeKeyring()
        with patch.dict(sys.modules, {"keyring": keyring}):
            google_workspace_tool._save_credentials(_payload())

        self.assertTrue(self.credential_path.is_file())
        self.assertTrue(self.key_path.is_file())
        self.assertEqual(self.key_path.read_text(encoding="ascii"), keyring.value)
        if os.name != "nt":
            self.assertEqual(self.key_path.stat().st_mode & 0o777, 0o600)

        unavailable_keyring = _FakeKeyring(unavailable=True)
        with patch.dict(sys.modules, {"keyring": unavailable_keyring}):
            loaded = google_workspace_tool._load_credentials()
        self.assertEqual(loaded, _payload())

    def test_legacy_keyring_only_store_repairs_private_fallback_after_decrypt(self):
        key = b"k" * 32
        encoded = base64.urlsafe_b64encode(key).decode("ascii")
        _write_envelope(self.credential_path, key)
        original_ciphertext = self.credential_path.read_bytes()
        keyring = _FakeKeyring(encoded)

        with patch.dict(sys.modules, {"keyring": keyring}):
            loaded = google_workspace_tool._load_credentials(repair_key=True)

        self.assertEqual(loaded, _payload())
        self.assertEqual(self.key_path.read_text(encoding="ascii"), encoded)
        self.assertEqual(self.credential_path.read_bytes(), original_ciphertext)

    def test_authenticated_keyring_candidate_repairs_stale_local_key(self):
        stale_key = b"s" * 32
        correct_key = b"c" * 32
        self.key_path.write_text(
            base64.urlsafe_b64encode(stale_key).decode("ascii"), encoding="ascii"
        )
        _write_envelope(self.credential_path, correct_key)
        original_ciphertext = self.credential_path.read_bytes()
        keyring = _FakeKeyring(base64.urlsafe_b64encode(correct_key).decode("ascii"))

        with patch.dict(sys.modules, {"keyring": keyring}):
            loaded = google_workspace_tool._load_credentials(repair_key=True)

        self.assertEqual(loaded, _payload())
        self.assertEqual(
            self.key_path.read_text(encoding="ascii"),
            base64.urlsafe_b64encode(correct_key).decode("ascii"),
        )
        self.assertEqual(self.credential_path.read_bytes(), original_ciphertext)

    def test_status_reports_unavailable_key_and_preserves_ciphertext(self):
        _write_envelope(self.credential_path, b"u" * 32)
        original_ciphertext = self.credential_path.read_bytes()
        unavailable_keyring = _FakeKeyring(unavailable=True)

        with patch.dict(sys.modules, {"keyring": unavailable_keyring}):
            status = json.loads(google_workspace_tool.google_workspace(action="status"))

        self.assertFalse(status["connected"])
        self.assertTrue(status["credential_file_present"])
        self.assertEqual(status["credential_store_status"], "unreadable")
        self.assertTrue(status["reauthorization_required"])
        self.assertIn("encryption key is unavailable", status["error"])
        self.assertEqual(self.credential_path.read_bytes(), original_ciphertext)
        self.assertFalse(self.key_path.exists())

    def test_status_validates_keyring_only_store_without_repairing_it(self):
        key = b"r" * 32
        _write_envelope(self.credential_path, key)
        keyring = _FakeKeyring(base64.urlsafe_b64encode(key).decode("ascii"))

        with patch.dict(sys.modules, {"keyring": keyring}):
            status = json.loads(google_workspace_tool.google_workspace(action="status"))

        self.assertTrue(status["connected"])
        self.assertEqual(status["credential_store_status"], "ready")
        self.assertFalse(self.key_path.exists())

    def test_malformed_ciphertext_status_is_truthful_and_non_destructive(self):
        malformed = b'{"version":1,"nonce":'
        self.credential_path.write_bytes(malformed)
        unavailable_keyring = _FakeKeyring(unavailable=True)

        with patch.dict(sys.modules, {"keyring": unavailable_keyring}):
            status = json.loads(google_workspace_tool.google_workspace(action="status"))

        self.assertFalse(status["connected"])
        self.assertEqual(status["credential_store_status"], "unreadable")
        self.assertIn("malformed and was preserved", status["error"])
        self.assertEqual(self.credential_path.read_bytes(), malformed)
        self.assertFalse(self.key_path.exists())


class GoogleAuthorizationTests(unittest.TestCase):
    def test_oauth_uses_bounded_explicit_loopback_listener(self):
        with tempfile.TemporaryDirectory() as temporary:
            client_path = Path(temporary) / "client.json"
            client_config = {"installed": {"client_id": "test-client"}}
            client_path.write_text(json.dumps(client_config), encoding="utf-8")
            flow = MagicMock()
            credentials = MagicMock()
            credentials.to_json.return_value = json.dumps(_payload()["token"])
            flow.run_local_server.return_value = credentials
            installed_app_flow = MagicMock()
            installed_app_flow.from_client_config.return_value = flow

            with (
                patch.object(
                    google_workspace_tool,
                    "_google_imports",
                    return_value=(MagicMock(), MagicMock(), installed_app_flow, MagicMock()),
                ),
                patch.object(google_workspace_tool, "_save_credentials") as save_credentials,
            ):
                result = json.loads(
                    google_workspace_tool._authorize(str(client_path))
                )

        self.assertTrue(result["ok"])
        installed_app_flow.from_client_config.assert_called_once_with(
            client_config, google_workspace_tool.SCOPES
        )
        call_kwargs = flow.run_local_server.call_args.kwargs
        self.assertEqual(call_kwargs["host"], "127.0.0.1")
        self.assertEqual(call_kwargs["bind_addr"], "127.0.0.1")
        self.assertEqual(call_kwargs["port"], 0)
        self.assertEqual(
            call_kwargs["timeout_seconds"],
            google_workspace_tool.OAUTH_WAIT_TIMEOUT_SECONDS,
        )
        save_credentials.assert_called_once()

    def test_oauth_timeout_does_not_replace_existing_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client_path = root / "client.json"
            credential_path = root / "google_oauth.enc"
            client_path.write_text(
                json.dumps({"installed": {"client_id": "test-client"}}),
                encoding="utf-8",
            )
            original = b"existing-encrypted-credentials"
            credential_path.write_bytes(original)
            flow = MagicMock()
            flow.run_local_server.side_effect = TimeoutError(
                "Timed out waiting for Google authorization"
            )
            installed_app_flow = MagicMock()
            installed_app_flow.from_client_config.return_value = flow

            with (
                patch.object(
                    google_workspace_tool,
                    "_google_imports",
                    return_value=(MagicMock(), MagicMock(), installed_app_flow, MagicMock()),
                ),
                patch.object(google_workspace_tool, "CREDENTIAL_PATH", credential_path),
                patch.object(google_workspace_tool, "_save_credentials") as save_credentials,
            ):
                result = json.loads(google_workspace_tool._authorize(str(client_path)))

            self.assertIn("Timed out", result["error"])
            save_credentials.assert_not_called()
            self.assertEqual(credential_path.read_bytes(), original)


class GoogleToolMetadataTests(unittest.TestCase):
    def test_authorize_timeout_exceeds_bounded_oauth_wait(self):
        metadata = get_tool_metadata("google_workspace", {"action": "authorize"})
        self.assertIsNotNone(metadata)
        self.assertGreater(
            metadata.default_timeout_seconds,
            google_workspace_tool.OAUTH_WAIT_TIMEOUT_SECONDS,
        )

    def test_birthdays_are_read_only_and_google_dependencies_are_complete(self):
        metadata = get_tool_metadata("google_workspace", {"action": "list_birthdays"})
        self.assertIsNotNone(metadata)
        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.side_effecting)
        self.assertEqual(
            set(TOOL_METADATA["google_workspace"].optional_dependencies),
            {
                "google-api-python-client",
                "google-auth-oauthlib",
                "cryptography",
                "keyring",
            },
        )


if __name__ == "__main__":
    unittest.main()
