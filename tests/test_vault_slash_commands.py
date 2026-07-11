"""Compatibility tests for web /vault slash commands."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.tool_runner import ToolCallResult, ToolCallSpec, ToolResultStatus, normalize_tool_calls
from agent.web import execute_command_web


def _ok_result(name: str, payload: dict) -> ToolCallResult:
    spec = ToolCallSpec(index=0, name=name, arguments={}, raw={})
    return ToolCallResult(
        spec=spec,
        content=json.dumps(payload),
        status=ToolResultStatus.SUCCESS,
    )


class VaultSlashCommandTests(unittest.TestCase):
    def setUp(self):
        self.session = {"options": {}}
        self.history: list[dict] = []

    def test_vault_help_lists_subcommands(self):
        text = execute_command_web("/vault help", self.session, self.history)
        self.assertIn("/vault list", text)
        self.assertIn("/vault search", text)
        self.assertIn("/vault status", text)
        self.assertIn("/vault read", text)

    def test_vault_list_uses_registered_tool(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "list_vaults",
                {"vaults": [{"collection": "notes", "indexed_chunks": 3}]},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web("/vault list", self.session, self.history)
        self.assertIn("notes", text)
        self.assertEqual(mock_execute.call_args[0][0].name, "list_vaults")

    def test_vault_search_uses_vault_search_not_legacy_name(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "vault_search",
                {"results": [{"source": "a.md", "score": 0.9, "text": "hello"}]},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web("/vault search alpha", self.session, self.history)
        self.assertIn("hello", text)
        self.assertEqual(mock_execute.call_args[0][0].name, "vault_search")

    def test_vault_delete_uses_delete_vault_item(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "delete_vault_item",
                {"deleted_chunks": 2},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web("/vault delete notes.md", self.session, self.history)
        self.assertIn("2", text)
        self.assertEqual(mock_execute.call_args[0][0].name, "delete_vault_item")

    def test_vault_alias_uses_register_vault_alias(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "register_vault_alias",
                {"success": True, "alias": "Notes", "collection": "notes"},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web("/vault alias Notes notes", self.session, self.history)
        self.assertIn("Notes", text)
        self.assertEqual(mock_execute.call_args[0][0].name, "register_vault_alias")

    def test_vault_rename_uses_rename_vault(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "rename_vault",
                {"renamed": True, "old_collection": "old", "new_collection": "new"},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web("/vault rename old new", self.session, self.history)
        self.assertIn("new", text)
        self.assertEqual(mock_execute.call_args[0][0].name, "rename_vault")

    def test_vault_aliases_handles_list_payload(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "list_vault_aliases",
                {"aliases": [{"alias": "Notes", "collection": "notes"}]},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web("/vault aliases", self.session, self.history)
        self.assertIn("Notes", text)
        self.assertIn("notes", text)

    def test_vault_read_passes_cursor_to_ordered_reader(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "vault_read",
                {"content": "page text", "next_cursor": "2:100", "complete": False},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web(
                    "/vault read --cursor 1:20 --collection lecture",
                    self.session,
                    self.history,
                )
        self.assertIn("page text", text)
        self.assertIn("2:100", text)
        spec = mock_execute.call_args[0][0]
        self.assertEqual(spec.name, "vault_read")
        self.assertEqual(spec.arguments["cursor"], "1:20")

    def test_vault_status_uses_checkpoint_action(self):
        with patch("agent.web.execute_tool_call") as mock_execute:
            mock_execute.return_value = _ok_result(
                "index_vault",
                {"jobs": [{"source": "slides.pdf", "indexed_pages": 20, "page_count": 900, "next_page": 21}]},
            )
            with patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls):
                text = execute_command_web(
                    "/vault status /tmp/slides.pdf --collection lecture",
                    self.session,
                    self.history,
                )
        self.assertIn("20/900", text)
        spec = mock_execute.call_args[0][0]
        self.assertEqual(spec.name, "index_vault")
        self.assertEqual(spec.arguments["action"], "status")

    def test_vault_add_forwards_visual_batch_options_without_forcing_generic_collection(self):
        with (
            patch("agent.web.execute_tool_call") as mock_execute,
            patch("agent.web.normalize_tool_calls", side_effect=normalize_tool_calls),
            patch("agent.web.os.path.exists", return_value=True),
            patch("agent.web.os.path.isfile", return_value=True),
        ):
            mock_execute.return_value = _ok_result(
                "index_vault",
                {"collection": "slides", "indexed_chunks": 20, "incomplete_pdf_count": 1,
                 "pdf_jobs": [{"source": "slides.pdf", "indexed_pages": 5, "page_count": 900,
                                "vision_pages": 5, "next_page": 6}]},
            )
            text = execute_command_web(
                "/vault add /tmp/slides.pdf --vision all --max-pages 5",
                self.session,
                self.history,
            )
        self.assertIn("checkpoint", text)
        spec = mock_execute.call_args[0][0]
        self.assertEqual(spec.arguments["vision_mode"], "all")
        self.assertEqual(spec.arguments["max_pages"], 5)
        self.assertNotIn("collection", spec.arguments)


if __name__ == "__main__":
    unittest.main()
