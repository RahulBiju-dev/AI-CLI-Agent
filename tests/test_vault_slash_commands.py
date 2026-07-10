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


if __name__ == "__main__":
    unittest.main()
