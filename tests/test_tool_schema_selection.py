"""Request-specific tool schema selection coverage."""

from __future__ import annotations

import unittest

from agent.core import select_tool_schemas
from tools.registry import TOOL_SCHEMAS


class ToolSchemaSelectionTests(unittest.TestCase):
    def setUp(self):
        self.session = {"options": {"num_ctx": 4096}}

    def _names(self, user_text: str) -> set[str]:
        messages = [{"role": "user", "content": user_text}]
        selected = select_tool_schemas(messages, self.session, TOOL_SCHEMAS)
        self.assertIsNotNone(selected)
        self.assertLessEqual(len(selected), 10)
        return {
            schema["function"]["name"]
            for schema in selected
            if schema.get("function", {}).get("name")
        }

    def test_vault_phrasing_selects_vault_tools(self):
        names = self._names("search my vault notes for the meeting summary")
        self.assertIn("vault_search", names)

    def test_spotify_phrasing_selects_spotify(self):
        names = self._names("play a song on spotify")
        self.assertIn("spotify_play", names)

    def test_spreadsheet_phrasing_selects_spreadsheet(self):
        names = self._names("open the excel spreadsheet and update cells")
        self.assertIn("spreadsheet", names)

    def test_browser_phrasing_selects_browser(self):
        names = self._names("open this website in the browser")
        self.assertIn("open_browser", names)

    def test_datetime_preflight_included_with_web_search(self):
        names = self._names("search the web for today's latest news headlines")
        self.assertIn("web_search", names)
        self.assertIn("get_current_datetime", names)

    def test_each_model_exposed_tool_has_recall_phrasing(self):
        phrases = {
            "get_current_datetime": "what is the current date and time today",
            "spreadsheet": "read the xlsx spreadsheet worksheet cells",
            "web_search": "search the internet for latest research news",
            "web_scrape": "scrape this webpage article url",
            "read_document": "extract text from this pdf document",
            "read_file": "read the text file lines on this path",
            "create_file": "create a new file and write content",
            "create_pdf": "create a pdf document with these notes",
            "export_vault_pdf": "export the complete entire vault to a reference pdf",
            "build_vault_notes_pdf": "generate refined lecture notes pdf from my entire vault",
            "spotify_play": "play music on spotify playlist",
            "open_browser": "open the browser to a website url",
            "view_code": "inspect the source code function class",
            "describe_image": "describe this screenshot image photo",
            "open_terminal_at_path": "open a terminal console at this folder directory",
            "launch_apps": "launch the desktop application app",
            "google_workspace": "list my google calendar events and tasks",
            "codebase_indexer": "index this repository codebase architecture",
            "index_vault": "index vault document folder embeddings",
            "vault_search": "semantic search vault knowledge documents",
            "vault_read": "read all vault chunks exhaustively in order",
            "delete_vault_item": "delete vault collection chunks from index",
            "list_vaults": "list vault collections indexes",
            "list_vault_aliases": "list vault aliases friendly names",
            "create_structured_note": "create an obsidian markdown note with tags",
            "knowledge_graph_builder": "build knowledge graph concepts relationships",
            "run_simulation": "run monte carlo simulation scenario probability",
            "api_orchestrator": "call api http endpoint request integration",
            "context_memory_optimizer": "optimize conversation context memory compact",
            "reasoning_chain_debugger": "audit claim evidence reasoning confidence",
            "automated_routine_executor": "run automated routine workflow trigger",
        }
        for tool_name, phrase in phrases.items():
            with self.subTest(tool=tool_name):
                names = self._names(phrase)
                self.assertIn(tool_name, names, f"{tool_name} not selected for: {phrase}")


if __name__ == "__main__":
    unittest.main()
