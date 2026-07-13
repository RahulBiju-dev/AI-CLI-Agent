"""Regression coverage for the recursive reliability pass over legacy tools."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import api_orchestrator as api_module
from tools import obsi_vault_writer
from tools.api_orchestrator import api_orchestrator
from tools.app_launcher import _is_safe_desktop_entry
from tools.automated_routine_executor import _validate as validate_routine
from tools.code import view_code
from tools.codebase_indexer import codebase_indexer
from tools.context_memory_optimizer import context_memory_optimizer
from tools.current_datetime import get_current_datetime
from tools.document import MAX_PAGE_SELECTION, _parse_page_spec, read_document
from tools.file import read_file
from tools.knowledge_graph_builder import knowledge_graph_builder
from tools.pdf_writer import _parse_notes_cursor
from tools.reasoning_chain_debugger import reasoning_chain_debugger
from tools.run_simulation import run_simulation
from tools.search import web_search
from tools.spreadsheet import _Sheet, _parse_range
from tools.vault_embeddings import embed_texts
from tools.vault_search import ordered_vault_records
from tools.vision_describer import describe_image
from tools.web_scraper import web_scrape


class _StreamingResponse:
    def __init__(self, chunks: list[str], *, status: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status
        self.ok = 200 <= status < 400
        self.encoding = "utf-8"
        self.headers = {
            "Content-Type": "text/plain",
            "Set-Cookie": "session=secret",
            "X-Token": "secret",
            "X-Request-ID": "safe",
        }
        self.closed = False

    def iter_content(self, **_kwargs):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class RemainingToolReliabilityTests(unittest.TestCase):
    def test_context_optimizer_rejects_malformed_nested_values(self):
        payload = json.loads(context_memory_optimizer(["not-an-object"]))
        self.assertIn("messages[0]", payload["error"])
        payload = json.loads(context_memory_optimizer([], critical_terms="wrong"))
        self.assertIn("critical_terms", payload["error"])

    def test_simulation_rejects_non_finite_and_expensive_inputs(self):
        payload = json.loads(run_simulation({"x": math.inf}, {"x": "x"}))
        self.assertIn("finite", payload["error"])
        payload = json.loads(run_simulation({"x": 2}, {"x": "x ** 101"}))
        self.assertIn("Power exponent", payload["error"])
        payload = json.loads(run_simulation(
            {"x": 1}, {"x": "x + 1"}, scenarios=[{"overrides": {"missing": 2}}]
        ))
        self.assertIn("unknown variables", payload["error"])
        payload = json.loads(run_simulation(
            {"x": 1}, {"x": "x"}, steps=10_000, trials=50,
            scenarios=[{"name": "a"}, {"name": "b"}],
        ))
        self.assertIn("across all scenarios", payload["error"])

    def test_simulation_still_runs_valid_recurrence(self):
        payload = json.loads(run_simulation({"x": 1}, {"x": "x + 1"}, steps=2))
        self.assertEqual(payload["scenarios"][0]["sample_trajectory"][-1]["x"], 3.0)

    def test_reasoning_debugger_normalizes_references_and_mermaid_ids(self):
        payload = json.loads(reasoning_chain_debugger(
            "done",
            [{"id": 'bad] --> injected["', "claim": "done", "depends_on": "wrong"}],
        ))
        self.assertFalse(payload["valid"])
        self.assertIn("depends_on must be an array", [item["issue"] for item in payload["issues"]])
        self.assertIn("n1[", payload["mermaid"])
        self.assertNotIn("bad] --> injected", payload["mermaid"])

    def test_reasoning_debugger_reports_duplicate_evidence(self):
        payload = json.loads(reasoning_chain_debugger(
            "claim",
            [{"claim": "claim", "evidence_ids": ["e1"]}],
            [{"id": "e1"}, {"id": "e1"}],
        ))
        self.assertFalse(payload["valid"])
        self.assertIn("Duplicate evidence id", [item["issue"] for item in payload["issues"]])

    def test_knowledge_graph_rejects_bad_query_weight_and_edge_ids(self):
        concepts = [{"id": "a"}, {"id": "b"}]
        payload = json.loads(knowledge_graph_builder(concepts, [], query="wrong"))
        self.assertIn("query must be", payload["error"])
        payload = json.loads(knowledge_graph_builder(
            concepts, [{"source": "a", "target": "b", "weight": float("nan")}]
        ))
        self.assertIn("Invalid graph", payload["error"])
        payload = json.loads(knowledge_graph_builder(concepts, [
            {"id": "same", "source": "a", "target": "b"},
            {"id": "same", "source": "a", "target": "b"},
        ]))
        self.assertIn("Duplicate relationship id", " ".join(payload["details"]))

    def test_api_orchestrator_validates_urls_and_literal_secrets(self):
        payload = json.loads(api_orchestrator({"url": "https://user:pass@example.com"}))
        self.assertIn("embedded credentials", payload["error"])
        payload = json.loads(api_orchestrator({
            "url": "https://example.com", "headers": {"Authorization": "Bearer literal"}
        }))
        self.assertIn("environment-variable", payload["error"])

    def test_api_orchestrator_streams_bounds_and_redacts_response(self):
        response = _StreamingResponse(["a" * 800, "b" * 800])
        with patch.object(api_module.requests, "request", return_value=response) as request:
            payload = json.loads(api_orchestrator({
                "url": "https://example.com/path?api_key=secret",
                "max_response_chars": 1000,
            }))
        self.assertEqual(len(payload["body"]), 1000)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["endpoint"], "https://example.com/path")
        self.assertNotIn("Set-Cookie", payload["headers"])
        self.assertNotIn("X-Token", payload["headers"])
        self.assertEqual(payload["headers"]["X-Request-ID"], "safe")
        self.assertTrue(response.closed)
        self.assertTrue(request.call_args.kwargs["stream"])

    def test_web_search_skips_malformed_backend_records(self):
        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def text(self, *_args, **_kwargs):
                return [None, {"href": "javascript:alert(1)"}, {
                    "href": "https://example.com", "title": "ok", "body": "snippet"
                }]

        fake_module = SimpleNamespace(DDGS=FakeDDGS)
        with patch.dict("sys.modules", {"ddgs": fake_module}):
            payload = json.loads(web_search("query", difficulty="easy"))
        self.assertEqual(payload["skipped_invalid_results"], 2)
        self.assertEqual(payload["results"][0]["url"], "https://example.com")

    def test_web_search_page_limit_bounds_attempts_not_only_successes(self):
        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def text(self, *_args, **_kwargs):
                return [
                    {"href": f"https://example.com/{index}", "title": str(index), "body": ""}
                    for index in range(3)
                ]

        with (
            patch.dict("sys.modules", {"ddgs": SimpleNamespace(DDGS=FakeDDGS)}),
            patch("tools.search.web_scrape", return_value='{"error":"failed"}') as scrape,
        ):
            web_search("query", include_content=True, max_pages=1)
        self.assertEqual(scrape.call_count, 1)

    def test_native_and_file_tools_reject_malformed_paths_without_crashing(self):
        self.assertTrue(describe_image(None).startswith("Error:"))
        self.assertIn("error", json.loads(read_file(None)))
        self.assertIn("error", json.loads(read_document(None)))
        self.assertIn("error", json.loads(view_code(None)))
        self.assertIn("error", json.loads(get_current_datetime("x" * 300)))
        self.assertIn("error", json.loads(web_scrape("https://" + "a" * 5000)))

    def test_embedding_helper_rejects_single_string_and_bad_timeout(self):
        with self.assertRaises(TypeError):
            embed_texts("not-a-sequence-of-documents")
        with self.assertRaises(ValueError):
            embed_texts(["document"], timeout="invalid")

    def test_linux_env_desktop_wrapper_allows_assignments_but_blocks_flags(self):
        base = {"name": "Example", "terminal": False}
        self.assertTrue(_is_safe_desktop_entry({**base, "exec": "env FOO=bar /usr/bin/example"}))
        self.assertFalse(_is_safe_desktop_entry({**base, "exec": "env -S 'bash -c id'"}))

    def test_routine_validator_rejects_invalid_runtime_values(self):
        routine = {
            "description": "bad delay",
            "triggers": ["run it"],
            "actions": [{"type": "delay", "seconds": float("nan")}],
        }
        self.assertIn("seconds must be between", " ".join(validate_routine(routine)))
        routine["actions"] = [{"type": "open_url", "url": "https://user:pass@example.com"}]
        self.assertIn("embedded credentials", " ".join(validate_routine(routine)))

    def test_note_writer_bounds_content_before_writing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            obsi_vault_writer, "VAULTS_DIR", directory
        ):
            payload = json.loads(obsi_vault_writer.create_structured_note(
                "title", "x" * (obsi_vault_writer.MAX_NOTE_CHARS + 1)
            ))
        self.assertIn("limit", payload["error"])

    def test_pdf_page_range_and_cursor_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "at most"):
            _parse_page_spec(f"1-{MAX_PAGE_SELECTION + 1}", MAX_PAGE_SELECTION + 1)
        with self.assertRaisesRegex(ValueError, "100-character"):
            _parse_notes_cursor("1" * 101)

    def test_spreadsheet_cell_range_length_is_bounded(self):
        sheet = _Sheet("Sheet1", 1, 1, lambda _row, _column: None)
        with self.assertRaisesRegex(ValueError, "64-character"):
            _parse_range("A" * 65 + "1", sheet)

    def test_vault_ordering_tolerates_malformed_legacy_metadata(self):
        collection = SimpleNamespace(get=lambda **_kwargs: {
            "ids": ["a", "b"],
            "metadatas": [
                {"source": "file", "page": "not-a-number", "chunk_index": None},
                None,
            ],
        })
        client = SimpleNamespace(get_collection=lambda **_kwargs: collection)
        with patch("tools.vault_search.get_chroma_client", return_value=client):
            records = ordered_vault_records("vault")
        self.assertEqual({item["id"] for item in records}, {"a", "b"})

    def test_codebase_model_name_is_bounded_before_index_access(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(codebase_indexer(directory, action="status", model="x" * 201))
        self.assertIn("model", payload["error"])


if __name__ == "__main__":
    unittest.main()
