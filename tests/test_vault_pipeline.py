import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import pdf_writer, vault_indexer, vault_search


class FakeCollection:
    def __init__(self, documents):
        self.documents = dict(documents)
        self.deleted = []

    def count(self):
        return len(self.documents)

    def get(self, ids=None, where=None, include=None, **kwargs):
        if ids is None:
            selected = list(self.documents)
        else:
            selected = [item_id for item_id in ids if item_id in self.documents]
        if where and "source" in where:
            selected = [
                item_id for item_id in selected
                if self.documents[item_id][1].get("source") == where["source"]
            ]
        return {
            "ids": selected,
            "documents": [self.documents[item_id][0] for item_id in selected],
            "metadatas": [self.documents[item_id][1] for item_id in selected],
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.documents[item_id] = (document, metadata)

    def delete(self, ids):
        self.deleted.extend(ids)
        for item_id in ids:
            self.documents.pop(item_id, None)


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        return self.collection


class VaultReadTests(unittest.TestCase):
    def test_partial_cursor_reads_oversized_chunk_without_gaps(self):
        document = "abcdefghijklmnopqrstuvwxyz" * 80
        collection = FakeCollection({
            "one": (document, {
                "source": "slides.pdf",
                "source_path": "/tmp/slides.pdf",
                "page": 1,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": len(document),
                "content_kind": "text+vision",
            })
        })
        cursor = 0
        recovered = []
        with patch.object(vault_search, "get_chroma_client", return_value=FakeClient(collection)):
            for _ in range(10):
                payload = json.loads(vault_search.read_vault(
                    collection="lecture",
                    cursor=cursor,
                    max_chunks=1,
                    max_chars=500,
                ))
                if payload["content"]:
                    recovered.append(payload["content"].split("\n", 1)[1])
                if payload["complete"]:
                    break
                cursor = payload["next_cursor"]

        self.assertEqual("".join(recovered), document)
        self.assertTrue(payload["complete"])

    def test_ordered_records_sort_by_source_page_and_chunk(self):
        collection = FakeCollection({
            "b": ("second", {"source": "slides.pdf", "page": 2, "chunk_index": 0}),
            "a2": ("later", {"source": "slides.pdf", "page": 1, "chunk_index": 1}),
            "a1": ("first", {"source": "slides.pdf", "page": 1, "chunk_index": 0}),
        })
        with patch.object(vault_search, "get_chroma_client", return_value=FakeClient(collection)):
            records = vault_search.ordered_vault_records("lecture")
        self.assertEqual([item["id"] for item in records], ["a1", "a2", "b"])


class IncrementalPdfIndexTests(unittest.TestCase):
    def test_pdf_index_resumes_from_atomic_page_checkpoint(self):
        class FakePage:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return f"page {self.number} text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(1), FakePage(2), FakePage(3)]

        collection = FakeCollection({
            "old-generation": ("old", {"source": "slides.pdf"})
        })
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            fake_pypdf = SimpleNamespace(PdfReader=FakeReader)
            with (
                patch.dict(sys.modules, {"pypdf": fake_pypdf}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(vault_indexer, "_describe_pdf_page", side_effect=lambda path, page, *args: f"visual {page}"),
                patch.object(vault_indexer, "embed_texts", side_effect=lambda docs, **kwargs: [[float(i)] for i, _ in enumerate(docs)]),
            ):
                first = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=None,
                    cancellation_token=None,
                )
                second = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=5,
                    resume_page=2,
                    cancellation_token=None,
                )

        self.assertFalse(first["complete"])
        self.assertEqual(first["next_page"], 2)
        self.assertTrue(second["complete"])
        self.assertEqual(second["indexed_pages"], 3)
        self.assertEqual(second["vision_pages"], 3)
        self.assertIn("old-generation", collection.deleted)
        page_one_ids = [item_id for item_id in collection.documents if "page::1::" in item_id]
        self.assertEqual(len(page_one_ids), 1)

    def test_visual_failures_remain_resumable_and_block_complete_claim(self):
        class FakePage:
            def extract_text(self):
                return "slide text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage()]

        collection = FakeCollection({})
        attempts = iter((RuntimeError("moondream unavailable"), "visual details"))

        def describe(*args):
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            return result

        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(vault_indexer, "_describe_pdf_page", side_effect=describe),
                patch.object(vault_indexer, "embed_texts", side_effect=lambda docs, **kwargs: [[1.0] for _ in docs]),
            ):
                first = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=None, cancellation_token=None,
                )
                second = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=1, cancellation_token=None,
                )

        self.assertFalse(first["complete"])
        self.assertEqual(first["vision_failed_pages"], [1])
        self.assertEqual(first["next_page"], 1)
        self.assertTrue(second["complete"])
        self.assertTrue(second["vision_complete"])


class PdfToolTests(unittest.TestCase):
    def test_create_pdf_is_non_overwriting_and_uses_export_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_render(path, title, content):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-test")

            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root),
                patch.object(pdf_writer, "_render_pdf", side_effect=fake_render),
            ):
                first = json.loads(pdf_writer.create_pdf("notes.pdf", "# Notes", "Lecture"))
                second = json.loads(pdf_writer.create_pdf("notes.pdf", "replacement", "Lecture"))
                unconfirmed = json.loads(pdf_writer.create_pdf(
                    "notes.pdf", "replacement", "Lecture", overwrite=True
                ))

            self.assertTrue(first["created"])
            self.assertEqual(Path(first["file_path"]), root / "notes.pdf")
            self.assertTrue(second["overwrite_required"])
            self.assertIn("confirmed=true", unconfirmed["error"])

    def test_refined_notes_job_resumes_and_finalizes(self):
        long_document = "A" * 7000
        collection = FakeCollection({
            "one": (long_document, {"source": "slides.pdf", "page": 1, "chunk_index": 0}),
            "two": ("final facts", {"source": "slides.pdf", "page": 2, "chunk_index": 0}),
        })
        records = [
            {"id": "one", "metadata": collection.documents["one"][1]},
            {"id": "two", "metadata": collection.documents["two"][1]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_render(path, title, content):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-notes")

            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root / "exports"),
                patch.object(pdf_writer, "PDF_JOBS_DIR", root / "jobs"),
                patch.object(vault_search, "ordered_vault_records", return_value=records),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(pdf_writer, "_generate_note_section", return_value="# Refined notes"),
                patch.object(pdf_writer, "_render_pdf", side_effect=fake_render),
            ):
                first = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture", file_path="notes.pdf", sections_per_run=1
                ))
                second = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture", file_path="notes.pdf",
                    cursor=first["next_cursor"], sections_per_run=1,
                ))

            self.assertFalse(first["complete"])
            self.assertTrue(second["created"])
            self.assertTrue(second["refined"])
            self.assertTrue((root / "exports" / "notes.pdf").is_file())

    def test_vault_export_removes_chunk_overlap_without_dropping_tail(self):
        collection = FakeCollection({
            "one": ("abcdefghij", {
                "source": "slides.pdf", "page": 1, "chunk_index": 0,
                "char_start": 0, "char_end": 10,
            }),
            "two": ("ijklmnop", {
                "source": "slides.pdf", "page": 1, "chunk_index": 1,
                "char_start": 8, "char_end": 16,
            }),
        })
        records = [
            {"id": "one", "metadata": collection.documents["one"][1]},
            {"id": "two", "metadata": collection.documents["two"][1]},
        ]
        captured = {}

        def fake_create_pdf(**kwargs):
            captured.update(kwargs)
            return json.dumps({"created": True, "file_path": "/tmp/export.pdf", "bytes": 10})

        with (
            patch.object(vault_search, "ordered_vault_records", return_value=records),
            patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
            patch.object(pdf_writer, "create_pdf", side_effect=fake_create_pdf),
        ):
            result = json.loads(pdf_writer.export_vault_pdf(
                collection="lecture", file_path="export.pdf"
            ))

        normalized = captured["content"].replace("\n", "")
        self.assertIn("abcdefghijklmnop", normalized)
        self.assertNotIn("abcdefghijijklmnop", normalized)
        self.assertEqual(result["exported_chunks"], 2)


if __name__ == "__main__":
    unittest.main()
