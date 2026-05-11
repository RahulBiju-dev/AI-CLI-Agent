"""Vault indexer: chunk local files and index embeddings into ChromaDB using Ollama."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import chromadb

from tools.document import extract_document_text
from tools.vault_embeddings import DEFAULT_EMBED_MODEL, embed_texts

SUPPORTED_INDEX_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".pdf",
    ".docx",
}
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_BATCH_SIZE = 16
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".chroma")
VAULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vaults")


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _positive_int(value: int | str | None, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def sanitize_collection_name(name: str) -> str:
    """Sanitize the collection name to meet ChromaDB requirements.
    Expected a name containing 3-63 characters from [a-zA-Z0-9._-],
    starting and ending with an alphanumeric character.
    """
    if not name:
        return "vault"
    # Replace invalid chars with underscores
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    # Strip leading/trailing non-alphanumeric chars
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    
    if not name:
        return "vault"
    if len(name) < 3:
        name = name.ljust(3, '0')
        
    return name[:63]


def get_chroma_client(path: str | None = None):
    """Return a persistent Chroma client shared by index and search tools."""
    persist_directory = path or CHROMA_DIR
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=persist_directory)

    from chromadb.config import Settings

    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory,
    ))


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    return [chunk["text"] for chunk in chunk_text_with_offsets(text, chunk_size, chunk_overlap)]


def chunk_text_with_offsets(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into readable chunks with character offsets.

    The splitter prefers paragraph/newline/sentence boundaries near the end of
    each window, which gives retrieval snippets more coherent context than hard
    character slicing.
    """
    chunk_size = _positive_int(chunk_size, DEFAULT_CHUNK_SIZE, minimum=500, maximum=20000)
    chunk_overlap = _positive_int(chunk_overlap, DEFAULT_CHUNK_OVERLAP, minimum=0, maximum=max(0, chunk_size // 2))
    text = text.replace("\r\n", "\n")
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(length, start + chunk_size)
        end = hard_end
        if hard_end < length:
            window = text[start:hard_end]
            boundary_candidates = [
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
            ]
            boundary = max(boundary_candidates)
            if boundary >= chunk_size // 2:
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "index": len(chunks),
                "text": chunk,
                "char_start": start,
                "char_end": end,
            })

        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    match = re.match(r"^---\s*\n.*?\n---\s*\n", text, flags=re.DOTALL)
    return text[match.end():].lstrip() if match else text


def _read_text_for_index(path: str) -> tuple[str, dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pdf", ".docx"}:
        return extract_document_text(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _strip_frontmatter(text)
    return text, {"document_type": ext.lstrip(".") or "text", "char_count": len(text)}


def _iter_indexable_files(vault_path: str):
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [name for name in dirs if name not in {".git", ".chroma", "__pycache__", "node_modules"}]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_INDEX_EXTENSIONS:
                yield os.path.join(root, fname)


def _flush_batch(collection, ids: list[str], docs: list[str], metadatas: list[dict], model: str) -> int:
    if not docs:
        return 0
    embeddings = embed_texts(docs, model=model)
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    return len(docs)


def _delete_existing_source(collection, source: str) -> None:
    try:
        collection.delete(where={"source": source})
    except Exception:
        # Older Chroma versions may raise when no rows match. Stale chunks are
        # less harmful than failing the entire indexing operation.
        pass


def index_vault(
    vault_path: Optional[str] = None,
    collection_name: str = "vault",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    file_path: Optional[str] = None,
    collection: Optional[str] = None,
):
    """Index either a vault folder or a single file.

    Returns a JSON string with status and next-step guidance for tool consumers.
    """
    if collection:
        collection_name = collection
        
    collection_name = sanitize_collection_name(collection_name)

    if not vault_path:
        vault_path = os.path.dirname(file_path) if file_path else VAULTS_DIR
    vault_path = os.path.abspath(vault_path)
    if not os.path.exists(vault_path):
        return _json({"error": f"vault path not found: {vault_path}"})

    if file_path:
        candidates = [os.path.abspath(file_path)]
    else:
        if not os.path.isdir(vault_path):
            return _json({"error": f"vault_path must be a folder when file_path is not provided: {vault_path}"})
        candidates = list(_iter_indexable_files(vault_path))

    batch_size = _positive_int(batch_size, DEFAULT_BATCH_SIZE, minimum=1, maximum=128)
    chunk_size = _positive_int(chunk_size, DEFAULT_CHUNK_SIZE, minimum=500, maximum=20000)
    chunk_overlap = _positive_int(chunk_overlap, DEFAULT_CHUNK_OVERLAP, minimum=0, maximum=max(0, chunk_size // 2))

    client = get_chroma_client()
    collection_obj = client.get_or_create_collection(name=collection_name)

    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict] = []
    indexed_chunks = 0
    indexed_files = 0
    skipped_files: list[dict] = []

    for path in candidates:
        if not os.path.exists(path):
            skipped_files.append({"file": path, "error": "file not found"})
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_INDEX_EXTENSIONS:
            skipped_files.append({"file": path, "error": f"unsupported extension: {ext}"})
            continue

        try:
            text, info = _read_text_for_index(path)
        except UnicodeDecodeError:
            skipped_files.append({"file": path, "error": "not UTF-8 text; use PDF/DOCX or plain text"})
            continue
        except Exception as exc:
            skipped_files.append({"file": path, "error": str(exc)})
            continue

        chunks = chunk_text_with_offsets(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            skipped_files.append({"file": path, "error": "no extractable text"})
            continue

        rel = os.path.relpath(path, vault_path) if os.path.isdir(vault_path) else os.path.basename(path)
        _delete_existing_source(collection_obj, rel)
        indexed_files += 1

        for chunk in chunks:
            chunk_index = chunk["index"]
            ids.append(f"{rel}::chunk::{chunk_index}")
            docs.append(chunk["text"])
            metadatas.append({
                "source": rel,
                "source_path": path,
                "filename": os.path.basename(path),
                "extension": ext,
                "chunk_index": chunk_index,
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "document_type": info.get("document_type", ext.lstrip(".")),
            })

            if len(docs) >= batch_size:
                indexed_chunks += _flush_batch(collection_obj, ids, docs, metadatas, model=model)
                ids, docs, metadatas = [], [], []

    if docs:
        indexed_chunks += _flush_batch(collection_obj, ids, docs, metadatas, model=model)

    return _json({
        "collection": collection_name,
        "persist_directory": CHROMA_DIR,
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files": skipped_files[:20],
        "skipped_count": len(skipped_files),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "guidance": "Use vault_search with a focused query to retrieve relevant chunks from large indexed files.",
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index a local vault into ChromaDB using Ollama embeddings.")
    parser.add_argument("--vault-path", default=None)
    parser.add_argument("--collection", default="vault")
    parser.add_argument("--file-path")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    print(index_vault(
        vault_path=args.vault_path,
        collection_name=args.collection,
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model=args.model,
        batch_size=args.batch_size,
    ))
