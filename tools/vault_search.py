"""Vault search tool: embed queries and query ChromaDB for relevant snippets."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from agent.cancellation import CancellationToken, OperationCancelled
from tools.vault_embeddings import DEFAULT_EMBED_MODEL, embed_query
from tools.vault_indexer import CHROMA_DIR, get_chroma_client, resolve_vault_alias

DEFAULT_TOP_K = 6
DEFAULT_MAX_CHARS = 7000
DEFAULT_READ_CHARS = 2800


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _positive_int(value: int | str | None, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _embed_query(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    cancellation_token: CancellationToken | None = None,
) -> List[float]:
    return embed_query(text, model=model, cancellation_token=cancellation_token)


def _query_collection(
    query: str,
    collection_name: str,
    model: str,
    top_k: int,
    source: str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Dict[str, Any]:
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{collection_name}' was not found in {CHROMA_DIR}. Index files first with index_vault."
        ) from exc

    collection_count = collection.count()
    if collection_count == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    emb = _embed_query(query, model=model, cancellation_token=cancellation_token)
    fetch_k = min(top_k, collection_count)
    where = None
    if source:
        if os.path.isabs(source):
            where = {"source_path": os.path.abspath(source)}
        else:
            fetch_k = min(max(top_k * 10, 50), collection_count)

    kwargs = {
        "query_embeddings": [emb],
        "n_results": fetch_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
        
    results = collection.query(**kwargs)
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    
    if source and not os.path.isabs(source):
        source_lower = source.lower()
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        f_docs, f_metas, f_dists = [], [], []
        for d, m, dist in zip(docs, metas, dists):
            m_source = str(m.get("source", "")).lower()
            m_file = str(m.get("filename", "")).lower()
            if source_lower in m_source or source_lower in m_file:
                f_docs.append(d)
                f_metas.append(m)
                f_dists.append(dist)
                if len(f_docs) >= top_k:
                    break
                    
        results = {
            "documents": [f_docs],
            "metadatas": [f_metas],
            "distances": [f_dists]
        }
        
    return results


def _flatten_results(results: Dict[str, Any], max_chars: int) -> tuple[list[dict], str]:
    docs = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    matches: list[dict] = []
    context_parts: list[str] = []
    used_chars = 0

    for index, doc in enumerate(docs[0] if docs else []):
        meta = metadatas[0][index] if metadatas and metadatas[0] and index < len(metadatas[0]) else {}
        distance = distances[0][index] if distances and distances[0] and index < len(distances[0]) else None
        text = (doc or "").strip()
        header = (
            f"Source: {meta.get('source', 'unknown')} | "
            f"chunk: {meta.get('chunk_index', index)} | "
            f"chars: {meta.get('char_start', '?')}-{meta.get('char_end', '?')}"
        )
        entry = f"{header}\n{text}\n---"
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(entry) > remaining:
            entry = entry[:max(0, remaining - 3)].rstrip() + "..."
        context_parts.append(entry)
        used_chars += len(entry)

        matches.append({
            "rank": index + 1,
            "source": meta.get("source"),
            "source_path": meta.get("source_path"),
            "filename": meta.get("filename"),
            "chunk_index": meta.get("chunk_index", index),
            "char_start": meta.get("char_start"),
            "char_end": meta.get("char_end"),
            "distance": distance,
            "similarity": round(1.0 / (1.0 + max(0.0, float(distance))), 6) if isinstance(distance, (int, float)) else None,
            "text": text[:1200] + ("..." if len(text) > 1200 else ""),
        })

    return matches, "\n\n".join(context_parts)


def search_vault(
    query: str,
    collection_name: str = "vault",
    model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    collection: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    source: str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """
    Search the indexed vault and return relevant snippets in a compact JSON format.

    This function generates an embedding for the provided query, queries the local
    ChromaDB collection for the most similar text chunks, and returns them along
    with metadata like source path and chunk index.

    Args:
        query (str): The search query to embed and look up.
        collection_name (str): The name or human-friendly alias of the ChromaDB collection.
        model (str): The embedding model used for the query.
        top_k (int): Number of top results to return.
        collection (str | None): An alias for collection_name.
        max_chars (int): The maximum character limit for the returned context string.
        source (str | None): Optional filepath to restrict the search to a specific file.

    Returns:
        str: A JSON-encoded string with search matches, extracted context string,
             and metadata, or an error message.
    """
    if collection:
        collection_name = collection

    # Resolve human-friendly aliases first, then sanitize as fallback
    collection_name = resolve_vault_alias(collection_name)
    
    if not query or not str(query).strip():
        return _json({"error": "query is required"})
    query = str(query).strip()
    if len(query) > 4000:
        return _json({"error": "query exceeds the 4000-character limit"})

    top_k_int = _positive_int(top_k, DEFAULT_TOP_K, minimum=1, maximum=20)
    max_chars_int = _positive_int(max_chars, DEFAULT_MAX_CHARS, minimum=1000, maximum=20000)

    try:
        results = _query_collection(
            query=query,
            collection_name=collection_name,
            model=model,
            top_k=top_k_int,
            source=source,
            cancellation_token=cancellation_token,
        )
        matches, context = _flatten_results(results, max_chars=max_chars_int)
        return _json({
            "collection": collection_name,
            "query": query,
            "top_k": top_k_int,
            "match_count": len(matches),
            "matches": matches,
            "context": context,
            "guidance": "Use source and chunk_index/char offsets to cite or retrieve nearby content with read_file/read_document when needed.",
        })
    except OperationCancelled:
        raise
    except Exception as exc:
        return _json({"error": str(exc), "collection": collection_name, "persist_directory": CHROMA_DIR})


def _source_matches(meta: dict, source: str | None) -> bool:
    if not source:
        return True
    requested = str(source).strip().casefold()
    if not requested:
        return True
    candidates = (
        str(meta.get("source") or ""),
        str(meta.get("source_path") or ""),
        str(meta.get("filename") or ""),
    )
    if os.path.isabs(source):
        return os.path.abspath(str(meta.get("source_path") or "")) == os.path.abspath(source)
    return any(requested in candidate.casefold() for candidate in candidates)


def ordered_vault_records(
    collection_name: str,
    source: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    cancellation_token: CancellationToken | None = None,
) -> list[dict]:
    """Return lightweight collection records in deterministic source/page order."""
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    client = get_chroma_client()
    collection_obj = client.get_collection(name=resolve_vault_alias(collection_name))
    raw = collection_obj.get(include=["metadatas"])
    ids = raw.get("ids", [])
    metadatas = raw.get("metadatas", [])
    records = []
    for item_id, metadata in zip(ids, metadatas):
        meta = metadata or {}
        if not _source_matches(meta, source):
            continue
        page = meta.get("page")
        try:
            page_number = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_number = None
        if start_page is not None and (page_number is None or page_number < start_page):
            continue
        if end_page is not None and (page_number is None or page_number > end_page):
            continue
        records.append({"id": item_id, "metadata": meta})

    records.sort(key=lambda item: (
        str(item["metadata"].get("source") or "").casefold(),
        int(item["metadata"].get("page") or 0),
        int(item["metadata"].get("chunk_index") or 0),
        int(item["metadata"].get("char_start") or 0),
        str(item["id"]),
    ))
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    return records


def read_vault(
    collection: str = "vault",
    cursor: int | str = 0,
    source: str | None = None,
    start_page: int | str | None = None,
    end_page: int | str | None = None,
    max_chunks: int | str = 8,
    max_chars: int | str = DEFAULT_READ_CHARS,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Read every vault chunk through a stable cursor instead of similarity search."""
    collection_name = resolve_vault_alias(collection)
    raw_cursor = str(cursor or "0").strip()
    try:
        if ":" in raw_cursor:
            record_raw, char_raw = raw_cursor.split(":", 1)
            cursor_int = max(0, min(int(record_raw), 10_000_000))
            cursor_char = max(0, min(int(char_raw), 10_000_000))
        else:
            cursor_int = max(0, min(int(raw_cursor), 10_000_000))
            cursor_char = 0
    except ValueError:
        return _json({"error": "cursor must be an integer or '<chunk>:<character>'"})
    max_chunks_int = _positive_int(max_chunks, 8, minimum=1, maximum=50)
    # Keep the complete cursor envelope below the shared low-VRAM tool-output cap.
    max_chars_int = _positive_int(max_chars, DEFAULT_READ_CHARS, minimum=500, maximum=3200)

    def optional_page(value) -> int | None:
        if value is None or str(value).strip() == "":
            return None
        return _positive_int(value, 1, minimum=1, maximum=1_000_000)

    start_page_int = optional_page(start_page)
    end_page_int = optional_page(end_page)
    if start_page_int and end_page_int and start_page_int > end_page_int:
        start_page_int, end_page_int = end_page_int, start_page_int

    try:
        records = ordered_vault_records(
            collection_name,
            source=source,
            start_page=start_page_int,
            end_page=end_page_int,
            cancellation_token=cancellation_token,
        )
        selected = records[cursor_int:cursor_int + max_chunks_int]
        if not selected:
            return _json({
                "collection": collection_name,
                "cursor": raw_cursor,
                "next_cursor": None,
                "total_chunks": len(records),
                "returned_chunks": 0,
                "complete": True,
                "content": "",
            })

        client = get_chroma_client()
        collection_obj = client.get_collection(name=collection_name)
        raw = collection_obj.get(
            ids=[item["id"] for item in selected],
            include=["documents", "metadatas"],
        )
        by_id = {
            item_id: (document, metadata or {})
            for item_id, document, metadata in zip(
                raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])
            )
        }
        parts = []
        used = 0
        consumed = 0
        next_cursor: int | str | None = cursor_int
        for selected_index, item in enumerate(selected):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            document, meta = by_id.get(item["id"], ("", item["metadata"]))
            document = str(document or "")
            offset = cursor_char if selected_index == 0 else 0
            if offset >= len(document):
                consumed += 1
                next_cursor = cursor_int + consumed
                continue
            header = (
                f"[source={meta.get('source', 'unknown')}"
                f" page={meta.get('page', '?')}"
                f" chunk={meta.get('chunk_index', '?')}"
                f" kind={meta.get('content_kind', 'text')}]"
            )
            remaining = max_chars_int - used
            text_budget = remaining - len(header) - 1
            if text_budget <= 0:
                break
            text_slice = document[offset:offset + text_budget]
            block = f"{header}\n{text_slice}"
            if not block:
                break
            parts.append(block)
            used += len(block) + 2
            if offset + len(text_slice) < len(document):
                next_cursor = f"{cursor_int + selected_index}:{offset + len(text_slice)}"
                break
            consumed += 1
            next_cursor = cursor_int + selected_index + 1

        complete = isinstance(next_cursor, int) and next_cursor >= len(records)
        return _json({
            "collection": collection_name,
            "cursor": raw_cursor,
            "next_cursor": None if complete else next_cursor,
            "total_chunks": len(records),
            "returned_chunks": consumed,
            "complete": complete,
            "source": source,
            "start_page": start_page_int,
            "end_page": end_page_int,
            "content": "\n\n".join(parts),
            "guidance": (
                "If complete is false, call vault_read again with next_cursor. "
                "Keep the same collection/source/page filters to walk the vault without gaps."
            ),
        })
    except OperationCancelled:
        raise
    except Exception as exc:
        return _json({"error": str(exc), "collection": collection_name})


def format_for_gemma(results: Dict[str, Any] | str, token_limit: int = 2048) -> str:
    """Format raw Chroma results or search_vault JSON into context text."""
    if isinstance(results, str):
        try:
            data = json.loads(results)
        except json.JSONDecodeError:
            return results
        if "context" in data:
            return data["context"]
        results = data

    max_chars = _positive_int(token_limit, 2048, minimum=128) * 4
    _, context = _flatten_results(results, max_chars=max_chars)
    return context


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search a ChromaDB vault collection using Ollama embeddings.")
    parser.add_argument("query")
    parser.add_argument("--collection", default="vault")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--source")
    args = parser.parse_args()
    print(search_vault(
        args.query,
        collection_name=args.collection,
        model=args.model,
        top_k=args.top_k,
        max_chars=args.max_chars,
        source=args.source,
    ))
