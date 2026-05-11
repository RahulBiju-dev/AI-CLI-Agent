"""Vault search tool: embed queries and query ChromaDB for relevant snippets."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from tools.vault_embeddings import DEFAULT_EMBED_MODEL, embed_query
from tools.vault_indexer import CHROMA_DIR, get_chroma_client

DEFAULT_TOP_K = 6
DEFAULT_MAX_CHARS = 7000


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


def _embed_query(text: str, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    return embed_query(text, model=model)


def _query_collection(
    query: str,
    collection_name: str,
    model: str,
    top_k: int,
    source: str | None = None,
) -> Dict[str, Any]:
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{collection_name}' was not found in {CHROMA_DIR}. Index files first with index_vault."
        ) from exc

    emb = _embed_query(query, model=model)
    where = None
    if source:
        where = {"source_path": source} if os.path.isabs(source) else {"source": source}
    kwargs = {
        "query_embeddings": [emb],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


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
) -> str:
    """Search the indexed vault and return compact JSON for the agent."""
    if collection:
        collection_name = collection
    if not query or not query.strip():
        return _json({"error": "query is required"})

    top_k_int = _positive_int(top_k, DEFAULT_TOP_K, minimum=1, maximum=20)
    max_chars_int = _positive_int(max_chars, DEFAULT_MAX_CHARS, minimum=1000, maximum=20000)

    try:
        results = _query_collection(
            query=query,
            collection_name=collection_name,
            model=model,
            top_k=top_k_int,
            source=source,
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
    except Exception as exc:
        return _json({"error": str(exc), "collection": collection_name, "persist_directory": CHROMA_DIR})


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
