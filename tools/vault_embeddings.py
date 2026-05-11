"""Shared Ollama embedding helpers for vault indexing/search."""

from __future__ import annotations

from typing import Any, List, Sequence

import requests

DEFAULT_EMBED_MODEL = "embeddinggemma"
OLLAMA_EMBED_URLS = (
    "http://127.0.0.1:11434/api/embed",
    "http://127.0.0.1:11434/embed",
)


def _as_plain_data(response: Any) -> Any:
    """Convert Ollama client response objects into plain Python data."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return response


def normalize_embeddings(response: Any) -> List[List[float]]:
    """Extract a list of embedding vectors from common Ollama response shapes."""
    data = _as_plain_data(response)

    if isinstance(data, dict):
        if "embeddings" in data:
            return [list(embedding) for embedding in data["embeddings"]]
        if "embedding" in data:
            return [list(data["embedding"])]

    if isinstance(data, list):
        if all(isinstance(item, (list, tuple)) for item in data):
            return [list(item) for item in data]
        if all(isinstance(item, dict) and "embedding" in item for item in data):
            return [list(item["embedding"]) for item in data]

    raise RuntimeError("Unexpected embedding response shape: %r" % (data,))


def _clean_inputs(texts: Sequence[str]) -> list[str]:
    cleaned = []
    for text in texts:
        value = str(text or "").strip()
        cleaned.append(value if value else " ")
    return cleaned


def _validate_embedding_count(embeddings: list[list[float]], expected: int) -> list[list[float]]:
    if len(embeddings) != expected:
        raise RuntimeError(f"Ollama returned {len(embeddings)} embedding(s) for {expected} input(s)")
    return embeddings


def embed_texts(texts: Sequence[str], model: str = DEFAULT_EMBED_MODEL, timeout: int = 60) -> List[List[float]]:
    """Embed one or more texts using Ollama, returning Chroma-compatible vectors."""
    inputs = _clean_inputs(texts)
    if not inputs:
        return []

    last_error: Exception | None = None
    payload = {"model": model, "input": inputs}

    for url in OLLAMA_EMBED_URLS:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return _validate_embedding_count(normalize_embeddings(resp.json()), len(inputs))
        except Exception as exc:
            last_error = exc

    try:
        import ollama

        if hasattr(ollama, "embed"):
            return _validate_embedding_count(normalize_embeddings(ollama.embed(model=model, input=inputs)), len(inputs))
        if hasattr(ollama, "Embeddings"):
            client = ollama.Embeddings()
            return _validate_embedding_count(normalize_embeddings(client.create(model=model, input=inputs)), len(inputs))
    except Exception as exc:
        last_error = exc

    raise RuntimeError("Failed to obtain embeddings via HTTP or ollama client: %s" % last_error)


def embed_query(text: str, model: str = DEFAULT_EMBED_MODEL, timeout: int = 30) -> List[float]:
    embeddings = embed_texts([text], model=model, timeout=timeout)
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding for query")
    return embeddings[0]
