"""Shared Ollama embedding helpers for vault indexing/search."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, List, Sequence

from agent.cancellation import CancellationToken
from agent.ollama_runtime import OllamaRuntimeError, OllamaService
from agent.runtime_config import get_runtime_config

_RUNTIME_CONFIG = get_runtime_config()
DEFAULT_EMBED_MODEL = _RUNTIME_CONFIG.embedding_model
_EMBED_SERVICE = OllamaService(_RUNTIME_CONFIG)


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

    raise RuntimeError("Unexpected embedding response shape: %s" % repr(data)[:500])


def _clean_inputs(texts: Sequence[str]) -> list[str]:
    cleaned = []
    for text in texts:
        value = str(text or "").strip()
        cleaned.append(value if value else " ")
    return cleaned


def _validate_embedding_count(embeddings: list[list[float]], expected: int) -> list[list[float]]:
    if len(embeddings) != expected:
        raise RuntimeError(f"Ollama returned {len(embeddings)} embedding(s) for {expected} input(s)")
    dimensions = set()
    normalized = []
    for index, embedding in enumerate(embeddings):
        if not embedding:
            raise RuntimeError(f"Ollama returned an empty embedding at index {index}")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Ollama returned a non-numeric embedding at index {index}") from exc
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(f"Ollama returned a non-finite embedding at index {index}")
        dimensions.add(len(vector))
        normalized.append(vector)
    if len(dimensions) > 1:
        raise RuntimeError(f"Ollama returned inconsistent embedding dimensions: {sorted(dimensions)}")
    return normalized


def embed_texts(
    texts: Sequence[str],
    model: str = DEFAULT_EMBED_MODEL,
    timeout: int = 60,
    cancellation_token: CancellationToken | None = None,
) -> List[List[float]]:
    """
    Embed one or more texts using Ollama, returning Chroma-compatible vectors.
    
    This function communicates with a local Ollama instance to generate vector
    embeddings for the provided strings through the shared resource coordinator.
    
    Args:
        texts (Sequence[str]): A list or tuple of string documents to embed.
        model (str): The Ollama model name to use for embeddings (e.g., 'embeddinggemma').
        timeout (int): The maximum number of seconds to wait for a network response.
        
    Returns:
        List[List[float]]: A list of floating-point vectors corresponding to the inputs.
        
    Raises:
        RuntimeError: If Ollama fails or the returned vectors do not match the inputs.
    """
    inputs = _clean_inputs(texts)
    if not inputs:
        return []

    timeout = max(1, min(int(timeout), 300))
    thread_id = threading.get_ident()
    coordinator = _EMBED_SERVICE.coordinator
    owner = (
        f"tool:{thread_id}"
        if coordinator.is_owned_by_current_context()
        else f"embedding:{thread_id}:{time.monotonic_ns()}"
    )
    try:
        response = _EMBED_SERVICE.embed(
            inputs,
            owner=owner,
            model=model,
            cancellation_token=cancellation_token,
            operation_timeout=timeout,
        )
        return _validate_embedding_count(normalize_embeddings(response), len(inputs))
    except OllamaRuntimeError as exc:
        raise RuntimeError(f"Failed to obtain coordinated Ollama embeddings: {exc}") from exc


def embed_query(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    timeout: int = 30,
    cancellation_token: CancellationToken | None = None,
) -> List[float]:
    embeddings = embed_texts(
        [text],
        model=model,
        timeout=timeout,
        cancellation_token=cancellation_token,
    )
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding for query")
    return embeddings[0]
