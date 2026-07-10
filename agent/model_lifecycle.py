"""Safe startup lifecycle for Selene's managed Ollama model alias."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.cancellation import CancellationToken
from agent.ollama_runtime import (
    ModelBuildRecord,
    OllamaModelMissingError,
    OllamaRuntimeError,
    OllamaService,
    OllamaUnavailableError,
    ParsedModelfile,
    parse_modelfile,
)
from agent.persistence import PersistenceError, atomic_write_json, read_json_preserved
from agent.platform_runtime import RuntimePaths, get_runtime_paths
from agent.runtime_config import RuntimeConfig, get_runtime_config


MANAGED_MODEL_NAME = "selene"
MODEL_METADATA_SCHEMA = 1


class ModelLifecycleError(OllamaRuntimeError):
    """A controlled startup failure that leaves existing models untouched."""


@dataclass(frozen=True)
class ModelStartupResult:
    model: str
    base_model: str | None
    action: str
    reason: str
    metadata_path: Path | None = None


def _metadata_path(paths: RuntimePaths, model: str) -> Path:
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    return paths.state_dir / "models" / f"{digest}.json"


def _expected_record(config: RuntimeConfig, parsed: ParsedModelfile) -> ModelBuildRecord:
    return ModelBuildRecord(
        schema_version=MODEL_METADATA_SCHEMA,
        model=config.chat_model,
        base_model=config.base_model,
        modelfile_sha256=parsed.sha256,
    )


def _load_record(path: Path) -> Mapping[str, Any] | None:
    try:
        return read_json_preserved(path, expected_type=dict)
    except FileNotFoundError:
        return None
    except PersistenceError as exc:
        raise ModelLifecycleError(
            f"Selene's model-build metadata is malformed and was preserved: {exc}"
        ) from exc


def _stale_reason(metadata: Mapping[str, Any] | None, expected: ModelBuildRecord) -> str | None:
    if metadata is None:
        return "No trusted build record exists for the managed model."
    expected_values = expected.as_dict()
    for key, message in (
        ("schema_version", "The build-record schema changed."),
        ("model", "The build record belongs to another model."),
        ("base_model", "The configured base model changed."),
        ("modelfile_sha256", "The bundled Modelfile changed."),
    ):
        if metadata.get(key) != expected_values[key]:
            return message
    return None


def ensure_managed_model(
    *,
    config: RuntimeConfig | None = None,
    service: OllamaService | None = None,
    modelfile_path: str | Path,
    runtime_paths: RuntimePaths | None = None,
    cancellation_token: CancellationToken | None = None,
) -> ModelStartupResult:
    """Verify or stage-build Selene's model without deleting a working alias."""
    runtime = config or get_runtime_config()
    ollama_service = service or OllamaService(runtime)
    paths = runtime_paths or get_runtime_paths()

    probe = ollama_service.probe(timeout=min(5.0, runtime.chat_timeout_seconds))
    if not probe.api_available:
        if probe.cli_installed:
            raise OllamaUnavailableError(
                "Ollama is installed but its local API is unavailable. Start Ollama and retry. "
                f"Details: {probe.reason}"
            )
        raise OllamaUnavailableError(
            "Ollama is not available. Install and start native Ollama, then retry. "
            f"Details: {probe.reason}"
        )

    # An explicitly selected non-Selene model is user-owned. Verify it, but
    # never rebuild, copy over, or delete it as part of Selene startup.
    if runtime.chat_model != MANAGED_MODEL_NAME:
        if not ollama_service.model_exists(runtime.chat_model):
            raise OllamaModelMissingError(
                f"Configured chat model {runtime.chat_model!r} is not installed."
            )
        return ModelStartupResult(
            model=runtime.chat_model,
            base_model=None,
            action="verified-external",
            reason="The explicitly configured model exists and was not modified.",
        )

    parsed = parse_modelfile(modelfile_path)
    expected = _expected_record(runtime, parsed)
    metadata_path = _metadata_path(paths, runtime.chat_model)
    metadata = _load_record(metadata_path)
    model_exists = ollama_service.model_exists(runtime.chat_model)
    stale_reason = _stale_reason(metadata, expected) if model_exists else "The managed model is missing."
    if model_exists and stale_reason is None:
        return ModelStartupResult(
            model=runtime.chat_model,
            base_model=runtime.base_model,
            action="ready",
            reason="The managed model and Modelfile build record match.",
            metadata_path=metadata_path,
        )

    if not ollama_service.model_exists(runtime.base_model):
        raise OllamaModelMissingError(
            f"Required base model {runtime.base_model!r} is not installed. "
            f"Install it with 'ollama pull {runtime.base_model}' and retry; Selene did not download a model."
        )

    staging_name = f"{MANAGED_MODEL_NAME}-build-{parsed.sha256[:8]}-{uuid.uuid4().hex[:8]}"
    try:
        ollama_service.install_model_staged(
            model=runtime.chat_model,
            staging_model=staging_name,
            base_model=runtime.base_model,
            system_prompt=parsed.system_prompt,
            parameters=parsed.parameters,
            owner="startup:model-build",
            cancellation_token=cancellation_token,
            operation_timeout=runtime.build_timeout_seconds,
        )
    except BaseException:
        # The staged installer publishes only after validation, so an existing
        # target alias remains usable when this path fails or is interrupted.
        raise

    try:
        atomic_write_json(metadata_path, expected.as_dict(), durable=True)
    except OSError as exc:
        raise ModelLifecycleError(
            f"Model was built, but Selene could not persist its build record at {metadata_path}: {exc}"
        ) from exc

    return ModelStartupResult(
        model=runtime.chat_model,
        base_model=runtime.base_model,
        action="rebuilt" if model_exists else "built",
        reason=stale_reason or "The managed model required a build.",
        metadata_path=metadata_path,
    )
