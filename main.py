#!/usr/bin/env python3
"""Selene entry point: validate local runtime, then launch CLI or web UI."""

import sys

from agent.model_lifecycle import ModelStartupResult, ensure_managed_model
from agent.ollama_runtime import InvalidModelfileError, OllamaRuntimeError, OllamaService
from agent.platform_runtime import get_runtime_paths, resource_path
from agent.runtime_config import RuntimeConfig, RuntimeConfigurationError, get_runtime_config

from rich.console import Console

_console = Console()


def _get_modelfile_path() -> str:
    """Return the path to the Modelfile, handling PyInstaller's temp directory.
    
    When running as a packaged PyInstaller executable, files are extracted to a 
    temporary _MEIPASS directory. Otherwise, they are relative to this script.
    
    Returns:
        str: Absolute path to the Modelfile.
    """
    return str(resource_path("Modelfile"))

def _ensure_model(
    config: RuntimeConfig | None = None,
    service: OllamaService | None = None,
) -> ModelStartupResult:
    """Verify or safely stage-build Selene's managed model alias."""
    runtime = config or get_runtime_config()
    return ensure_managed_model(
        config=runtime,
        service=service,
        modelfile_path=_get_modelfile_path(),
    )


def main() -> None:
    """Main execution point for the application."""
    service: OllamaService | None = None
    try:
        runtime = get_runtime_config(refresh=True)
        runtime_paths = get_runtime_paths()
        _console.print(f"[cyan]Runtime profile:[/] {runtime.profile.value}")
        _console.print(f"[dim]{runtime.selection_reason}[/]")
        _console.print(
            f"[dim]Runtime data: {runtime_paths.data_dir} ({runtime_paths.source})[/]"
        )
        for warning in runtime.warnings:
            _console.print(f"[yellow]⚠ {warning}[/]")

        service = OllamaService(runtime)
        _console.print(f"[cyan bold]⟳  Verifying local model '{runtime.chat_model}'…[/]")
        model_result = _ensure_model(runtime, service)
        if model_result.action in {"built", "rebuilt"}:
            _console.print(
                f"[cyan bold]✓  Model {model_result.action} safely through a verified staging alias.[/]"
            )
        else:
            _console.print(f"[cyan bold]✓  Model ready ({model_result.action}).[/]")
        _console.print()

        if "--cli" in sys.argv:
            from agent.core import run

            run()
        else:
            from agent.web import start_web_server

            start_web_server() 
    except KeyboardInterrupt:
        _console.print("\n[dim]Interrupted — goodbye.[/]")
    except (RuntimeConfigurationError, InvalidModelfileError, OllamaRuntimeError) as exc:
        _console.print(f"\n[red bold]Selene startup failed:[/] {exc}", style="red")
        raise SystemExit(1) from None
    finally:
        if service is not None:
            service.coordinator.shutdown(cancel_active=True, wait=False)
        tool_runner = sys.modules.get("agent.tool_runner")
        shutdown = getattr(tool_runner, "shutdown_tool_runner", None)
        if callable(shutdown):
            shutdown(wait=False)


if __name__ == "__main__":
    main()
