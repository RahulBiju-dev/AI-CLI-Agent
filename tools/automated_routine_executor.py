"""Store, preview, and safely execute reusable local workflow macros."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

from agent.cancellation import CancellationToken, OperationCancelled
from agent.persistence import PersistenceError, atomic_write_json, read_json_preserved
from agent.platform_runtime import (
    get_runtime_paths,
    open_url_native,
    path_is_within,
    spawn_detached,
    terminate_process_tree,
    validate_http_url,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = get_runtime_paths().data_dir
STORE_PATH = DATA_DIR / "routines.json"
LEGACY_STORE_PATH = PROJECT_ROOT / ".selene" / "routines.json"
MAX_ACTIONS = 50
MAX_TRIGGERS = 25
MAX_ROUTINE_NAME_CHARS = 200
MAX_COMMAND_ARGUMENTS = 100
MAX_ARGUMENT_CHARS = 4096
AUTOMATIC_ACTION_TYPES = {"open_app", "delay", "tool"}
AUTOMATIC_TOOL_NAMES = {"open_app", "launch_apps"}
CONFIRMATION_TOOL_NAMES = {*AUTOMATIC_TOOL_NAMES, "open_terminal_at_path"}
ALLOWED_ROUTINE_COMMANDS = {"python", "pytest", "git", "ls", "cat", "echo", "grep", "node", "npm"}
_STORE_LOCK = threading.RLock()


def _active_store_path() -> Path:
    """Choose one routine store without silently copying or moving legacy data."""
    if STORE_PATH.exists() or not LEGACY_STORE_PATH.is_file():
        return STORE_PATH
    return LEGACY_STORE_PATH


def _load() -> dict[str, dict]:
    store = _active_store_path()
    try:
        routines = read_json_preserved(store, expected_type=dict)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise PersistenceError(
            f"Routine state at '{store}' could not be read and was preserved: {exc}"
        ) from exc
    invalid = [
        name
        for name, value in routines.items()
        if not isinstance(name, str)
        or not isinstance(value, dict)
        or not isinstance(value.get("triggers", []), list)
        or not isinstance(value.get("actions", []), list)
        or any(not isinstance(item, dict) for item in value.get("actions", []))
    ]
    if invalid:
        raise PersistenceError(
            f"Routine state at '{store}' contains invalid records and was preserved without modification"
        )
    return routines


def _save(routines: dict[str, dict]) -> None:
    atomic_write_json(_active_store_path(), routines)


def _resolve(routines: dict[str, dict], name: str | None, trigger: str | None) -> tuple[str | None, dict | None]:
    if name and name in routines:
        return name, routines[name]
    normalized = (trigger or "").strip().casefold()
    exact_matches = []
    phrase_matches = []
    for routine_name, routine in routines.items():
        values = [routine_name, *routine.get("triggers", [])]
        normalized_values = [str(value).strip().casefold() for value in values if str(value).strip()]
        if normalized in normalized_values:
            exact_matches.append((routine_name, routine))
        elif any(value in normalized for value in normalized_values):
            phrase_matches.append((routine_name, routine))
    if len(exact_matches) == 1:
        return exact_matches[0]
    return phrase_matches[0] if not exact_matches and len(phrase_matches) == 1 else (None, None)


def _normalized_triggers(routine: dict, legacy_trigger: str | None = None) -> list[str]:
    """Normalize and deduplicate trigger phrases while accepting the old argument."""
    raw_triggers = routine.get("triggers", [])
    if not isinstance(raw_triggers, list):
        return []
    values = [*raw_triggers, *([legacy_trigger] if legacy_trigger else [])]
    triggers = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        cleaned = value.strip()
        normalized = cleaned.casefold()
        if normalized not in seen:
            seen.add(normalized)
            triggers.append(cleaned)
    return triggers


def _validate(routine: dict) -> list[str]:
    errors = []
    description = routine.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("routine.description must clearly describe the routine and cannot be empty")
    elif len(description.strip()) > 500:
        errors.append("routine.description may contain at most 500 characters")
    raw_triggers = routine.get("triggers")
    if not isinstance(raw_triggers, list) or not raw_triggers:
        errors.append("routine.triggers must be a non-empty array of user phrases")
    elif len(raw_triggers) > MAX_TRIGGERS:
        errors.append(f"A routine may contain at most {MAX_TRIGGERS} triggers")
    else:
        for index, value in enumerate(raw_triggers):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"triggers[{index}] must be a non-empty string")
            elif len(value.strip()) > 200:
                errors.append(f"triggers[{index}] may contain at most 200 characters")
    actions = routine.get("actions", [])
    if not isinstance(actions, list) or not actions:
        errors.append("routine.actions must be a non-empty array")
        return errors
    if len(actions) > MAX_ACTIONS:
        errors.append(f"A routine may contain at most {MAX_ACTIONS} actions")
    for index, item in enumerate(actions):
        if not isinstance(item, dict) or item.get("type") not in {"command", "open_app", "open_url", "delay", "tool"}:
            errors.append(f"actions[{index}] has an unsupported type")
        elif item.get("type") == "command":
            if not isinstance(item.get("argv"), list) or not item.get("argv"):
                errors.append(f"actions[{index}].argv must be a non-empty argument array; shell strings are not accepted")
            else:
                argv = item["argv"]
                if len(argv) > MAX_COMMAND_ARGUMENTS:
                    errors.append(f"actions[{index}].argv may contain at most {MAX_COMMAND_ARGUMENTS} arguments")
                invalid_arg = any(
                    not isinstance(value, str) or not value or len(value) > MAX_ARGUMENT_CHARS or "\0" in value
                    for value in argv
                )
                if invalid_arg:
                    errors.append(f"actions[{index}].argv entries must be non-empty strings of at most {MAX_ARGUMENT_CHARS} characters")
                executable = str(argv[0])
                if executable not in ALLOWED_ROUTINE_COMMANDS:
                    errors.append(f"actions[{index}].argv[0] must be an allowed command ({', '.join(sorted(ALLOWED_ROUTINE_COMMANDS))}); found '{executable}'")
                try:
                    timeout = float(item.get("timeout", 60))
                    if not math.isfinite(timeout) or not 1 <= timeout <= 600:
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"actions[{index}].timeout must be between 1 and 600 seconds")
                cwd = item.get("cwd", ".")
                if not isinstance(cwd, str) or not cwd.strip() or len(cwd) > MAX_ARGUMENT_CHARS or "\0" in cwd:
                    errors.append(f"actions[{index}].cwd must be a valid project-relative path")
                else:
                    try:
                        requested_cwd = (PROJECT_ROOT / cwd).resolve()
                    except (OSError, RuntimeError):
                        requested_cwd = None
                    if requested_cwd is None or not path_is_within(requested_cwd, PROJECT_ROOT):
                        errors.append(f"actions[{index}].cwd must stay inside the project workspace")
        elif item.get("type") == "open_app":
            app_name = item.get("app_name")
            if not isinstance(app_name, str) or not app_name.strip() or len(app_name) > 128:
                errors.append(f"actions[{index}].app_name must be an installed application display name")
        elif item.get("type") == "tool":
            if not isinstance(item.get("tool_name"), str) or not item["tool_name"].strip():
                errors.append(f"actions[{index}].tool_name must be a registered tool name")
            elif item["tool_name"] == "automated_routine_executor":
                errors.append(f"actions[{index}] cannot recursively call automated_routine_executor")
            if not isinstance(item.get("arguments", {}), dict):
                errors.append(f"actions[{index}].arguments must be an object")
        elif item.get("type") == "open_url":
            try:
                validate_http_url(item.get("url"))
            except ValueError as exc:
                errors.append(f"actions[{index}].url is invalid: {exc}")
        elif item.get("type") == "delay":
            try:
                seconds = float(item.get("seconds", 1))
                if not math.isfinite(seconds) or not 0 <= seconds <= 30:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                errors.append(f"actions[{index}].seconds must be between 0 and 30")
        if isinstance(item, dict) and "continue_on_error" in item and not isinstance(item["continue_on_error"], bool):
            errors.append(f"actions[{index}].continue_on_error must be boolean")
    if routine.get("allow_automatic") is True:
        unsafe = sorted({
            str(item.get("type"))
            for item in actions
            if isinstance(item, dict) and (
                item.get("type") not in AUTOMATIC_ACTION_TYPES
                or (item.get("type") == "tool" and item.get("tool_name") not in AUTOMATIC_TOOL_NAMES)
            )
        })
        if unsafe:
            errors.append(
                "allow_automatic is limited to app-launch tools and delay actions; found: "
                + ", ".join(unsafe)
            )
    return errors


def _trigger_matches(routine: dict, trigger: str | None) -> bool:
    normalized = (trigger or "").strip().casefold()
    return bool(normalized) and any(
        normalized == str(value).strip().casefold()
        for value in routine.get("triggers", [])
    )


def _is_safe_automatic_routine(routine: dict) -> bool:
    """Recheck stored data before bypassing per-run confirmation."""
    actions = routine.get("actions")
    if not isinstance(actions, list) or not actions:
        return False
    for item in actions:
        if not isinstance(item, dict) or item.get("type") not in AUTOMATIC_ACTION_TYPES:
            return False
        if item.get("type") == "open_app":
            has_name = isinstance(item.get("app_name"), str) and bool(item["app_name"].strip())
            legacy_argv = item.get("argv")
            has_legacy_name = isinstance(legacy_argv, list) and bool(legacy_argv)
            if not has_name and not has_legacy_name:
                return False
        if item.get("type") == "tool":
            if item.get("tool_name") not in AUTOMATIC_TOOL_NAMES:
                return False
            if not isinstance(item.get("arguments", {}), dict):
                return False
    return True


def _run_registered_tool(
    tool_name: str,
    arguments: dict,
    action_type: str = "tool",
    cancellation_token: CancellationToken | None = None,
) -> dict:
    """Call tools through the shared registry used by normal agent tool calls."""
    if tool_name == "automated_routine_executor":
        raise ValueError("A routine cannot recursively invoke itself")

    # Imported lazily because registry.py imports this module while constructing
    # the shared dispatch map.
    from agent.tool_runner import execute_tool_call, normalize_tool_calls
    from tools.registry import TOOL_DISPATCH

    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        raise ValueError(f"Unknown registered tool: {tool_name}")

    call_arguments = dict(arguments)
    # Reuse the routine-level approval for tools whose only side effect is the
    # already-previewed launch. Terminal launching is deliberately excluded
    # from AUTOMATIC_TOOL_NAMES, so it can never bypass per-run confirmation.
    if tool_name in CONFIRMATION_TOOL_NAMES:
        call_arguments["confirmed"] = True
    spec = normalize_tool_calls([{
        "function": {"name": tool_name, "arguments": call_arguments}
    }])[0]
    execution = execute_tool_call(spec, cancellation_token=cancellation_token)
    raw_result = execution.content
    if isinstance(raw_result, str):
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            result = {"output": raw_result}
    else:
        result = raw_result

    failed = not execution.ok or (isinstance(result, dict) and (
        "error" in result or result.get("success") is False or result.get("ok") is False
    ))
    return {
        "type": action_type,
        "ok": not failed,
        "tool_name": tool_name,
        "status": execution.status.value,
        "result": result,
    }


def _run_action(
    item: dict,
    cancellation_token: CancellationToken | None = None,
) -> dict:
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    action_type = item["type"]
    if action_type == "delay":
        seconds = max(0.0, min(float(item.get("seconds", 1)), 30.0))
        if cancellation_token and cancellation_token.wait(seconds):
            cancellation_token.raise_if_cancelled()
        elif not cancellation_token:
            time.sleep(seconds)
        return {"type": action_type, "ok": True, "seconds": seconds}
    if action_type == "open_url":
        launch = open_url_native(item["url"])
        return {"type": action_type, **launch.as_dict(), "url": item["url"]}
    if action_type == "open_app":
        app_name = item.get("app_name")
        if not app_name and isinstance(item.get("argv"), list) and item["argv"]:
            app_name = str(item["argv"][0])
        if not app_name:
            raise ValueError("open_app requires app_name; command arguments are not permitted")
        result = _run_registered_tool(
            "open_app",
            {"app_name": str(app_name)},
            action_type,
            cancellation_token,
        )
        result["app_name"] = app_name
        return result
    if action_type == "tool":
        return _run_registered_tool(
            item["tool_name"],
            item.get("arguments", {}),
            cancellation_token=cancellation_token,
        )
    argv = [str(value) for value in item["argv"]]
    if not argv:
        raise ValueError("argv cannot be empty")
    if argv[0] not in ALLOWED_ROUTINE_COMMANDS:
        raise ValueError(f"Command '{argv[0]}' is not permitted")
    requested_cwd = (PROJECT_ROOT / str(item.get("cwd", "."))).resolve()
    if not path_is_within(requested_cwd, PROJECT_ROOT):
        raise ValueError("Command cwd must stay inside the project workspace")
    timeout = max(1.0, min(float(item.get("timeout", 60)), 600.0))
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        handle = spawn_detached(
            argv,
            cwd=requested_cwd,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        deadline = time.monotonic() + timeout
        timed_out = False
        while handle.poll() is None:
            if cancellation_token and cancellation_token.wait(0.05):
                if not terminate_process_tree(handle):
                    raise RuntimeError(
                        "Cancellation was requested, but termination of the owned process tree could not be confirmed"
                    )
                cancellation_token.raise_if_cancelled()
            if time.monotonic() >= deadline:
                timed_out = True
                termination_confirmed = terminate_process_tree(handle)
                break
            if not cancellation_token:
                time.sleep(0.05)
        returncode = handle.process.poll()

        def output_tail(stream) -> str:
            stream.flush()
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 12000))
            return stream.read().decode("utf-8", errors="replace")

        result = {
            "type": action_type,
            "ok": not timed_out and returncode == 0,
            "argv": argv,
            "returncode": returncode,
            "stdout": output_tail(stdout_file),
            "stderr": output_tail(stderr_file),
        }
        if timed_out:
            if termination_confirmed:
                result["error"] = (
                    f"Command exceeded its {timeout:g}s timeout and its owned process tree was stopped"
                )
            else:
                result["error"] = (
                    f"Command exceeded its {timeout:g}s timeout; termination of its owned process tree "
                    "could not be confirmed"
                )
        return result


def automated_routine_executor(
    action: str,
    name: str | None = None,
    routine: dict | None = None,
    trigger: str | None = None,
    dry_run: bool = False,
    confirmed: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Manage workflow macros with per-run or narrowly scoped persistent approval."""
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    try:
        with _STORE_LOCK:
            routines = _load()
    except PersistenceError as exc:
        return json.dumps({
            "error": str(exc),
            "store": str(_active_store_path()),
            "preserved": True,
        }, ensure_ascii=False)
    if action == "list":
        items = [{"name": key, "description": value.get("description", ""), "triggers": value.get("triggers", []), "action_count": len(value.get("actions", []))} for key, value in sorted(routines.items())]
        return json.dumps({"routines": items, "store": str(_active_store_path())}, ensure_ascii=False)
    if action == "define":
        if not name or not routine:
            return json.dumps({"error": "name and routine are required for define"})
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_ROUTINE_NAME_CHARS or any(ord(char) < 32 for char in name):
            return json.dumps({"error": f"name must contain 1-{MAX_ROUTINE_NAME_CHARS} printable characters"})
        name = name.strip()
        candidate = dict(routine)
        candidate["description"] = str(candidate.get("description", "")).strip()
        candidate["triggers"] = _normalized_triggers(candidate, trigger)
        errors = _validate(candidate)
        if errors:
            return json.dumps({"error": "Invalid routine", "details": errors})
        wants_automatic = candidate.get("allow_automatic") is True
        if wants_automatic and not confirmed:
            return json.dumps({
                "error": "Persistent automatic execution requires confirmed=true after the user approves the preview.",
                "routine": candidate,
            }, ensure_ascii=False)
        try:
            with _STORE_LOCK:
                routines = _load()
                routines[name] = {
                    "description": candidate["description"],
                    "triggers": candidate["triggers"],
                    "actions": candidate["actions"],
                    "automatic_approved": wants_automatic and confirmed is True,
                }
                _save(routines)
        except (PersistenceError, OSError, TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc), "store": str(_active_store_path()), "preserved": True})
        return json.dumps({
            "ok": True,
            "defined": name,
            "description": candidate["description"],
            "triggers": candidate["triggers"],
            "action_count": len(candidate["actions"]),
            "automatic_approved": wants_automatic and confirmed is True,
            "store": str(_active_store_path()),
        })
    if action == "delete":
        if not confirmed:
            return json.dumps({"error": "Deleting a routine requires confirmed=true"})
        try:
            with _STORE_LOCK:
                routines = _load()
                if not name or name not in routines:
                    return json.dumps({"error": "Routine not found"})
                del routines[name]
                _save(routines)
        except (PersistenceError, OSError, TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc), "store": str(_active_store_path()), "preserved": True})
        return json.dumps({"ok": True, "deleted": name})
    if action not in {"show", "run"}:
        return json.dumps({"error": "action must be list, define, show, run, or delete"})
    resolved_name, selected = _resolve(routines, name, trigger)
    if not selected:
        return json.dumps({"error": "No unique routine matched", "name": name, "trigger": trigger})
    automatic_trigger = (
        selected.get("automatic_approved") is True
        and _trigger_matches(selected, trigger)
        and _is_safe_automatic_routine(selected)
    )
    # ``show`` is always a preview. ``run`` executes unless the caller
    # explicitly asks for a dry run; requiring dry_run=false as well as
    # action="run" made approved routine calls silently do nothing.
    if action == "show" or dry_run is True:
        requirement = (
            "This exact trigger is persistently approved; call run with the trigger."
            if automatic_trigger
            else "Call run with confirmed=true after user approval."
        )
        return json.dumps({
            "name": resolved_name,
            "routine": selected,
            "dry_run": True,
            "automatic_trigger": automatic_trigger,
            "execution_required": requirement,
        }, ensure_ascii=False)
    if not confirmed and not automatic_trigger:
        return json.dumps({"error": "Routine execution requires confirmed=true after the user reviews the preview"})
    results = []
    for index, item in enumerate(selected["actions"]):
        try:
            result = _run_action(item, cancellation_token)
        except OperationCancelled:
            raise
        except Exception as exc:
            result = {"type": item.get("type"), "ok": False, "error": str(exc)}
        results.append({"index": index, **result})
        if not result.get("ok") and item.get("continue_on_error") is not True:
            break
    ok = len(results) == len(selected["actions"]) and all(item.get("ok") for item in results)
    return json.dumps({"ok": ok, "name": resolved_name, "results": results}, ensure_ascii=False)
