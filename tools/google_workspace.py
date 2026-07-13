"""Google Calendar and Tasks integration with encrypted local OAuth storage."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.persistence import atomic_write_bytes
from agent.platform_runtime import get_runtime_paths


SCOPES = (
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
)
DATA_DIR = get_runtime_paths().data_dir
CREDENTIAL_PATH = DATA_DIR / "google_oauth.enc"
KEY_PATH = DATA_DIR / ".credential-key"
KEYRING_SERVICE = "selene-agent"
KEYRING_USER = "google-oauth-encryption"
MAX_RESULTS = 100
OAUTH_WAIT_TIMEOUT_SECONDS = 240
_CREDENTIAL_AAD = b"selene-google-v1"


class CredentialStoreError(RuntimeError):
    """A safe, actionable encrypted-credential storage failure."""


def _result(**values: Any) -> str:
    return json.dumps(values, ensure_ascii=False, default=str)


def _safe_error(exc: Exception) -> str:
    """Return a useful error without echoing common credential fields."""
    raw_message = str(exc).strip()
    message = (raw_message.splitlines()[0] if raw_message else type(exc).__name__)[:500]
    message = re.sub(
        r"(?i)(access_token|refresh_token|client_secret|authorization|api[_-]?key)([=:'\" ]+)[^, }&]+",
        r"\1\2[redacted]",
        message,
    )
    return message


def _atomic_private_write(path: Path, data: bytes) -> None:
    atomic_write_bytes(path, data, private=True)


def _encode_key(key: bytes) -> str:
    if len(key) != 32:
        raise CredentialStoreError("The Google credential encryption key is invalid")
    return base64.urlsafe_b64encode(key).decode("ascii")


def _decode_key(encoded: str, source: str) -> bytes:
    try:
        key = base64.b64decode(encoded.encode("ascii"), altchars=b"-_", validate=True)
    except (AttributeError, UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise CredentialStoreError(
            f"The Google credential encryption key in {source} is invalid"
        ) from exc
    if len(key) != 32:
        raise CredentialStoreError(
            f"The Google credential encryption key in {source} is invalid"
        )
    return key


def _key_candidates() -> list[bytes]:
    """Return distinct usable local/keyring keys without modifying either store."""
    candidates: list[bytes] = []

    if KEY_PATH.is_file():
        try:
            encoded = KEY_PATH.read_text(encoding="ascii").strip()
            candidates.append(_decode_key(encoded, "the local fallback file"))
        except (OSError, UnicodeError, CredentialStoreError):
            # A keyring copy may still unlock the credentials. Do not replace
            # the malformed fallback until a candidate authenticates them.
            pass

    try:
        import keyring

        encoded = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
        if encoded:
            key = _decode_key(encoded, "the OS keyring")
            if key not in candidates:
                candidates.append(key)
    except Exception:
        # keyring can be installed while its SecretService/desktop backend is
        # unavailable. The private file is the deterministic fallback.
        pass
    return candidates


def _persist_key(key: bytes) -> None:
    """Persist the local fallback first, then mirror it to keyring best-effort."""
    encoded = _encode_key(key)
    local_is_current = False
    try:
        local_is_current = _decode_key(
            KEY_PATH.read_text(encoding="ascii").strip(), "the local fallback file"
        ) == key
        if os.name != "nt" and KEY_PATH.stat().st_mode & 0o077:
            local_is_current = False
    except (OSError, UnicodeError, CredentialStoreError):
        pass
    if not local_is_current:
        _atomic_private_write(KEY_PATH, encoded.encode("ascii"))
    try:
        import keyring

        if keyring.get_password(KEYRING_SERVICE, KEYRING_USER) != encoded:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USER, encoded)
    except Exception:
        # Fedora headless sessions and packaged apps may not have an unlocked
        # SecretService. The owner-only local copy remains authoritative.
        pass


def _encryption_key(create: bool) -> bytes:
    """Load a durable key, creating one only for an explicit credential save."""
    candidates = _key_candidates()
    if candidates:
        key = candidates[0]
        if create:
            _persist_key(key)
        return key
    if not create:
        raise CredentialStoreError(
            "The Google credential encryption key is unavailable. Run authorize again "
            "with a Google Desktop OAuth client JSON file; the existing encrypted "
            "credential file was preserved."
        )
    key = os.urandom(32)
    _persist_key(key)
    return key


def _save_credentials(payload: dict[str, Any], *, encryption_key: bytes | None = None) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if encryption_key is None:
        key = _encryption_key(create=True)
    else:
        key = encryption_key
        # Always establish the private fallback before replacing the
        # ciphertext. A later keyring outage can therefore never strand a
        # refreshed token.
        _persist_key(key)
    nonce = os.urandom(12)
    plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _CREDENTIAL_AAD)
    envelope = {
        "version": 1,
        "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
        "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
    }
    _atomic_private_write(CREDENTIAL_PATH, json.dumps(envelope).encode("utf-8"))


def _decode_envelope_value(envelope: dict[str, Any], field: str) -> bytes:
    encoded = envelope.get(field)
    if not isinstance(encoded, str):
        raise CredentialStoreError(
            "The encrypted Google credential file is malformed and was preserved"
        )
    try:
        return base64.b64decode(encoded.encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise CredentialStoreError(
            "The encrypted Google credential file is malformed and was preserved"
        ) from exc


def _load_credentials_with_key(*, repair_key: bool) -> tuple[dict[str, Any], bytes]:
    if not CREDENTIAL_PATH.is_file():
        raise CredentialStoreError("Google is not connected. Run the authorize action first.")
    try:
        envelope = json.loads(CREDENTIAL_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CredentialStoreError(
            "The encrypted Google credential file is malformed and was preserved"
        ) from exc
    if not isinstance(envelope, dict):
        raise CredentialStoreError(
            "The encrypted Google credential file is malformed and was preserved"
        )
    if envelope.get("version") != 1:
        raise CredentialStoreError(
            "Unsupported encrypted Google credential format; the file was preserved"
        )
    nonce = _decode_envelope_value(envelope, "nonce")
    ciphertext = _decode_envelope_value(envelope, "ciphertext")
    if len(nonce) != 12 or len(ciphertext) < 16:
        raise CredentialStoreError(
            "The encrypted Google credential file is malformed and was preserved"
        )

    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    candidates = _key_candidates()
    if not candidates:
        # Keep one consistent, actionable message for the common Fedora case
        # where keyring exists but has no usable SecretService backend.
        _encryption_key(create=False)
    plaintext = None
    selected_key = None
    for candidate in candidates:
        try:
            plaintext = AESGCM(candidate).decrypt(nonce, ciphertext, _CREDENTIAL_AAD)
            selected_key = candidate
            break
        except InvalidTag:
            continue
    if plaintext is None or selected_key is None:
        raise CredentialStoreError(
            "The encrypted Google credentials cannot be unlocked with the available key. "
            "Run authorize again with a Google Desktop OAuth client JSON file; the existing "
            "credential file was preserved."
        )
    try:
        payload = json.loads(plaintext)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CredentialStoreError(
            "The decrypted Google credential payload is malformed; the file was preserved"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("token"), dict):
        raise CredentialStoreError(
            "The decrypted Google credential payload is invalid; the file was preserved"
        )
    if repair_key:
        # Repair only after AES-GCM authenticated the candidate. This safely
        # handles old keyring-only files and stale/mismatched key copies.
        _persist_key(selected_key)
    return payload, selected_key


def _load_credentials(*, repair_key: bool = False) -> dict[str, Any]:
    payload, _ = _load_credentials_with_key(repair_key=repair_key)
    return payload


def _google_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google dependencies are missing. Install requirements.txt and restart Selene."
        ) from exc
    return Request, Credentials, InstalledAppFlow, build


def _authorize(client_secrets_file: str | None) -> str:
    if not client_secrets_file:
        return _result(
            error="client_secrets_file is required for first-time authorization",
            setup=(
                "Create a Desktop OAuth client in Google Cloud, enable Calendar API and "
                "Tasks API, download its JSON, then call authorize with that file path."
            ),
        )
    path = Path(os.path.abspath(os.path.expanduser(client_secrets_file)))
    if not path.is_file():
        return _result(error="The OAuth client JSON file does not exist")
    try:
        client_config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(client_config, dict) or "installed" not in client_config:
            return _result(error="The file must be a Google Desktop-app OAuth client JSON file")
        _, _, InstalledAppFlow, _ = _google_imports()
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        credentials = flow.run_local_server(
            host="127.0.0.1",
            bind_addr="127.0.0.1",
            port=0,
            timeout_seconds=OAUTH_WAIT_TIMEOUT_SECONDS,
            access_type="offline",
            prompt="consent",
            open_browser=True,
            authorization_prompt_message="Opening Google authorization in your browser...",
            success_message="Selene is connected. You may close this tab.",
        )
        _save_credentials({
            "client_config": client_config,
            "token": json.loads(credentials.to_json()),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        })
        return _result(
            ok=True,
            connected=True,
            credential_file=str(CREDENTIAL_PATH),
            encrypted=True,
            note="The downloaded source JSON is not modified; delete it yourself after verifying the connection.",
        )
    except Exception as exc:
        return _result(error=_safe_error(exc))


def _services() -> tuple[Any, Any]:
    Request, Credentials, _, build = _google_imports()
    stored, encryption_key = _load_credentials_with_key(repair_key=True)
    credentials = Credentials.from_authorized_user_info(stored["token"], SCOPES)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        stored["token"] = json.loads(credentials.to_json())
        stored["saved_at"] = datetime.now(timezone.utc).isoformat()
        _save_credentials(stored, encryption_key=encryption_key)
    if not credentials.valid:
        raise RuntimeError("Google authorization is invalid or expired; authorize again")
    return (
        build("calendar", "v3", credentials=credentials, cache_discovery=False),
        build("tasks", "v1", credentials=credentials, cache_discovery=False),
    )


def _limit(value: int | None) -> int:
    try:
        return max(1, min(int(value or 25), MAX_RESULTS))
    except (TypeError, ValueError):
        return 25


def _event_time(value: str, timezone_name: str | None) -> dict[str, str]:
    value = value.strip()
    if len(value) == 10:
        return {"date": value}
    result = {"dateTime": value}
    if timezone_name:
        result["timeZone"] = timezone_name
    return result


def _task_due(value: str) -> str:
    """Google Tasks accepts RFC3339 but only retains the due-date portion."""
    value = value.strip()
    return f"{value}T00:00:00.000Z" if len(value) == 10 else value


def _selected(item: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Keep API responses useful to the model without flooding its context."""
    return {field: item[field] for field in fields if item.get(field) is not None}


def _calendar_summary(item: dict[str, Any]) -> dict[str, Any]:
    return _selected(item, ("id", "summary", "primary", "accessRole", "timeZone"))


def _event_summary(item: dict[str, Any]) -> dict[str, Any]:
    event = _selected(item, (
        "id", "summary", "location", "start", "end", "status", "eventType",
        "birthdayProperties", "occurrence_date", "recurringEventId",
    ))
    if item.get("description"):
        description = str(item["description"])
        event["description"] = description[:240] + ("…" if len(description) > 240 else "")
    if item.get("attendees"):
        event["attendees"] = [
            _selected(person, ("email", "displayName", "responseStatus", "self"))
            for person in item["attendees"][:10]
        ]
        if len(item["attendees"]) > 10:
            event["additional_attendee_count"] = len(item["attendees"]) - 10
    return event


def _task_list_summary(item: dict[str, Any]) -> dict[str, Any]:
    return _selected(item, ("id", "title", "updated"))


def _task_summary(item: dict[str, Any]) -> dict[str, Any]:
    return _selected(item, (
        "id", "title", "notes", "due", "status", "completed", "parent",
        "position", "webViewLink", "updated",
    ))


def _birthday_window(days_ahead: int | None, timezone_name: str | None) -> tuple[date, date, Any]:
    try:
        days = max(1, min(int(days_ahead or 90), 3660))
    except (TypeError, ValueError):
        days = 90
    try:
        zone = ZoneInfo(timezone_name) if timezone_name else datetime.now().astimezone().tzinfo
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone_name}") from exc
    start_date = datetime.now(zone).date()
    return start_date, start_date + timedelta(days=days + 1), zone


def _annual_occurrence(month: int, day: int, start_date: date, end_date: date) -> date | None:
    for year in range(start_date.year, end_date.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            # Google defines a Feb 29 birthday recurrence as the last day of
            # February when that date does not exist in the occurrence year.
            if month == 2 and day == 29:
                candidate = date(year, 2, 28)
            else:
                continue
        if start_date <= candidate < end_date:
            return candidate
    return None


def _upcoming_birthday(item: dict[str, Any], start_date: date, end_date: date) -> dict[str, Any] | None:
    properties = item.get("birthdayProperties") or {}
    birthday_type = properties.get("type", "birthday")
    if birthday_type not in {"birthday", "self"}:
        return None
    raw_date = str((item.get("start") or {}).get("date") or "")
    try:
        stored_date = date.fromisoformat(raw_date)
    except ValueError:
        return None
    occurrence = _annual_occurrence(stored_date.month, stored_date.day, start_date, end_date)
    if occurrence is None:
        return None
    normalized = dict(item)
    normalized["start"] = {"date": occurrence.isoformat()}
    normalized["end"] = {"date": (occurrence + timedelta(days=1)).isoformat()}
    normalized["occurrence_date"] = occurrence.isoformat()
    return _event_summary(normalized)


def _birthday_key(item: dict[str, Any]) -> tuple[str, str, str]:
    properties = item.get("birthdayProperties") or {}
    owner = str(
        properties.get("contact")
        or item.get("recurringEventId")
        or item.get("id")
        or item.get("summary")
        or ""
    )
    return (owner.casefold(), str(item.get("occurrence_date", "")), str(properties.get("type", "birthday")))


def _calendar_action(calendar: Any, action: str, params: dict[str, Any]) -> dict[str, Any]:
    calendar_id = str(params.get("calendar_id") or "primary")
    if action == "list_calendars":
        response = calendar.calendarList().list(maxResults=_limit(params.get("max_results"))).execute()
        return {"calendars": [_calendar_summary(item) for item in response.get("items", [])]}
    if action == "list_events":
        kwargs = {
            "calendarId": calendar_id,
            "maxResults": _limit(params.get("max_results")),
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if params.get("time_min"):
            kwargs["timeMin"] = params["time_min"]
        if params.get("time_max"):
            kwargs["timeMax"] = params["time_max"]
        if params.get("query"):
            kwargs["q"] = params["query"]
        response = calendar.events().list(**kwargs).execute()
        return {
            "events": [_event_summary(item) for item in response.get("items", [])],
            "next_page_token": response.get("nextPageToken"),
        }
    if action == "list_birthdays":
        start_date, end_date, zone = _birthday_window(params.get("days_ahead"), params.get("timezone"))
        midnight = datetime.combine(start_date, datetime.min.time(), tzinfo=zone)
        end_midnight = datetime.combine(end_date, datetime.min.time(), tzinfo=zone)
        max_results = _limit(params.get("max_results") or 100)
        candidates: list[dict[str, Any]] = []

        # Current instances are authoritative. Fetch recurring masters as a
        # fallback because contact birthdays can expose their stored source
        # year on some Calendar/API combinations.
        for single_events, bounded in ((True, True), (False, False)):
            page_token = None
            for _ in range(5):
                kwargs: dict[str, Any] = {
                    "calendarId": calendar_id,
                    "eventTypes": ["birthday"],
                    "singleEvents": single_events,
                    "maxResults": max_results,
                }
                if bounded:
                    kwargs.update({
                        "orderBy": "startTime",
                        "timeMin": midnight.isoformat(),
                        "timeMax": end_midnight.isoformat(),
                    })
                if page_token:
                    kwargs["pageToken"] = page_token
                response = calendar.events().list(**kwargs).execute()
                candidates.extend(response.get("items", []))
                page_token = response.get("nextPageToken")
                if not page_token or len(candidates) >= max_results * 2:
                    break

        birthdays: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in candidates:
            normalized = _upcoming_birthday(item, start_date, end_date)
            if normalized is not None:
                birthdays.setdefault(_birthday_key(normalized), normalized)
        ordered = sorted(
            birthdays.values(),
            key=lambda item: (item.get("occurrence_date", ""), str(item.get("summary", "")).casefold()),
        )[:max_results]
        return {
            "birthdays": ordered,
            "window_start": start_date.isoformat(),
            "window_end_inclusive": (end_date - timedelta(days=1)).isoformat(),
            "timezone": str(zone),
            "count": len(ordered),
        }
    if action == "create_event":
        if not params.get("summary") or not params.get("start") or not params.get("end"):
            raise ValueError("summary, start, and end are required to create an event")
        body = {
            "summary": params["summary"],
            "start": _event_time(params["start"], params.get("timezone")),
            "end": _event_time(params["end"], params.get("timezone")),
        }
        for field in ("description", "location"):
            if params.get(field) is not None:
                body[field] = params[field]
        if params.get("attendees"):
            body["attendees"] = [{"email": email} for email in params["attendees"]]
        event = calendar.events().insert(calendarId=calendar_id, body=body, sendUpdates="all").execute()
        return {"ok": True, "event": event}
    if action == "update_event":
        if not params.get("event_id"):
            raise ValueError("event_id is required to update an event")
        body: dict[str, Any] = {}
        for field in ("summary", "description", "location"):
            if params.get(field) is not None:
                body[field] = params[field]
        for field in ("start", "end"):
            if params.get(field) is not None:
                body[field] = _event_time(params[field], params.get("timezone"))
        if params.get("attendees") is not None:
            body["attendees"] = [{"email": email} for email in params["attendees"]]
        if not body:
            raise ValueError("At least one event field must be supplied")
        event = calendar.events().patch(
            calendarId=calendar_id, eventId=params["event_id"], body=body, sendUpdates="all"
        ).execute()
        return {"ok": True, "event": event}
    if action == "delete_event":
        if not params.get("event_id") or params.get("confirmed") is not True:
            raise ValueError("Deleting an event requires event_id and confirmed=true")
        calendar.events().delete(
            calendarId=calendar_id, eventId=params["event_id"], sendUpdates="all"
        ).execute()
        return {"ok": True, "deleted_event_id": params["event_id"]}
    raise ValueError(f"Unsupported Calendar action: {action}")


def _task_action(tasks: Any, action: str, params: dict[str, Any]) -> dict[str, Any]:
    tasklist_id = str(params.get("tasklist_id") or "@default")
    if action == "list_task_lists":
        response = tasks.tasklists().list(maxResults=_limit(params.get("max_results"))).execute()
        return {"task_lists": [_task_list_summary(item) for item in response.get("items", [])]}
    if action == "list_tasks":
        response = tasks.tasks().list(
            tasklist=tasklist_id,
            maxResults=_limit(params.get("max_results")),
            showCompleted=params.get("show_completed", True),
            showHidden=False,
        ).execute()
        items = response.get("items", [])
        query = str(params.get("query") or "").strip().casefold()
        if query:
            terms = [term for term in re.findall(r"[\w#:+.-]+", query) if term]

            def matches_query(item: dict[str, Any]) -> bool:
                haystack = f"{item.get('title', '')} {item.get('notes', '')}".casefold()
                return query in haystack or all(term in haystack for term in terms)

            items = [item for item in items if matches_query(item)]
        return {
            "tasks": [_task_summary(item) for item in items],
            "query": params.get("query") or None,
            "next_page_token": response.get("nextPageToken"),
        }
    if action == "create_task":
        if not params.get("title"):
            raise ValueError("title is required to create a task")
        body = {"title": params["title"]}
        if params.get("notes") is not None:
            body["notes"] = params["notes"]
        if params.get("due") is not None:
            body["due"] = _task_due(params["due"])
        request = tasks.tasks().insert(tasklist=tasklist_id, body=body)
        task = request.execute()
        return {"ok": True, "task": task}
    if action == "update_task":
        if not params.get("task_id"):
            raise ValueError("task_id is required to update a task")
        body = {}
        for field in ("title", "notes", "status"):
            if params.get(field) is not None:
                body[field] = params[field]
        if params.get("due") is not None:
            body["due"] = _task_due(params["due"])
        if not body:
            raise ValueError("At least one task field must be supplied")
        task = tasks.tasks().patch(tasklist=tasklist_id, task=params["task_id"], body=body).execute()
        return {"ok": True, "task": task}
    if action == "delete_task":
        if not params.get("task_id") or params.get("confirmed") is not True:
            raise ValueError("Deleting a task requires task_id and confirmed=true")
        tasks.tasks().delete(tasklist=tasklist_id, task=params["task_id"]).execute()
        return {"ok": True, "deleted_task_id": params["task_id"]}
    raise ValueError(f"Unsupported Tasks action: {action}")


def google_workspace(
    action: str,
    client_secrets_file: str | None = None,
    calendar_id: str = "primary",
    event_id: str | None = None,
    summary: str | None = None,
    description: str | None = None,
    location: str | None = None,
    start: str | None = None,
    end: str | None = None,
    timezone: str | None = None,
    days_ahead: int = 90,
    attendees: list[str] | None = None,
    time_min: str | None = None,
    time_max: str | None = None,
    query: str | None = None,
    tasklist_id: str = "@default",
    task_id: str | None = None,
    title: str | None = None,
    notes: str | None = None,
    due: str | None = None,
    status: str | None = None,
    show_completed: bool = True,
    max_results: int = 25,
    confirmed: bool = False,
) -> str:
    """Authorize and manage the user's Google Calendar events and Tasks."""
    action = str(action or "").strip().lower()
    if action == "status":
        credential_file_present = CREDENTIAL_PATH.is_file()
        status_result: dict[str, Any] = {
            "connected": False,
            "encrypted": True,
            "credential_file": str(CREDENTIAL_PATH),
            "credential_file_present": credential_file_present,
        }
        if not credential_file_present:
            status_result.update({
                "credential_store_status": "missing",
                "authorization_required": True,
            })
            return _result(**status_result)
        try:
            # This is a local decryptability check only: no Google imports,
            # network requests, token refreshes, or key-store repairs.
            _load_credentials(repair_key=False)
        except ImportError:
            status_result.update({
                "credential_store_status": "dependency_unavailable",
                "error": (
                    "Google credential encryption support is unavailable. "
                    "Install cryptography and restart Selene."
                ),
            })
        except Exception as exc:
            status_result.update({
                "credential_store_status": "unreadable",
                "reauthorization_required": True,
                "error": _safe_error(exc),
            })
        else:
            status_result.update({
                "connected": True,
                "credential_store_status": "ready",
                "authorization_required": False,
            })
        return _result(**status_result)
    if action == "authorize":
        return _authorize(client_secrets_file)
    if action == "disconnect":
        if confirmed is not True:
            return _result(error="Disconnecting Google requires confirmed=true")
        try:
            CREDENTIAL_PATH.unlink(missing_ok=True)
            return _result(ok=True, connected=False)
        except OSError as exc:
            return _result(error=str(exc))

    params = locals()
    try:
        calendar, tasks = _services()
        if action in {"list_calendars", "list_events", "list_birthdays", "create_event", "update_event", "delete_event"}:
            return _result(**_calendar_action(calendar, action, params))
        if action in {"list_task_lists", "list_tasks", "create_task", "update_task", "delete_task"}:
            return _result(**_task_action(tasks, action, params))
        return _result(error=f"Unknown Google action: {action}")
    except Exception as exc:
        # Do not include request objects, tokens, or raw response bodies in tool
        # output; Google client exceptions can otherwise echo sensitive data.
        return _result(error=_safe_error(exc), action=action)
