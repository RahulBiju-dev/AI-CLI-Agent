"""Resilient HTTP API execution with bounded retries and secret-safe auth."""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


RETRYABLE = {408, 425, 429, 500, 502, 503, 504}
MAX_URL_CHARS = 4_096
MAX_ENDPOINTS = 10
_SENSITIVE_HEADER_MARKERS = ("authorization", "api-key", "apikey", "token", "cookie", "secret")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def _secret(name: str | None) -> str | None:
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise ValueError("Credential environment-variable names must use letters, digits, and underscores")
    return os.environ.get(name, "")


def _validate_endpoint(value: Any, label: str = "URL") -> str:
    endpoint = str(value or "").strip()
    if not endpoint or len(endpoint) > MAX_URL_CHARS or any(ord(char) < 32 for char in endpoint):
        raise ValueError(f"{label} is empty or invalid")
    try:
        parsed = urlparse(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must use http or https and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain embedded credentials")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{label} port is outside the valid range")
    return endpoint


def _redacted_url(value: str) -> str:
    """Keep endpoint identity in audit output without echoing query secrets."""
    parsed = urlparse(value)
    return parsed._replace(params="", query="", fragment="").geturl()


def _is_sensitive_header(name: object) -> bool:
    normalized = str(name).strip().casefold().replace("_", "-")
    return any(marker in normalized for marker in _SENSITIVE_HEADER_MARKERS)


def _safe_response_headers(headers: Any) -> dict[str, str]:
    if not hasattr(headers, "items"):
        return {}
    safe: dict[str, str] = {}
    used = 0
    for key, value in headers.items():
        key_text = str(key)[:200]
        value_text = str(value)[:4_000]
        if _is_sensitive_header(key_text) or "\r" in value_text or "\n" in value_text:
            continue
        if used + len(key_text) + len(value_text) > 16_000 or len(safe) >= 100:
            break
        safe[key_text] = value_text
        used += len(key_text) + len(value_text)
    return safe


def _bounded_response_text(response: Any, max_chars: int) -> tuple[str, bool]:
    """Read a streamed response without allowing an unbounded body allocation."""
    parts: list[str] = []
    used = 0
    truncated = False
    for chunk in response.iter_content(chunk_size=65_536, decode_unicode=True):
        if not chunk:
            continue
        if isinstance(chunk, bytes):
            try:
                chunk = chunk.decode(getattr(response, "encoding", None) or "utf-8", errors="replace")
            except LookupError:
                chunk = chunk.decode("utf-8", errors="replace")
        else:
            chunk = str(chunk)
        remaining = max_chars + 1 - used
        if remaining <= 0:
            truncated = True
            break
        parts.append(chunk[:remaining])
        used += min(len(chunk), remaining)
        if len(chunk) > remaining or used > max_chars:
            truncated = True
            break
    body = "".join(parts)
    if len(body) > max_chars:
        truncated = True
        body = body[:max_chars]
    return body, truncated


def _prepare_auth(auth: dict, headers: dict[str, str], timeout: float) -> tuple[Any, dict[str, str]]:
    kind = auth.get("type", "none")
    request_auth = None
    if kind == "bearer":
        token = _secret(auth.get("token_env"))
        if not token:
            raise ValueError("Bearer token environment variable is unset")
        headers["Authorization"] = f"Bearer {token}"
    elif kind == "api_key":
        value = _secret(auth.get("value_env"))
        if not value:
            raise ValueError("API key environment variable is unset")
        header_name = str(auth.get("header", "X-API-Key"))
        if not _HEADER_NAME.fullmatch(header_name):
            raise ValueError("API key header name is invalid")
        headers[header_name] = value
    elif kind == "basic":
        username = _secret(auth.get("username_env"))
        password = _secret(auth.get("password_env"))
        if not username or not password:
            raise ValueError("Basic auth environment variables are unset")
        request_auth = (username, password)
    elif kind == "oauth2_client_credentials":
        token_url = _validate_endpoint(auth.get("token_url"), "OAuth token_url")
        client_id = _secret(auth.get("client_id_env"))
        client_secret = _secret(auth.get("client_secret_env"))
        if not token_url or not client_id or not client_secret:
            raise ValueError("OAuth token_url/client environment variables are required")
        response = requests.post(
            token_url,
            data={"grant_type": "client_credentials", "scope": auth.get("scope", "")},
            auth=(client_id, client_secret), timeout=timeout, stream=True,
        )
        try:
            response.raise_for_status()
            token_body, token_truncated = _bounded_response_text(response, 100_000)
            if token_truncated:
                raise ValueError("OAuth response exceeded the 100,000-character limit")
            try:
                token_payload = json.loads(token_body)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("OAuth response was not valid JSON") from exc
            token = token_payload.get("access_token") if isinstance(token_payload, dict) else None
        finally:
            response.close()
        if not token:
            raise ValueError("OAuth response did not contain access_token")
        headers["Authorization"] = f"Bearer {token}"
    elif kind != "none":
        raise ValueError(f"Unsupported auth type: {kind}")
    return request_auth, headers


def _documentation_suggestions(primary: str, documentation: dict | None) -> list[str]:
    urls: list[str] = []
    if isinstance(documentation, dict) and isinstance(documentation.get("paths"), dict):
        base = str(documentation.get("base_url") or f"{urlparse(primary).scheme}://{urlparse(primary).netloc}")
        for path, definition in documentation["paths"].items():
            if len(urls) >= 10:
                break
            deprecated = isinstance(definition, dict) and definition.get("deprecated") is True
            candidate = urljoin(base.rstrip("/") + "/", str(path).lstrip("/"))
            try:
                candidate = _validate_endpoint(candidate, "documentation endpoint")
            except ValueError:
                continue
            if not deprecated and candidate not in urls:
                urls.append(_redacted_url(candidate))
    return urls


def api_orchestrator(
    request: dict,
    auth: dict | None = None,
    retry: dict | None = None,
    alternative_endpoints: list[str] | None = None,
    documentation: dict | None = None,
) -> str:
    """Execute an API call with auth refresh, backoff, and endpoint failover."""
    if not isinstance(request, dict):
        return json.dumps({"error": "request must be an object"})
    if auth is not None and not isinstance(auth, dict):
        return json.dumps({"error": "auth must be an object"})
    if retry is not None and not isinstance(retry, dict):
        return json.dumps({"error": "retry must be an object"})
    if documentation is not None and not isinstance(documentation, dict):
        return json.dumps({"error": "documentation must be an object"})
    if alternative_endpoints is not None and not isinstance(alternative_endpoints, list):
        return json.dumps({"error": "alternative_endpoints must be an array"})
    method = str(request.get("method", "GET")).upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        return json.dumps({"error": f"Unsupported HTTP method: {method}"})
    policy = retry or {}
    try:
        attempts = max(1, min(int(policy.get("max_attempts", 3)), 6))
        raw_base_delay = float(policy.get("base_delay", 0.5))
        raw_timeout = float(request.get("timeout", 20))
        max_chars = max(1000, min(int(request.get("max_response_chars", 20000)), 100000))
    except (TypeError, ValueError, OverflowError):
        return json.dumps({"error": "retry/timeout/response limits must be numeric"})
    if not math.isfinite(raw_base_delay) or not math.isfinite(raw_timeout):
        return json.dumps({"error": "retry delay and timeout must be finite"})
    base_delay = max(0.0, min(raw_base_delay, 10.0))
    timeout = max(0.5, min(raw_timeout, 120.0))
    if len(alternative_endpoints or []) + 1 > MAX_ENDPOINTS:
        return json.dumps({"error": f"At most {MAX_ENDPOINTS} total endpoints are allowed"})
    try:
        primary = _validate_endpoint(request.get("url"), "request.url")
        urls = [primary] + [
            _validate_endpoint(url, f"alternative_endpoints[{index}]")
            for index, url in enumerate(alternative_endpoints or [])
        ]
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    suggestions = _documentation_suggestions(primary, documentation)
    audit: list[dict] = []

    raw_headers = request.get("headers", {})
    if not isinstance(raw_headers, dict):
        return json.dumps({"error": "request.headers must be an object"})
    if len(raw_headers) > 100:
        return json.dumps({"error": "request.headers may contain at most 100 entries"})
    literal_secret_headers = sorted(str(key) for key in raw_headers if _is_sensitive_header(key))
    if literal_secret_headers:
        return json.dumps({
            "error": "Sensitive request headers must be configured through auth environment-variable names",
            "headers": literal_secret_headers,
        })
    headers = {str(key): str(value) for key, value in raw_headers.items()}
    invalid_header_names = sorted(name for name in headers if not _HEADER_NAME.fullmatch(name))
    if invalid_header_names:
        return json.dumps({"error": "request contains invalid header names", "headers": invalid_header_names})
    if any(len(value) > 16_384 for value in headers.values()):
        return json.dumps({"error": "request header values may contain at most 16,384 characters"})
    if any("\r" in value or "\n" in value for value in headers.values()):
        return json.dumps({"error": "request header values cannot contain newlines"})
    try:
        request_auth, headers = _prepare_auth(auth or {}, headers, timeout)
    except requests.RequestException:
        return json.dumps({
            "error": "Authentication endpoint request failed",
            "secret_policy": "Credentials must be supplied through environment-variable names",
        })
    except Exception as exc:
        return json.dumps({"error": str(exc), "secret_policy": "Credentials must be supplied through environment-variable names"})

    last_error = "No request attempted"
    for endpoint_index, endpoint in enumerate(urls):
        for attempt in range(1, attempts + 1):
            response = None
            try:
                response = requests.request(
                    method, endpoint, headers=headers, params=request.get("params"),
                    json=request.get("json"), data=request.get("data"), auth=request_auth,
                    timeout=timeout, allow_redirects=bool(request.get("allow_redirects", True)),
                    stream=True,
                )
                deprecation_header = response.headers.get("Deprecation", "").strip().lower()
                deprecated = response.status_code == 410 or bool(response.headers.get("Sunset")) or deprecation_header not in {"", "false", "0"}
                safe_endpoint = _redacted_url(endpoint)
                audit.append({"endpoint": safe_endpoint, "attempt": attempt, "status": response.status_code, "deprecated": deprecated})
                if response.status_code == 401 and (auth or {}).get("type") == "oauth2_client_credentials" and attempt < attempts:
                    request_auth, headers = _prepare_auth(auth or {}, headers, timeout)
                    audit[-1]["auth_refreshed"] = True
                    continue
                if response.status_code not in RETRYABLE and response.status_code != 404 and not deprecated:
                    body, truncated = _bounded_response_text(response, max_chars)
                    return json.dumps({
                        "ok": response.ok, "status": response.status_code, "endpoint": safe_endpoint,
                        "headers": _safe_response_headers(response.headers),
                        "body": body, "truncated": truncated, "attempts": audit,
                    }, ensure_ascii=False)
                last_error = f"HTTP {response.status_code}"
                if deprecated or response.status_code in {404, 410}:
                    break
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after), 30.0) if retry_after and retry_after.isdigit() else min(base_delay * (2 ** (attempt - 1)), 30.0)
                if attempt < attempts:
                    time.sleep(delay)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: request failed"
                audit.append({"endpoint": _redacted_url(endpoint), "attempt": attempt, "error": type(exc).__name__})
                if attempt < attempts:
                    time.sleep(min(base_delay * (2 ** (attempt - 1)), 30.0))
            except (TypeError, ValueError, UnicodeError) as exc:
                last_error = f"Invalid HTTP response: {type(exc).__name__}"
                audit.append({"endpoint": _redacted_url(endpoint), "attempt": attempt, "error": type(exc).__name__})
                break
            finally:
                if response is not None:
                    response.close()
        if endpoint_index + 1 >= len(urls):
            break
    return json.dumps({
        "ok": False,
        "error": last_error,
        "attempts": audit,
        "alternatives_considered": [_redacted_url(url) for url in urls[1:]],
        "documentation_suggestions": suggestions,
    }, ensure_ascii=False)
