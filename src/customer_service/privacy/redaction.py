"""Redact direct identifiers before data leaves the application process."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
ORDER_ID = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
SECRET_VALUE = re.compile(
    r"(?i)\b(?:api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"
)
SENSITIVE_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "cookie", "password", "secret", "token"}
)


def redact_text(text: str) -> str:
    """Replace common direct identifiers with stable, non-reversible labels."""

    redacted = EMAIL.sub("[EMAIL]", text)
    redacted = PHONE.sub("[PHONE]", redacted)
    redacted = CARD.sub("[CARD]", redacted)
    redacted = ORDER_ID.sub("[ORDER_ID]", redacted)
    return SECRET_VALUE.sub("[SECRET]", redacted)


def stable_thread_reference(thread_id: str, salt: str) -> str:
    """Return a trace-safe correlation value without exposing caller input."""

    digest = hashlib.sha256(f"{salt}:{thread_id}".encode()).hexdigest()[:16]
    return f"thread-{digest}"


def sanitize_for_trace(value: Any, *, thread_id_salt: str = "") -> Any:
    """Recursively sanitize LangChain/LangSmith-compatible payload data."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in SENSITIVE_KEYS or normalized_key.endswith("_secret"):
                safe[str(key)] = "[SECRET]"
            elif normalized_key == "thread_id" and isinstance(item, str):
                safe[str(key)] = stable_thread_reference(item, thread_id_salt)
            else:
                safe[str(key)] = sanitize_for_trace(item, thread_id_salt=thread_id_salt)
        return safe
    if isinstance(value, tuple):
        return tuple(sanitize_for_trace(item, thread_id_salt=thread_id_salt) for item in value)
    if isinstance(value, list):
        return [sanitize_for_trace(item, thread_id_salt=thread_id_salt) for item in value]
    if hasattr(value, "model_dump"):
        return sanitize_for_trace(value.model_dump(), thread_id_salt=thread_id_salt)
    return value
