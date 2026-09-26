"""Provider-neutral extraction of customer-facing text from message content."""

from __future__ import annotations

from typing import Any


def content_to_text(content: str | list[Any]) -> str:
    """Return text blocks only, intentionally dropping provider metadata."""

    if isinstance(content, str):
        return content
    return "".join(
        item
        if isinstance(item, str)
        else item["text"]
        if isinstance(item, dict) and isinstance(item.get("text"), str)
        else ""
        for item in content
    )
