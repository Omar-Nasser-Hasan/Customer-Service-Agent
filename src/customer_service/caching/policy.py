"""Make unsupported Gemini caching a visible, testable policy rather than a no-op."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheDecision:
    node_name: str
    status: str
    reason: str
    fingerprint: str


def static_context_fingerprint(*, prompt_revision: str, model: str, tool_schemas: list[dict[str, Any]]) -> str:
    """Version static context without including conversation or account data."""

    payload = json.dumps(
        {"prompt_revision": prompt_revision, "model": model, "tool_schemas": tool_schemas},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def decide_cache_policy(
    *,
    node_name: str,
    prompt_revision: str,
    model: str,
    tool_schemas: list[dict[str, Any]] | None = None,
    has_system_instruction: bool = True,
) -> CacheDecision:
    """Return an explicit decision for Gemini cached-content compatibility.

    Gemini rejects cached content combined with the LangChain assistant's tool
    binding/system instruction payload. We therefore never pretend that a
    normal response, transactional tool result, or unsupported model call was
    cached. A future provider-native adapter can turn an eligible ``miss`` into
    a real cache create/hit without changing callers.
    """

    schemas = tool_schemas or []
    fingerprint = static_context_fingerprint(
        prompt_revision=prompt_revision,
        model=model,
        tool_schemas=schemas,
    )
    if schemas:
        return CacheDecision(node_name, "bypass", "langchain_tool_binding_incompatible", fingerprint)
    if has_system_instruction:
        return CacheDecision(node_name, "bypass", "system_instruction_incompatible", fingerprint)
    return CacheDecision(node_name, "bypass", "below_provider_static_context_threshold", fingerprint)


def record_cache_decision(decision: CacheDecision) -> None:
    """Emit only non-sensitive cache metadata for operational visibility."""

    LOGGER.info(
        "model_cache_decision",
        extra={
            "node": decision.node_name,
            "cache_status": decision.status,
            "cache_reason": decision.reason,
            "cache_fingerprint": decision.fingerprint,
        },
    )
