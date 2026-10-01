"""Safety gate: decides whether a customer message is allowed, refused or escalated.

Three layers run in order, and the first one that is sure decides:

1. deterministic.py  instant English keyword rules
2. jev.py            fast classifier that answers only when it is clearly sure
3. gemini.py         careful classifier that decides whenever Jev cannot

If Gemini also fails, the conversation is escalated to a human (fail closed).
This module owns that order and turns the final Verdict into a graph state update.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Callable, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from customer_service.config.settings import Settings
from customer_service.nodes.safety_check import deterministic
from customer_service.nodes.safety_check.gemini import (
    GeminiJudge,
    SafetyDecision,
    SafetyJudge,
    create_structured_judge,
)
from customer_service.nodes.safety_check.jev import JevFallback
from customer_service.nodes.safety_check.verdict import Verdict
from customer_service.privacy.redaction import redact_text
from customer_service.state.models import AgentState, EscalationReason

# SafetyDecision, SafetyJudge and create_structured_judge are re-exported
# because graph wiring and tests import them from this module.
__all__ = [
    "SAFE_REFUSAL",
    "Classifier",
    "SafetyDecision",
    "SafetyJudge",
    "build_node",
    "classify",
    "create_structured_judge",
    "latest_customer_message",
    "redact_sensitive_text",
]

LOGGER = logging.getLogger(__name__)

SAFE_REFUSAL = "I can help with customer-support questions, but I can't help with that request."


class Classifier(Protocol):
    """A safety layer: returns a Verdict, or raises when it cannot decide."""

    def check(self, text: str) -> Verdict: ...


def redact_sensitive_text(text: str) -> str:
    """Remove direct identifiers before text reaches any external model."""

    return redact_text(text)


def latest_customer_message(state: AgentState) -> str:
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def classify(text: str, *, gemini: Classifier, jev: Classifier | None = None) -> Verdict:
    """Run the layers in order and return the final verdict. Never raises."""

    started = time.monotonic()
    jev_fallback: str | None = None

    verdict = deterministic.check(text)
    if verdict is None:
        redacted = redact_sensitive_text(text)
        if jev is not None:
            try:
                verdict = jev.check(redacted)
            except JevFallback as fallback:
                jev_fallback = fallback.reason
            except Exception:
                jev_fallback = "jev_error"
        if verdict is None:
            try:
                verdict = gemini.check(redacted)
            except Exception as error:
                verdict = Verdict(
                    action="escalate",
                    source="fail_closed",
                    reason=EscalationReason.SAFETY_UNCERTAIN,
                    detail=type(error).__name__,
                )

    _log_verdict(verdict, jev_fallback, started)
    return verdict


def _log_verdict(verdict: Verdict, jev_fallback: str | None, started: float) -> None:
    """Log which layer decided and how long it took. Never log message text."""

    LOGGER.info(
        "safety_verdict",
        extra={
            "safety_action": verdict.action,
            "safety_source": verdict.source,
            "safety_detail": verdict.detail or "-",
            "jev_fallback": jev_fallback or "-",
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        },
    )


def _state_update(verdict: Verdict) -> dict[str, object]:
    update: dict[str, object] = {
        "last_activity_at": datetime.now(UTC),
        "safety_action": verdict.action,
    }
    if verdict.action == "refuse":
        update["messages"] = [AIMessage(content=SAFE_REFUSAL)]
    elif verdict.action == "escalate":
        update["escalation_reason"] = verdict.reason
    return update


def build_node(
    *, settings: Settings, judge: SafetyJudge, jev: Classifier | None = None
) -> Callable[[AgentState], dict[str, object]]:
    """`judge` is the Gemini classifier. `jev` is optional; without it Gemini decides alone."""

    gemini = GeminiJudge(judge, settings)

    def safety_check(state: AgentState) -> dict[str, object]:
        verdict = classify(latest_customer_message(state), gemini=gemini, jev=jev)
        return _state_update(verdict)

    return safety_check