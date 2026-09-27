"""Hybrid safety gate for customer messages before agent execution."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Callable, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from customer_service.config.settings import Settings
from customer_service.nodes.safety_check.prompt import build_system_prompt
from customer_service.privacy.redaction import redact_text
from customer_service.state.models import AgentState, EscalationReason


SAFE_REFUSAL = "I can help with customer-support questions, but I can't help with that request."

_INJECTION = re.compile(
    r"\b(ignore (?:all |any )?(?:previous|prior)|system prompt|developer message|"
    r"reveal .*instructions|jailbreak|act as (?:a )?(?:system|developer))\b",
    re.IGNORECASE,
)
_ABUSE = re.compile(r"\b(fuck|shit|asshole|bitch|kill yourself)\b", re.IGNORECASE)
_HUMAN_REQUEST = re.compile(
    r"\b(?:human|representative|support agent|manager|real person)\b|"
    r"(?:need|want|talk|speak|connect).{0,24}\bperson\b",
    re.IGNORECASE,
)
_RETURN_EXCEPTION = re.compile(
    r"\b(damaged|defective|wrong item|missing item|broken item)\b", re.IGNORECASE
)
_BILLING_DISPUTE = re.compile(
    r"\b(unauthorized|fraud|duplicate|charged twice|wrong charge|incorrect charge|wrong amount)\b",
    re.IGNORECASE,
)


class SafetyDecision(BaseModel):
    action: Literal["allow", "refuse", "escalate"]

    model_config = ConfigDict(extra="forbid")


class SafetyJudge(Protocol):
    def invoke(self, input: object) -> SafetyDecision | dict[str, Any]: ...


def create_structured_judge(model: BaseChatModel) -> SafetyJudge:
    """Use Gemini's native JSON-schema mode, not its tool-call emulation."""

    return model.with_structured_output(SafetyDecision, method="json_schema")


def redact_sensitive_text(text: str) -> str:
    """Remove direct identifiers before ambiguous text reaches the safety model."""

    return redact_text(text)


def latest_customer_message(state: AgentState) -> str:
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def _deterministic_decision(text: str) -> tuple[str, EscalationReason | None] | None:
    if _INJECTION.search(text) or _ABUSE.search(text):
        return "refuse", None
    if _HUMAN_REQUEST.search(text):
        return "escalate", EscalationReason.HUMAN_REQUEST
    if _RETURN_EXCEPTION.search(text):
        return "escalate", EscalationReason.RETURN_EXCEPTION
    if _BILLING_DISPUTE.search(text):
        return "escalate", EscalationReason.BILLING_DISPUTE
    return None


def build_node(
    *, settings: Settings, judge: SafetyJudge
) -> Callable[[AgentState], dict[str, object]]:
    prompt = SystemMessage(content=build_system_prompt(settings))

    def safety_check(state: AgentState) -> dict[str, object]:
        text = latest_customer_message(state)
        common = {"last_activity_at": datetime.now(UTC)}
        if decision := _deterministic_decision(text):
            action, reason = decision
            if action == "refuse":
                return {
                    **common,
                    "safety_action": "refuse",
                    "messages": [AIMessage(content=SAFE_REFUSAL)],
                }
            return {
                **common,
                "safety_action": "escalate",
                "escalation_reason": reason,
            }

        request = HumanMessage(content=redact_sensitive_text(text))
        try:
            decision = SafetyDecision.model_validate(judge.invoke([prompt, request]))
        except Exception:
            return {
                **common,
                "safety_action": "escalate",
                "escalation_reason": EscalationReason.SAFETY_UNCERTAIN,
            }

        if decision.action == "refuse":
            return {
                **common,
                "safety_action": "refuse",
                "messages": [AIMessage(content=SAFE_REFUSAL)],
            }
        if decision.action == "escalate":
            return {
                **common,
                "safety_action": "escalate",
                "escalation_reason": EscalationReason.SAFETY_UNCERTAIN,
            }
        return {**common, "safety_action": "allow"}

    return safety_check
