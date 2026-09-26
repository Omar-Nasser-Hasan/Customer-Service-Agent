"""Create a deterministic, PII-minimized summary before human takeover."""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage

from customer_service.nodes.safety_check.node import latest_customer_message, redact_sensitive_text
from customer_service.state.models import AgentState, HandoffSummary


HANDOFF_ACKNOWLEDGEMENT = "I'm connecting you with a support specialist. They'll continue here shortly."


def _mask_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    local, domain = email.split("@", maxsplit=1)
    return f"{local[:1]}***@{domain}"


def _attempted_tools(state: AgentState) -> list[str]:
    attempts: list[str] = []
    for message in state.messages:
        if isinstance(message, AIMessage):
            attempts.extend(str(call["name"]) for call in message.tool_calls)
    return list(dict.fromkeys(attempts))


def _clean_customer_request(text: str) -> str:
    return re.sub(r"\s+", " ", redact_sensitive_text(text)).strip()[:1_000]


def build_node() -> Callable[[AgentState], dict[str, object]]:
    def handoff_summary(state: AgentState) -> dict[str, object]:
        if state.escalation_reason is None:
            raise ValueError("A handoff requires an escalation reason")
        summary = HandoffSummary(
            reason=state.escalation_reason,
            latest_customer_request=_clean_customer_request(latest_customer_message(state)),
            attempted_tools=_attempted_tools(state),
            verified=state.verified,
            order_id=state.customer.order_id,
            masked_email=_mask_email(state.customer.email),
        )
        return {
            "escalated": True,
            "case_status": "open",
            "handoff_summary": summary,
            "promo_matches": [],
            "draft_response": None,
            "draft_message_id": None,
            "promo_line": None,
            "messages": [AIMessage(content=HANDOFF_ACKNOWLEDGEMENT)],
        }

    return handoff_summary
