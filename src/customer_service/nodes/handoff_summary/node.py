"""Create a deterministic, PII-minimized summary before human takeover."""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage

from customer_service.nodes.safety_check.node import latest_customer_message
from customer_service.privacy.redaction import redact_text
from customer_service.state.models import AgentState, HandoffSummary
from customer_service.verification.grant import VerificationPolicy
from customer_service.verification.messages import detect_language, message


# The English text is unchanged; the acknowledgement now also exists in Arabic.
HANDOFF_ACKNOWLEDGEMENT = message("handoff_ack", "en")


def _mask_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    local, domain = email.split("@", maxsplit=1)
    return f"{local[:1]}***@{domain}"


def _attempted_tools(state: AgentState) -> list[str]:
    attempts: list[str] = []
    for item in state.messages:
        if isinstance(item, AIMessage):
            attempts.extend(str(call["name"]) for call in item.tool_calls)
    return list(dict.fromkeys(attempts))


def _clean_customer_request(text: str) -> str:
    return re.sub(r"\s+", " ", redact_text(text)).strip()[:1_000]


def build_node(*, policy: VerificationPolicy | None = None) -> Callable[[AgentState], dict[str, object]]:
    grant = policy or VerificationPolicy()

    def handoff_summary(state: AgentState) -> dict[str, object]:
        if state.escalation_reason is None:
            raise ValueError("A handoff requires an escalation reason")
        latest = latest_customer_message(state)
        summary = HandoffSummary(
            reason=state.escalation_reason,
            latest_customer_request=_clean_customer_request(latest),
            attempted_tools=_attempted_tools(state),
            # Expiry applies here too: a stale grant is not "verified" for the agent.
            verified=grant.is_valid(state.verified_email, state.verified_at),
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
            "messages": [AIMessage(content=message("handoff_ack", detect_language(latest)))],
        }

    return handoff_summary
