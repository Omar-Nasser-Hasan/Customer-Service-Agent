"""Deterministically decide whether promotion judging is appropriate."""

from __future__ import annotations

import json
import re
from typing import Callable

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from customer_service.infrastructure.orders import OrderRepository
from customer_service.infrastructure.promotions import PromotionRepository
from customer_service.state.message_text import content_to_text
from customer_service.state.models import AgentState
from customer_service.verification.grant import VerificationPolicy


# Deliberately conservative, English-only prototype heuristic. Replace with
# transcript-tuned sentiment evaluation before customer launch.
NEGATIVE_SENTIMENT_TERMS = frozenset(
    {
        "angry",
        "annoyed",
        "awful",
        "bad",
        "broken",
        "complaint",
        "disappointed",
        "furious",
        "hate",
        "horrible",
        "issue",
        "late",
        "never",
        "problem",
        "refund",
        "scam",
        "terrible",
        "unhappy",
        "upset",
        "wrong",
    }
)


def _message_text(message: HumanMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _conversation_tokens(state: AgentState) -> set[str]:
    text = " ".join(
        _message_text(message).casefold()
        for message in state.messages
        if isinstance(message, HumanMessage)
    )
    return set(re.findall(r"[a-z0-9]+", text))


def _latest_support_draft(state: AgentState) -> AIMessage | None:
    for message in reversed(state.messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message
    return None


def _has_negative_sentiment(tokens: set[str]) -> bool:
    return bool(tokens.intersection(NEGATIVE_SENTIMENT_TERMS))


def _answered_from_a_found_tool_result(messages: list[AnyMessage]) -> bool:
    """True when this turn's reply is backed by a FAQ or order that was found.

    Greetings, requests for information, refusals, and "I can't help" replies
    have no found tool result behind them, so they never earn a promotion. The
    check reads tool output, not keywords, so it works in Arabic too.
    """

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return False
        if isinstance(message, ToolMessage):
            try:
                payload = json.loads(content_to_text(message.content))
            except ValueError:
                continue  # e.g. a gate's "not executed" notice
            if isinstance(payload, dict) and payload.get("found") is True:
                return True
    return False


def _no_promotion() -> dict[str, object]:
    return {
        "promo_matches": [],
        "draft_response": None,
        "draft_message_id": None,
        "promo_line": None,
    }


def build_node(
    *,
    promotion_repository: PromotionRepository,
    order_repository: OrderRepository,
    policy: VerificationPolicy | None = None,
) -> Callable[[AgentState], dict[str, object]]:
    grant = policy or VerificationPolicy()

    def promo_prefilter(state: AgentState) -> dict[str, object]:
        draft = _latest_support_draft(state)
        if draft is None or state.already_promoted:
            return _no_promotion()

        if not _answered_from_a_found_tool_result(state.messages):
            return _no_promotion()

        tokens = _conversation_tokens(state)
        if _has_negative_sentiment(tokens):
            return _no_promotion()

        verified = grant.is_valid(state.verified_email, state.verified_at)
        # Order state is derived only from an order the verified email owns.
        order = (
            order_repository.verify_identity(state.customer.order_id, state.verified_email)
            if verified and state.customer.order_id and state.verified_email
            else None
        )
        matches = promotion_repository.active_matches(
            conversation_tokens=tokens,
            verified=verified,
            order_status=order.status if order else None,
            delivered=bool(order and order.delivered_at),
        )
        if not matches:
            return _no_promotion()

        draft_text = content_to_text(draft.content)
        return {
            "promo_matches": matches,
            "draft_response": draft_text,
            "draft_message_id": draft.id,
            "promo_line": None,
        }

    return promo_prefilter
