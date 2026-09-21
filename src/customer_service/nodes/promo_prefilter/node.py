"""Deterministically decide whether promotion judging is appropriate."""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage

from customer_service.infrastructure.orders import OrderRepository
from customer_service.infrastructure.promotions import PromotionRepository
from customer_service.state.models import AgentState


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


def build_node(
    *,
    promotion_repository: PromotionRepository,
    order_repository: OrderRepository,
) -> Callable[[AgentState], dict[str, object]]:
    def promo_prefilter(state: AgentState) -> dict[str, object]:
        draft = _latest_support_draft(state)
        if draft is None or state.already_promoted:
            return {
                "promo_matches": [],
                "draft_response": None,
                "draft_message_id": None,
                "promo_line": None,
            }

        tokens = _conversation_tokens(state)
        if _has_negative_sentiment(tokens):
            return {
                "promo_matches": [],
                "draft_response": None,
                "draft_message_id": None,
                "promo_line": None,
            }

        order = (
            order_repository.find(state.customer.order_id)
            if state.verified and state.customer.order_id
            else None
        )
        matches = promotion_repository.active_matches(
            conversation_tokens=tokens,
            verified=state.verified,
            order_status=order.status if order else None,
            delivered=bool(order and order.delivered_at),
        )
        if not matches:
            return {
                "promo_matches": [],
                "draft_response": None,
                "draft_message_id": None,
                "promo_line": None,
            }

        draft_text = draft.content if isinstance(draft.content, str) else str(draft.content)
        return {
            "promo_matches": matches,
            "draft_response": draft_text,
            "draft_message_id": draft.id,
            "promo_line": None,
        }

    return promo_prefilter
