"""Deterministic identity gate for account-specific tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from customer_service.infrastructure.orders import OrderRepository
from customer_service.state.models import AgentState, CustomerContext


ACCOUNT_SPECIFIC_TOOLS = frozenset({"order_status", "returns", "billing"})
_ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_ACCOUNT_INTENT_WORDS = frozenset(
    {"order", "status", "track", "tracking", "return", "refund", "billing", "bill", "charge"}
)


@dataclass(frozen=True)
class IdentityClaim:
    order_id: str | None
    email: str | None

    @property
    def complete(self) -> bool:
        return bool(self.order_id and self.email)


def _text(message: AnyMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def extract_identity_claim(state: AgentState) -> IdentityClaim:
    """Collect the most recently supplied order ID and email from customer messages."""

    order_id = state.customer.order_id
    email = state.customer.email
    for message in state.messages:
        if not isinstance(message, HumanMessage):
            continue
        text = _text(message)
        if matches := _ORDER_ID_PATTERN.findall(text):
            order_id = matches[-1].upper()
        if matches := _EMAIL_PATTERN.findall(text):
            email = matches[-1].casefold()
    return IdentityClaim(order_id=order_id, email=email)


def pending_account_tool_calls(state: AgentState) -> list[dict[str, object]]:
    """Return unexecuted account-tool calls from the latest AI message, if any."""

    for message in reversed(state.messages):
        if isinstance(message, AIMessage):
            return [
                call
                for call in message.tool_calls
                if call["name"] in ACCOUNT_SPECIFIC_TOOLS
            ]
    return []


def _recent_account_intent(state: AgentState) -> bool:
    human_text = " ".join(_text(message).casefold() for message in state.messages if isinstance(message, HumanMessage))
    tokens = set(re.findall(r"[a-z]+", human_text))
    return bool(tokens.intersection(_ACCOUNT_INTENT_WORDS))


def requires_identity_verification(state: AgentState) -> bool:
    """Route only unverified account work through this node."""

    if state.verified:
        return False
    if pending_account_tool_calls(state):
        return True
    return extract_identity_claim(state).complete and _recent_account_intent(state)


def has_unverified_account_intent(state: AgentState) -> bool:
    """Identify a verification prompt that must not be followed by a promotion."""

    return not state.verified and _recent_account_intent(state)


def build_node(repository: OrderRepository) -> Callable[[AgentState], dict[str, object]]:
    def verify_identity(state: AgentState) -> dict[str, object]:
        claim = extract_identity_claim(state)
        intercepted_calls = pending_account_tool_calls(state)

        def resolve_intercepted(message: str) -> list[ToolMessage]:
            return [
                ToolMessage(content=message, tool_call_id=str(call["id"]))
                for call in intercepted_calls
            ]

        if not claim.complete:
            return {
                "verified": False,
                "messages": [
                    *resolve_intercepted("Account lookup was not executed: identity verification is required."),
                    AIMessage(
                        content="To protect your account, please send both your order ID and the email used for the order."
                    ),
                ],
            }

        order = repository.verify_identity(claim.order_id, claim.email)
        customer = CustomerContext(order_id=claim.order_id, email=claim.email)
        if order is None:
            return {
                "customer": customer,
                "verified": False,
                "messages": [
                    *resolve_intercepted("Account lookup was not executed: identity verification failed."),
                    AIMessage(
                        content="I couldn't verify that order ID and email combination. Please check both and try again."
                    ),
                ],
            }

        customer.name = order.customer_name
        return {
            "customer": customer,
            "verified": True,
            "messages": resolve_intercepted(
                "Identity verified for this conversation. Call the requested account tool again."
            ),
        }

    return verify_identity
