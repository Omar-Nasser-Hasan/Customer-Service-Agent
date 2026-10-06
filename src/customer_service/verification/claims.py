"""Order-ID and email claims found in customer messages."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AnyMessage, HumanMessage

from customer_service.state.message_text import content_to_text

# Arabic-Indic (U+0660..) and Eastern Arabic-Indic (U+06F0..) digits become ASCII.
_DIGIT_MAP = str.maketrans(
    {
        **{chr(0x0660 + value): str(value) for value in range(10)},
        **{chr(0x06F0 + value): str(value) for value in range(10)},
    }
)

# Matches ORD-1001, ord 1001, ORD1001. The lookbehind is ASCII-only so an order
# ID glued to Arabic text still matches, while "order 1001" does not.
_ORDER_ID = re.compile(r"(?<![A-Za-z0-9])ORD[\s_-]?(\d+)(?!\d)", re.IGNORECASE)
# ASCII-only on purpose: \w would swallow adjacent Arabic letters into the address.
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


@dataclass(frozen=True)
class Claim:
    order_id: str
    email: str
    message_id: str | None


def normalize_order_id(value: str) -> str | None:
    """Return the canonical ``ORD-<digits>`` form, or None if no order ID is present."""

    match = _ORDER_ID.search(value.translate(_DIGIT_MAP))
    return f"ORD-{match.group(1)}" if match else None


def fields_in(text: str) -> tuple[str | None, str | None]:
    """Last order ID and last email in one text, both normalized."""

    text = text.translate(_DIGIT_MAP)
    orders = _ORDER_ID.findall(text)
    emails = _EMAIL.findall(text)
    return (
        f"ORD-{orders[-1]}" if orders else None,
        emails[-1].casefold() if emails else None,
    )


def _text(message: AnyMessage) -> str:
    return content_to_text(message.content)


def _unused_customer_messages(
    messages: Sequence[AnyMessage], last_used_id: str | None
) -> list[AnyMessage]:
    """Customer messages after the last one that was already used as a claim."""

    customer = [message for message in messages if isinstance(message, HumanMessage)]
    if last_used_id:
        for index in range(len(customer) - 1, -1, -1):
            if customer[index].id == last_used_id:
                return customer[index + 1 :]
    return customer


def latest_customer_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _text(message)
    return ""


def find_claim(messages: Sequence[AnyMessage], last_used_id: str | None) -> Claim | None:
    """A complete order ID + email pair, or None.

    The pair must be in the latest customer message, or completed by the one
    customer message just before it. The latest message has to contribute at
    least one field, so an old message can never revive itself as a claim, and
    messages at or before ``last_used_id`` are spent.
    """

    unused = _unused_customer_messages(messages, last_used_id)
    if not unused:
        return None
    latest = unused[-1]
    order_id, email = fields_in(_text(latest))
    if order_id is None and email is None:
        return None
    if len(unused) >= 2:
        previous_order, previous_email = fields_in(_text(unused[-2]))
        order_id = order_id or previous_order
        email = email or previous_email
    if order_id and email:
        return Claim(order_id=order_id, email=email, message_id=latest.id)
    return None


def partial_fields(messages: Sequence[AnyMessage], last_used_id: str | None) -> tuple[bool, bool]:
    """Whether unused customer messages contain an order ID, and an email."""

    has_order = has_email = False
    for message in _unused_customer_messages(messages, last_used_id):
        order_id, email = fields_in(_text(message))
        has_order = has_order or order_id is not None
        has_email = has_email or email is not None
    return has_order, has_email
