"""Authorize account-tool calls: a valid verification and an order the verified email owns."""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage

from customer_service.infrastructure.orders import OrderRepository
from customer_service.state.models import AgentState, EscalationReason
from customer_service.verification.attempts import (
    AttemptLimiter,
    InMemoryAttemptLimiter,
    recent_failures,
)
from customer_service.verification.claims import (
    latest_customer_text,
    normalize_order_id,
    partial_fields,
)
from customer_service.verification.grant import VerificationPolicy
from customer_service.verification.messages import detect_language, message

LOGGER = logging.getLogger(__name__)

ACCOUNT_SPECIFIC_TOOLS = frozenset({"order_status", "returns", "billing"})


def pending_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """Unexecuted tool calls of the AI message that was just produced, if any."""

    if not state.messages:
        return []
    last = state.messages[-1]
    return list(last.tool_calls) if isinstance(last, AIMessage) else []


def has_pending_account_calls(state: AgentState) -> bool:
    return any(call["name"] in ACCOUNT_SPECIFIC_TOOLS for call in pending_tool_calls(state))


def _requested_order(call: dict[str, Any]) -> str:
    value = (call.get("args") or {}).get("order_id")
    return value if isinstance(value, str) else ""


def build_node(
    repository: OrderRepository,
    *,
    policy: VerificationPolicy | None = None,
    limiter: AttemptLimiter | None = None,
    max_failures: int = 3,
) -> Callable[[AgentState], dict[str, object]]:
    grant = policy or VerificationPolicy()
    attempts = limiter or InMemoryAttemptLimiter()

    def account_gate(state: AgentState) -> dict[str, object]:
        calls = pending_tool_calls(state)
        account_calls = [call for call in calls if call["name"] in ACCOUNT_SPECIFIC_TOOLS]
        if not account_calls:
            return {}

        def resolve(text: str) -> list[ToolMessage]:
            # Every call in the AI message needs an answer, not only the account
            # ones, or the provider rejects the history on the next turn.
            return [ToolMessage(content=text, tool_call_id=str(call["id"])) for call in calls]

        language = detect_language(latest_customer_text(state.messages))

        if not grant.is_valid(state.verified_email, state.verified_at):
            expired = grant.is_expired(state.verified_email, state.verified_at)
            LOGGER.info("account_gate_denied", extra={"outcome": "expired" if expired else "unverified"})
            if expired:
                key, reason = "expired", "identity verification has expired."
            else:
                has_order, has_email = partial_fields(state.messages, state.last_claim_message_id)
                key = (
                    "ask_email" if has_order and not has_email
                    else "ask_order_id" if has_email and not has_order
                    else "ask_both"
                )
                reason = "identity verification is required."
            return {
                "messages": [
                    *resolve(f"Account lookup was not executed: {reason}"),
                    AIMessage(content=message(key, language)),
                ]
            }

        requested = [_requested_order(call) for call in account_calls]
        normalized = [normalize_order_id(value) for value in requested]
        now = grant.now()
        try:
            owned = [
                grant.owned_order(repository, value, state.verified_email, state.verified_at)
                is not None
                for value in requested
            ]
            # An unparseable order ID is a model slip, not a customer guess.
            counted = any(
                order_id is not None and not is_owned
                for order_id, is_owned in zip(normalized, owned, strict=True)
            )
            if counted:
                attempts.record_failure(email=state.verified_email, order_id=None)
        except Exception as error:
            LOGGER.warning("account_gate_error", extra={"error_type": type(error).__name__})
            return {
                "messages": resolve("Account lookup was not executed: lookup unavailable."),
                "escalation_reason": EscalationReason.TOOL_FAILURE,
            }

        if all(owned):
            # The most recently looked-up order is the "current" one.
            return {"customer": state.customer.model_copy(update={"order_id": normalized[-1]})}

        update: dict[str, object] = {}
        failures = recent_failures(state.failed_attempts, now)
        if counted:
            failures = [*failures, now]
            update["failed_attempts"] = failures
        LOGGER.info(
            "account_gate_denied",
            extra={"outcome": "ownership_denied", "thread_failures": len(failures)},
        )
        tool_messages = resolve("Account lookup was not executed: order not found.")
        if counted and len(failures) >= max_failures:
            return {
                **update,
                "messages": tool_messages,
                "escalation_reason": EscalationReason.VERIFICATION_FAILED,
            }
        return {
            **update,
            "messages": [*tool_messages, AIMessage(content=message("order_not_found", language))],
        }

    return account_gate
