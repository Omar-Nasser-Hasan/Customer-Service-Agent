"""Deterministic identity check for an order ID + email claim, run before the assistant."""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.messages import AIMessage

from customer_service.infrastructure.orders import OrderRepository
from customer_service.state.models import AgentState, EscalationReason
from customer_service.verification.attempts import (
    AttemptLimiter,
    InMemoryAttemptLimiter,
    recent_failures,
)
from customer_service.verification.claims import find_claim, latest_customer_text
from customer_service.verification.grant import VerificationPolicy
from customer_service.verification.messages import detect_language, message

LOGGER = logging.getLogger(__name__)


def has_unused_claim(state: AgentState) -> bool:
    """Route here only when the customer just supplied a complete, unused pair."""

    return find_claim(state.messages, state.last_claim_message_id) is not None


def build_node(
    repository: OrderRepository,
    *,
    policy: VerificationPolicy | None = None,
    limiter: AttemptLimiter | None = None,
    max_failures: int = 3,
) -> Callable[[AgentState], dict[str, object]]:
    grant = policy or VerificationPolicy()
    attempts = limiter or InMemoryAttemptLimiter()

    def verify_identity(state: AgentState) -> dict[str, object]:
        claim = find_claim(state.messages, state.last_claim_message_id)
        if claim is None:
            return {}

        now = grant.now()
        failures = recent_failures(state.failed_attempts, now)
        # The claim is spent whatever happens, so it is never counted twice.
        spent = {"last_claim_message_id": claim.message_id}

        try:
            blocked = len(failures) >= max_failures or attempts.is_blocked(
                email=claim.email, order_id=claim.order_id
            )
            order = None if blocked else repository.verify_identity(claim.order_id, claim.email)
            if not blocked and order is None:
                attempts.record_failure(email=claim.email, order_id=claim.order_id)
        except Exception as error:
            # Infrastructure trouble is not the customer's failed attempt.
            LOGGER.warning("verification_error", extra={"error_type": type(error).__name__})
            return {"escalation_reason": EscalationReason.TOOL_FAILURE}

        if blocked:
            LOGGER.info("verification_blocked", extra={"outcome": "blocked"})
            return {
                **spent,
                "failed_attempts": failures,
                "escalation_reason": EscalationReason.VERIFICATION_FAILED,
            }

        if order is not None:
            LOGGER.info("verification_succeeded", extra={"outcome": "verified"})
            # A new full pair replaces any earlier identity, one at a time.
            customer = state.customer.model_copy(
                update={
                    "email": claim.email,
                    "order_id": claim.order_id,
                    "name": order.customer_name,
                }
            )
            return {
                **spent,
                "customer": customer,
                "verified_email": claim.email,
                "verified_at": now,
            }

        failures = [*failures, now]
        LOGGER.info(
            "verification_failed", extra={"outcome": "failed", "thread_failures": len(failures)}
        )
        if len(failures) >= max_failures:
            return {
                **spent,
                "failed_attempts": failures,
                "escalation_reason": EscalationReason.VERIFICATION_FAILED,
            }
        language = detect_language(latest_customer_text(state.messages))
        # An existing verification is left alone: a typo must not log anyone out.
        return {
            **spent,
            "failed_attempts": failures,
            "messages": [AIMessage(content=message("claim_failed", language))],
        }

    return verify_identity
