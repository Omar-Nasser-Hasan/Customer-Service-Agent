from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage

from customer_service.infrastructure.orders import default_order_repository
from customer_service.nodes.verify_identity.node import build_node, has_unused_claim
from customer_service.state.models import AgentState, EscalationReason
from customer_service.verification.attempts import InMemoryAttemptLimiter
from customer_service.verification.grant import VerificationPolicy
from customer_service.verification.messages import message

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def human(text: str, message_id: str) -> HumanMessage:
    return HumanMessage(content=text, id=message_id)


def node(limiter: InMemoryAttemptLimiter | None = None, max_failures: int = 3):
    return build_node(
        default_order_repository(),
        policy=VerificationPolicy(clock=lambda: NOW),
        limiter=limiter or InMemoryAttemptLimiter(clock=lambda: NOW),
        max_failures=max_failures,
    )


def test_without_a_complete_claim_the_node_does_nothing() -> None:
    state = AgentState(messages=[human("Please track order ORD-1001 for me.", "m1")])

    assert not has_unused_claim(state)
    assert node()(state) == {}


def test_matching_pair_verifies_and_stores_only_the_identity() -> None:
    state = AgentState(messages=[human("ORD-1001, Alice@Example.com", "m1")])

    update = node()(state)

    assert update["verified_email"] == "alice@example.com"
    assert update["verified_at"] == NOW
    assert update["last_claim_message_id"] == "m1"
    assert update["customer"].order_id == "ORD-1001"
    assert update["customer"].email == "alice@example.com"
    assert "messages" not in update and "failed_attempts" not in update


def test_new_pair_replaces_the_previous_identity_without_losing_other_context() -> None:
    state = AgentState(
        messages=[human("my order ORD-1002, ben@example.com", "m2")],
        verified_email="alice@example.com",
        verified_at=NOW - timedelta(minutes=5),
        customer={"email": "alice@example.com", "order_id": "ORD-1001", "phone": "keep-me"},
    )

    update = node()(state)

    assert update["verified_email"] == "ben@example.com"
    assert update["customer"].phone == "keep-me"
    assert update["customer"].order_id == "ORD-1002"


def test_mismatch_counts_once_asks_again_and_stores_nothing_from_the_claim() -> None:
    state = AgentState(messages=[human("ORD-1001 wrong@example.com", "m1")])

    update = node()(state)

    assert update["failed_attempts"] == [NOW]
    assert update["last_claim_message_id"] == "m1"
    assert "customer" not in update and "verified_email" not in update
    assert update["messages"][0].content == message("claim_failed", "en")


def test_nonexistent_order_and_wrong_email_get_the_same_answer() -> None:
    wrong_email = node()(AgentState(messages=[human("ORD-1001 wrong@example.com", "m1")]))
    no_such_order = node()(AgentState(messages=[human("ORD-9999 alice@example.com", "m1")]))

    assert wrong_email["messages"][0].content == no_such_order["messages"][0].content


def test_failure_message_follows_the_customers_language() -> None:
    state = AgentState(messages=[human("رقم طلبي ORD-1001 وبريدي wrong@example.com", "m1")])

    assert node()(state)["messages"][0].content == message("claim_failed", "ar")


def test_a_failed_check_keeps_an_existing_verification() -> None:
    state = AgentState(
        messages=[human("ORD-1001 wrong@example.com", "m3")],
        verified_email="alice@example.com",
        verified_at=NOW,
    )

    update = node()(state)

    assert "verified_email" not in update and "verified_at" not in update


def test_third_failure_in_a_day_opens_a_handoff_instead_of_a_message() -> None:
    state = AgentState(
        messages=[human("ORD-1001 wrong@example.com", "m3")],
        failed_attempts=[NOW - timedelta(hours=2), NOW - timedelta(hours=1)],
    )

    update = node()(state)

    assert update["escalation_reason"] == EscalationReason.VERIFICATION_FAILED
    assert len(update["failed_attempts"]) == 3
    assert "messages" not in update


def test_failures_older_than_a_day_no_longer_count() -> None:
    state = AgentState(
        messages=[human("ORD-1001 wrong@example.com", "m3")],
        failed_attempts=[NOW - timedelta(hours=30), NOW - timedelta(hours=26)],
    )

    update = node()(state)

    assert update["failed_attempts"] == [NOW]
    assert "escalation_reason" not in update


def test_a_correct_pair_does_not_clear_earlier_failures() -> None:
    state = AgentState(
        messages=[human("ORD-1001 alice@example.com", "m3")],
        failed_attempts=[NOW - timedelta(hours=1)],
    )

    update = node()(state)

    assert update["verified_email"] == "alice@example.com"
    assert "failed_attempts" not in update  # state keeps its list untouched


def test_shared_limiter_blocks_even_a_correct_pair_and_hands_off() -> None:
    limiter = InMemoryAttemptLimiter(limit=1, clock=lambda: NOW)
    limiter.record_failure(email="alice@example.com", order_id=None)
    state = AgentState(messages=[human("ORD-1001 alice@example.com", "m1")])

    update = node(limiter)(state)

    assert update["escalation_reason"] == EscalationReason.VERIFICATION_FAILED
    assert "verified_email" not in update


def test_failures_are_recorded_for_the_shared_limiter() -> None:
    limiter = InMemoryAttemptLimiter(limit=1, clock=lambda: NOW)

    node(limiter)(AgentState(messages=[human("ORD-1001 wrong@example.com", "m1")]))

    assert limiter.is_blocked(email="wrong@example.com", order_id=None)
    assert limiter.is_blocked(email=None, order_id="ORD-1001")


def test_lookup_error_is_a_tool_failure_not_a_counted_attempt() -> None:
    class Exploding:
        def verify_identity(self, order_id: str, email: str):
            raise RuntimeError("commerce API down")

    failing = build_node(Exploding(), policy=VerificationPolicy(clock=lambda: NOW))  # type: ignore[arg-type]

    update = failing(AgentState(messages=[human("ORD-1001 alice@example.com", "m1")]))

    assert update == {"escalation_reason": EscalationReason.TOOL_FAILURE}


def test_limiter_error_fails_closed_into_a_handoff() -> None:
    class BrokenLimiter:
        def is_blocked(self, **_: object) -> bool:
            raise RuntimeError("database down")

        def record_failure(self, **_: object) -> None:
            raise RuntimeError("database down")

    update = node(BrokenLimiter())(  # type: ignore[arg-type]
        AgentState(messages=[human("ORD-1001 alice@example.com", "m1")])
    )

    assert update == {"escalation_reason": EscalationReason.TOOL_FAILURE}


def test_an_old_claim_is_not_used_twice() -> None:
    state = AgentState(
        messages=[human("ORD-1001 alice@example.com", "m1"), AIMessage(content="ok")],
        last_claim_message_id="m1",
    )

    assert not has_unused_claim(state)
    assert node()(state) == {}
