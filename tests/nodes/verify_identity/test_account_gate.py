from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from customer_service.infrastructure.orders import default_order_repository
from customer_service.nodes.verify_identity.gate import (
    build_node,
    has_pending_account_calls,
    pending_tool_calls,
)
from customer_service.state.models import AgentState, EscalationReason
from customer_service.verification.attempts import InMemoryAttemptLimiter
from customer_service.verification.grant import VerificationPolicy
from customer_service.verification.messages import message

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def call(name: str, order_id: str, call_id: str) -> dict[str, object]:
    return {"name": name, "args": {"order_id": order_id}, "id": call_id}


def state_with(
    *calls: dict[str, object],
    customer_text: str = "Where is my order?",
    verified_email: str | None = None,
    verified_at: datetime | None = None,
    failed_attempts: list[datetime] | None = None,
) -> AgentState:
    return AgentState(
        messages=[
            HumanMessage(content=customer_text, id="h1"),
            AIMessage(content="", tool_calls=list(calls), id="a1"),
        ],
        verified_email=verified_email,
        verified_at=verified_at,
        failed_attempts=failed_attempts or [],
    )


def gate(limiter: InMemoryAttemptLimiter | None = None, max_failures: int = 3):
    return build_node(
        default_order_repository(),
        policy=VerificationPolicy(clock=lambda: NOW),
        limiter=limiter or InMemoryAttemptLimiter(clock=lambda: NOW),
        max_failures=max_failures,
    )


def test_calls_without_account_tools_pass_untouched() -> None:
    state = state_with({"name": "faq_lookup", "args": {"query": "shipping"}, "id": "f1"})

    assert not has_pending_account_calls(state)
    assert gate()(state) == {}


def test_pending_calls_come_only_from_the_message_just_produced() -> None:
    state = AgentState(messages=[HumanMessage(content="hi", id="h1")])

    assert pending_tool_calls(state) == []
    assert pending_tool_calls(AgentState()) == []


def test_unverified_call_is_resolved_and_asks_for_both_fields() -> None:
    update = gate()(state_with(call("billing", "ORD-1001", "c1")))

    tool_message, reply = update["messages"]
    assert isinstance(tool_message, ToolMessage) and tool_message.tool_call_id == "c1"
    assert tool_message.content == "Account lookup was not executed: identity verification is required."
    assert reply.content == message("ask_both", "en")
    assert "failed_attempts" not in update


def test_unverified_call_asks_only_for_the_missing_field() -> None:
    with_order = gate()(state_with(call("billing", "ORD-1001", "c1"), customer_text="Why was ORD-1001 charged?"))
    with_email = gate()(state_with(call("billing", "ORD-1001", "c1"), customer_text="I am alice@example.com"))

    assert with_order["messages"][-1].content == message("ask_email", "en")
    assert with_email["messages"][-1].content == message("ask_order_id", "en")


def test_expired_grant_asks_to_send_the_pair_again_and_counts_nothing() -> None:
    update = gate()(
        state_with(
            call("order_status", "ORD-1001", "c1"),
            verified_email="alice@example.com",
            verified_at=NOW - timedelta(hours=3),
        )
    )

    assert update["messages"][-1].content == message("expired", "en")
    assert "has expired" in update["messages"][0].content
    assert "failed_attempts" not in update


def test_every_call_in_a_mixed_message_gets_a_tool_response() -> None:
    update = gate()(
        state_with(
            {"name": "faq_lookup", "args": {"query": "shipping"}, "id": "faq"},
            call("order_status", "ORD-1001", "status"),
        )
    )

    tool_messages = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["faq", "status"]
    assert isinstance(update["messages"][-1], AIMessage)


def test_owned_orders_pass_and_the_latest_becomes_current() -> None:
    state = state_with(
        call("order_status", "ORD-1001", "c1"),
        call("billing", "ord 1001", "c2"),
        verified_email="alice@example.com",
        verified_at=NOW,
    )

    update = gate()(state)

    assert set(update) == {"customer"}
    assert update["customer"].order_id == "ORD-1001"


def test_someone_elses_order_and_a_missing_order_look_identical_and_count_once() -> None:
    kwargs = {"verified_email": "alice@example.com", "verified_at": NOW}
    someone_elses = gate()(state_with(call("order_status", "ORD-1002", "c1"), **kwargs))
    nonexistent = gate()(state_with(call("order_status", "ORD-9999", "c1"), **kwargs))

    assert someone_elses["messages"][0].content == nonexistent["messages"][0].content
    assert someone_elses["messages"][-1].content == message("order_not_found", "en")
    assert someone_elses["failed_attempts"] == [NOW] == nonexistent["failed_attempts"]


def test_one_bad_order_in_a_batch_denies_the_whole_batch_with_one_failure() -> None:
    update = gate()(
        state_with(
            call("order_status", "ORD-1001", "ok"),
            call("billing", "ORD-1002", "bad"),
            verified_email="alice@example.com",
            verified_at=NOW,
        )
    )

    assert [m.tool_call_id for m in update["messages"] if isinstance(m, ToolMessage)] == ["ok", "bad"]
    assert len(update["failed_attempts"]) == 1


def test_an_unparseable_order_id_is_denied_but_not_counted() -> None:
    update = gate()(
        state_with(
            call("order_status", "", "c1"),
            verified_email="alice@example.com",
            verified_at=NOW,
        )
    )

    assert update["messages"][-1].content == message("order_not_found", "en")
    assert "failed_attempts" not in update


def test_the_third_denial_opens_a_handoff() -> None:
    update = gate()(
        state_with(
            call("order_status", "ORD-1002", "c1"),
            verified_email="alice@example.com",
            verified_at=NOW,
            failed_attempts=[NOW - timedelta(hours=3), NOW - timedelta(hours=2)],
        )
    )

    assert update["escalation_reason"] == EscalationReason.VERIFICATION_FAILED
    assert isinstance(update["messages"][-1], ToolMessage)  # resolved, no reply: handoff acks
    assert len(update["failed_attempts"]) == 3


def test_denial_feeds_the_shared_limiter_under_the_verified_email_only() -> None:
    limiter = InMemoryAttemptLimiter(limit=1, clock=lambda: NOW)

    gate(limiter)(
        state_with(
            call("order_status", "ORD-1002", "c1"),
            verified_email="alice@example.com",
            verified_at=NOW,
        )
    )

    assert limiter.is_blocked(email="alice@example.com", order_id=None)
    # The probed order is not penalised, so a verified user cannot lock its real owner.
    assert not limiter.is_blocked(email=None, order_id="ORD-1002")


def test_reply_language_follows_the_customer() -> None:
    update = gate()(
        state_with(
            call("order_status", "ORD-1002", "c1"),
            customer_text="وماذا عن ORD-1002؟",
            verified_email="alice@example.com",
            verified_at=NOW,
        )
    )

    assert update["messages"][-1].content == message("order_not_found", "ar")


def test_lookup_error_resolves_calls_and_escalates_as_a_tool_failure() -> None:
    class Exploding:
        def verify_identity(self, order_id: str, email: str):
            raise RuntimeError("commerce API down")

    broken = build_node(Exploding(), policy=VerificationPolicy(clock=lambda: NOW))  # type: ignore[arg-type]

    update = broken(
        state_with(
            call("order_status", "ORD-1001", "c1"),
            verified_email="alice@example.com",
            verified_at=NOW,
        )
    )

    assert update["escalation_reason"] == EscalationReason.TOOL_FAILURE
    assert [m.tool_call_id for m in update["messages"]] == ["c1"]
    assert "failed_attempts" not in update
