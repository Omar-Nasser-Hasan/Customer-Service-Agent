"""End-to-end identity verification flows through the real graph."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, ConfigDict, Field

from customer_service.config.settings import Settings
from customer_service.graph.build import build_graph
from customer_service.graph.response import final_reply
from customer_service.infrastructure.orders import (
    OrderRecord,
    OrderRepository,
    default_order_repository,
)
from customer_service.services.handoffs import HandoffService
from customer_service.state.models import EscalationReason
from customer_service.verification.attempts import InMemoryAttemptLimiter
from customer_service.verification.grant import VerificationPolicy
from customer_service.verification.messages import message
from tests.conftest import (
    ScriptedChatModel,
    StaticFaqRepository,
    allowing_safety_judge,
    declining_promo_judge,
)

START_TIME = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START_TIME

    def __call__(self) -> datetime:
        return self.now


def repository_with_second_order() -> OrderRepository:
    """Alice owns two orders; the default fixtures give each customer one."""

    base = default_order_repository()
    second = OrderRecord(
        order_id="ORD-2001",
        status="processing",
        status_detail="We are preparing your order.",
        updated_at=START_TIME,
        customer_email="alice@example.com",
        customer_name="Alice Morgan",
        payment_status="paid",
        charged_amount=Decimal("15.00"),
        currency="USD",
        charge_description="Souqly order ORD-2001",
    )
    return OrderRepository([base.find("ORD-1001"), base.find("ORD-1002"), base.find("ORD-1003"), second])  # type: ignore[list-item]


def tool_call(name: str, order_id: str, call_id: str = "call") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": {"order_id": order_id}, "id": call_id}]
    )


def make_graph(
    responses: list[AIMessage],
    *,
    clock: Clock | None = None,
    limiter: InMemoryAttemptLimiter | None = None,
    repository: OrderRepository | None = None,
    faq: StaticFaqRepository | None = None,
):
    model = ScriptedChatModel(responses=responses)
    graph = build_graph(
        settings=Settings(),
        repository=repository or default_order_repository(),
        model=model,
        promo_judge=declining_promo_judge(),
        safety_judge=allowing_safety_judge(),
        faq_repository=faq,
        attempt_limiter=limiter,
        verification_policy=VerificationPolicy(clock=clock) if clock else None,
    )
    return graph, model


def config(thread_id: str) -> dict[str, object]:
    return {"configurable": {"thread_id": thread_id}}


def say(text: str) -> dict[str, object]:
    return {"messages": [HumanMessage(content=text)]}


def tool_messages(state: dict[str, object]) -> list[ToolMessage]:
    return [item for item in state["messages"] if isinstance(item, ToolMessage)]  # type: ignore[union-attr]


async def test_early_verification_makes_the_happy_path_two_model_calls() -> None:
    graph, model = make_graph([tool_call("order_status", "ORD-1001"), AIMessage(content="It shipped.")])

    state = await graph.ainvoke(say("Where is ORD-1001? My email is alice@example.com"), config("happy"))

    assert model.response_index == 2
    assert state["verified_email"] == "alice@example.com"
    assert state["customer"].order_id == "ORD-1001"
    assert json.loads(tool_messages(state)[0].content)["found"] is True
    assert final_reply(state) == "It shipped."


async def test_verified_customer_asks_about_their_other_order_with_only_the_order_id() -> None:
    graph, _ = make_graph(
        [
            AIMessage(content="Thanks, how can I help?"),
            tool_call("order_status", "ORD-2001"),
            AIMessage(content="Order ORD-2001 is being prepared."),
        ],
        repository=repository_with_second_order(),
    )
    cfg = config("other-order")

    await graph.ainvoke(say("ORD-1001 alice@example.com"), cfg)
    state = await graph.ainvoke(say("What about ORD-2001?"), cfg)

    assert json.loads(tool_messages(state)[-1].content)["found"] is True
    assert state["customer"].order_id == "ORD-2001"  # the most recent lookup
    assert state.get("failed_attempts", []) == []
    assert final_reply(state) == "Order ORD-2001 is being prepared."


async def test_verified_customer_cannot_read_someone_elses_order() -> None:
    graph, _ = make_graph(
        [AIMessage(content="How can I help?"), tool_call("billing", "ORD-1002")]
    )
    cfg = config("not-yours")

    await graph.ainvoke(say("ORD-1001 alice@example.com"), cfg)
    state = await graph.ainvoke(say("And what was ORD-1002 charged?"), cfg)

    assert final_reply(state) == message("order_not_found", "en")
    assert "not executed" in tool_messages(state)[-1].content
    assert len(state["failed_attempts"]) == 1
    transcript = " ".join(str(item.content) for item in state["messages"])
    assert "ben@example.com" not in transcript and "42.50" not in transcript


async def test_split_pair_across_two_messages_verifies() -> None:
    graph, _ = make_graph(
        [AIMessage(content="Please send the email too."), AIMessage(content="Thanks, you're verified.")]
    )
    cfg = config("split")

    first = await graph.ainvoke(say("My order is ORD-1001"), cfg)
    second = await graph.ainvoke(say("alice@example.com"), cfg)

    assert first.get("verified_email") is None
    assert second["verified_email"] == "alice@example.com"


async def test_arabic_digits_and_arabic_failure_message() -> None:
    graph, _ = make_graph([AIMessage(content="تم التحقق.")])

    good = await graph.ainvoke(say("رقم طلبي ORD-١٠٠١ وبريدي alice@example.com"), config("ar-good"))
    bad = await graph.ainvoke(say("رقم طلبي ORD-1001 وبريدي wrong@example.com"), config("ar-bad"))

    assert good["verified_email"] == "alice@example.com"
    assert final_reply(bad) == message("claim_failed", "ar")


async def test_a_failed_claim_does_not_log_out_an_existing_verification() -> None:
    graph, _ = make_graph([AIMessage(content="How can I help?")])
    cfg = config("typo")

    await graph.ainvoke(say("ORD-1001 alice@example.com"), cfg)
    state = await graph.ainvoke(say("ORD-1001 alice@exmaple.com"), cfg)

    assert final_reply(state) == message("claim_failed", "en")
    assert state["verified_email"] == "alice@example.com"
    assert len(state["failed_attempts"]) == 1


async def test_a_new_full_pair_switches_identity_and_a_stray_email_does_not() -> None:
    graph, _ = make_graph(
        [AIMessage(content="Hi Alice."), AIMessage(content="Noted."), AIMessage(content="Hi Ben.")]
    )
    cfg = config("switch")

    await graph.ainvoke(say("ORD-1001 alice@example.com"), cfg)
    stray = await graph.ainvoke(say("my friend is ben@example.com by the way"), cfg)
    switched = await graph.ainvoke(say("my other order ORD-1002, ben@example.com"), cfg)

    assert stray["verified_email"] == "alice@example.com"
    assert switched["verified_email"] == "ben@example.com"
    assert switched["customer"].order_id == "ORD-1002"


async def test_verification_expires_and_the_customer_is_asked_again() -> None:
    clock = Clock()
    graph, _ = make_graph(
        [AIMessage(content="How can I help?"), tool_call("order_status", "ORD-1001")], clock=clock
    )
    cfg = config("expiry")

    await graph.ainvoke(say("ORD-1001 alice@example.com"), cfg)
    clock.now += timedelta(hours=3)
    state = await graph.ainvoke(say("Where is my order?"), cfg)

    assert final_reply(state) == message("expired", "en")
    assert "failed_attempts" not in state or state["failed_attempts"] == []


async def test_mixed_tool_calls_are_all_answered_when_the_gate_denies() -> None:
    mixed = AIMessage(
        content="",
        tool_calls=[
            {"name": "faq_lookup", "args": {"query": "shipping"}, "id": "faq"},
            {"name": "order_status", "args": {"order_id": "ORD-1001"}, "id": "status"},
        ],
    )
    graph, _ = make_graph([mixed], faq=StaticFaqRepository())

    state = await graph.ainvoke(say("How long is shipping, and where is my order?"), config("mixed"))

    assert [item.tool_call_id for item in tool_messages(state)] == ["faq", "status"]
    assert final_reply(state) == message("ask_both", "en")


async def test_third_failure_opens_a_handoff_and_resolution_resets_the_counter() -> None:
    graph, model = make_graph([AIMessage(content="unused")])
    cfg = config("guessing")

    first = await graph.ainvoke(say("ORD-1001 one@example.com"), cfg)
    second = await graph.ainvoke(say("ORD-1001 two@example.com"), cfg)
    third = await graph.ainvoke(say("ORD-1001 three@example.com"), cfg)

    assert final_reply(first) == final_reply(second) == message("claim_failed", "en")
    assert third["escalation_reason"] == EscalationReason.VERIFICATION_FAILED
    assert third["__interrupt__"]
    assert final_reply(third) == message("handoff_ack", "en")
    assert third["handoff_summary"].verified is False
    assert model.response_index == 0

    resolved = await HandoffService(graph).resolve("guessing")
    assert resolved["failed_attempts"] == []


async def test_handoff_acknowledgement_is_arabic_for_an_arabic_customer() -> None:
    graph, _ = make_graph([AIMessage(content="unused")])
    cfg = config("guessing-ar")

    state = {}
    for index in range(3):
        state = await graph.ainvoke(say(f"طلبي ORD-1001 وبريدي guess{index}@example.com"), cfg)

    assert final_reply(state) == message("handoff_ack", "ar")


async def test_shared_limiter_sends_a_correct_pair_to_a_human_after_too_many_failures() -> None:
    limiter = InMemoryAttemptLimiter(limit=1)
    limiter.record_failure(email="alice@example.com", order_id=None)
    graph, model = make_graph([AIMessage(content="unused")], limiter=limiter)

    state = await graph.ainvoke(say("ORD-1001 alice@example.com"), config("blocked"))

    assert state["escalation_reason"] == EscalationReason.VERIFICATION_FAILED
    assert state.get("verified_email") is None
    assert model.response_index == 0


async def test_failures_in_different_threads_add_up_in_the_shared_limiter() -> None:
    limiter = InMemoryAttemptLimiter(limit=2)
    graph, _ = make_graph([AIMessage(content="unused")], limiter=limiter)

    await graph.ainvoke(say("ORD-1001 guess1@example.com"), config("thread-a"))
    await graph.ainvoke(say("ORD-1001 guess2@example.com"), config("thread-b"))
    state = await graph.ainvoke(say("ORD-1001 alice@example.com"), config("thread-c"))

    # Two failures against ORD-1001 elsewhere blocked it for every thread.
    assert state["escalation_reason"] == EscalationReason.VERIFICATION_FAILED


async def test_threads_created_before_this_change_still_load() -> None:
    saver = InMemorySaver()

    class OldState(BaseModel):
        messages: Annotated[list, add_messages] = Field(default_factory=list)
        verified: bool = False
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    old = StateGraph(OldState)
    old.add_node("reply", lambda state: {"verified": True, "messages": [AIMessage(content="old reply")]})
    old.add_edge(START, "reply")
    old.add_edge("reply", END)
    await old.compile(checkpointer=saver).ainvoke(say("hello"), config("legacy"))

    model = ScriptedChatModel(responses=[AIMessage(content="new reply")])
    graph = build_graph(
        settings=Settings(),
        repository=default_order_repository(),
        model=model,
        promo_judge=declining_promo_judge(),
        safety_judge=allowing_safety_judge(),
        checkpointer=saver,
    )
    state = await graph.ainvoke(say("hello again"), config("legacy"))

    assert final_reply(state) == "new reply"
    assert len(state["messages"]) == 4
    # The old boolean grants nothing: this customer has to verify once.
    assert state.get("verified_email") is None
