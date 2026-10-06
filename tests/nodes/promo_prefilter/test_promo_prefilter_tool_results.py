from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from customer_service.infrastructure.orders import default_order_repository
from customer_service.infrastructure.promotions import default_promotion_repository
from customer_service.nodes.promo_prefilter.node import build_node
from customer_service.state.models import AgentState

EMPTY = {"promo_matches": [], "draft_response": None, "draft_message_id": None, "promo_line": None}


def prefilter(messages: list) -> dict[str, object]:
    node = build_node(
        promotion_repository=default_promotion_repository(),
        order_repository=default_order_repository(),
    )
    return node(AgentState(messages=messages))


def answered(tool_content: str, reply: str = "Shipping takes 3 to 5 business days.") -> list:
    return [
        HumanMessage(content="How long is shipping?"),
        AIMessage(content="", tool_calls=[{"name": "faq_lookup", "args": {"query": "q"}, "id": "t"}]),
        ToolMessage(content=tool_content, tool_call_id="t"),
        AIMessage(id="draft", content=reply),
    ]


def test_a_found_tool_result_this_turn_allows_promotion_matching() -> None:
    update = prefilter(answered('{"found": true, "faq_id": "shipping-times"}'))

    assert update["promo_matches"] and update["draft_message_id"] == "draft"


def test_a_not_found_tool_result_never_earns_a_promotion() -> None:
    assert prefilter(answered('{"found": false, "message": "No matching FAQ was found."}')) == EMPTY


def test_a_plain_reply_never_earns_a_promotion() -> None:
    plain = [
        HumanMessage(content="How long is shipping?"),
        AIMessage(id="draft", content="Please send your order ID and email."),
    ]

    assert prefilter(plain) == EMPTY


def test_a_gate_notice_is_not_a_found_result() -> None:
    notice = answered("Account lookup was not executed: identity verification is required.")

    assert prefilter(notice) == EMPTY


def test_a_found_result_from_an_earlier_turn_does_not_count() -> None:
    earlier_turn = [
        *answered('{"found": true, "faq_id": "shipping-times"}'),
        HumanMessage(content="What categories do you sell?"),
        AIMessage(id="draft-2", content="We sell many categories."),
    ]

    assert prefilter(earlier_turn) == EMPTY


def test_the_found_result_check_does_not_depend_on_the_conversation_language() -> None:
    from customer_service.nodes.promo_prefilter.node import _answered_from_a_found_tool_result

    arabic = [
        HumanMessage(content="كم يستغرق الشحن؟"),
        AIMessage(content="", tool_calls=[{"name": "faq_lookup", "args": {"query": "الشحن"}, "id": "t"}]),
        ToolMessage(content='{"found": true, "faq_id": "shipping-times"}', tool_call_id="t"),
        AIMessage(id="draft", content="تصل معظم الطلبات خلال 3 إلى 5 أيام عمل."),
    ]
    greeting = [HumanMessage(content="مرحبا"), AIMessage(id="draft", content="أهلا بك! كيف أساعدك؟")]

    assert _answered_from_a_found_tool_result(arabic) is True
    assert _answered_from_a_found_tool_result(greeting) is False
