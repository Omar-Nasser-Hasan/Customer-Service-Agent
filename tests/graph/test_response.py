from langchain_core.messages import AIMessage, HumanMessage

from customer_service.graph.response import final_reply
from customer_service.nodes.promo_prefilter.node import build_node as build_prefilter
from customer_service.infrastructure.orders import default_order_repository
from customer_service.infrastructure.promotions import default_promotion_repository
from customer_service.state.models import AgentState


def test_final_reply_extracts_text_from_provider_content_blocks() -> None:
    reply = final_reply(
        {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "text", "text": "A customer-facing answer.", "extras": {"secret": "x"}}
                    ]
                )
            ]
        }
    )

    assert reply == "A customer-facing answer."


def test_promo_prefilter_extracts_provider_content_block_text() -> None:
    node = build_prefilter(
        promotion_repository=default_promotion_repository(),
        order_repository=default_order_repository(),
    )
    update = node(
        AgentState(
            messages=[
                HumanMessage(content="How long is shipping?"),
                AIMessage(
                    id="provider-message",
                    content=[{"type": "text", "text": "Shipping takes 3 to 5 business days.", "extras": {}}],
                )
            ]
        )
    )

    assert update["draft_response"] == "Shipping takes 3 to 5 business days."
