from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from customer_service.infrastructure.orders import default_order_repository
from customer_service.infrastructure.promotions import default_promotion_repository
from customer_service.nodes.promo_prefilter.node import build_node
from customer_service.state.models import AgentState, PromoMatch


def test_prefilter_clears_stale_promo_state_when_thread_was_already_promoted() -> None:
    node = build_node(
        promotion_repository=default_promotion_repository(),
        order_repository=default_order_repository(),
    )

    update = node(
        AgentState(
            messages=[
                HumanMessage(content="How long does shipping take?"),
                AIMessage(id="draft", content="Shipping takes 3 to 5 business days."),
            ],
            already_promoted=True,
            promo_matches=[PromoMatch(promo_id="STALE", text="stale", relevance_score=1)],
            draft_response="stale draft",
            draft_message_id="stale-id",
            promo_line="stale line",
        )
    )

    assert update == {
        "promo_matches": [],
        "draft_response": None,
        "draft_message_id": None,
        "promo_line": None,
    }
