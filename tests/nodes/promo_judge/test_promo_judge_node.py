from __future__ import annotations

from langchain_core.messages import AIMessage

from customer_service.config.settings import Settings
from customer_service.nodes.promo_judge.node import PromoDecision, build_node
from customer_service.state.models import AgentState, PromoMatch
from tests.conftest import ScriptedPromoJudge


def test_declined_promo_leaves_no_promo_line() -> None:
    node = build_node(
        settings=Settings(),
        judge=ScriptedPromoJudge([PromoDecision(include_promo=False)]),
    )
    update = node(
        AgentState(
            messages=[AIMessage(id="draft", content="Shipping takes 3 to 5 business days.")],
            draft_response="Shipping takes 3 to 5 business days.",
            draft_message_id="draft",
            promo_matches=[PromoMatch(promo_id="P1", text="Catalog text", relevance_score=1)],
        )
    )

    assert update == {"promo_line": None}


def test_invalid_structured_judge_output_fails_closed() -> None:
    node = build_node(
        settings=Settings(),
        judge=ScriptedPromoJudge([{"include_promo": True, "selected_promo_id": None}]),
    )
    update = node(
        AgentState(
            messages=[AIMessage(id="draft", content="Shipping takes 3 to 5 business days.")],
            draft_response="Shipping takes 3 to 5 business days.",
            draft_message_id="draft",
            promo_matches=[PromoMatch(promo_id="P1", text="Catalog text", relevance_score=1)],
        )
    )

    assert update == {"promo_line": None}
