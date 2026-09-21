from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from customer_service.api.app import create_app
from customer_service.config.settings import Settings
from customer_service.graph.build import build_graph
from customer_service.infrastructure.orders import default_order_repository
from customer_service.nodes.promo_judge.node import PromoDecision
from tests.conftest import ScriptedChatModel, ScriptedPromoJudge, declining_promo_judge


def test_message_endpoint_returns_agent_reply_and_keeps_thread_history() -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Please send your order ID."),
            AIMessage(content="Thanks. Order ORD-1001 has shipped."),
        ]
    )
    graph = build_graph(
        settings=Settings(),
        repository=default_order_repository(),
        model=model,
        promo_judge=declining_promo_judge(),
    )
    client = TestClient(create_app(lambda: graph))

    first = client.post(
        "/conversations/customer-42/messages",
        json={"message": "Where is my order?"},
    )
    second = client.post(
        "/conversations/customer-42/messages",
        json={"message": "It is ORD-1001."},
    )

    assert first.status_code == 200
    assert first.json() == {
        "thread_id": "customer-42",
        "reply": "Please send your order ID.",
    }
    assert second.status_code == 200
    assert second.json() == {
        "thread_id": "customer-42",
        "reply": "Thanks. Order ORD-1001 has shipped.",
    }


def test_message_endpoint_rejects_blank_message() -> None:
    client = TestClient(create_app(lambda: None))

    response = client.post("/conversations/customer-42/messages", json={"message": "  "})

    assert response.status_code == 422


def test_message_endpoint_returns_a_catalog_promotion_when_the_judge_approves() -> None:
    graph = build_graph(
        settings=Settings(),
        repository=default_order_repository(),
        model=ScriptedChatModel(
            responses=[AIMessage(content="Most orders arrive within 3 to 5 business days.")]
        ),
        promo_judge=ScriptedPromoJudge(
            [PromoDecision(include_promo=True, selected_promo_id="DEMO-SHIP-10")]
        ),
    )
    client = TestClient(create_app(lambda: graph))

    response = client.post(
        "/conversations/promo-api/messages",
        json={"message": "How long does shipping take?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": "promo-api",
        "reply": (
            "Most orders arrive within 3 to 5 business days.\n\n"
            "[Development sample] For a future order, use code DEMO-SHIP10 for 10% off standard shipping."
        ),
    }
