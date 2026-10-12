from __future__ import annotations

import json
from datetime import UTC, date, datetime

from customer_service.infrastructure.faqs import FaqEntry
from customer_service.infrastructure.orders import default_order_repository
from customer_service.tools.billing import create_billing_tool
from customer_service.tools.faq_lookup import create_faq_lookup_tool
from customer_service.tools.returns import create_returns_tool

# What the graph injects from state for a customer who verified as Alice.
ALICE = {"verified_email": "alice@example.com", "verified_at": datetime.now(UTC)}
CARLA = {"verified_email": "carla@example.com", "verified_at": datetime.now(UTC)}


def test_billing_returns_only_charge_information() -> None:
    result = json.loads(
        create_billing_tool(default_order_repository()).invoke({"order_id": "ORD-1001", **ALICE})
    )

    assert result == {
        "found": True,
        "order_id": "ORD-1001",
        "payment_status": "paid",
        "amount": "899.99",
        "currency": "EGP",
        "description": "Souqly order ORD-1001",
    }


def test_billing_refuses_without_a_grant_or_for_someone_elses_order() -> None:
    tool = create_billing_tool(default_order_repository())

    ungranted = json.loads(tool.invoke({"order_id": "ORD-1001"}))
    someone_elses = json.loads(tool.invoke({"order_id": "ORD-1002", **ALICE}))
    nonexistent = json.loads(tool.invoke({"order_id": "ORD-9999", **ALICE}))

    assert ungranted == {"found": False, "order_id": "ORD-1001"}
    assert someone_elses == {"found": False, "order_id": "ORD-1002"}
    assert nonexistent == {"found": False, "order_id": "ORD-9999"}


def test_returns_rejects_orders_that_have_not_been_delivered() -> None:
    tool = create_returns_tool(default_order_repository(), today=lambda: date(2026, 9, 20))

    result = json.loads(tool.invoke({"order_id": "ORD-1001", **ALICE}))

    assert result["eligible"] is False
    assert result["reason"] == "Returns can be started after the order is delivered."


def test_returns_calculates_the_14_day_window() -> None:
    repository = default_order_repository()
    eligible = json.loads(
        create_returns_tool(repository, today=lambda: date(2026, 9, 30)).invoke(
            {"order_id": "ORD-1003", **CARLA}
        )
    )
    expired = json.loads(
        create_returns_tool(repository, today=lambda: date(2026, 10, 1)).invoke(
            {"order_id": "ORD-1003", **CARLA}
        )
    )

    assert eligible["eligible"] is True
    assert eligible["return_deadline"] == "2026-09-30"
    assert expired["eligible"] is False
    assert expired["reason"] == "The 14-day return window has ended."


def test_returns_refuses_another_customers_delivered_order() -> None:
    tool = create_returns_tool(default_order_repository(), today=lambda: date(2026, 9, 20))

    assert json.loads(tool.invoke({"order_id": "ORD-1003", **ALICE})) == {
        "found": False,
        "order_id": "ORD-1003",
    }


def test_the_model_cannot_see_or_supply_the_injected_grant() -> None:
    for tool in (
        create_billing_tool(default_order_repository()),
        create_returns_tool(default_order_repository()),
    ):
        assert set(tool.tool_call_schema.model_json_schema()["properties"]) == {"order_id"}


async def test_faq_lookup_returns_matches_and_no_match_result() -> None:
    class Repository:
        async def search(self, query: str, *, language: str = "en") -> FaqEntry | None:
            assert language in {"en", "ar"}
            return FaqEntry(id="shipping-times", title="Shipping times", answer="Most orders arrive within 3 to 5 business days after dispatch.") if query == "How long does delivery take?" else None

    tool = create_faq_lookup_tool(Repository())

    shipping = json.loads(await tool.ainvoke({"query": "How long does delivery take?"}))
    unknown = json.loads(await tool.ainvoke({"query": "Tell me about cryptocurrency mining."}))

    assert shipping["faq_id"] == "shipping-times"
    assert "3 to 5 business days" in shipping["answer"]
    assert unknown == {"found": False, "message": "No matching FAQ was found."}
