from __future__ import annotations

import json
from datetime import date

from customer_service.infrastructure.faqs import default_faq_repository
from customer_service.infrastructure.orders import default_order_repository
from customer_service.tools.billing import create_billing_tool
from customer_service.tools.faq_lookup import create_faq_lookup_tool
from customer_service.tools.returns import create_returns_tool


def test_billing_returns_only_charge_information() -> None:
    result = json.loads(
        create_billing_tool(default_order_repository()).invoke({"order_id": "ORD-1001"})
    )

    assert result == {
        "found": True,
        "order_id": "ORD-1001",
        "payment_status": "paid",
        "amount": "89.99",
        "currency": "USD",
        "description": "Souqly order ORD-1001",
    }


def test_returns_rejects_orders_that_have_not_been_delivered() -> None:
    tool = create_returns_tool(default_order_repository(), today=lambda: date(2026, 9, 20))

    result = json.loads(tool.invoke({"order_id": "ORD-1001"}))

    assert result["eligible"] is False
    assert result["reason"] == "Returns can be started after the order is delivered."


def test_returns_calculates_the_14_day_window() -> None:
    repository = default_order_repository()
    eligible = json.loads(
        create_returns_tool(repository, today=lambda: date(2026, 9, 30)).invoke(
            {"order_id": "ORD-1003"}
        )
    )
    expired = json.loads(
        create_returns_tool(repository, today=lambda: date(2026, 10, 1)).invoke(
            {"order_id": "ORD-1003"}
        )
    )

    assert eligible["eligible"] is True
    assert eligible["return_deadline"] == "2026-09-30"
    assert expired["eligible"] is False
    assert expired["reason"] == "The 14-day return window has ended."


def test_faq_lookup_returns_matches_and_no_match_result() -> None:
    tool = create_faq_lookup_tool(default_faq_repository())

    shipping = json.loads(tool.invoke({"query": "How long does delivery take?"}))
    unknown = json.loads(tool.invoke({"query": "Tell me about cryptocurrency mining."}))

    assert shipping["faq_id"] == "shipping-times"
    assert "3 to 5 business days" in shipping["answer"]
    assert unknown == {"found": False, "message": "No matching FAQ was found."}
