from __future__ import annotations

import json

from customer_service.infrastructure.orders import default_order_repository
from customer_service.tools.order_status import create_order_status_tool


def test_order_status_returns_customer_safe_seeded_order() -> None:
    tool = create_order_status_tool(default_order_repository())

    result = json.loads(tool.invoke({"order_id": "ord-1001"}))

    assert result == {
        "found": True,
        "order_id": "ORD-1001",
        "status": "shipped",
        "status_detail": "Your package left our warehouse and is with the carrier.",
        "updated_at": "2026-09-17T08:30:00Z",
        "estimated_delivery": "2026-09-21",
        "tracking_reference": "TRK-781204",
    }


def test_order_status_returns_structured_not_found_result() -> None:
    tool = create_order_status_tool(default_order_repository())

    result = json.loads(tool.invoke({"order_id": "ORD-9999"}))

    assert result == {
        "found": False,
        "order_id": "ORD-9999",
        "message": "No order matches that order ID.",
    }
