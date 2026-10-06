from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from customer_service.infrastructure.orders import default_order_repository
from customer_service.tools.order_status import create_order_status_tool
from customer_service.verification.grant import VerificationPolicy

ALICE = {"verified_email": "alice@example.com", "verified_at": datetime.now(UTC)}


def test_order_status_returns_customer_safe_seeded_order() -> None:
    tool = create_order_status_tool(default_order_repository())

    result = json.loads(tool.invoke({"order_id": "ord-1001", **ALICE}))

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

    result = json.loads(tool.invoke({"order_id": "ORD-9999", **ALICE}))

    assert result == {
        "found": False,
        "order_id": "ORD-9999",
        "message": "No order matches that order ID.",
    }


def test_unverified_expired_and_foreign_orders_are_indistinguishable_from_missing() -> None:
    tool = create_order_status_tool(default_order_repository())
    expired = {"verified_email": "alice@example.com", "verified_at": datetime.now(UTC) - timedelta(hours=3)}

    unverified = json.loads(tool.invoke({"order_id": "ORD-1001"}))
    after_expiry = json.loads(tool.invoke({"order_id": "ORD-1001", **expired}))
    foreign = json.loads(tool.invoke({"order_id": "ORD-1002", **ALICE}))

    for result, order_id in ((unverified, "ORD-1001"), (after_expiry, "ORD-1001"), (foreign, "ORD-1002")):
        assert result == {
            "found": False,
            "order_id": order_id,
            "message": "No order matches that order ID.",
        }


def test_tool_uses_the_graphs_policy_clock() -> None:
    start = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    now = [start]
    tool = create_order_status_tool(
        default_order_repository(), policy=VerificationPolicy(clock=lambda: now[0])
    )
    grant = {"verified_email": "alice@example.com", "verified_at": start}

    assert json.loads(tool.invoke({"order_id": "ORD-1001", **grant}))["found"] is True
    now[0] = start + timedelta(hours=2)
    assert json.loads(tool.invoke({"order_id": "ORD-1001", **grant}))["found"] is False
