"""Deterministic order-status tool."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.orders import OrderRepository


def create_order_status_tool(repository: OrderRepository) -> BaseTool:
    @tool("order_status")
    def order_status(order_id: str) -> str:
        """Look up the current status of an order by its order ID."""

        normalized_id = order_id.strip().upper()
        order = repository.find(normalized_id)
        if order is None:
            return json.dumps(
                {
                    "found": False,
                    "order_id": normalized_id,
                    "message": "No order matches that order ID.",
                }
            )
        return json.dumps(
            {
                "found": True,
                "order_id": order.order_id,
                "status": order.status,
                "status_detail": order.status_detail,
                "updated_at": order.updated_at.isoformat().replace("+00:00", "Z"),
                "estimated_delivery": str(order.estimated_delivery)
                if order.estimated_delivery
                else None,
                "tracking_reference": order.tracking_reference,
            }
        )

    return order_status
