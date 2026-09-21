"""Verified-order billing lookup tool."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.orders import OrderRepository


def create_billing_tool(repository: OrderRepository) -> BaseTool:
    @tool("billing")
    def billing(order_id: str) -> str:
        """Explain the recorded charge for a verified order."""

        order = repository.find(order_id)
        if order is None:
            return json.dumps({"found": False, "order_id": order_id.upper()})
        return json.dumps(
            {
                "found": True,
                "order_id": order.order_id,
                "payment_status": order.payment_status,
                "amount": str(order.charged_amount),
                "currency": order.currency,
                "description": order.charge_description,
            }
        )

    return billing
