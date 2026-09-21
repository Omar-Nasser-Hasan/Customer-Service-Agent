"""Deterministic return-eligibility tool for verified orders."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Callable

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.orders import OrderRepository


RETURN_WINDOW_DAYS = 14


def create_returns_tool(
    repository: OrderRepository, *, today: Callable[[], date] = date.today
) -> BaseTool:
    @tool("returns")
    def returns(order_id: str) -> str:
        """Check whether a verified order is eligible for a return."""

        order = repository.find(order_id)
        if order is None:
            return json.dumps({"found": False, "order_id": order_id.upper()})
        if order.delivered_at is None:
            return json.dumps(
                {
                    "found": True,
                    "order_id": order.order_id,
                    "eligible": False,
                    "reason": "Returns can be started after the order is delivered.",
                }
            )

        deadline = order.delivered_at + timedelta(days=RETURN_WINDOW_DAYS)
        eligible = today() <= deadline
        return json.dumps(
            {
                "found": True,
                "order_id": order.order_id,
                "eligible": eligible,
                "return_deadline": str(deadline),
                "reason": "The order is within the 14-day return window."
                if eligible
                else "The 14-day return window has ended.",
            }
        )

    return returns
