"""Deterministic return-eligibility tool, scoped to the verified customer's own orders."""

import json
from datetime import date, timedelta
from typing import Callable

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.orders import OrderRepository
from customer_service.tools.injected import VerifiedAt, VerifiedEmail
from customer_service.verification.claims import normalize_order_id
from customer_service.verification.grant import VerificationPolicy


RETURN_WINDOW_DAYS = 14


def create_returns_tool(
    repository: OrderRepository,
    *,
    today: Callable[[], date] = date.today,
    policy: VerificationPolicy | None = None,
) -> BaseTool:
    grant = policy or VerificationPolicy()

    @tool("returns")
    def returns(
        order_id: str,
        verified_email: VerifiedEmail = None,
        verified_at: VerifiedAt = None,
    ) -> str:
        """Check whether a verified order is eligible for a return."""

        normalized_id = normalize_order_id(order_id) or order_id.strip().upper()
        order = grant.owned_order(repository, normalized_id, verified_email, verified_at)
        if order is None:
            return json.dumps({"found": False, "order_id": normalized_id})
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
