"""Deterministic order-status tool, scoped to the verified customer's own orders."""

import json

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.orders import OrderRepository
from customer_service.tools.injected import VerifiedAt, VerifiedEmail
from customer_service.verification.claims import normalize_order_id
from customer_service.verification.grant import VerificationPolicy


def create_order_status_tool(
    repository: OrderRepository, *, policy: VerificationPolicy | None = None
) -> BaseTool:
    grant = policy or VerificationPolicy()

    @tool("order_status")
    def order_status(
        order_id: str,
        verified_email: VerifiedEmail = None,
        verified_at: VerifiedAt = None,
    ) -> str:
        """Look up the current status of an order by its order ID."""

        normalized_id = normalize_order_id(order_id) or order_id.strip().upper()
        # Unverified, expired, someone else's, and nonexistent all look the same.
        order = grant.owned_order(repository, normalized_id, verified_email, verified_at)
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
