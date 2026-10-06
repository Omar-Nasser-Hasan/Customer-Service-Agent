"""Verified-order billing lookup tool, scoped to the verified customer's own orders."""

import json

from langchain_core.tools import BaseTool, tool

from customer_service.infrastructure.orders import OrderRepository
from customer_service.tools.injected import VerifiedAt, VerifiedEmail
from customer_service.verification.claims import normalize_order_id
from customer_service.verification.grant import VerificationPolicy


def create_billing_tool(
    repository: OrderRepository, *, policy: VerificationPolicy | None = None
) -> BaseTool:
    grant = policy or VerificationPolicy()

    @tool("billing")
    def billing(
        order_id: str,
        verified_email: VerifiedEmail = None,
        verified_at: VerifiedAt = None,
    ) -> str:
        """Explain the recorded charge for a verified order."""

        normalized_id = normalize_order_id(order_id) or order_id.strip().upper()
        order = grant.owned_order(repository, normalized_id, verified_email, verified_at)
        if order is None:
            return json.dumps({"found": False, "order_id": normalized_id})
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
