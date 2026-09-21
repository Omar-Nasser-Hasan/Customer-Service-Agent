"""Local order data adapter, replaceable by a commerce integration later."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


OrderState = Literal[
    "processing",
    "packed",
    "shipped",
    "out_for_delivery",
    "delivered",
    "cancelled",
]


class OrderRecord(BaseModel):
    order_id: str
    status: OrderState
    status_detail: str
    updated_at: datetime
    estimated_delivery: date | None = None
    tracking_reference: str | None = None
    customer_email: str = Field(exclude=True)
    customer_name: str | None = Field(default=None, exclude=True)
    delivered_at: date | None = Field(default=None, exclude=True)
    payment_status: Literal["paid", "pending", "refunded"] = Field(exclude=True)
    charged_amount: Decimal = Field(exclude=True)
    currency: str = Field(exclude=True)
    charge_description: str = Field(exclude=True)

    model_config = ConfigDict(extra="forbid")


class OrderRepository:
    def __init__(self, orders: list[OrderRecord]) -> None:
        self._by_order_id = {order.order_id.upper(): order for order in orders}
        if len(self._by_order_id) != len(orders):
            raise ValueError("Seed orders must have unique order IDs")

    @classmethod
    def from_path(cls, path: Path) -> "OrderRepository":
        raw_orders = json.loads(path.read_text(encoding="utf-8"))
        return cls(TypeAdapter(list[OrderRecord]).validate_python(raw_orders))

    def find(self, order_id: str) -> OrderRecord | None:
        return self._by_order_id.get(order_id.strip().upper())

    def verify_identity(self, order_id: str, email: str) -> OrderRecord | None:
        order = self.find(order_id)
        if order is None or order.customer_email.casefold() != email.strip().casefold():
            return None
        return order


def default_order_repository() -> OrderRepository:
    return OrderRepository.from_path(Path(__file__).parents[1] / "data" / "orders.json")
