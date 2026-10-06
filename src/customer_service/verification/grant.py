"""The single definition of "verified": an email, a timestamp, and ownership of an order."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from customer_service.infrastructure.orders import OrderRecord, OrderRepository
from customer_service.verification.claims import normalize_order_id

DEFAULT_TTL = timedelta(hours=2)


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class VerificationPolicy:
    """Shared by the claim node, the account gate, the tools, and the prompts.

    One object means one clock and one TTL everywhere, so a grant can never be
    valid for the assistant but expired for the tool.
    """

    ttl: timedelta = DEFAULT_TTL
    clock: Callable[[], datetime] = utc_now

    def now(self) -> datetime:
        return self.clock()

    def is_valid(self, verified_email: str | None, verified_at: datetime | None) -> bool:
        if not verified_email or verified_at is None:
            return False
        started = verified_at if verified_at.tzinfo else verified_at.replace(tzinfo=UTC)
        return self.now() - started < self.ttl

    def is_expired(self, verified_email: str | None, verified_at: datetime | None) -> bool:
        """A grant exists but has run out, as opposed to never having been given."""

        return bool(verified_email and verified_at is not None) and not self.is_valid(
            verified_email, verified_at
        )

    def owned_order(
        self,
        repository: OrderRepository,
        order_id: str,
        verified_email: str | None,
        verified_at: datetime | None,
    ) -> OrderRecord | None:
        """The order, only if the grant is valid and the verified email owns it.

        None covers an expired grant, a nonexistent order, and someone else's
        order alike, so callers cannot tell the cases apart.
        """

        if not self.is_valid(verified_email, verified_at):
            return None
        normalized = normalize_order_id(order_id)
        if normalized is None or verified_email is None:
            return None
        return repository.verify_identity(normalized, verified_email)
