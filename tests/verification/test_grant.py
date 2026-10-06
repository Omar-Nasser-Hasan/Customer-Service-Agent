from __future__ import annotations

from datetime import UTC, datetime, timedelta

from customer_service.infrastructure.orders import default_order_repository
from customer_service.verification.grant import VerificationPolicy
from customer_service.verification.messages import CATALOG, detect_language, message

START = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now


def test_grant_is_valid_for_two_hours_then_expires() -> None:
    clock = Clock()
    policy = VerificationPolicy(clock=clock)

    assert policy.is_valid("alice@example.com", START)
    clock.now = START + timedelta(hours=1, minutes=59)
    assert policy.is_valid("alice@example.com", START)
    clock.now = START + timedelta(hours=2)
    assert not policy.is_valid("alice@example.com", START)
    assert policy.is_expired("alice@example.com", START)


def test_missing_grant_is_not_valid_and_not_expired() -> None:
    policy = VerificationPolicy(clock=Clock())

    assert not policy.is_valid(None, None)
    assert not policy.is_valid("alice@example.com", None)
    assert not policy.is_valid(None, START)
    assert not policy.is_expired(None, None)


def test_a_naive_timestamp_is_treated_as_utc() -> None:
    policy = VerificationPolicy(clock=Clock())

    assert policy.is_valid("alice@example.com", START.replace(tzinfo=None))


def test_owned_order_requires_a_valid_grant_and_ownership() -> None:
    clock = Clock()
    policy = VerificationPolicy(clock=clock)
    repository = default_order_repository()

    assert policy.owned_order(repository, "ORD-1001", "alice@example.com", START) is not None
    # Same answer for someone else's order, a nonexistent order, and junk input.
    assert policy.owned_order(repository, "ORD-1002", "alice@example.com", START) is None
    assert policy.owned_order(repository, "ORD-9999", "alice@example.com", START) is None
    assert policy.owned_order(repository, "not an order", "alice@example.com", START) is None
    # Formats are normalized the same way the claim parser does it.
    assert policy.owned_order(repository, "ord 1001", "Alice@Example.com", START) is not None

    clock.now = START + timedelta(hours=3)
    assert policy.owned_order(repository, "ORD-1001", "alice@example.com", START) is None


def test_language_follows_arabic_script() -> None:
    assert detect_language("Where is my order?") == "en"
    assert detect_language("أين طلبي؟") == "ar"
    assert detect_language("ORD-1001 وبريدي alice@example.com") == "ar"


def test_every_catalog_entry_exists_in_both_languages() -> None:
    for key, texts in CATALOG.items():
        assert set(texts) == {"en", "ar"}, key
        assert all(text.strip() for text in texts.values()), key


def test_failure_messages_never_say_which_field_was_wrong() -> None:
    for language in ("en", "ar"):
        for key in ("claim_failed", "order_not_found"):
            text = message(key, language).casefold()
            assert "email was" not in text and "order id was" not in text
