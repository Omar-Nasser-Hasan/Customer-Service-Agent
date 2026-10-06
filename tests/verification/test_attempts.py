from __future__ import annotations

from datetime import UTC, datetime, timedelta

from customer_service.verification.attempts import (
    FAILURE_WINDOW,
    InMemoryAttemptLimiter,
    recent_failures,
)

START = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now


def test_five_failures_for_one_email_block_it_across_orders() -> None:
    limiter = InMemoryAttemptLimiter(clock=Clock())

    for index in range(4):
        limiter.record_failure(email="alice@example.com", order_id=f"ORD-{index}")
    assert not limiter.is_blocked(email="alice@example.com", order_id="ORD-77")
    limiter.record_failure(email="Alice@Example.com", order_id="ORD-4")

    assert limiter.is_blocked(email="alice@example.com", order_id="ORD-77")
    assert not limiter.is_blocked(email="ben@example.com", order_id="ORD-77")


def test_five_failures_for_one_order_block_it_across_emails() -> None:
    limiter = InMemoryAttemptLimiter(clock=Clock())

    for index in range(5):
        limiter.record_failure(email=f"guess{index}@example.com", order_id="ord-1001")

    assert limiter.is_blocked(email="alice@example.com", order_id="ORD-1001")
    assert not limiter.is_blocked(email="alice@example.com", order_id="ORD-1002")


def test_failures_stop_counting_after_the_window() -> None:
    clock = Clock()
    limiter = InMemoryAttemptLimiter(limit=2, clock=clock)
    limiter.record_failure(email="a@example.com", order_id=None)
    limiter.record_failure(email="a@example.com", order_id=None)
    assert limiter.is_blocked(email="a@example.com", order_id=None)

    clock.now = START + FAILURE_WINDOW + timedelta(seconds=1)

    assert not limiter.is_blocked(email="a@example.com", order_id=None)


def test_nothing_to_key_on_means_never_blocked() -> None:
    limiter = InMemoryAttemptLimiter(limit=1, clock=Clock())

    limiter.record_failure(email=None, order_id=None)

    assert not limiter.is_blocked(email=None, order_id=None)


def test_recent_failures_drops_old_and_accepts_naive_timestamps() -> None:
    now = START
    failures = [
        now - timedelta(hours=25),
        now - timedelta(hours=23),
        (now - timedelta(hours=1)).replace(tzinfo=None),
    ]

    assert recent_failures(failures, now) == failures[1:]
