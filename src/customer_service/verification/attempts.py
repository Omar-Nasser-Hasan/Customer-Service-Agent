"""Failure counting that outlives a single conversation thread.

The per-thread counter lives in graph state. These limiters count failures per
hashed email and per hashed order ID across every thread and channel, so a new
thread (or the public HTTP endpoint) cannot reset a guessing attempt.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import threading
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

KEY_FAILURE_LIMIT = 5
FAILURE_WINDOW = timedelta(hours=24)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def recent_failures(failures: list[datetime], now: datetime) -> list[datetime]:
    """Failures inside the rolling window; older ones stop counting."""

    cutoff = now - FAILURE_WINDOW
    return [failure for failure in failures if _aware(failure) > cutoff]


def _keys(email: str | None, order_id: str | None) -> list[str]:
    keys: list[str] = []
    if email:
        keys.append(f"email:{email.strip().casefold()}")
    if order_id:
        keys.append(f"order:{order_id.strip().upper()}")
    return keys


class AttemptLimiter(Protocol):
    """Synchronous on purpose: graph nodes are sync and run in a worker thread."""

    def is_blocked(self, *, email: str | None, order_id: str | None) -> bool: ...

    def record_failure(self, *, email: str | None, order_id: str | None) -> None: ...


class InMemoryAttemptLimiter:
    """Process-local limiter for tests and injected graphs."""

    def __init__(
        self,
        *,
        limit: int = KEY_FAILURE_LIMIT,
        window: timedelta = FAILURE_WINDOW,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._limit, self._window, self._clock = limit, window, clock
        self._failures: dict[str, list[datetime]] = defaultdict(list)
        self._lock = threading.Lock()

    def _recent(self, key: str, now: datetime) -> list[datetime]:
        cutoff = now - self._window
        self._failures[key] = [failure for failure in self._failures[key] if failure > cutoff]
        return self._failures[key]

    def is_blocked(self, *, email: str | None, order_id: str | None) -> bool:
        with self._lock:
            now = self._clock()
            return any(
                len(self._recent(key, now)) >= self._limit for key in _keys(email, order_id)
            )

    def record_failure(self, *, email: str | None, order_id: str | None) -> None:
        with self._lock:
            now = self._clock()
            for key in _keys(email, order_id):
                self._recent(key, now).append(now)


class PostgresAttemptLimiter:
    """Shared limiter. Only HMAC digests of emails and order IDs are stored."""

    def __init__(
        self,
        pool: ConnectionPool,
        *,
        hash_key: str,
        limit: int = KEY_FAILURE_LIMIT,
        window: timedelta = FAILURE_WINDOW,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._pool, self._limit, self._window, self._clock = pool, limit, window, clock
        self._hash_key = hash_key.encode()

    def setup(self) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS verification_failures (
                       key_hash TEXT NOT NULL,
                       created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                   )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS verification_failures_key_idx "
                "ON verification_failures (key_hash, created_at)"
            )

    def _hashed(self, email: str | None, order_id: str | None) -> list[str]:
        return [
            hmac.new(self._hash_key, key.encode(), hashlib.sha256).hexdigest()
            for key in _keys(email, order_id)
        ]

    def is_blocked(self, *, email: str | None, order_id: str | None) -> bool:
        hashed = self._hashed(email, order_id)
        if not hashed:
            return False
        cutoff = self._clock() - self._window
        with self._pool.connection() as connection:
            rows = connection.execute(
                "SELECT key_hash, count(*) FROM verification_failures "
                "WHERE key_hash = ANY(%s) AND created_at > %s GROUP BY key_hash",
                (hashed, cutoff),
            ).fetchall()
        return any(count >= self._limit for _, count in rows)

    def record_failure(self, *, email: str | None, order_id: str | None) -> None:
        hashed = self._hashed(email, order_id)
        if not hashed:
            return
        now = self._clock()
        with self._pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO verification_failures (key_hash, created_at) VALUES (%s, %s)",
                    [(key_hash, now) for key_hash in hashed],
                )

    def prune(self) -> None:
        cutoff = self._clock() - 2 * self._window
        with self._pool.connection() as connection:
            connection.execute("DELETE FROM verification_failures WHERE created_at < %s", (cutoff,))


@asynccontextmanager
async def postgres_attempt_limiter(
    dsn: str, *, hash_key: str, limit: int = KEY_FAILURE_LIMIT
) -> AsyncIterator[PostgresAttemptLimiter]:
    """Application-lifetime limiter on a small synchronous pool.

    Graph nodes are synchronous and LangGraph runs them in worker threads, so
    the limiter uses a sync pool instead of the async pools used elsewhere.
    """

    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=4, open=False)
    await asyncio.to_thread(pool.open, wait=True)
    try:
        limiter = PostgresAttemptLimiter(pool, hash_key=hash_key, limit=limit)
        await asyncio.to_thread(limiter.setup)
        yield limiter
    finally:
        await asyncio.to_thread(pool.close)
