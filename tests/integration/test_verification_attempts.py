from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from docker.errors import DockerException
from psycopg_pool import ConnectionPool
from testcontainers.community.postgres import PostgresContainer

from customer_service.verification.attempts import PostgresAttemptLimiter


@pytest.fixture
def pool():
    try:
        with PostgresContainer("pgvector/pgvector:pg16") as postgres:
            dsn = postgres.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
            connection_pool = ConnectionPool(conninfo=dsn, open=False)
            connection_pool.open(wait=True)
            try:
                yield connection_pool
            finally:
                connection_pool.close()
    except DockerException as error:
        pytest.skip(f"Docker daemon is unavailable: {error}")


@pytest.mark.integration
def test_failures_are_shared_hashed_windowed_and_prunable(pool: ConnectionPool) -> None:
    now = [datetime.now(UTC)]
    limiter = PostgresAttemptLimiter(pool, hash_key="test-key", limit=3, clock=lambda: now[0])
    limiter.setup()
    limiter.setup()  # idempotent

    # Three guesses against one order, each with a different email.
    for index in range(3):
        limiter.record_failure(email=f"guess{index}@example.com", order_id="ORD-1001")

    assert limiter.is_blocked(email="alice@example.com", order_id="ORD-1001")
    assert not limiter.is_blocked(email="alice@example.com", order_id="ORD-1002")
    assert not limiter.is_blocked(email=None, order_id=None)

    with pool.connection() as connection:
        stored = [row[0] for row in connection.execute("SELECT key_hash FROM verification_failures")]
    assert stored and all(len(value) == 64 for value in stored)
    assert not any("ORD-1001" in value or "guess" in value for value in stored)

    # Failures stop counting after the window, and old rows are pruned.
    now[0] += timedelta(hours=25)
    assert not limiter.is_blocked(email="alice@example.com", order_id="ORD-1001")
    now[0] += timedelta(hours=30)
    limiter.prune()
    with pool.connection() as connection:
        assert connection.execute("SELECT count(*) FROM verification_failures").fetchone()[0] == 0
