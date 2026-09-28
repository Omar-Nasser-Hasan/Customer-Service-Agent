"""Postgres persistence for the staff inbox and WhatsApp transport.

This module deliberately stores a projection, not LangGraph checkpoint internals.
It is safe to rebuild from durable handoff state and keeps case queries stable as
the graph evolves.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
import asyncio
import sys
from typing import AsyncIterator, Iterable
from uuid import UUID, uuid4

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from customer_service.operations.models import CaseEvent, CaseMessage, CaseRecord, OutboxRecord, StaffIdentity
from customer_service.privacy.redaction import redact_text


class CaseConflictError(ValueError):
    pass


class CasePermissionError(ValueError):
    pass


class CaseNotFoundError(ValueError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS support_cases (
    case_id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('open','claimed','resolved')),
    assigned_to_sub TEXT,
    assigned_to_email TEXT,
    handoff_summary JSONB NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS support_cases_status_updated_idx ON support_cases(status, updated_at DESC);
CREATE TABLE IF NOT EXISTS case_messages (
    message_id UUID PRIMARY KEY,
    case_id UUID NOT NULL REFERENCES support_cases(case_id) ON DELETE CASCADE,
    direction TEXT NOT NULL CHECK (direction IN ('customer','bot','staff')),
    content TEXT NOT NULL,
    author_sub TEXT,
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    provider_message_id TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS case_messages_case_created_idx ON case_messages(case_id, created_at);
CREATE TABLE IF NOT EXISTS whatsapp_contacts (
    thread_id TEXT PRIMARY KEY,
    encrypted_phone TEXT NOT NULL,
    last_inbound_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS webhook_events (
    provider_event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS outbound_outbox (
    outbox_id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL,
    case_id UUID REFERENCES support_cases(case_id) ON DELETE SET NULL,
    message_id UUID REFERENCES case_messages(message_id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('text','template')),
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider_message_id TEXT UNIQUE,
    error_kind TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS outbound_outbox_due_idx ON outbound_outbox(status, next_attempt_at);
CREATE TABLE IF NOT EXISTS audit_events (
    audit_id UUID PRIMARY KEY,
    case_id UUID REFERENCES support_cases(case_id) ON DELETE CASCADE,
    actor_sub TEXT,
    actor_email TEXT,
    action TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS websocket_tokens (
    token_id TEXT PRIMARY KEY,
    staff_sub TEXT NOT NULL,
    staff_email TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ
);
ALTER TABLE websocket_tokens ADD COLUMN IF NOT EXISTS staff_email TEXT;
"""


def _model_values(row: dict[str, object]) -> dict[str, object]:
    """Adapt Psycopg UUID objects to the string IDs exposed by our API models."""

    return {key: str(value) if isinstance(value, UUID) else value for key, value in row.items()}


def _case(row: dict[str, object]) -> CaseRecord:
    return CaseRecord(**_model_values(row))


class OperationsRepository:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def setup(self) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(SCHEMA)
            await conn.commit()

    async def create_or_reconcile_case(self, thread_id: str, summary: dict[str, object]) -> CaseRecord:
        case_id = str(uuid4())
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """INSERT INTO support_cases (case_id, thread_id, status, handoff_summary)
                       VALUES (%s, %s, 'open', %s::jsonb)
                       ON CONFLICT (thread_id) DO UPDATE SET
                         handoff_summary = EXCLUDED.handoff_summary,
                         updated_at = now(), version = support_cases.version + 1
                       RETURNING *""",
                    (case_id, thread_id, json.dumps(summary)),
                )
                row = await cur.fetchone()
            await conn.commit()
        result = _case(row)
        await self.notify(CaseEvent(event="case_created", case_id=result.case_id, version=result.version))
        return result

    async def list_cases(self, status: str | None = None, limit: int = 50) -> list[CaseRecord]:
        query = "SELECT * FROM support_cases"
        params: tuple[object, ...] = ()
        if status:
            query += " WHERE status = %s"
            params = (status,)
        query += " ORDER BY updated_at DESC LIMIT %s"
        params += (limit,)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, params)
                return [_case(row) for row in await cur.fetchall()]

    async def get_case(self, case_id: str) -> CaseRecord:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT * FROM support_cases WHERE case_id = %s", (case_id,))
                row = await cur.fetchone()
        if row is None:
            raise CaseNotFoundError(case_id)
        return _case(row)

    async def get_by_thread(self, thread_id: str) -> CaseRecord | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT * FROM support_cases WHERE thread_id = %s", (thread_id,))
                row = await cur.fetchone()
        return _case(row) if row else None

    async def messages(self, case_id: str) -> list[CaseMessage]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT * FROM case_messages WHERE case_id=%s ORDER BY created_at", (case_id,))
                return [CaseMessage(**_model_values(row)) for row in await cur.fetchall()]

    async def claim(self, case_id: str, actor: StaffIdentity, version: int) -> CaseRecord:
        return await self._ownership_change(case_id, actor, version, "claim")

    async def release(self, case_id: str, actor: StaffIdentity, version: int) -> CaseRecord:
        return await self._ownership_change(case_id, actor, version, "release")

    async def _ownership_change(self, case_id: str, actor: StaffIdentity, version: int, action: str) -> CaseRecord:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                if action == "claim":
                    await cur.execute(
                        """UPDATE support_cases SET status='claimed', assigned_to_sub=%s,
                           assigned_to_email=%s, version=version+1, updated_at=now()
                           WHERE case_id=%s AND status='open' AND version=%s RETURNING *""",
                        (actor.sub, actor.email, case_id, version),
                    )
                else:
                    await cur.execute(
                        """UPDATE support_cases SET status='open', assigned_to_sub=NULL,
                           assigned_to_email=NULL, version=version+1, updated_at=now()
                           WHERE case_id=%s AND status='claimed' AND assigned_to_sub=%s AND version=%s RETURNING *""",
                        (case_id, actor.sub, version),
                    )
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            await self._raise_ownership_error(case_id, actor, version)
        result = _case(row)
        await self.audit(result.case_id, actor, f"case_{'claimed' if action == 'claim' else 'released'}")
        await self.notify(CaseEvent(event="case_updated", case_id=result.case_id, version=result.version))
        return result

    async def _raise_ownership_error(self, case_id: str, actor: StaffIdentity, version: int) -> None:
        existing = await self.get_case(case_id)
        if existing.version != version or existing.status == "claimed":
            raise CaseConflictError("case ownership changed")
        raise CasePermissionError(f"{actor.email} does not own this case")

    async def add_message(self, case_id: str, direction: str, content: str, *, author: StaffIdentity | None = None,
                          delivery_status: str = "pending", message_id: str | None = None,
                          provider_message_id: str | None = None) -> CaseMessage:
        message_id = message_id or str(uuid4())
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """INSERT INTO case_messages(message_id,case_id,direction,content,author_sub,delivery_status,provider_message_id)
                    VALUES(%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (provider_message_id) DO UPDATE SET content = EXCLUDED.content
                    RETURNING *""",
                    (message_id, case_id, direction, content, author.sub if author else None, delivery_status, provider_message_id),
                )
                row = await cur.fetchone()
                await cur.execute("UPDATE support_cases SET updated_at=now(), version=version+1 WHERE case_id=%s RETURNING version", (case_id,))
                version = (await cur.fetchone())["version"]
            await conn.commit()
        result = CaseMessage(**_model_values(row))
        await self.notify(CaseEvent(event="message", case_id=case_id, version=version, message_id=result.message_id))
        return result

    async def queue_outbox(self, thread_id: str, *, case_id: str | None, message_id: str | None,
                           kind: str, payload: dict[str, object]) -> OutboxRecord:
        outbox_id = str(uuid4())
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """INSERT INTO outbound_outbox(outbox_id,thread_id,case_id,message_id,kind,payload)
                    VALUES(%s,%s,%s,%s,%s,%s::jsonb) RETURNING *""",
                    (outbox_id, thread_id, case_id, message_id, kind, json.dumps(payload)),
                )
                row = await cur.fetchone()
            await conn.commit()
        return OutboxRecord(**_model_values(row))

    async def record_contact(self, thread_id: str, encrypted_phone: str, received_at: datetime | None = None) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO whatsapp_contacts(thread_id,encrypted_phone,last_inbound_at)
                    VALUES(%s,%s,%s) ON CONFLICT(thread_id) DO UPDATE SET encrypted_phone=EXCLUDED.encrypted_phone,
                    last_inbound_at=EXCLUDED.last_inbound_at,updated_at=now()""",
                    (thread_id, encrypted_phone, received_at or datetime.now(UTC)),
                )
            await conn.commit()

    async def last_inbound_at(self, thread_id: str) -> datetime | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT last_inbound_at FROM whatsapp_contacts WHERE thread_id=%s", (thread_id,))
                row = await cur.fetchone()
        return row["last_inbound_at"] if row else None

    async def encrypted_phone(self, thread_id: str) -> str | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT encrypted_phone FROM whatsapp_contacts WHERE thread_id=%s", (thread_id,))
                row = await cur.fetchone()
        return row["encrypted_phone"] if row else None

    async def mark_message_delivery(self, message_id: str | None, status: str, provider_message_id: str | None = None) -> None:
        if not message_id and not provider_message_id:
            return
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE case_messages SET delivery_status=%s,provider_message_id=COALESCE(%s,provider_message_id) WHERE message_id=%s OR provider_message_id=%s", (status, provider_message_id, message_id, provider_message_id))
            await conn.commit()

    async def record_webhook(self, event_id: str, event_type: str, payload: dict[str, object]) -> bool:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO webhook_events(provider_event_id,event_type,payload) VALUES(%s,%s,%s::jsonb) ON CONFLICT DO NOTHING", (event_id, event_type, json.dumps(payload)))
                inserted = cur.rowcount == 1
            await conn.commit()
        return inserted

    async def claim_webhooks(self, limit: int = 20) -> list[dict]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("""WITH claimed AS (SELECT provider_event_id FROM webhook_events
                  WHERE status IN ('pending','retry') AND next_attempt_at <= now() ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT %s)
                  UPDATE webhook_events e SET status='processing', attempts=attempts+1 FROM claimed
                  WHERE e.provider_event_id=claimed.provider_event_id RETURNING e.*""", (limit,))
                rows = await cur.fetchall()
            await conn.commit()
        return rows

    async def complete_webhook(self, event_id: str) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE webhook_events SET status='done',processed_at=now() WHERE provider_event_id=%s", (event_id,))
            await conn.commit()

    async def retry_webhook(self, event_id: str, attempts: int, max_attempts: int, error: Exception) -> None:
        status = "failed" if attempts >= max_attempts else "retry"
        next_at = datetime.now(UTC) + timedelta(seconds=min(300, 2 ** attempts))
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE webhook_events SET status=%s,next_attempt_at=%s WHERE provider_event_id=%s",
                    (status, next_at, event_id),
                )
            await conn.commit()

    async def claim_outbox(self, limit: int = 20) -> list[OutboxRecord]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("""WITH claimed AS (SELECT outbox_id FROM outbound_outbox
                  WHERE status IN ('pending','retry') AND next_attempt_at <= now() ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT %s)
                  UPDATE outbound_outbox o SET status='processing',attempts=attempts+1,updated_at=now() FROM claimed
                  WHERE o.outbox_id=claimed.outbox_id RETURNING o.*""", (limit,))
                rows = await cur.fetchall()
            await conn.commit()
        return [OutboxRecord(**_model_values(row)) for row in rows]

    async def complete_outbox(self, outbox_id: str, provider_message_id: str) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE outbound_outbox SET status='sent',provider_message_id=%s,updated_at=now() WHERE outbox_id=%s", (provider_message_id, outbox_id))
            await conn.commit()

    async def retry_outbox(self, outbox_id: str, attempts: int, max_attempts: int, error: Exception) -> None:
        status = "failed" if attempts >= max_attempts else "retry"
        next_at = datetime.now(UTC) + timedelta(seconds=min(300, 2 ** attempts))
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE outbound_outbox SET status=%s,next_attempt_at=%s,error_kind=%s,updated_at=now() WHERE outbox_id=%s", (status, next_at, type(error).__name__, outbox_id))
            await conn.commit()

    async def audit(self, case_id: str, actor: StaffIdentity | None, action: str, metadata: dict[str, object] | None = None) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO audit_events(audit_id,case_id,actor_sub,actor_email,action,metadata) VALUES(%s,%s,%s,%s,%s,%s::jsonb)", (str(uuid4()), case_id, actor.sub if actor else None, actor.email if actor else None, action, json.dumps(metadata or {})))
            await conn.commit()

    async def notify(self, event: CaseEvent) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT pg_notify('customer_service_case_events', %s)", (event.model_dump_json(),))
            await conn.commit()

    async def issue_ws_token(self, token_id: str, staff_sub: str, staff_email: str, expires_at: datetime) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO websocket_tokens(token_id,staff_sub,staff_email,expires_at) VALUES(%s,%s,%s,%s)", (token_id, staff_sub, staff_email, expires_at))
            await conn.commit()

    async def consume_ws_token(self, token_id: str, staff_sub: str) -> str | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("UPDATE websocket_tokens SET used_at=now() WHERE token_id=%s AND staff_sub=%s AND used_at IS NULL AND expires_at > now() RETURNING staff_email", (token_id, staff_sub))
                row = await cur.fetchone()
            await conn.commit()
        return row["staff_email"] if row else None

    async def recover_interrupted_work(self) -> None:
        """Return unsent outbound work leased by a crashed worker to retry.

        Webhook processing performs several durable writes before it can be
        marked complete. Replaying an event that was interrupted after its
        reply was queued could therefore send a customer the same answer
        twice. Ordinary processing failures are explicitly retried by the
        worker; processing webhook rows left by a hard process stop remain
        available for deliberate operator reconciliation until source-event
        idempotency is added.
        """

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE outbound_outbox SET status='retry', next_attempt_at=now(), updated_at=now() WHERE status='processing'")
            await conn.commit()

    async def prune_resolved(self, days: int = 90) -> int:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM support_cases WHERE status='resolved' AND resolved_at < now() - (%s || ' days')::interval", (days,))
                count = cur.rowcount
                await cur.execute("DELETE FROM outbound_outbox WHERE created_at < now() - (%s || ' days')::interval AND case_id IS NULL", (days,))
                await cur.execute("DELETE FROM webhook_events WHERE created_at < now() - (%s || ' days')::interval", (days,))
                await cur.execute("DELETE FROM websocket_tokens WHERE expires_at < now()")
            await conn.commit()
        return count


@asynccontextmanager
async def operations_repository_runtime(dsn: str) -> AsyncIterator[OperationsRepository]:
    pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=10, open=False)
    await pool.open()
    repository = OperationsRepository(pool)
    await repository.setup()
    try:
        yield repository
    finally:
        await pool.close()
