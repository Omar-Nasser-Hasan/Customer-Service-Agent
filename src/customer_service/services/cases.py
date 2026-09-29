"""Synchronize staff case projection with durable LangGraph handoffs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from psycopg.rows import dict_row

from customer_service.config.settings import Settings, WhatsAppTemplate
from customer_service.operations.models import CaseEvent, CaseMessage, CaseRecord, StaffIdentity
from customer_service.operations.repository import CaseConflictError, CasePermissionError, OperationsRepository, _model_values
from customer_service.services.handoffs import HandoffNotOpenError, HandoffService


class CustomerWindowClosedError(ValueError):
    pass


class TemplateValidationError(ValueError):
    pass


class CaseService:
    """Applies ownership rules before changing graph or delivery state."""

    def __init__(self, graph: Any, handoffs: HandoffService, repository: OperationsRepository, settings: Settings) -> None:
        self.graph, self.handoffs, self.repository, self.settings = graph, handoffs, repository, settings

    async def reconcile_handoff(self, thread_id: str, state: dict[str, object] | None = None) -> CaseRecord | None:
        state = state or (await self.graph.aget_state({"configurable": {"thread_id": thread_id}})).values
        if not (state.get("escalated") and state.get("case_status") in {"open", "claimed"}):
            return None
        summary = state.get("handoff_summary")
        if hasattr(summary, "model_dump"):
            summary = summary.model_dump(mode="json")
        case = await self.repository.create_or_reconcile_case(thread_id, summary if isinstance(summary, dict) else {"reason": "safety_uncertain", "latest_customer_request": "", "verified": False})
        await self._backfill_visible_messages(case, state)
        return case

    async def _backfill_visible_messages(self, case: CaseRecord, state: dict[str, object]) -> None:
        recorded = await self.repository.messages(case.case_id)
        existing = {message.message_id for message in recorded} | {
            message.provider_message_id for message in recorded if message.provider_message_id
        }
        for message in state.get("messages", []):
            if not isinstance(message, (HumanMessage, AIMessage)) or not getattr(message, "id", None) or message.id in existing:
                continue
            content = str(message.content)
            if content:
                await self.repository.add_message(
                    case.case_id,
                    "customer" if isinstance(message, HumanMessage) else "bot",
                    content,
                    provider_message_id=message.id,
                )

    async def record_active_customer_message(self, thread_id: str, content: str, message_id: str | None = None) -> CaseMessage | None:
        config = {"configurable": {"thread_id": thread_id}}
        await self.graph.aupdate_state(config, {"messages": [HumanMessage(content=content, id=message_id)], "last_activity_at": datetime.now(UTC)})
        case = await self.repository.get_by_thread(thread_id)
        return (
            await self.repository.add_message(case.case_id, "customer", content, provider_message_id=message_id)
            if case
            else None
        )

    async def claim(self, case_id: str, actor: StaffIdentity, version: int) -> CaseRecord:
        case = await self.repository.claim(case_id, actor, version)
        await self.graph.aupdate_state({"configurable": {"thread_id": case.thread_id}}, {"case_status": "claimed", "assigned_to": actor.sub})
        return case

    async def release(self, case_id: str, actor: StaffIdentity, version: int) -> CaseRecord:
        case = await self.repository.release(case_id, actor, version)
        await self.graph.aupdate_state({"configurable": {"thread_id": case.thread_id}}, {"case_status": "open", "assigned_to": None})
        return case

    async def queue_staff_reply(self, case_id: str, actor: StaffIdentity, version: int, *, text: str | None = None, template_id: str | None = None, parameters: list[str] | None = None) -> CaseMessage:
        case = await self._owned_case(case_id, actor, version)
        if text is not None:
            if not await self._inside_service_window(case.thread_id):
                raise CustomerWindowClosedError("Free-text replies require an active WhatsApp customer-service window")
            kind, payload, display = "text", {"text": text}, text
        else:
            template = self._template(template_id or "", parameters or [])
            kind = "template"
            payload = {"template_id": template.template_id, "name": template.name, "language": template.language, "parameters": parameters or []}
            display = f"Template: {template.name}"
        message = await self.repository.add_message(case_id, "staff", display, author=actor, delivery_status="queued")
        await self.graph.aupdate_state({"configurable": {"thread_id": case.thread_id}}, {"messages": [AIMessage(content=text or display, id=message.message_id, name="staff")]})
        await self.repository.queue_outbox(case.thread_id, case_id=case_id, message_id=message.message_id, kind=kind, payload=payload)
        await self.repository.audit(case_id, actor, "case_replied", {"message_id": message.message_id, "kind": kind})
        return message

    async def resolve(self, case_id: str, actor: StaffIdentity, version: int) -> CaseRecord:
        case = await self._owned_case(case_id, actor, version)
        try:
            await self.handoffs.resolve(case.thread_id)
        except HandoffNotOpenError as error:
            raise CaseConflictError("The graph handoff is no longer open") from error
        async with self.repository.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("UPDATE support_cases SET status='resolved',version=version+1,updated_at=now(),resolved_at=now() WHERE case_id=%s AND status='claimed' AND assigned_to_sub=%s AND version=%s RETURNING *", (case_id, actor.sub, version))
                row = await cur.fetchone()
            await conn.commit()
        if not row:
            raise CaseConflictError("case changed while resolving")
        result = CaseRecord(**_model_values(row))
        await self.repository.audit(case_id, actor, "case_resolved")
        await self.repository.notify(CaseEvent(event="case_updated", case_id=case_id, version=result.version))
        return result

    async def _owned_case(self, case_id: str, actor: StaffIdentity, version: int) -> CaseRecord:
        case = await self.repository.get_case(case_id)
        if case.version != version:
            raise CaseConflictError("case version is stale")
        if case.status != "claimed" or case.assigned_to_sub != actor.sub:
            raise CasePermissionError("Only the current claimant may take this action")
        return case

    async def _inside_service_window(self, thread_id: str) -> bool:
        last = await self.repository.last_inbound_at(thread_id)
        return bool(last and last >= datetime.now(UTC) - timedelta(hours=self.settings.whatsapp_service_window_hours))

    def _template(self, template_id: str, parameters: list[str]) -> WhatsAppTemplate:
        template = next((item for item in self.settings.whatsapp_template_catalog if item.template_id == template_id), None)
        if not template or len(parameters) != template.parameter_count:
            raise TemplateValidationError("Template is not approved or parameters do not match its deployment schema")
        return template
