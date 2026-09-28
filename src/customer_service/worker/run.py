from __future__ import annotations

import asyncio
import logging
import time
from customer_service.api.dependencies import ApplicationRuntime
from customer_service.graph.response import final_reply
from customer_service.transport.whatsapp import WhatsAppTransport

LOGGER = logging.getLogger(__name__)


class WhatsAppWorker:
    def __init__(self, runtime: ApplicationRuntime) -> None:
        self.runtime = runtime
        self.transport = WhatsAppTransport(runtime.settings)

    async def once(self) -> None:
        for event in await self.runtime.operations.claim_webhooks():
            try:
                await self._process_event(event)
                await self.runtime.operations.complete_webhook(event["provider_event_id"])
            except Exception as error:
                LOGGER.warning("webhook_processing_failed", extra={"error_type": type(error).__name__})
                await self.runtime.operations.retry_webhook(
                    event["provider_event_id"],
                    event["attempts"],
                    self.runtime.settings.worker_max_attempts,
                    error,
                )
        for outbox in await self.runtime.operations.claim_outbox():
            try:
                encrypted = await self.runtime.operations.encrypted_phone(outbox.thread_id)
                if not encrypted:
                    raise ValueError("Missing encrypted WhatsApp contact")
                provider_id = await self.transport.send(self.transport.decrypt_phone(encrypted), outbox.kind, outbox.payload)
                await self.runtime.operations.complete_outbox(outbox.outbox_id, provider_id)
                await self.runtime.operations.mark_message_delivery(outbox.message_id, "sent", provider_id)
            except Exception as error:
                await self.runtime.operations.retry_outbox(outbox.outbox_id, outbox.attempts, self.runtime.settings.worker_max_attempts, error)

    async def _process_event(self, event: dict) -> None:
        payload = event["payload"]
        if event["event_type"] == "status":
            status = payload["status"]
            await self.runtime.operations.mark_message_delivery(None, status.get("status", "sent"), status.get("id"))
            return
        message = payload["message"]
        phone = str(message.get("from", ""))
        if not phone:
            return
        thread_id = self.transport.thread_id(phone)
        await self.runtime.operations.record_contact(thread_id, self.transport.encrypt_phone(phone))
        if message.get("type") != "text":
            await self.runtime.operations.queue_outbox(thread_id, case_id=None, message_id=None, kind="text", payload={"text": "I can help with text messages. Please send your question as text."})
            return
        content = str(message.get("text", {}).get("body", "")).strip()
        if not content:
            return
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.runtime.graph.aget_state(config)
        if snapshot.values.get("escalated") and snapshot.values.get("case_status") in {"open", "claimed"}:
            await self.runtime.cases.record_active_customer_message(thread_id, content, message_id=message.get("id"))
            return
        # Only shown when we're actually about to reply, matching Meta's
        # guidance. A failed typing-indicator call is cosmetic and must
        # never block or break the real reply.
        if event_id := message.get("id"):
            try:
                await self.transport.send_typing_indicator(str(event_id))
            except Exception as error:
                LOGGER.warning("typing_indicator_failed", extra={"error_type": type(error).__name__})
        # This is currently the only place tracing callbacks are attached for
        # WhatsApp-originated turns. Without this line, LANGSMITH_TRACING=true
        # has no effect on messages that arrive over WhatsApp — the API path
        # attaches its own callbacks separately in api/bot/conversations.py.
        config["callbacks"] = self.runtime.observability.callbacks(thread_id)
        started = time.monotonic()
        state = await self.runtime.graph.ainvoke({"messages": [("human", content)]}, config=config)
        LOGGER.info("graph_invoke_completed", extra={"elapsed_seconds": round(time.monotonic() - started, 2)})
        if state.get("escalated") and state.get("case_status") in {"open", "claimed"}:
            await self.runtime.cases.reconcile_handoff(thread_id, state)
        reply = final_reply(state)
        if reply:
            case = await self.runtime.operations.get_by_thread(thread_id)
            message_row = await self.runtime.operations.add_message(case.case_id, "bot", reply, delivery_status="queued") if case else None
            await self.runtime.operations.queue_outbox(thread_id, case_id=case.case_id if case else None, message_id=message_row.message_id if message_row else None, kind="text", payload={"text": reply})

    async def close(self) -> None:
        await self.transport.close()


async def main() -> None:
    runtime = ApplicationRuntime()
    await runtime.start()
    await runtime.operations.recover_interrupted_work()
    worker = WhatsAppWorker(runtime)
    try:
        while True:
            await worker.once()
            await asyncio.sleep(runtime.settings.worker_poll_seconds)
    finally:
        await worker.close()
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
