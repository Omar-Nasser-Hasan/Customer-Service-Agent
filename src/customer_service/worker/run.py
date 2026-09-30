from __future__ import annotations

import asyncio
import logging
import time
from customer_service.api.dependencies import ApplicationRuntime
from customer_service.graph.response import final_reply
from customer_service.transport.whatsapp import WhatsAppTransport
from customer_service.observability.logs import configure_logging
LOGGER = logging.getLogger("customer_service.worker")


class WhatsAppWorker:
    def __init__(self, runtime: ApplicationRuntime) -> None:
        self.runtime = runtime
        self.transport = WhatsAppTransport(runtime.settings)

    async def once(self) -> None:
        webhooks = await self.runtime.operations.claim_webhooks()
        if webhooks:
            LOGGER.info("webhooks_claimed", extra={"count": len(webhooks)})
        for event in webhooks:
            try:
                await self._process_event(event)
                await self.runtime.operations.complete_webhook(event["provider_event_id"])
            except Exception as error:
                LOGGER.warning(
                    "webhook_processing_failed",
                    extra={"error_type": type(error).__name__, "event_type": event.get("event_type")},
                )
                await self.runtime.operations.retry_webhook(
                    event["provider_event_id"],
                    event["attempts"],
                    self.runtime.settings.worker_max_attempts,
                    error,
                )
        outbox_items = await self.runtime.operations.claim_outbox()
        if outbox_items:
            LOGGER.info("outbox_claimed", extra={"count": len(outbox_items)})
        for outbox in outbox_items:
            started = time.monotonic()
            try:
                encrypted = await self.runtime.operations.encrypted_phone(outbox.thread_id)
                if not encrypted:
                    raise ValueError("Missing encrypted WhatsApp contact")
                provider_id = await self.transport.send(self.transport.decrypt_phone(encrypted), outbox.kind, outbox.payload)
                await self.runtime.operations.complete_outbox(outbox.outbox_id, provider_id)
                await self.runtime.operations.mark_message_delivery(outbox.message_id, "sent", provider_id)
                LOGGER.info(
                    "outbox_sent",
                    extra={"kind": outbox.kind, "attempts": outbox.attempts, "elapsed_seconds": round(time.monotonic() - started, 2)},
                )
            except Exception as error:
                LOGGER.warning(
                    "outbox_send_failed",
                    extra={"kind": outbox.kind, "attempts": outbox.attempts, "error_type": type(error).__name__},
                )
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
        thread_ref = self.runtime.observability.thread_reference(thread_id)
        await self.runtime.operations.record_contact(thread_id, self.transport.encrypt_phone(phone))
        if message.get("type") != "text":
            LOGGER.info("non_text_message_received", extra={"thread_ref": thread_ref, "message_type": message.get("type")})
            await self.runtime.operations.queue_outbox(thread_id, case_id=None, message_id=None, kind="text", payload={"text": "I can help with text messages. Please send your question as text."})
            return
        content = str(message.get("text", {}).get("body", "")).strip()
        if not content:
            return
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.runtime.graph.aget_state(config)
        if snapshot.values.get("escalated") and snapshot.values.get("case_status") in {"open", "claimed"}:
            LOGGER.info("message_recorded_during_active_handoff", extra={"thread_ref": thread_ref})
            await self.runtime.cases.record_active_customer_message(thread_id, content, message_id=message.get("id"))
            return
        if event_id := message.get("id"):
            try:
                await self.transport.send_typing_indicator(str(event_id))
            except Exception as error:
                LOGGER.warning("typing_indicator_failed", extra={"thread_ref": thread_ref, "error_type": type(error).__name__})
        config["callbacks"] = self.runtime.observability.callbacks(thread_id)
        started = time.monotonic()
        state = await self.runtime.graph.ainvoke({"messages": [("human", content)]}, config=config)
        LOGGER.info(
            "graph_invoke_completed",
            extra={
                "thread_ref": thread_ref,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "escalated": bool(state.get("escalated")),
            },
        )
        if state.get("escalated") and state.get("case_status") in {"open", "claimed"}:
            await self.runtime.cases.reconcile_handoff(thread_id, state)
        reply = final_reply(state)
        if reply:
            case = await self.runtime.operations.get_by_thread(thread_id)
            message_row = await self.runtime.operations.add_message(case.case_id, "bot", reply, delivery_status="queued") if case else None
            await self.runtime.operations.queue_outbox(thread_id, case_id=case.case_id if case else None, message_id=message_row.message_id if message_row else None, kind="text", payload={"text": reply})
            LOGGER.info("reply_queued", extra={"thread_ref": thread_ref, "reply_length": len(reply)})

    async def close(self) -> None:
        await self.transport.close()


async def main() -> None:
    configure_logging()
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
