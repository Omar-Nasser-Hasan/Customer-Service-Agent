from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response

from customer_service.api.dependencies import ApplicationRuntime
from customer_service.transport.whatsapp import WhatsAppTransport


def create_router(runtime_provider: Any) -> APIRouter:
    router = APIRouter(prefix="/bot/whatsapp", tags=["whatsapp"])

    @router.get("/webhook")
    async def verify(hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None, hub_verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None, hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None) -> Response:
        settings = runtime_provider().settings
        expected = settings.require_secret("META_VERIFY_TOKEN", settings.meta_verify_token)
        if hub_mode != "subscribe" or not hub_verify_token or hub_verify_token != expected:
            raise HTTPException(403, "Meta verification failed")
        return Response(content=hub_challenge or "", media_type="text/plain")

    @router.post("/webhook")
    async def receive(request: Request, x_hub_signature_256: Annotated[str | None, Header()] = None) -> dict[str, str]:
        runtime: ApplicationRuntime = runtime_provider()
        body = await request.body()
        secret = runtime.settings.require_secret("META_APP_SECRET", runtime.settings.meta_app_secret)
        if not WhatsAppTransport.valid_signature(body, x_hub_signature_256, secret):
            raise HTTPException(403, "Meta signature failed")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise HTTPException(400, "Malformed Meta payload") from error
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for message in value.get("messages", []):
                    event_id = message.get("id")
                    if event_id:
                        await runtime.operations.record_webhook(str(event_id), "message", {"message": message, "contacts": value.get("contacts", [])})
                for status in value.get("statuses", []):
                    event_id = status.get("id")
                    if event_id:
                        await runtime.operations.record_webhook(str(event_id), "status", {"status": status})
        return {"status": "accepted"}

    return router
