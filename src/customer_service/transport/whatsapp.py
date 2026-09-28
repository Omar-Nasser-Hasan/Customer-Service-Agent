from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
from cryptography.fernet import Fernet

from customer_service.config.settings import Settings


class WhatsAppTransport:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=15)
        key = settings.require_secret("WHATSAPP_PHONE_ENCRYPTION_KEY", settings.whatsapp_phone_encryption_key)
        self.fernet = Fernet(key.encode())

    @staticmethod
    def valid_signature(body: bytes, signature: str | None, secret: str) -> bool:
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return bool(signature and hmac.compare_digest(expected, signature))

    def thread_id(self, phone: str) -> str:
        key = self.settings.require_secret("WHATSAPP_THREAD_HMAC_KEY", self.settings.whatsapp_thread_hmac_key)
        return "wa_" + hmac.new(key.encode(), phone.encode(), hashlib.sha256).hexdigest()

    def encrypt_phone(self, phone: str) -> str:
        return self.fernet.encrypt(phone.encode()).decode()

    def decrypt_phone(self, encrypted_phone: str) -> str:
        return self.fernet.decrypt(encrypted_phone.encode()).decode()

    async def send(self, destination: str, kind: str, payload: dict[str, object]) -> str:
        access_token = self.settings.require_secret("META_ACCESS_TOKEN", self.settings.meta_access_token)
        phone_id = self.settings.meta_phone_number_id
        if not phone_id:
            raise ValueError("Missing required configuration: META_PHONE_NUMBER_ID")
        body: dict[str, Any] = {"messaging_product": "whatsapp", "to": destination}
        if kind == "text":
            body.update({"type": "text", "text": {"body": payload["text"]}})
        else:
            body.update({"type": "template", "template": {"name": payload["name"], "language": {"code": payload["language"]}, "components": [{"type": "body", "parameters": [{"type": "text", "text": item} for item in payload.get("parameters", [])]}]}})
        response = await self.client.post(f"https://graph.facebook.com/{self.settings.meta_graph_api_version}/{phone_id}/messages", headers={"Authorization": f"Bearer {access_token}"}, json=body)
        response.raise_for_status()
        data = response.json()
        return str(data["messages"][0]["id"])

    async def send_typing_indicator(self, message_id: str) -> None:
        """Mark an inbound message read and show typing, per Meta's typing-indicator API.

        Auto-dismissed once we send our real reply, or after 25 seconds,
        whichever comes first. Unlike send(), the response carries no
        message id of its own, so nothing is parsed from it.
        """

        access_token = self.settings.require_secret("META_ACCESS_TOKEN", self.settings.meta_access_token)
        phone_id = self.settings.meta_phone_number_id
        if not phone_id:
            raise ValueError("Missing required configuration: META_PHONE_NUMBER_ID")
        body = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
            "typing_indicator": {"type": "text"},
        }
        response = await self.client.post(
            f"https://graph.facebook.com/{self.settings.meta_graph_api_version}/{phone_id}/messages",
            headers={"Authorization": f"Bearer {access_token}"},
            json=body,
        )
        response.raise_for_status()

    async def close(self) -> None:
        await self.client.aclose()
