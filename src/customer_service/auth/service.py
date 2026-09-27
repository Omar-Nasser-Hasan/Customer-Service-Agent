"""Session, CSRF, and Google ID-token validation at the API trust boundary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import secrets
from typing import Any
from uuid import uuid4

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from customer_service.config.settings import Settings
from customer_service.operations.models import StaffIdentity
from customer_service.operations.repository import OperationsRepository


class AuthenticationError(ValueError):
    pass


class CsrfError(AuthenticationError):
    pass


class StaffAuthService:
    def __init__(self, settings: Settings, repository: OperationsRepository) -> None:
        self.settings = settings
        self.repository = repository
        secret = settings.require_secret("STAFF_SESSION_SECRET", settings.staff_session_secret)
        self.serializer = URLSafeTimedSerializer(secret, salt="customer-service-staff-v1")

    @property
    def allowed_emails(self) -> set[str]:
        return {email.lower().strip() for email in self.settings.admin_allowed_emails}

    def _validate_allowlist(self, identity: StaffIdentity) -> StaffIdentity:
        if not identity.email or identity.email.lower() not in self.allowed_emails:
            raise AuthenticationError("This Google account is not authorized for staff access")
        return identity

    def verify_google_credential(self, credential: str) -> StaffIdentity:
        audience = self.settings.google_oauth_client_id
        if not audience:
            raise AuthenticationError("Google staff login is not configured")
        try:
            claims = id_token.verify_oauth2_token(credential, google_requests.Request(), audience=audience)
        except Exception as error:
            raise AuthenticationError("Google ID token could not be verified") from error
        issuer = claims.get("iss")
        if issuer not in {"accounts.google.com", "https://accounts.google.com"}:
            raise AuthenticationError("Google ID token issuer is invalid")
        if not claims.get("email_verified") or not claims.get("sub") or not claims.get("email"):
            raise AuthenticationError("Google ID token lacks a verified staff identity")
        return self._validate_allowlist(StaffIdentity(sub=str(claims["sub"]), email=str(claims["email"]).lower()))

    def mint_session(self, identity: StaffIdentity) -> str:
        return self.serializer.dumps({"sub": identity.sub, "email": identity.email})

    def read_session(self, token: str | None) -> StaffIdentity:
        if not token:
            raise AuthenticationError("Staff session is required")
        try:
            payload = self.serializer.loads(token, max_age=self.settings.staff_session_ttl_seconds)
        except (BadSignature, SignatureExpired) as error:
            raise AuthenticationError("Staff session is invalid or expired") from error
        return self._validate_allowlist(StaffIdentity(sub=str(payload["sub"]), email=str(payload["email"]).lower()))

    def mint_csrf(self) -> str:
        return secrets.token_urlsafe(32)

    def validate_csrf(self, cookie: str | None, header: str | None) -> None:
        if not cookie or not header or not secrets.compare_digest(cookie, header):
            raise CsrfError("A valid CSRF token is required")

    def validate_origin(self, origin: str | None) -> None:
        if origin != self.settings.frontend_origin:
            raise CsrfError("State-changing staff requests require the configured frontend origin")

    async def mint_websocket_token(self, identity: StaffIdentity) -> str:
        token_id = str(uuid4())
        expires = datetime.now(UTC) + timedelta(seconds=self.settings.websocket_token_ttl_seconds)
        await self.repository.issue_ws_token(token_id, identity.sub, identity.email, expires)
        return self.serializer.dumps({"token_id": token_id, "sub": identity.sub, "purpose": "websocket"})

    async def consume_websocket_token(self, token: str | None) -> StaffIdentity:
        if not token:
            raise AuthenticationError("WebSocket token is required")
        try:
            payload = self.serializer.loads(token, max_age=self.settings.websocket_token_ttl_seconds)
        except (BadSignature, SignatureExpired) as error:
            raise AuthenticationError("WebSocket token is invalid or expired") from error
        if payload.get("purpose") != "websocket":
            raise AuthenticationError("WebSocket token purpose is invalid")
        # The session is intentionally not transported in the URL. The stored
        # one-time token binds the signed token to a single staff subject.
        email = await self.repository.consume_ws_token(str(payload["token_id"]), str(payload["sub"]))
        if not email:
            raise AuthenticationError("WebSocket token is already used or expired")
        return self._validate_allowlist(StaffIdentity(sub=str(payload["sub"]), email=email))
