from datetime import UTC, datetime, timedelta

import pytest

from customer_service.auth.service import AuthenticationError, CsrfError, StaffAuthService
from customer_service.config.settings import Settings
from customer_service.operations.models import StaffIdentity


class Tokens:
    def __init__(self): self.issued = None
    async def issue_ws_token(self, *args): self.issued = args
    async def consume_ws_token(self, token_id, sub): return "agent@example.com"


def auth():
    return StaffAuthService(Settings(staff_session_secret="secret", admin_allowed_emails=["agent@example.com"]), Tokens())


def test_session_rechecks_allowlist():
    service = auth(); token = service.mint_session(StaffIdentity(sub="google-sub", email="agent@example.com"))
    assert service.read_session(token).sub == "google-sub"
    service.settings.admin_allowed_emails.clear()
    with pytest.raises(AuthenticationError): service.read_session(token)


def test_csrf_and_origin_fail_closed():
    service = auth()
    service.validate_csrf("same", "same")
    with pytest.raises(CsrfError): service.validate_csrf("a", "b")
    with pytest.raises(CsrfError): service.validate_origin("https://evil.example")
