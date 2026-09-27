from datetime import UTC, datetime, timedelta

import pytest

from customer_service.config.settings import Settings, WhatsAppTemplate
from customer_service.operations.models import CaseRecord, StaffIdentity
from customer_service.services.cases import CaseService, CustomerWindowClosedError, TemplateValidationError


def service():
    settings = Settings(whatsapp_template_catalog=[WhatsAppTemplate(template_id="outside", name="outside_window", language="en", parameter_count=1)])
    value = CaseService(None, None, None, settings)
    return value


def test_template_requires_deployment_allowlist_and_exact_parameters():
    value = service()
    assert value._template("outside", ["A"]).name == "outside_window"
    with pytest.raises(TemplateValidationError): value._template("unknown", [])
    with pytest.raises(TemplateValidationError): value._template("outside", [])
