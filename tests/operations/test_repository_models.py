from uuid import uuid4

from customer_service.operations.repository import _model_values


def test_model_values_converts_postgres_uuid_columns_to_public_string_ids():
    identifier = uuid4()

    result = _model_values({"outbox_id": identifier, "payload": {"text": "hello"}})

    assert result["outbox_id"] == str(identifier)
    assert result["payload"] == {"text": "hello"}
