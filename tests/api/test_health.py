from __future__ import annotations

from fastapi.testclient import TestClient

from customer_service.api.app import create_app


def test_health_check_requires_no_model_configuration() -> None:
    client = TestClient(create_app(lambda: None))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
