from __future__ import annotations

from fastapi.testclient import TestClient

from customer_service.api.app import create_app


def test_health_check_requires_no_model_configuration() -> None:
    client = TestClient(create_app(lambda: None))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "observability": "disabled"}


def test_partial_node_model_environment_mapping_keeps_evaluation_default() -> None:
    from customer_service.config.settings import Settings

    settings = Settings(node_models={"assistant": {"provider": "google_genai", "model": "custom"}})

    assert settings.model_for("eval_judge").model == "gemini-3.1-flash-lite"
