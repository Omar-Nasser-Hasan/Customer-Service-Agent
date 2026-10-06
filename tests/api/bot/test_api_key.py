from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from customer_service.api.app import create_app
from customer_service.api.bot.conversations import create_router
from customer_service.config.settings import Settings

URL = "/conversations/thread-1/messages"
BODY = {"message": "hello"}


class StubGraph:
    async def aget_state(self, config: object) -> SimpleNamespace:
        return SimpleNamespace(values={})

    async def ainvoke(self, payload: object, config: object = None) -> dict[str, object]:
        return {"messages": [AIMessage(content="hi")]}


def client(api_key: str | None) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(lambda: StubGraph(), api_key=api_key))
    return TestClient(app)


def test_missing_or_wrong_key_is_rejected() -> None:
    protected = client("s3cret")

    assert protected.post(URL, json=BODY).status_code == 401
    assert protected.post(URL, json=BODY, headers={"X-API-Key": "nope"}).status_code == 401
    assert protected.post(URL, json=BODY, headers={"X-API-Key": "s3cret-but-longer"}).status_code == 401


def test_a_non_ascii_key_is_rejected_cleanly_not_with_a_server_error() -> None:
    response = client("s3cret").post(URL, json=BODY, headers={"X-API-Key": "مفتاح".encode("utf-8")})

    assert response.status_code == 401


def test_the_correct_key_reaches_the_graph() -> None:
    response = client("s3cret").post(URL, json=BODY, headers={"X-API-Key": "s3cret"})

    assert response.status_code == 200
    assert response.json()["reply"] == "hi"


def test_no_configured_key_keeps_local_development_open() -> None:
    assert client(None).post(URL, json=BODY).status_code == 200


def test_production_refuses_to_start_without_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "customer_service.api.app.get_settings",
        lambda: Settings(app_environment="production", bot_api_key=None),
    )

    with pytest.raises(ValueError, match="BOT_API_KEY"):
        create_app()
