from __future__ import annotations

from uuid import uuid4

from customer_service.config.settings import Settings
from customer_service.observability.runtime import ObservabilityRuntime, _RedactingLangSmithHandler


class RecordingClient:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.updated: list[tuple[object, dict[str, object]]] = []

    def create_run(self, **kwargs: object) -> None:
        self.created.append(kwargs)

    def update_run(self, run_id: object, **kwargs: object) -> None:
        # Mirrors langsmith.Client.update_run, where run_id is required.
        self.updated.append((run_id, kwargs))


def test_trace_handler_finishes_runs_by_run_id() -> None:
    runtime = ObservabilityRuntime(Settings(langsmith_tracing=True, trace_thread_hash_salt="test-salt"))
    client = RecordingClient()
    runtime._client = client
    runtime._status = "healthy"
    handler = _RedactingLangSmithHandler(runtime, "visible-thread")
    run_id = uuid4()

    handler.on_chain_end({"message": "email alice@example.com"}, run_id=run_id)

    assert runtime.status == "healthy"
    assert client.updated[0][0] == run_id
    assert client.updated[0][1]["outputs"] == {"message": "email [EMAIL]"}


class FailingClient:
    def create_run(self, **_: object) -> None:
        raise RuntimeError("telemetry unavailable")


def test_trace_handler_sends_only_redacted_hashed_data() -> None:
    runtime = ObservabilityRuntime(Settings(langsmith_tracing=True, trace_thread_hash_salt="test-salt"))
    client = RecordingClient()
    runtime._client = client
    runtime._status = "healthy"
    handler = _RedactingLangSmithHandler(runtime, "visible-thread")

    handler.on_chain_start(
        {"name": "graph"},
        {"message": "email alice@example.com order ORD-1001", "thread_id": "visible-thread"},
        run_id=uuid4(),
    )

    sent = client.created[0]
    assert sent["inputs"] == {
        "message": "email [EMAIL] order [ORDER_ID]",
        "thread_id": handler.thread_reference,
    }
    assert sent["extra"]["metadata"]["thread_reference"] == handler.thread_reference


def test_trace_delivery_failure_degrades_without_raising() -> None:
    runtime = ObservabilityRuntime(Settings(langsmith_tracing=True))
    runtime._client = FailingClient()
    runtime._status = "healthy"
    handler = _RedactingLangSmithHandler(runtime, "thread")

    handler.on_chain_start({"name": "graph"}, {"message": "hello"}, run_id=uuid4())

    assert runtime.status == "degraded"
    assert runtime.last_error == "RuntimeError"
