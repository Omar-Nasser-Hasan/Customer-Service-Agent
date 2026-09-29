"""Redacted LangSmith callback integration that cannot interrupt support flows."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from customer_service.config.settings import Settings
from customer_service.privacy.redaction import sanitize_for_trace, stable_thread_reference

LOGGER = logging.getLogger(__name__)


class ObservabilityRuntime:
    """Own optional tracing state and produce redacting callback handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None
        self._status = "disabled"
        self._last_error: str | None = None

    def start(self) -> None:
        if not self.settings.langsmith_tracing:
            return
        try:
            from langsmith import Client

            api_key = self.settings.require_secret("LANGSMITH_API_KEY", self.settings.langsmith_api_key)
            self._client = Client(api_key=api_key, api_url=self.settings.langsmith_endpoint)
            self._status = "healthy"
        except Exception as error:  # Observability is intentionally fail-open.
            self.mark_failure(error)

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def callbacks(self, thread_id: str) -> list[BaseCallbackHandler]:
        if self._client is None or not self._sampled(thread_id):
            return []
        return [_RedactingLangSmithHandler(self, thread_id)]
    
    def thread_reference(self, thread_id: str) -> str:
        """A stable, non-reversible id safe to place in ordinary application
        logs — derived the same way trace metadata is, so a log line and its
        matching LangSmith run can be lined up by eye without ever writing
        the real thread id anywhere outside the database."""

        return stable_thread_reference(thread_id, self._salt())

    def mark_failure(self, error: Exception) -> None:
        self._status = "degraded"
        self._last_error = type(error).__name__
        LOGGER.warning("observability_delivery_failed", extra={"error_type": self._last_error})

    def _sampled(self, thread_id: str) -> bool:
        if self.settings.langsmith_sample_rate <= 0:
            return False
        if self.settings.langsmith_sample_rate >= 1:
            return True
        salt = self._salt()
        bucket = int(hashlib.sha256(f"{salt}:{thread_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return bucket < self.settings.langsmith_sample_rate

    def _salt(self) -> str:
        secret = self.settings.trace_thread_hash_salt
        return secret.get_secret_value() if secret is not None else "local-development-salt"


class _RedactingLangSmithHandler(BaseCallbackHandler):
    """Send only sanitized callback payloads through LangSmith's batching client."""

    raise_error = False

    def __init__(self, runtime: ObservabilityRuntime, thread_id: str) -> None:
        self.runtime = runtime
        self.thread_reference = stable_thread_reference(thread_id, runtime._salt())

    def _start(
        self,
        run_type: str,
        serialized: dict[str, Any] | None,
        inputs: Any,
        run_id: UUID,
        parent_run_id: UUID | None,
    ) -> None:
        try:
            name = (serialized or {}).get("name") or (serialized or {}).get("id", [run_type])[-1]
            self.runtime._client.create_run(
                id=run_id,
                name=str(name),
                run_type=run_type,
                inputs=sanitize_for_trace(inputs, thread_id_salt=self.runtime._salt()),
                parent_run_id=parent_run_id,
                project_name=self.runtime.settings.langsmith_project,
                extra={
                    "metadata": {
                        "thread_reference": self.thread_reference,
                        "gemini_api_tier": self.runtime.settings.gemini_api_tier.value,
                    }
                },
                start_time=datetime.now(UTC),
            )
        except Exception as error:
            self.runtime.mark_failure(error)

    def _end(self, outputs: Any, run_id: UUID, error: BaseException | None = None) -> None:
        try:
            kwargs: dict[str, Any] = {"id": run_id, "end_time": datetime.now(UTC)}
            if error is not None:
                kwargs["error"] = type(error).__name__
            else:
                kwargs["outputs"] = sanitize_for_trace(outputs, thread_id_salt=self.runtime._salt())
            self.runtime._client.update_run(**kwargs)
        except Exception as callback_error:
            self.runtime.mark_failure(callback_error)

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **_: Any) -> None:
        self._start("chain", serialized, inputs, run_id, parent_run_id)

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **_: Any) -> None:
        self._end(outputs, run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        self._end({}, run_id, error)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, parent_run_id: UUID | None = None, **_: Any) -> None:
        self._start("llm", serialized, {"prompts": prompts}, run_id, parent_run_id)

    def on_llm_end(self, response: Any, *, run_id: UUID, **_: Any) -> None:
        self._end(response, run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        self._end({}, run_id, error)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: UUID | None = None, **_: Any) -> None:
        self._start("tool", serialized, {"input": input_str}, run_id, parent_run_id)

    def on_tool_end(self, output: Any, *, run_id: UUID, **_: Any) -> None:
        self._end({"output": output}, run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        self._end({}, run_id, error)
