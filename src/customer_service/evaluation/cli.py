"""Run the synthetic contracts against a running API and write a JSON report."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx

from customer_service.config.settings import get_settings
from customer_service.evaluation.runner import run_contracts, score_completed_answers
from customer_service.llm.factory import create_chat_model


async def _run() -> int:
    base_url = os.environ.get("EVALUATION_BASE_URL", "http://127.0.0.1:8000")

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        run_namespace = uuid4().hex[:12]

        async def send(thread_id: str, message: str) -> dict[str, object]:
            isolated_thread_id = f"eval-{run_namespace}-{thread_id}"
            response = await client.post(
                f"/conversations/{isolated_thread_id}/messages", json={"message": message}
            )
            response.raise_for_status()
            return response.json()

        settings = get_settings()
        report, candidates = await run_contracts(send, tier=settings.gemini_api_tier.value)
        report = await score_completed_answers(
            report,
            candidates,
            settings=settings,
            model=create_chat_model("eval_judge", settings),
        )
    path = Path(os.environ.get("EVALUATION_REPORT", "artifacts/live-evaluation.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(path)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
