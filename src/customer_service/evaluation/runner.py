"""Run contract checks locally and optional qualitative checks against Gemini."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from importlib.resources import files
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from customer_service.config.settings import Settings
from customer_service.evaluation.models import (
    ContractResult,
    EvaluationCase,
    EvaluationReport,
    JudgeVerdict,
)
from customer_service.evaluation.prompt import build_judge_prompt

SendMessage = Callable[[str, str], Awaitable[dict[str, Any]]]


def load_cases() -> list[EvaluationCase]:
    payload = files("customer_service.data").joinpath("evaluations.json").read_text(encoding="utf-8")
    return [EvaluationCase.model_validate(item) for item in json.loads(payload)]


async def run_contracts(send_message: SendMessage, *, tier: str) -> tuple[EvaluationReport, list[tuple[EvaluationCase, str]]]:
    results: list[ContractResult] = []
    qualitative: list[tuple[EvaluationCase, str]] = []
    for case in load_cases():
        response = await send_message(case.thread_id, case.message)
        status = response.get("status")
        reply = response.get("reply") or ""
        failures: list[str] = []
        if status != case.expected_status:
            failures.append(f"expected status {case.expected_status}, got {status}")
        for forbidden in case.forbidden_fragments:
            if forbidden.lower() in reply.lower():
                failures.append(f"reply leaked forbidden fragment {forbidden!r}")
        results.append(
            ContractResult(case_id=case.case_id, passed=not failures, detail="; ".join(failures) or "ok")
        )
        if case.qualitative and not failures:
            qualitative.append((case, reply))
    return EvaluationReport(tier=tier, contracts=results), qualitative


async def score_completed_answers(
    report: EvaluationReport,
    candidates: list[tuple[EvaluationCase, str]],
    *,
    settings: Settings,
    model: BaseChatModel,
) -> EvaluationReport:
    """Apply the strict structured evaluator only to completed customer answers."""

    judge = model.with_structured_output(JudgeVerdict, method="json_schema")
    verdicts: list[JudgeVerdict] = []
    for case, reply in candidates:
        verdict = await judge.ainvoke(
            build_judge_prompt(settings, customer_message=case.message, reply=reply)
        )
        verdicts.append(JudgeVerdict.model_validate(verdict))
    return report.model_copy(update={"judge_scores": verdicts})
