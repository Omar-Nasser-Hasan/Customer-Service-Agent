from __future__ import annotations

import pytest

from customer_service.evaluation.runner import load_cases, run_contracts


def test_fixture_corpus_is_versioned_and_contains_required_paths() -> None:
    case_ids = {case.case_id for case in load_cases()}
    assert {"public-faq", "verified-status", "identity-mismatch", "human-handoff", "injection-refusal"} <= case_ids


@pytest.mark.asyncio
async def test_contract_runner_detects_status_and_leakage_failures() -> None:
    async def send(thread_id: str, message: str) -> dict[str, object]:
        if thread_id == "eval-mismatch":
            return {"status": "completed", "reply": "alice@example.com"}
        return {"status": "completed", "reply": "safe reply"}

    report, _ = await run_contracts(send, tier="free")

    mismatch = next(result for result in report.contracts if result.case_id == "identity-mismatch")
    handoff = next(result for result in report.contracts if result.case_id == "human-handoff")
    assert not mismatch.passed
    assert not handoff.passed
