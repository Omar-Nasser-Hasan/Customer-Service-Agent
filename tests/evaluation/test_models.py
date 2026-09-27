import pytest
from pydantic import ValidationError

from customer_service.evaluation.models import ContractResult, EvaluationReport, JudgeVerdict


def test_evaluation_gate_requires_contracts_and_quality_thresholds() -> None:
    report = EvaluationReport(
        tier="free",
        contracts=[ContractResult(case_id="ok", passed=True, detail="ok")],
        judge_scores=[JudgeVerdict(score=4, rationale="helpful"), JudgeVerdict(score=4, rationale="safe")],
    )
    assert report.passed

    assert not report.model_copy(update={"judge_scores": [JudgeVerdict(score=2, rationale="bad")]}).passed


def test_malformed_judge_output_is_rejected() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate({"score": 6, "rationale": "invalid"})
