"""Contracts for versioned synthetic evaluation cases and reports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    case_id: str
    thread_id: str
    message: str = Field(min_length=1)
    expected_status: str
    forbidden_fragments: list[str] = Field(default_factory=list)
    qualitative: bool = False

    model_config = ConfigDict(extra="forbid")


class ContractResult(BaseModel):
    case_id: str
    passed: bool
    detail: str


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1, max_length=500)

    model_config = ConfigDict(extra="forbid")


class EvaluationReport(BaseModel):
    tier: str
    contracts: list[ContractResult]
    judge_scores: list[JudgeVerdict] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        contracts_ok = all(result.passed for result in self.contracts)
        if not self.judge_scores:
            return contracts_ok
        mean = sum(verdict.score for verdict in self.judge_scores) / len(self.judge_scores)
        return contracts_ok and mean >= 4 and all(verdict.score >= 3 for verdict in self.judge_scores)
