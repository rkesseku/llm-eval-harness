"""
models.py

Pydantic data models for evaluation inputs and outputs.

A TestCase represents one thing to evaluate (input, expected output, metadata).
An EvalResult represents the outcome of running one evaluator on one test case.
An EvalReport aggregates results across all evaluators and test cases.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    """Coarse pass/fail verdict for an evaluation."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"  # the evaluator itself crashed


class TestCase(BaseModel):
    """One evaluation example: an input to feed the system, an expected output, and metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., description="Stable unique identifier for this test case.")
    input: str = Field(..., description="The prompt or query to evaluate.")
    expected_output: str | None = Field(
        default=None,
        description="Ground-truth answer for correctness judging. Optional.",
    )
    context: str | None = Field(
        default=None,
        description="Retrieved context for groundedness checks (RAG scenarios).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary tags for filtering or grouping (e.g. ['safety', 'edge-case']).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra application-specific fields.",
    )


class EvalResult(BaseModel):
    """The outcome of one evaluator running on one test case."""

    model_config = ConfigDict(extra="forbid")

    test_case_id: str
    evaluator_name: str
    actual_output: str = Field(..., description="What the system-under-test produced.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized score in [0, 1]. 1.0 = perfect."
    )
    verdict: Verdict
    reasoning: str = Field(
        default="",
        description="Human-readable explanation (esp. for LLM-as-judge evaluators).",
    )
    latency_ms: float = Field(..., ge=0.0)
    tokens_used: int = Field(..., ge=0)
    error: str | None = Field(
        default=None, description="If verdict is ERROR, the exception message."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalReport(BaseModel):
    """Aggregated results across all evaluators and test cases."""

    model_config = ConfigDict(extra="forbid")

    results: list[EvalResult]
    total_test_cases: int
    total_evaluators: int

    @property
    def pass_rate(self) -> float:
        """Fraction of results with PASS verdict. Returns 0.0 if no results."""
        if not self.results:
            return 0.0
        passes = sum(1 for r in self.results if r.verdict == Verdict.PASS)
        return passes / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def by_evaluator(self) -> dict[str, list[EvalResult]]:
        """Group results by evaluator name."""
        grouped: dict[str, list[EvalResult]] = {}
        for r in self.results:
            grouped.setdefault(r.evaluator_name, []).append(r)
        return grouped