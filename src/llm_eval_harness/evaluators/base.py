"""
evaluators/base.py

Abstract base class for all evaluators (a.k.a. judges).

Subclasses implement `_evaluate(...)` which returns a (score, verdict, reasoning,
tokens) tuple. The base class handles timing, error handling, and packaging
into a uniform `EvalResult`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_eval_harness.models import EvalResult, TestCase, Verdict


@dataclass(frozen=True, slots=True)
class JudgeOutput:
    """Raw judging output before being packaged into an EvalResult."""

    score: float
    verdict: Verdict
    reasoning: str
    tokens_used: int


class Evaluator(ABC):
    """Abstract base class for all evaluators.

    Each concrete evaluator scores a single (test_case, actual_output) pair on
    one dimension (correctness, groundedness, safety, etc.).

    Subclasses must:
      - set `name` (used to identify results in reports)
      - implement `_evaluate(...)`
    """

    name: str = "unnamed_evaluator"

    @abstractmethod
    def _evaluate(self, test_case: TestCase, actual_output: str) -> JudgeOutput:
        """Concrete judging logic. Implemented by subclasses."""
        ...

    def evaluate(self, test_case: TestCase, actual_output: str) -> EvalResult:
        """Public entry point. Handles timing, errors, and result packaging."""
        start = time.perf_counter()

        try:
            judgment = self._evaluate(test_case, actual_output)
            error: str | None = None
        except Exception as exc:  # broad on purpose: one judge crash != run crash
            judgment = JudgeOutput(
                score=0.0,
                verdict=Verdict.ERROR,
                reasoning=f"Evaluator raised: {exc!r}",
                tokens_used=0,
            )
            error = str(exc)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return EvalResult(
            test_case_id=test_case.id,
            evaluator_name=self.name,
            actual_output=actual_output,
            score=judgment.score,
            verdict=judgment.verdict,
            reasoning=judgment.reasoning,
            latency_ms=elapsed_ms,
            tokens_used=judgment.tokens_used,
            error=error,
        )