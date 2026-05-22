"""
runner.py

Orchestrates running a set of evaluators against a set of test cases.

Workflow:
  1. For each test case, invoke the system-under-test to produce an actual output.
  2. For each evaluator, score the (test_case, actual_output) pair.
  3. Aggregate every EvalResult into a single EvalReport.

The system-under-test is just an LLMClient -- which means in tests you can
substitute a FakeLLMClient and avoid all network traffic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from llm_eval_harness.evaluators.base import Evaluator
from llm_eval_harness.llm_client import LLMClient
from llm_eval_harness.models import EvalReport, EvalResult, TestCase


@dataclass(frozen=True, slots=True)
class SystemUnderTestConfig:
    """Configuration for the LLM being evaluated (NOT the grader)."""

    system_prompt: str = "You are a helpful, concise assistant."
    temperature: float = 0.2
    max_tokens: int = 512


class EvalRunner:
    """Runs a suite of evaluators over a collection of test cases."""

    def __init__(
        self,
        *,
        system_client: LLMClient,
        evaluators: Sequence[Evaluator],
        system_config: SystemUnderTestConfig | None = None,
    ) -> None:
        if not evaluators:
            raise ValueError("At least one evaluator is required.")
        self._system_client = system_client
        self._evaluators = list(evaluators)
        self._config = system_config or SystemUnderTestConfig()

    def run(self, test_cases: Iterable[TestCase]) -> EvalReport:
        """Run all evaluators over all test cases and return an aggregated report."""
        test_cases = list(test_cases)
        results: list[EvalResult] = []

        for test_case in test_cases:
            actual_output = self._invoke_system(test_case)
            for evaluator in self._evaluators:
                result = evaluator.evaluate(test_case, actual_output)
                results.append(result)

        return EvalReport(
            results=results,
            total_test_cases=len(test_cases),
            total_evaluators=len(self._evaluators),
        )

    def _invoke_system(self, test_case: TestCase) -> str:
        """Call the system-under-test on one test case and return its response text."""
        completion = self._system_client.complete(
            system=self._config.system_prompt,
            user=test_case.input,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return completion.text