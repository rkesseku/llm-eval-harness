"""Tests for EvalRunner."""

from __future__ import annotations

import pytest

from llm_eval_harness.evaluators.correctness import CorrectnessEvaluator
from llm_eval_harness.llm_client import FakeLLMClient
from llm_eval_harness.models import TestCase, Verdict
from llm_eval_harness.runner import EvalRunner, SystemUnderTestConfig


def _passing_grader() -> FakeLLMClient:
    return FakeLLMClient(response='{"score": 5, "reasoning": "ok"}', tokens=20)


class TestRunner:
    def test_rejects_empty_evaluators(self) -> None:
        sut = FakeLLMClient(response="x")
        with pytest.raises(ValueError):
            EvalRunner(system_client=sut, evaluators=[])

    def test_runs_all_combinations(self) -> None:
        sut = FakeLLMClient(response="Paris", tokens=10)
        evaluator = CorrectnessEvaluator(_passing_grader())
        runner = EvalRunner(system_client=sut, evaluators=[evaluator])

        test_cases = [
            TestCase(id="q1", input="?", expected_output="Paris"),
            TestCase(id="q2", input="?", expected_output="Paris"),
            TestCase(id="q3", input="?", expected_output="Paris"),
        ]
        report = runner.run(test_cases)

        assert report.total_test_cases == 3
        assert report.total_evaluators == 1
        assert len(report.results) == 3
        assert report.pass_rate == 1.0

    def test_multiple_evaluators_multiply_results(self) -> None:
        sut = FakeLLMClient(response="Paris", tokens=10)
        e1 = CorrectnessEvaluator(_passing_grader())
        e2 = CorrectnessEvaluator(_passing_grader())  # same kind, different instance
        runner = EvalRunner(system_client=sut, evaluators=[e1, e2])

        test_cases = [TestCase(id=f"q{i}", input="?", expected_output="x") for i in range(4)]
        report = runner.run(test_cases)

        # 4 test cases x 2 evaluators = 8 results
        assert len(report.results) == 8

    def test_system_under_test_receives_config(self) -> None:
        sut = FakeLLMClient(response="x", tokens=10)
        evaluator = CorrectnessEvaluator(_passing_grader())
        config = SystemUnderTestConfig(
            system_prompt="custom prompt", temperature=0.7, max_tokens=99
        )
        runner = EvalRunner(system_client=sut, evaluators=[evaluator], system_config=config)

        runner.run([TestCase(id="t1", input="hi", expected_output="x")])

        # FakeLLMClient records every call -- assert config was passed through
        assert len(sut.calls) == 1
        assert sut.calls[0]["system"] == "custom prompt"
        assert sut.calls[0]["temperature"] == 0.7
        assert sut.calls[0]["max_tokens"] == 99

    def test_continues_on_evaluator_error(self) -> None:
        sut = FakeLLMClient(response="x", tokens=10)
        # Test case with no expected_output -> CorrectnessEvaluator raises -> ERROR verdict
        evaluator = CorrectnessEvaluator(_passing_grader())
        runner = EvalRunner(system_client=sut, evaluators=[evaluator])

        test_cases = [
            TestCase(id="bad", input="?", expected_output=None),  # will ERROR
            TestCase(id="good", input="?", expected_output="x"),  # will PASS
        ]
        report = runner.run(test_cases)

        assert len(report.results) == 2
        verdicts = {r.test_case_id: r.verdict for r in report.results}
        assert verdicts["bad"] is Verdict.ERROR
        assert verdicts["good"] is Verdict.PASS