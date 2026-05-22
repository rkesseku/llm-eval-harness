"""Tests for CorrectnessEvaluator."""

from __future__ import annotations

import pytest

from llm_eval_harness.evaluators.correctness import CorrectnessEvaluator
from llm_eval_harness.llm_client import FakeLLMClient
from llm_eval_harness.models import TestCase, Verdict


def _make_test_case() -> TestCase:
    return TestCase(id="t1", input="Capital of France?", expected_output="Paris")


class TestScoreNormalization:
    @pytest.mark.parametrize(
        "raw_score,expected_normalized",
        [
            (1, 0.0),
            (2, 0.25),
            (3, 0.5),
            (4, 0.75),
            (5, 1.0),
        ],
    )
    def test_normalization(self, raw_score: int, expected_normalized: float) -> None:
        grader = FakeLLMClient(
            response=f'{{"score": {raw_score}, "reasoning": "x"}}', tokens=10
        )
        evaluator = CorrectnessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="Paris.")
        assert result.score == pytest.approx(expected_normalized)


class TestVerdictThreshold:
    def test_pass_at_default_threshold(self) -> None:
        grader = FakeLLMClient(response='{"score": 4, "reasoning": "good"}', tokens=10)
        evaluator = CorrectnessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="Paris")
        # 4/5 = 0.75, default threshold 0.8 -> FAIL
        assert result.verdict is Verdict.FAIL

    def test_pass_with_lower_threshold(self) -> None:
        grader = FakeLLMClient(response='{"score": 4, "reasoning": "good"}', tokens=10)
        evaluator = CorrectnessEvaluator(grader, pass_threshold=0.7)
        result = evaluator.evaluate(_make_test_case(), actual_output="Paris")
        assert result.verdict is Verdict.PASS

    def test_rejects_invalid_threshold(self) -> None:
        grader = FakeLLMClient(response="{}", tokens=0)
        with pytest.raises(ValueError):
            CorrectnessEvaluator(grader, pass_threshold=1.5)


class TestErrorHandling:
    def test_missing_expected_output_yields_error_verdict(self) -> None:
        grader = FakeLLMClient(response='{"score": 5, "reasoning": "x"}', tokens=10)
        evaluator = CorrectnessEvaluator(grader)
        tc = TestCase(id="t1", input="?", expected_output=None)  # no ground truth
        result = evaluator.evaluate(tc, actual_output="something")
        assert result.verdict is Verdict.ERROR
        assert result.error is not None
        assert "expected_output" in result.error

    def test_malformed_grader_output_yields_error_verdict(self) -> None:
        grader = FakeLLMClient(response="this is not json", tokens=10)
        evaluator = CorrectnessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="Paris")
        assert result.verdict is Verdict.ERROR

    def test_out_of_range_grader_score_yields_error_verdict(self) -> None:
        grader = FakeLLMClient(response='{"score": 99, "reasoning": "x"}', tokens=10)
        evaluator = CorrectnessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="Paris")
        assert result.verdict is Verdict.ERROR


class TestMarkdownFenceTolerance:
    def test_strips_json_code_fence(self) -> None:
        # Some models wrap JSON in markdown fences despite instructions; we tolerate it.
        grader = FakeLLMClient(
            response='```json\n{"score": 5, "reasoning": "ok"}\n```', tokens=10
        )
        evaluator = CorrectnessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="Paris")
        assert result.verdict is Verdict.PASS
        assert result.score == 1.0