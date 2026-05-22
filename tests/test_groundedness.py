"""Tests for GroundednessEvaluator."""

from __future__ import annotations

import pytest

from llm_eval_harness.evaluators.groundedness import GroundednessEvaluator
from llm_eval_harness.llm_client import FakeLLMClient
from llm_eval_harness.models import TestCase, Verdict


def _make_test_case() -> TestCase:
    return TestCase(
        id="rag1",
        input="When was the Eiffel Tower built?",
        context="The Eiffel Tower was completed in 1889.",
    )


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
        evaluator = GroundednessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="It was built in 1889.")
        assert result.score == pytest.approx(expected_normalized)


class TestVerdictThreshold:
    def test_fail_at_default_threshold(self) -> None:
        grader = FakeLLMClient(response='{"score": 4, "reasoning": "minor leak"}', tokens=10)
        evaluator = GroundednessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="Built in 1889.")
        # 4/5 = 0.75, default threshold 0.8 -> FAIL
        assert result.verdict is Verdict.FAIL

    def test_pass_with_lower_threshold(self) -> None:
        grader = FakeLLMClient(response='{"score": 4, "reasoning": "minor leak"}', tokens=10)
        evaluator = GroundednessEvaluator(grader, pass_threshold=0.7)
        result = evaluator.evaluate(_make_test_case(), actual_output="Built in 1889.")
        assert result.verdict is Verdict.PASS

    def test_rejects_invalid_threshold(self) -> None:
        grader = FakeLLMClient(response="{}", tokens=0)
        with pytest.raises(ValueError):
            GroundednessEvaluator(grader, pass_threshold=-0.1)


class TestErrorHandling:
    def test_missing_context_yields_error_verdict(self) -> None:
        grader = FakeLLMClient(response='{"score": 5, "reasoning": "ok"}', tokens=10)
        evaluator = GroundednessEvaluator(grader)
        tc = TestCase(id="no_ctx", input="?", context=None)
        result = evaluator.evaluate(tc, actual_output="something")
        assert result.verdict is Verdict.ERROR
        assert result.error is not None
        assert "context" in result.error.lower()

    def test_empty_context_yields_error_verdict(self) -> None:
        grader = FakeLLMClient(response='{"score": 5, "reasoning": "ok"}', tokens=10)
        evaluator = GroundednessEvaluator(grader)
        tc = TestCase(id="empty_ctx", input="?", context="   ")  # whitespace only
        result = evaluator.evaluate(tc, actual_output="something")
        assert result.verdict is Verdict.ERROR

    def test_malformed_grader_output_yields_error_verdict(self) -> None:
        grader = FakeLLMClient(response="not json at all", tokens=10)
        evaluator = GroundednessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="x")
        assert result.verdict is Verdict.ERROR

    def test_out_of_range_grader_score_yields_error_verdict(self) -> None:
        grader = FakeLLMClient(response='{"score": 0, "reasoning": "x"}', tokens=10)
        evaluator = GroundednessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="x")
        assert result.verdict is Verdict.ERROR


class TestMarkdownFenceTolerance:
    def test_strips_json_code_fence(self) -> None:
        grader = FakeLLMClient(
            response='```json\n{"score": 5, "reasoning": "ok"}\n```', tokens=10
        )
        evaluator = GroundednessEvaluator(grader)
        result = evaluator.evaluate(_make_test_case(), actual_output="x")
        assert result.verdict is Verdict.PASS
        assert result.score == 1.0