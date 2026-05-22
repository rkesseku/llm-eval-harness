"""Tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_eval_harness.models import EvalReport, EvalResult, TestCase, Verdict


class TestTestCase:
    def test_minimal_construction(self) -> None:
        tc = TestCase(id="t1", input="hello")
        assert tc.id == "t1"
        assert tc.input == "hello"
        assert tc.expected_output is None
        assert tc.tags == []
        assert tc.metadata == {}

    def test_full_construction(self) -> None:
        tc = TestCase(
            id="t1",
            input="Q?",
            expected_output="A",
            context="some context",
            tags=["safety", "edge-case"],
            metadata={"source": "team-survey"},
        )
        assert tc.tags == ["safety", "edge-case"]
        assert tc.metadata["source"] == "team-survey"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TestCase(id="t1", input="hi", typo_field="oops")  # type: ignore[call-arg]

    def test_immutable(self) -> None:
        tc = TestCase(id="t1", input="hi")
        with pytest.raises(ValidationError):
            tc.id = "t2"  # type: ignore[misc]


class TestEvalResult:
    def test_score_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            EvalResult(
                test_case_id="t1",
                evaluator_name="x",
                actual_output="",
                score=-0.1,
                verdict=Verdict.FAIL,
                latency_ms=1.0,
                tokens_used=10,
            )

    def test_score_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            EvalResult(
                test_case_id="t1",
                evaluator_name="x",
                actual_output="",
                score=1.5,
                verdict=Verdict.PASS,
                latency_ms=1.0,
                tokens_used=10,
            )

    def test_valid(self) -> None:
        r = EvalResult(
            test_case_id="t1",
            evaluator_name="correctness",
            actual_output="Paris",
            score=1.0,
            verdict=Verdict.PASS,
            latency_ms=42.0,
            tokens_used=100,
        )
        assert r.score == 1.0
        assert r.verdict is Verdict.PASS


class TestEvalReport:
    def _make_result(self, *, verdict: Verdict, score: float, tokens: int) -> EvalResult:
        return EvalResult(
            test_case_id="t1",
            evaluator_name="correctness",
            actual_output="x",
            score=score,
            verdict=verdict,
            latency_ms=10.0,
            tokens_used=tokens,
        )

    def test_empty_report(self) -> None:
        report = EvalReport(results=[], total_test_cases=0, total_evaluators=0)
        assert report.pass_rate == 0.0
        assert report.total_tokens == 0
        assert report.avg_latency_ms == 0.0

    def test_pass_rate(self) -> None:
        results = [
            self._make_result(verdict=Verdict.PASS, score=1.0, tokens=10),
            self._make_result(verdict=Verdict.PASS, score=1.0, tokens=20),
            self._make_result(verdict=Verdict.FAIL, score=0.2, tokens=15),
        ]
        report = EvalReport(results=results, total_test_cases=3, total_evaluators=1)
        assert report.pass_rate == pytest.approx(2 / 3)
        assert report.total_tokens == 45
        assert report.avg_latency_ms == pytest.approx(10.0)

    def test_by_evaluator_groups_correctly(self) -> None:
        r1 = self._make_result(verdict=Verdict.PASS, score=1.0, tokens=10)
        r2 = EvalResult(
            test_case_id="t2",
            evaluator_name="groundedness",  # different evaluator
            actual_output="x",
            score=0.5,
            verdict=Verdict.FAIL,
            latency_ms=10.0,
            tokens_used=20,
        )
        report = EvalReport(results=[r1, r2], total_test_cases=2, total_evaluators=2)
        groups = report.by_evaluator()
        assert set(groups.keys()) == {"correctness", "groundedness"}
        assert len(groups["correctness"]) == 1
        assert len(groups["groundedness"]) == 1