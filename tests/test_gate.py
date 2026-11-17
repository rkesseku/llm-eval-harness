"""Tests for latency/cost metrics, pricing, thresholds, baseline comparison, and reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_harness.evaluators.correctness import CorrectnessEvaluator
from llm_eval_harness.evaluators.safety import SafetyEvaluator
from llm_eval_harness.gate import (
    RegressionTolerance,
    Thresholds,
    compare_to_baseline,
    evaluate_gate,
    load_thresholds,
)
from llm_eval_harness.llm_client import FakeLLMClient, ScriptedLLMClient
from llm_eval_harness.metrics import (
    Baseline,
    RunSummary,
    load_baseline,
    percentile,
    save_baseline,
    summarize,
)
from llm_eval_harness.models import TestCase
from llm_eval_harness.pricing import ModelPrice, PricingTable, load_pricing
from llm_eval_harness.report import render_markdown
from llm_eval_harness.runner import EvalRunner

PRICING = PricingTable(
    models={"scripted": ModelPrice(prompt_per_million=1.0, completion_per_million=2.0)}
)


def _summary(**kw: object) -> RunSummary:
    base: dict[str, object] = {
        "n_cases": 4,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 200.0,
        "mean_cost_per_case": 0.001,
        "total_cost": 0.004,
        "total_prompt_tokens": 400,
        "total_completion_tokens": 40,
        "pass_rates": {"safety": 1.0, "correctness": 0.75},
    }
    base.update(kw)
    return RunSummary(**base)  # type: ignore[arg-type]


class TestPercentile:
    def test_basic(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert percentile(vals, 50) == 30.0
        assert percentile(vals, 0) == 10.0
        assert percentile(vals, 100) == 50.0
        assert percentile(vals, 95) == pytest.approx(48.0)

    def test_interpolates_and_unsorted(self) -> None:
        assert percentile([40.0, 10.0, 30.0, 20.0], 50) == 25.0

    def test_edge_cases(self) -> None:
        assert percentile([], 50) == 0.0
        assert percentile([7.0], 95) == 7.0
        with pytest.raises(ValueError):
            percentile([1.0], 101)


class TestPricing:
    def test_cost(self) -> None:
        # 1M prompt tokens @ $1 + 500k completion @ $2 = $2
        assert PRICING.cost("scripted", 1_000_000, 500_000) == pytest.approx(2.0)

    def test_unknown_model_raises_without_default(self) -> None:
        with pytest.raises(KeyError):
            PRICING.cost("mystery", 1, 1)

    def test_default_price(self) -> None:
        table = PricingTable(default=ModelPrice(prompt_per_million=1, completion_per_million=1))
        assert table.cost("anything", 1_000_000, 0) == pytest.approx(1.0)

    def test_load_yaml_and_json(self, tmp_path: Path) -> None:
        y = tmp_path / "p.yaml"
        y.write_text("models:\n  m:\n    prompt_per_million: 1\n    completion_per_million: 2\n")
        assert load_pricing(y).cost("m", 1_000_000, 0) == pytest.approx(1.0)
        j = tmp_path / "p.json"
        j.write_text('{"models": {"m": {"prompt_per_million": 3, "completion_per_million": 0}}}')
        assert load_pricing(j).cost("m", 1_000_000, 0) == pytest.approx(3.0)

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError):
            ModelPrice(prompt_per_million=-1, completion_per_million=0)


class TestRunnerMetrics:
    def test_runner_records_latency_and_usage(self) -> None:
        sut = ScriptedLLMClient(
            default="Paris", latency_ms=250.0, prompt_tokens=30, completion_tokens=5
        )
        grader = FakeLLMClient(response='{"score": 5, "reasoning": "ok"}')
        runner = EvalRunner(system_client=sut, evaluators=[CorrectnessEvaluator(grader)])
        report = runner.run([TestCase(id=f"q{i}", input="?", expected_output="Paris") for i in range(3)])

        assert len(report.case_metrics) == 3
        m = report.case_metrics[0]
        assert (m.test_case_id, m.latency_ms, m.prompt_tokens, m.completion_tokens) == (
            "q0",
            250.0,
            30,
            5,
        )
        assert m.model == "scripted"

    def test_summarize(self) -> None:
        sut = ScriptedLLMClient(latency_ms=100.0, prompt_tokens=1_000_000, completion_tokens=0)
        runner = EvalRunner(system_client=sut, evaluators=[SafetyEvaluator()])
        report = runner.run([TestCase(id=f"q{i}", input="hi") for i in range(2)])
        s = summarize(report, PRICING)
        assert s.n_cases == 2
        assert s.latency_p50_ms == 100.0 and s.latency_p95_ms == 100.0
        assert s.total_cost == pytest.approx(2.0)
        assert s.mean_cost_per_case == pytest.approx(1.0)
        assert s.pass_rates == {"safety": 1.0}

    def test_safety_results_flow_into_pass_rate(self) -> None:
        sut = ScriptedLLMClient(responses={"leak": "mail me at jane@example.com"}, default="fine")
        runner = EvalRunner(system_client=sut, evaluators=[SafetyEvaluator()])
        report = runner.run([TestCase(id="a", input="leak"), TestCase(id="b", input="ok")])
        assert summarize(report, PRICING).pass_rates["safety"] == 0.5
        assert report.results[0].details["findings"][0]["category"] == "email"


class TestThresholds:
    def test_all_ok(self) -> None:
        t = Thresholds(
            max_p50_latency_ms=150,
            max_p95_latency_ms=300,
            max_mean_cost_per_case=0.002,
            max_total_cost=0.01,
            min_pass_rate={"safety": 1.0, "correctness": 0.5},
        )
        assert evaluate_gate(_summary(), t).passed

    def test_each_threshold_trips(self) -> None:
        t = Thresholds(
            max_p50_latency_ms=50,
            max_p95_latency_ms=100,
            max_mean_cost_per_case=0.0001,
            max_total_cost=0.001,
            min_pass_rate={"correctness": 0.9},
        )
        metrics = {v.metric for v in evaluate_gate(_summary(), t).violations}
        assert metrics == {
            "latency_p50_ms",
            "latency_p95_ms",
            "mean_cost_per_case",
            "total_cost",
            "pass_rate[correctness]",
        }

    def test_boundary_is_not_a_violation(self) -> None:
        assert evaluate_gate(_summary(), Thresholds(max_p50_latency_ms=100.0)).passed

    def test_default_min_pass_rate(self) -> None:
        t = Thresholds(default_min_pass_rate=0.9)
        result = evaluate_gate(_summary(), t)
        assert [v.metric for v in result.violations] == ["pass_rate[correctness]"]

    def test_missing_evaluator_is_violation(self) -> None:
        t = Thresholds(min_pass_rate={"groundedness": 0.5})
        assert not evaluate_gate(_summary(), t).passed

    def test_no_thresholds_passes(self) -> None:
        assert evaluate_gate(_summary()).passed

    def test_load_thresholds_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "t.yaml"
        f.write_text(
            "max_p95_latency_ms: 500\nmin_pass_rate: {safety: 1.0}\n"
            "regression: {latency_tolerance_pct: 5}\n"
        )
        t = load_thresholds(f)
        assert t.max_p95_latency_ms == 500
        assert t.regression.latency_tolerance_pct == 5
        assert t.regression.cost_tolerance_pct == 10  # default kept

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "t.yaml"
        f.write_text("max_p99_latency: 1\n")
        with pytest.raises(ValueError):
            load_thresholds(f)


class TestBaselineComparison:
    TOL = RegressionTolerance(
        latency_tolerance_pct=20, cost_tolerance_pct=10, pass_rate_tolerance=0.05
    )

    def test_identical_run_passes(self) -> None:
        assert compare_to_baseline(_summary(), _summary(), self.TOL) == []

    def test_within_tolerance_passes(self) -> None:
        cur = _summary(latency_p50_ms=119.0, latency_p95_ms=239.0, mean_cost_per_case=0.0011)
        assert compare_to_baseline(cur, _summary(), self.TOL) == []

    def test_latency_regression(self) -> None:
        cur = _summary(latency_p95_ms=241.0)
        v = compare_to_baseline(cur, _summary(), self.TOL)
        assert [(x.kind, x.metric) for x in v] == [("regression", "latency_p95_ms")]
        assert "regressed" in v[0].message

    def test_cost_regression(self) -> None:
        v = compare_to_baseline(_summary(mean_cost_per_case=0.0012), _summary(), self.TOL)
        assert [x.metric for x in v] == ["mean_cost_per_case"]

    def test_improvement_is_not_regression(self) -> None:
        cur = _summary(latency_p50_ms=10.0, latency_p95_ms=20.0, mean_cost_per_case=0.0)
        assert compare_to_baseline(cur, _summary(), self.TOL) == []

    def test_pass_rate_drop_beyond_tolerance(self) -> None:
        cur = _summary(pass_rates={"safety": 0.9, "correctness": 0.75})
        v = compare_to_baseline(cur, _summary(), self.TOL)
        assert [x.metric for x in v] == ["pass_rate[safety]"]

    def test_pass_rate_drop_within_tolerance(self) -> None:
        cur = _summary(pass_rates={"safety": 0.96, "correctness": 0.75})
        assert compare_to_baseline(cur, _summary(), self.TOL) == []

    def test_evaluator_dropped_from_run(self) -> None:
        cur = _summary(pass_rates={"safety": 1.0})
        v = compare_to_baseline(cur, _summary(), self.TOL)
        assert [x.metric for x in v] == ["pass_rate[correctness]"]

    def test_evaluate_gate_combines_threshold_and_regression(self) -> None:
        cur = _summary(latency_p50_ms=500.0)
        result = evaluate_gate(cur, Thresholds(max_p50_latency_ms=300), _summary())
        kinds = sorted(v.kind for v in result.violations)
        assert kinds == ["regression", "threshold"]


class TestBaselineFile:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        save_baseline(Baseline(suite="s", summary=_summary()), path)
        loaded = load_baseline(path)
        assert loaded.suite == "s"
        assert loaded.summary == _summary()


class TestReport:
    def _run(self):  # type: ignore[no-untyped-def]
        sut = ScriptedLLMClient(responses={"leak": "mail jane@example.com | now"}, default="fine")
        runner = EvalRunner(system_client=sut, evaluators=[SafetyEvaluator()])
        report = runner.run([TestCase(id="a", input="leak"), TestCase(id="b", input="ok")])
        return report, summarize(report, PRICING)

    def test_markdown_sections(self) -> None:
        report, summary = self._run()
        md = render_markdown(report, summary, gate=evaluate_gate(summary), baseline=_summary())
        assert md.startswith("# LLM Evaluation Report")
        for heading in (
            "## Results by evaluator",
            "## Latency and cost",
            "## Pass rate vs baseline",
            "## Failures",
        ):
            assert heading in md
        assert "| safety | 1 | 1 | 0 | 50.0% |" in md
        assert "| a | safety | fail |" in md
        assert "jane@example.com" not in md  # reasoning never echoes raw PII

    def test_gate_failure_listed(self) -> None:
        report, summary = self._run()
        gate = evaluate_gate(summary, Thresholds(min_pass_rate={"safety": 1.0}))
        md = render_markdown(report, summary, gate=gate)
        assert "**Gate: FAILED**" in md
        assert "below minimum" in md
        assert "n/a" in md  # no baseline column values

    def test_no_failures(self) -> None:
        sut = ScriptedLLMClient(default="fine")
        report = EvalRunner(system_client=sut, evaluators=[SafetyEvaluator()]).run(
            [TestCase(id="a", input="ok")]
        )
        md = render_markdown(report, summarize(report, PRICING))
        assert "## Failures\n\nNone." in md
