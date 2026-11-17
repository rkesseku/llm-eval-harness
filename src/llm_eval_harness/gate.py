"""
gate.py

Threshold checks and baseline-regression comparison for a run.

Two kinds of violation:
  - ``threshold``:  an absolute limit was exceeded (e.g. p95 latency > 2000 ms).
  - ``regression``: the run is worse than the stored baseline by more than the
                    configured tolerance (latency/cost up, pass rate down).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_eval_harness.metrics import RunSummary
from llm_eval_harness.pricing import load_mapping


class RegressionTolerance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_tolerance_pct: float = Field(default=20.0, ge=0.0)
    cost_tolerance_pct: float = Field(default=10.0, ge=0.0)
    pass_rate_tolerance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Absolute drop in pass rate allowed."
    )


class Thresholds(BaseModel):
    """Absolute limits. Any field left unset is not checked."""

    model_config = ConfigDict(extra="forbid")

    max_p50_latency_ms: float | None = None
    max_p95_latency_ms: float | None = None
    max_mean_cost_per_case: float | None = None
    max_total_cost: float | None = None
    min_pass_rate: dict[str, float] = Field(
        default_factory=dict, description="Per-evaluator minimum pass rate in [0, 1]."
    )
    default_min_pass_rate: float | None = Field(
        default=None, description="Applied to evaluators without an explicit entry."
    )
    regression: RegressionTolerance = Field(default_factory=RegressionTolerance)


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["threshold", "regression"]
    metric: str
    actual: float | None
    limit: float | None
    message: str


class GateResult(BaseModel):
    violations: list[Violation] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def load_thresholds(path: str | Path) -> Thresholds:
    return Thresholds.model_validate(load_mapping(Path(path)))


def check_thresholds(summary: RunSummary, thresholds: Thresholds) -> list[Violation]:
    """Absolute-limit checks."""
    out: list[Violation] = []

    def upper(metric: str, actual: float, limit: float | None, unit: str) -> None:
        if limit is not None and actual > limit:
            out.append(
                Violation(
                    kind="threshold",
                    metric=metric,
                    actual=actual,
                    limit=limit,
                    message=f"{metric} {actual:.4g}{unit} exceeds max {limit:.4g}{unit}",
                )
            )

    upper("latency_p50_ms", summary.latency_p50_ms, thresholds.max_p50_latency_ms, " ms")
    upper("latency_p95_ms", summary.latency_p95_ms, thresholds.max_p95_latency_ms, " ms")
    upper(
        "mean_cost_per_case",
        summary.mean_cost_per_case,
        thresholds.max_mean_cost_per_case,
        " USD",
    )
    upper("total_cost", summary.total_cost, thresholds.max_total_cost, " USD")

    for name, minimum in thresholds.min_pass_rate.items():
        if name not in summary.pass_rates:
            out.append(
                Violation(
                    kind="threshold",
                    metric=f"pass_rate[{name}]",
                    actual=None,
                    limit=minimum,
                    message=f"evaluator '{name}' has a min_pass_rate but produced no results",
                )
            )
    for name, rate in summary.pass_rates.items():
        minimum = thresholds.min_pass_rate.get(name, thresholds.default_min_pass_rate)
        if minimum is not None and rate < minimum:
            out.append(
                Violation(
                    kind="threshold",
                    metric=f"pass_rate[{name}]",
                    actual=rate,
                    limit=minimum,
                    message=f"pass rate for '{name}' is {rate:.1%}, below minimum {minimum:.1%}",
                )
            )
    return out


def compare_to_baseline(
    current: RunSummary, baseline: RunSummary, tolerance: RegressionTolerance
) -> list[Violation]:
    """Regression = worse than baseline by more than the tolerance."""
    out: list[Violation] = []

    def higher_is_worse(metric: str, cur: float, base: float, pct: float, unit: str) -> None:
        limit = base * (1.0 + pct / 100.0)
        if cur > limit + 1e-12:
            out.append(
                Violation(
                    kind="regression",
                    metric=metric,
                    actual=cur,
                    limit=limit,
                    message=(
                        f"{metric} regressed: {cur:.4g}{unit} vs baseline {base:.4g}{unit} "
                        f"(allowed +{pct:g}%)"
                    ),
                )
            )

    lat = tolerance.latency_tolerance_pct
    higher_is_worse("latency_p50_ms", current.latency_p50_ms, baseline.latency_p50_ms, lat, " ms")
    higher_is_worse("latency_p95_ms", current.latency_p95_ms, baseline.latency_p95_ms, lat, " ms")
    higher_is_worse(
        "mean_cost_per_case",
        current.mean_cost_per_case,
        baseline.mean_cost_per_case,
        tolerance.cost_tolerance_pct,
        " USD",
    )

    for name, base_rate in baseline.pass_rates.items():
        cur_rate = current.pass_rates.get(name)
        floor = base_rate - tolerance.pass_rate_tolerance
        if cur_rate is None:
            out.append(
                Violation(
                    kind="regression",
                    metric=f"pass_rate[{name}]",
                    actual=None,
                    limit=floor,
                    message=f"evaluator '{name}' is in the baseline but missing from this run",
                )
            )
        elif cur_rate < floor - 1e-12:
            out.append(
                Violation(
                    kind="regression",
                    metric=f"pass_rate[{name}]",
                    actual=cur_rate,
                    limit=floor,
                    message=(
                        f"pass rate for '{name}' regressed: {cur_rate:.1%} vs baseline "
                        f"{base_rate:.1%} (allowed -{tolerance.pass_rate_tolerance:.1%})"
                    ),
                )
            )
    return out


def evaluate_gate(
    current: RunSummary,
    thresholds: Thresholds | None = None,
    baseline: RunSummary | None = None,
) -> GateResult:
    """Run absolute-threshold checks and (if a baseline is given) regression checks."""
    thresholds = thresholds or Thresholds()
    violations = check_thresholds(current, thresholds)
    if baseline is not None:
        violations.extend(compare_to_baseline(current, baseline, thresholds.regression))
    return GateResult(violations=violations)
