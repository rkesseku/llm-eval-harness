"""
metrics.py

Aggregates a run into the numbers the regression gate cares about: latency
percentiles, token usage, cost, and per-evaluator pass rates.

Latency and cost describe the system-under-test calls only (one per test case).
Evaluator/grader token usage is reported separately in the EvalReport.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from llm_eval_harness.models import EvalReport, Verdict
from llm_eval_harness.pricing import PricingTable


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]). Returns 0.0 for empty input."""
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


class RunSummary(BaseModel):
    """Gate-relevant aggregate of one run."""

    model_config = ConfigDict(extra="forbid")

    n_cases: int
    latency_p50_ms: float
    latency_p95_ms: float
    mean_cost_per_case: float
    total_cost: float
    total_prompt_tokens: int
    total_completion_tokens: int
    pass_rates: dict[str, float] = Field(default_factory=dict)


class Baseline(BaseModel):
    """A stored RunSummary to compare future runs against."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    suite: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: RunSummary


def summarize(report: EvalReport, pricing: PricingTable) -> RunSummary:
    """Reduce an EvalReport to a RunSummary (costs computed from the pricing table)."""
    metrics = report.case_metrics
    latencies = [m.latency_ms for m in metrics]
    costs = [pricing.cost(m.model, m.prompt_tokens, m.completion_tokens) for m in metrics]
    total_cost = sum(costs)

    pass_rates: dict[str, float] = {}
    for name, results in report.by_evaluator().items():
        passes = sum(1 for r in results if r.verdict == Verdict.PASS)
        pass_rates[name] = passes / len(results) if results else 0.0

    return RunSummary(
        n_cases=len(metrics),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        mean_cost_per_case=total_cost / len(metrics) if metrics else 0.0,
        total_cost=total_cost,
        total_prompt_tokens=sum(m.prompt_tokens for m in metrics),
        total_completion_tokens=sum(m.completion_tokens for m in metrics),
        pass_rates=pass_rates,
    )


def save_baseline(baseline: Baseline, path: str | Path) -> None:
    Path(path).write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> Baseline:
    return Baseline.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
