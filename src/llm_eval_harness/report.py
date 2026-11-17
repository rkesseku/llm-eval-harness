"""
report.py

Render an evaluation run as a Markdown report (for CI job summaries, PR comments,
or artifacts).
"""

from __future__ import annotations

from llm_eval_harness.gate import GateResult
from llm_eval_harness.metrics import RunSummary
from llm_eval_harness.models import EvalReport, Verdict


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _row(label: str, cur: float, base: float | None, digits: int) -> str:
    fmt = f".{digits}f"
    if base is None:
        return f"| {label} | {cur:{fmt}} | n/a | n/a |"
    diff = cur - base
    pct = f" ({diff / base:+.1%})" if base != 0 else ""
    return f"| {label} | {cur:{fmt}} | {base:{fmt}} | {diff:+{fmt}}{pct} |"


def render_markdown(
    report: EvalReport,
    summary: RunSummary,
    *,
    gate: GateResult | None = None,
    baseline: RunSummary | None = None,
    title: str = "LLM Evaluation Report",
    max_failures: int = 50,
) -> str:
    """Render the run as Markdown: gate status, per-evaluator table, latency/cost, failures."""
    lines: list[str] = [f"# {title}", ""]

    if gate is not None:
        lines += [f"**Gate: {'PASSED' if gate.passed else 'FAILED'}**", ""]
        if gate.violations:
            lines += [f"- [{v.kind}] {v.message}" for v in gate.violations]
            lines.append("")

    lines += [
        (
            f"{summary.n_cases} test case(s) x {report.total_evaluators} evaluator(s) = "
            f"{len(report.results)} result(s). Overall pass rate: {report.pass_rate:.1%}."
        ),
        "",
        "## Results by evaluator",
        "",
        "| Evaluator | Passed | Failed | Errors | Pass rate | Mean score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, results in report.by_evaluator().items():
        passed = sum(1 for r in results if r.verdict == Verdict.PASS)
        failed = sum(1 for r in results if r.verdict == Verdict.FAIL)
        errors = sum(1 for r in results if r.verdict == Verdict.ERROR)
        mean_score = sum(r.score for r in results) / len(results)
        lines.append(
            f"| {name} | {passed} | {failed} | {errors} | "
            f"{passed / len(results):.1%} | {mean_score:.2f} |"
        )

    b = baseline
    lines += [
        "",
        "## Latency and cost (system under test)",
        "",
        "| Metric | Current | Baseline | Delta |",
        "|---|---:|---:|---:|",
        _row("Latency p50 (ms)", summary.latency_p50_ms, b and b.latency_p50_ms, 1),
        _row("Latency p95 (ms)", summary.latency_p95_ms, b and b.latency_p95_ms, 1),
        _row("Mean cost / case (USD)", summary.mean_cost_per_case, b and b.mean_cost_per_case, 6),
        _row("Total cost (USD)", summary.total_cost, b and b.total_cost, 6),
        _row("Prompt tokens", summary.total_prompt_tokens, b and b.total_prompt_tokens, 0),
        _row(
            "Completion tokens",
            summary.total_completion_tokens,
            b and b.total_completion_tokens,
            0,
        ),
    ]

    if b is not None and summary.pass_rates:
        lines += [
            "",
            "## Pass rate vs baseline",
            "",
            "| Evaluator | Current | Baseline |",
            "|---|---:|---:|",
        ]
        for name, rate in summary.pass_rates.items():
            base_rate = b.pass_rates.get(name)
            shown = "n/a" if base_rate is None else f"{base_rate:.1%}"
            lines.append(f"| {name} | {rate:.1%} | {shown} |")

    bad = [r for r in report.results if r.verdict != Verdict.PASS]
    lines += ["", "## Failures", ""]
    if not bad:
        lines.append("None.")
    else:
        lines += ["| Case | Evaluator | Verdict | Score | Reasoning |", "|---|---|---|---:|---|"]
        for r in bad[:max_failures]:
            lines.append(
                f"| {_escape(r.test_case_id)} | {r.evaluator_name} | {r.verdict.value} | "
                f"{r.score:.2f} | {_escape(r.reasoning)[:200]} |"
            )
        if len(bad) > max_failures:
            lines += ["", f"... and {len(bad) - max_failures} more."]

    lines.append("")
    return "\n".join(lines)
