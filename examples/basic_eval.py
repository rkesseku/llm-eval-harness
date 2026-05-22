"""
examples/basic_eval.py

End-to-end demo of the LLM evaluation harness against a real Groq model.

Runs a small geography Q&A suite through:
  - System-under-test: Llama 3.3 70B (the model being evaluated)
  - Grader:            Llama 3.3 70B (LLM-as-judge for correctness)

Both use the same model in this demo, but in production you'd typically use
a cheaper/faster grader (e.g. llama-3.1-8b-instant) and reserve the expensive
model for the system-under-test.

Run from project root:
    python examples/basic_eval.py
"""

from __future__ import annotations

from llm_eval_harness.evaluators.correctness import CorrectnessEvaluator
from llm_eval_harness.llm_client import GroqClient
from llm_eval_harness.models import TestCase, Verdict
from llm_eval_harness.runner import EvalRunner, SystemUnderTestConfig


TEST_CASES: list[TestCase] = [
    TestCase(
        id="geo_001",
        input="What is the capital of France?",
        expected_output="Paris",
        tags=["geography", "easy"],
    ),
    TestCase(
        id="geo_002",
        input="What is the capital of Australia?",
        expected_output="Canberra (commonly mistaken for Sydney).",
        tags=["geography", "tricky"],
    ),
    TestCase(
        id="geo_003",
        input="What is the capital of Brazil?",
        expected_output="Brasilia",
        tags=["geography", "easy"],
    ),
    TestCase(
        id="geo_004",
        input="What is the largest country in South America?",
        expected_output="Brazil",
        tags=["geography", "easy"],
    ),
    TestCase(
        id="geo_005",
        input="Which African country was never colonized?",
        expected_output="Ethiopia",
        tags=["geography", "history"],
    ),
]


def main() -> int:
    print("=" * 70)
    print("LLM Evaluation Harness — Basic Demo")
    print("=" * 70)

    # System-under-test: the model whose answers we are evaluating.
    sut = GroqClient(model="llama-3.3-70b-versatile")

    # Grader: the LLM that judges correctness. Same model in this demo.
    grader = GroqClient(model="llama-3.3-70b-versatile")

    evaluators = [CorrectnessEvaluator(grader, pass_threshold=0.75)]

    runner = EvalRunner(
        system_client=sut,
        evaluators=evaluators,
        system_config=SystemUnderTestConfig(
            system_prompt="You are a concise factual assistant. Answer in one short sentence.",
            temperature=0.0,
            max_tokens=100,
        ),
    )

    print(f"\nRunning {len(TEST_CASES)} test cases through {len(evaluators)} evaluator(s)...\n")
    report = runner.run(TEST_CASES)

    # Per-result detail
    print("-" * 70)
    print(f"{'ID':<10} {'EVALUATOR':<14} {'VERDICT':<8} {'SCORE':<7} REASONING")
    print("-" * 70)
    for r in report.results:
        icon = {
            Verdict.PASS: "✓",
            Verdict.FAIL: "✗",
            Verdict.ERROR: "!",
        }[r.verdict]
        reasoning = (r.reasoning[:60] + "...") if len(r.reasoning) > 60 else r.reasoning
        print(
            f"{r.test_case_id:<10} {r.evaluator_name:<14} {icon} {r.verdict.value:<5} "
            f"{r.score:>5.2f}  {reasoning}"
        )

    # Aggregate summary
    print("-" * 70)
    print(f"\n  Pass rate:    {report.pass_rate:.0%} ({sum(1 for r in report.results if r.verdict == Verdict.PASS)}/{len(report.results)})")
    print(f"  Total tokens: {report.total_tokens:,}")
    print(f"  Avg latency:  {report.avg_latency_ms:,.0f} ms")
    print()

    # Exit code reflects pass/fail (useful in CI later)
    return 0 if report.pass_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())