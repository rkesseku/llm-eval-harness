"""
examples/basic_eval.py

End-to-end demo of the LLM evaluation harness against real Groq models.

Scenario: a simplified RAG system answering questions about a small knowledge
base. Each test case provides (a) a question, (b) the retrieved context the
system should use, and (c) the expected answer.

We run two evaluators on every test case:
  - CorrectnessEvaluator: does the answer match the ground truth?
  - GroundednessEvaluator: is the answer supported by the retrieved context?

A senior engineer reads this combined output as: "correct + grounded = ship";
"correct + ungrounded = the model is leaking training data into RAG answers";
"incorrect + grounded = the retriever pulled the wrong docs"; etc.

Run from project root:
    python examples/basic_eval.py
"""

from __future__ import annotations

from llm_eval_harness.evaluators.correctness import CorrectnessEvaluator
from llm_eval_harness.evaluators.groundedness import GroundednessEvaluator
from llm_eval_harness.llm_client import GroqClient
from llm_eval_harness.models import TestCase, Verdict
from llm_eval_harness.runner import EvalRunner, SystemUnderTestConfig


# A tiny "knowledge base" of facts the system-under-test should answer from.
TEST_CASES: list[TestCase] = [
    TestCase(
        id="rag_001",
        input="When was the Eiffel Tower completed?",
        context=(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
            "Mars in Paris, France. It was completed in 1889 and stands "
            "330 meters tall."
        ),
        expected_output="1889",
        tags=["rag", "easy"],
    ),
    TestCase(
        id="rag_002",
        input="How tall is the Eiffel Tower?",
        context=(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
            "Mars in Paris, France. It was completed in 1889 and stands "
            "330 meters tall."
        ),
        expected_output="330 meters tall.",
        tags=["rag", "easy"],
    ),
    TestCase(
        id="rag_003",
        input="What is the population of Tokyo?",
        context=(
            "Tokyo is the capital of Japan and the most populous metropolitan "
            "area in the world, with over 37 million residents in the greater "
            "Tokyo region as of 2023."
        ),
        expected_output="Over 37 million in the greater Tokyo region.",
        tags=["rag", "easy"],
    ),
    TestCase(
        id="rag_004",
        input="Who designed the Eiffel Tower?",
        context=(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
            "Mars in Paris, France. It was completed in 1889."
        ),
        # Ground truth says "I don't know" because the context doesn't say.
        # A good RAG system should refuse rather than recall from training data.
        expected_output=(
            "The context does not say who designed the Eiffel Tower."
        ),
        tags=["rag", "groundedness-trap"],
    ),
    TestCase(
        id="rag_005",
        input="What is the capital of France?",
        context=(
            "France is a country in Western Europe with a population of "
            "approximately 67 million people."
        ),
        # Context doesn't mention Paris -- a strict RAG should refuse.
        expected_output=(
            "The context does not mention the capital of France."
        ),
        tags=["rag", "groundedness-trap"],
    ),
]


SYSTEM_PROMPT = (
    "You are a careful RAG assistant. Answer the user's question using ONLY "
    "the provided context. If the context does not contain the answer, say "
    "so explicitly rather than guessing or using outside knowledge. Be concise."
)


def _format_user_prompt(test_case: TestCase) -> str:
    """Inject context + question into the user message (standard RAG pattern)."""
    return (
        f"CONTEXT:\n{test_case.context}\n\n"
        f"QUESTION:\n{test_case.input}"
    )


def main() -> int:
    print("=" * 78)
    print("LLM Evaluation Harness — RAG Demo (Correctness + Groundedness)")
    print("=" * 78)

    # System-under-test and grader: same model here, different roles.
    sut = GroqClient(model="llama-3.3-70b-versatile")
    grader = GroqClient(model="llama-3.3-70b-versatile")

    evaluators = [
        CorrectnessEvaluator(grader, pass_threshold=0.75),
        GroundednessEvaluator(grader, pass_threshold=0.75),
    ]

    # The standard EvalRunner feeds `test_case.input` to the SUT as-is. For RAG
    # we need to inject the context too, so we wrap the runner by pre-formatting
    # each test case's input. We mutate by building new TestCase objects so
    # `expected_output` and `context` are preserved for the evaluators.
    rag_test_cases = [
        TestCase(
            id=tc.id,
            input=_format_user_prompt(tc),
            expected_output=tc.expected_output,
            context=tc.context,
            tags=tc.tags,
            metadata=tc.metadata,
        )
        for tc in TEST_CASES
    ]

    runner = EvalRunner(
        system_client=sut,
        evaluators=evaluators,
        system_config=SystemUnderTestConfig(
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=150,
        ),
    )

    print(f"\nRunning {len(rag_test_cases)} test cases × {len(evaluators)} evaluators "
          f"= {len(rag_test_cases) * len(evaluators)} evaluations...\n")
    report = runner.run(rag_test_cases)

    # Per-result detail, grouped by test case for readability
    icon = {Verdict.PASS: "✓", Verdict.FAIL: "✗", Verdict.ERROR: "!"}

    by_case: dict[str, list] = {}
    for r in report.results:
        by_case.setdefault(r.test_case_id, []).append(r)

    print("-" * 78)
    for tc_id, results in by_case.items():
        # Find the original test case for context display
        original = next(tc for tc in TEST_CASES if tc.id == tc_id)
        print(f"\n[{tc_id}]  {original.input}")
        # Show what the SUT actually answered (same for both evaluators, take first)
        actual = results[0].actual_output.strip().replace("\n", " ")
        actual_short = (actual[:90] + "...") if len(actual) > 90 else actual
        print(f"  SUT answered: {actual_short}")
        for r in results:
            reasoning_short = (r.reasoning[:60] + "...") if len(r.reasoning) > 60 else r.reasoning
            print(
                f"    {icon[r.verdict]} {r.evaluator_name:<14} "
                f"score={r.score:.2f}  {reasoning_short}"
            )

    print("\n" + "-" * 78)

    # Aggregate summary
    passes = sum(1 for r in report.results if r.verdict == Verdict.PASS)
    print(f"\n  Overall pass rate:    {report.pass_rate:.0%}  ({passes}/{len(report.results)})")
    print(f"  Total tokens:         {report.total_tokens:,}")
    print(f"  Avg latency / eval:   {report.avg_latency_ms:,.0f} ms")

    # Per-evaluator breakdown
    print("\n  Pass rate by evaluator:")
    for evaluator_name, results in report.by_evaluator().items():
        eval_passes = sum(1 for r in results if r.verdict == Verdict.PASS)
        pass_rate = eval_passes / len(results) if results else 0.0
        print(f"    {evaluator_name:<14}  {pass_rate:.0%}  ({eval_passes}/{len(results)})")
    print()

    # Exit code reflects pass/fail (useful for CI later this session)
    return 0 if report.pass_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())