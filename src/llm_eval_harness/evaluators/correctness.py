"""
evaluators/correctness.py

LLM-as-judge correctness scoring.

Given an expected answer and an actual answer, ask a grader LLM to score
semantic correctness on a 1-5 scale and explain its reasoning.
"""

from __future__ import annotations

import json
import re
from textwrap import dedent

from llm_eval_harness.evaluators.base import Evaluator, JudgeOutput
from llm_eval_harness.llm_client import LLMClient
from llm_eval_harness.models import TestCase, Verdict


GRADER_SYSTEM_PROMPT = dedent("""
    You are a strict but fair grader for an AI assistant's answers.

    You will be given:
    - A QUESTION
    - The EXPECTED answer (ground truth)
    - The ACTUAL answer (from the system under test)

    Score the ACTUAL answer's semantic correctness against the EXPECTED answer
    on a 1-5 scale:
      5 = Fully correct and complete.
      4 = Correct but missing minor detail.
      3 = Partially correct; some claims right, some wrong or missing.
      2 = Mostly incorrect; only superficial overlap.
      1 = Completely wrong, off-topic, or refuses to answer.

    Respond ONLY with valid JSON in this exact shape (no markdown, no prose):
    {"score": <int 1-5>, "reasoning": "<one short sentence>"}
""").strip()


class CorrectnessEvaluator(Evaluator):
    """Scores semantic correctness against an expected answer via LLM-as-judge."""

    name = "correctness"

    # Verdicts: PASS for score >= pass_threshold (on normalized 0-1 scale), else FAIL.
    DEFAULT_PASS_THRESHOLD = 0.8

    def __init__(
        self,
        judge_client: LLMClient,
        *,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    ) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0, 1]")
        self._client = judge_client
        self._pass_threshold = pass_threshold

    def _evaluate(self, test_case: TestCase, actual_output: str) -> JudgeOutput:
        if test_case.expected_output is None:
            raise ValueError(
                f"CorrectnessEvaluator requires test_case.expected_output, "
                f"but test case '{test_case.id}' has none."
            )

        user_prompt = dedent(f"""
            QUESTION:
            {test_case.input}

            EXPECTED:
            {test_case.expected_output}

            ACTUAL:
            {actual_output}
        """).strip()

        completion = self._client.complete(
            system=GRADER_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,  # deterministic grading
            max_tokens=200,
        )

        score_raw, reasoning = self._parse_grader_output(completion.text)
        normalized = (score_raw - 1) / 4  # map 1-5 -> 0-1
        verdict = Verdict.PASS if normalized >= self._pass_threshold else Verdict.FAIL

        return JudgeOutput(
            score=normalized,
            verdict=verdict,
            reasoning=reasoning,
            tokens_used=completion.total_tokens,
        )

    @staticmethod
    def _parse_grader_output(text: str) -> tuple[int, str]:
        """Extract score and reasoning from the grader's JSON response.

        Robust to stray markdown fences or surrounding whitespace.
        """
        cleaned = text.strip()
        # Strip markdown fences if present (defensive against grader misbehavior).
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Grader returned non-JSON output: {text!r}"
            ) from exc

        score = int(data["score"])
        if score < 1 or score > 5:
            raise ValueError(f"Grader returned out-of-range score: {score}")

        reasoning = str(data.get("reasoning", "")).strip()
        return score, reasoning