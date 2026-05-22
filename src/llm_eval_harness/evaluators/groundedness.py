"""
evaluators/groundedness.py

LLM-as-judge groundedness scoring for RAG-style outputs.

Where the correctness evaluator asks "is the answer correct?", groundedness asks
"is every claim in the answer supported by the retrieved context?" -- the
canonical hallucination check for production RAG systems.

A grounded answer says only what the context says. An ungrounded answer adds
claims that the context does not support, regardless of whether those extra
claims happen to be true.
"""

from __future__ import annotations

import json
import re
from textwrap import dedent

from llm_eval_harness.evaluators.base import Evaluator, JudgeOutput
from llm_eval_harness.llm_client import LLMClient
from llm_eval_harness.models import TestCase, Verdict


GRADER_SYSTEM_PROMPT = dedent("""
    You are a strict groundedness grader for a retrieval-augmented system.

    You will be given:
    - A QUESTION
    - The CONTEXT that was retrieved to answer it
    - The ANSWER the system produced

    Your job: determine how well every factual claim in the ANSWER is supported
    by the CONTEXT. The ANSWER is only allowed to use information from the
    CONTEXT -- even claims that are true in the real world but absent from the
    CONTEXT count as unsupported.

    Score on a 1-5 scale:
      5 = Every claim is directly supported by the context. Fully grounded.
      4 = Almost fully grounded; at most one minor unsupported detail.
      3 = Mixed; some claims are supported, others are not.
      2 = Mostly ungrounded; only a small portion is supported.
      1 = Completely ungrounded or contradicts the context.

    Treat hedged speech ("I don't know based on the provided context") as a
    grounded response and score it 5.

    Respond ONLY with valid JSON in this exact shape (no markdown, no prose):
    {"score": <int 1-5>, "reasoning": "<one short sentence>"}
""").strip()


class GroundednessEvaluator(Evaluator):
    """Scores whether the system's answer is supported by the retrieved context."""

    name = "groundedness"

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
        if test_case.context is None or not test_case.context.strip():
            raise ValueError(
                f"GroundednessEvaluator requires test_case.context, "
                f"but test case '{test_case.id}' has none."
            )

        user_prompt = dedent(f"""
            QUESTION:
            {test_case.input}

            CONTEXT:
            {test_case.context}

            ANSWER:
            {actual_output}
        """).strip()

        completion = self._client.complete(
            system=GRADER_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,
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
        """Extract score and reasoning from the grader's JSON response."""
        cleaned = text.strip()
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