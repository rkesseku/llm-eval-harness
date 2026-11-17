"""
suite.py

Declarative evaluation suites (YAML or JSON) so a run can be driven from the CLI / CI.

Example (see ``examples/suite.yaml``)::

    name: offline-demo
    system_prompt: "Answer using only the context."
    rag_prompt: true                # feed "CONTEXT: ... QUESTION: ..." to the system under test
    system_client: {type: scripted, model: scripted, latency_ms: 120, responses: {...}}
    grader_client: {type: scripted, default: '{"score": 5, "reasoning": "ok"}'}
    evaluators:
      - {type: safety, expect_refusal: null}
      - {type: correctness, pass_threshold: 0.75}
      - {type: groundedness, grader: {type: scripted, responses: {...}}}
    test_cases:
      - {id: q1, input: "...", context: "...", expected_output: "..."}

Client ``type`` is ``scripted`` (offline, deterministic, no secrets) or ``groq``
(real API, needs ``GROQ_API_KEY`` and the ``groq`` package).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_eval_harness.evaluators.base import Evaluator
from llm_eval_harness.llm_client import LLMClient, ScriptedLLMClient
from llm_eval_harness.models import TestCase
from llm_eval_harness.pricing import load_mapping
from llm_eval_harness.runner import EvalRunner, SystemUnderTestConfig


class ClientSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "scripted"
    model: str | None = None
    # scripted-only
    responses: dict[str, str] = Field(default_factory=dict)
    default: str = "OK"
    latency_ms: float = 100.0
    prompt_tokens: int = 50
    completion_tokens: int = 20


class EvaluatorSpec(BaseModel):
    """``type`` plus evaluator kwargs; ``grader`` optionally overrides the suite grader."""

    model_config = ConfigDict(extra="allow")

    type: str
    grader: ClientSpec | None = None


class SuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "suite"
    system_prompt: str = "You are a helpful, concise assistant."
    temperature: float = 0.0
    max_tokens: int = 512
    rag_prompt: bool = False
    system_client: ClientSpec = Field(default_factory=ClientSpec)
    grader_client: ClientSpec = Field(default_factory=ClientSpec)
    evaluators: list[EvaluatorSpec]
    test_cases: list[TestCase]


def load_suite(path: str | Path) -> SuiteSpec:
    return SuiteSpec.model_validate(load_mapping(Path(path)))


def build_client(spec: ClientSpec) -> LLMClient:
    if spec.type == "scripted":
        return ScriptedLLMClient(
            responses=spec.responses,
            default=spec.default,
            latency_ms=spec.latency_ms,
            prompt_tokens=spec.prompt_tokens,
            completion_tokens=spec.completion_tokens,
            model=spec.model or "scripted",
        )
    if spec.type == "groq":
        from llm_eval_harness.llm_client import GroqClient

        return GroqClient(model=spec.model)
    raise ValueError(f"Unknown client type {spec.type!r} (expected 'scripted' or 'groq').")


def build_evaluator(spec: EvaluatorSpec, default_grader: ClientSpec) -> Evaluator:
    kwargs: dict[str, Any] = dict(spec.model_extra or {})
    if spec.type == "safety":
        from llm_eval_harness.evaluators.safety import SafetyEvaluator

        return SafetyEvaluator(**kwargs)

    grader = build_client(spec.grader or default_grader)
    if spec.type == "correctness":
        from llm_eval_harness.evaluators.correctness import CorrectnessEvaluator

        return CorrectnessEvaluator(grader, **kwargs)
    if spec.type == "groundedness":
        from llm_eval_harness.evaluators.groundedness import GroundednessEvaluator

        return GroundednessEvaluator(grader, **kwargs)
    raise ValueError(
        f"Unknown evaluator type {spec.type!r} (expected safety, correctness, groundedness)."
    )


def _rag_input(tc: TestCase) -> str:
    return f"CONTEXT:\n{tc.context}\n\nQUESTION:\n{tc.input}"


def build_runner(suite: SuiteSpec) -> tuple[EvalRunner, list[TestCase]]:
    """Construct the runner and the (possibly RAG-formatted) test cases for a suite."""
    evaluators = [build_evaluator(e, suite.grader_client) for e in suite.evaluators]
    runner = EvalRunner(
        system_client=build_client(suite.system_client),
        evaluators=evaluators,
        system_config=SystemUnderTestConfig(
            system_prompt=suite.system_prompt,
            temperature=suite.temperature,
            max_tokens=suite.max_tokens,
        ),
    )
    cases = suite.test_cases
    if suite.rag_prompt:
        cases = [
            tc.model_copy(update={"input": _rag_input(tc)}) if tc.context else tc for tc in cases
        ]
    return runner, cases
