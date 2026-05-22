"""
llm_client.py

Thin wrapper around an LLM provider with timing and token tracking.

The LLMClient protocol defines what every backend must implement: a single
`complete(...)` call that returns a CompletionResult. This lets evaluators and
the runner depend on the protocol, not on any specific vendor SDK -- so we can
swap Groq for OpenAI, Anthropic, or a fake in-memory client without touching
business logic.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dotenv import load_dotenv
from groq import Groq


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Result of one LLM call. Lightweight container, not a Pydantic model."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str


@runtime_checkable
class LLMClient(Protocol):
    """Protocol every LLM backend must implement."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> CompletionResult: ...


class GroqClient:
    """Concrete LLMClient backed by Groq's hosted models."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        load_dotenv()
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self._client = Groq(api_key=key)
        self.model = model or self.DEFAULT_MODEL

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> CompletionResult:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content or ""
        usage = response.usage

        return CompletionResult(
            text=text,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=elapsed_ms,
            model=self.model,
        )


class FakeLLMClient:
    """In-memory LLM client for tests. Returns canned responses.

    Use this in pytest so tests don't hit the network or burn API quota.
    """

    def __init__(self, *, response: str = "OK", tokens: int = 10) -> None:
        self._response = response
        self._tokens = tokens
        self.calls: list[dict[str, object]] = []  # for assertions in tests

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> CompletionResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return CompletionResult(
            text=self._response,
            prompt_tokens=self._tokens,
            completion_tokens=self._tokens,
            total_tokens=self._tokens * 2,
            latency_ms=0.1,
            model="fake",
        )