# LLM Evaluation & Guardrails Harness

# LLM Evaluation & Guardrails Harness

[![CI](https://github.com/rkesseku/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/llm-eval-harness/actions/workflows/ci.yml)

A reusable evaluation framework for LLM applications. Score correctness, track cost and latency, and catch regressions before promoting prompt or model changes to production.

## Why this exists

In production LLM systems, every prompt tweak or model upgrade is a potential regression. This harness gives engineering teams a defensible answer to *"did this change make things better or worse?"* — with structured, reproducible evaluation runs that produce machine-readable reports.

## What's implemented today

- ✅ **LLM-as-judge correctness scoring** — graded on a 1–5 semantic scale, normalized to [0, 1]
- ✅ **Provider-agnostic LLM client** — `Protocol`-based interface; Groq backend included; pluggable for OpenAI / Anthropic / Ollama
- ✅ **Test-case + result data models** — Pydantic v2, strict validation, serializable to JSON
- ✅ **Runner / orchestrator** — runs N test cases × M evaluators, returns an aggregated report with pass-rate, token totals, and latency stats
- ✅ **Fake LLM client** — for fast, deterministic, network-free unit tests
- ✅ **27 unit tests** — covering data validation, evaluator behavior, error paths, and the runner

## Roadmap

- [ ] Groundedness evaluator (RAG-aware: verifies answers stay anchored to retrieved context)
- [ ] Safety evaluator (PII detection, refusal patterns, prompt-injection signals)
- [ ] Cost & latency regression thresholds with CI-friendly exit codes
- [ ] Additional providers (OpenAI, Anthropic, Ollama)
- [ ] Concurrent execution (asyncio) for large test suites
- [ ] HTML / Markdown report rendering
- [ ] GitHub Actions CI integration example

## Quick start

```bash
# Clone and set up
git clone git@github.com:rkesseku/llm-eval-harness.git
cd llm-eval-harness
python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash
# source .venv/bin/activate       # Mac/Linux

# Install (with dev tools: pytest, ruff, mypy)
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (get one free at https://console.groq.com)

# Sanity check
python -m llm_eval_harness.hello_llm

# Run the demo
python examples/basic_eval.py

# Run tests
pytest -v
```

## Example output

Running `python examples/basic_eval.py`:

```
======================================================================
LLM Evaluation Harness — Basic Demo
======================================================================

Running 5 test cases through 1 evaluator(s)...

----------------------------------------------------------------------
ID         EVALUATOR      VERDICT  SCORE   REASONING
----------------------------------------------------------------------
geo_001    correctness    ✓ pass    1.00  The actual answer fully matches...
geo_002    correctness    ✓ pass    1.00  Accurately identifies Canberra...
geo_003    correctness    ✓ pass    1.00  The actual answer fully matches...
geo_004    correctness    ✓ pass    1.00  Matches in meaning...
geo_005    correctness    ✓ pass    1.00  Fully and correctly identifies Ethiopia...
----------------------------------------------------------------------

  Pass rate:    100% (5/5)
  Total tokens: 1,270
  Avg latency:  273 ms
```

## Architecture

```
src/llm_eval_harness/
├── models.py              # Pydantic data models (TestCase, EvalResult, EvalReport)
├── llm_client.py          # LLMClient protocol + GroqClient + FakeLLMClient
├── evaluators/
│   ├── base.py            # Abstract Evaluator base class (handles timing + errors)
│   └── correctness.py     # LLM-as-judge correctness scoring
└── runner.py              # EvalRunner: orchestrates evaluator × test-case grid

tests/                     # 27 unit tests, no network calls
examples/basic_eval.py     # End-to-end demo against real Groq models
```

### Design notes

- **Two distinct LLM clients per eval run.** The *system-under-test* client is the LLM being evaluated; each evaluator carries its own *grader* client. In production you'd typically use a cheaper, faster grader.
- **`Protocol` over inheritance.** `LLMClient` is a structural type — any class with the right `complete(...)` signature works. This is what makes `FakeLLMClient` and `GroqClient` interchangeable.
- **Errors don't kill the run.** If an evaluator raises, the base class catches it and emits an `ERROR` verdict so the rest of the run continues. The error is surfaced in the report.
- **All scores normalized to [0, 1].** Different evaluators may use native 1–5, 1–10, or boolean scales internally; reports aggregate cleanly because everything's on the same axis.

## License

MIT