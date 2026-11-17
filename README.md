# LLM Evaluation & Guardrails Harness

[![CI](https://github.com/rkesseku/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/llm-eval-harness/actions/workflows/ci.yml)

A reusable evaluation framework for LLM applications. Score correctness, track cost and latency, and catch regressions before promoting prompt or model changes to production.

## Why this exists

In production LLM systems, every prompt tweak or model upgrade is a potential regression. This harness gives engineering teams a defensible answer to *"did this change make things better or worse?"* — with structured, reproducible evaluation runs that produce machine-readable reports.

## What's implemented today

- ✅ **LLM-as-judge correctness scoring** — graded on a 1–5 semantic scale, normalized to [0, 1]
- ✅ **Groundedness evaluator** — LLM-as-judge check that every claim in a RAG answer is supported by the retrieved context
- ✅ **Safety evaluator (deterministic, no LLM or network)** — validated PII detection with a redaction helper (email, US phone, SSN with invalid-range rejection, credit card with Luhn check, IPv4, optional IBAN with mod-97 check), configurable refusal detection (expected-but-missing / unexpected), and prompt-injection signal detection in inputs and retrieved context. Findings are stored per result and never contain raw PII
- ✅ **Latency & cost tracking** — per-case latency and prompt/completion tokens of the system-under-test are recorded in the report; a pricing table (YAML/JSON) turns tokens into cost
- ✅ **Regression gate** — absolute thresholds (p50/p95 latency, mean cost per case, max total cost, per-evaluator min pass rate) plus comparison against a stored baseline JSON with configurable tolerances
- ✅ **`llm-eval` CLI** — `llm-eval run --suite ... --baseline ... --thresholds ...` exits non-zero on any violation, so it can block a merge or promotion
- ✅ **Markdown reports** — per-evaluator summary, latency/cost vs baseline, failures table
- ✅ **CI integration** — GitHub Actions runs the test matrix (Python 3.10–3.12) and an `eval-gate` job that runs an offline example suite through the CLI against a committed baseline, with no secrets required
- ✅ **Provider-agnostic LLM client** — `Protocol`-based interface; Groq backend included; fake and scripted offline clients for deterministic tests and CI
- ✅ **Pydantic v2 data models, runner/orchestrator** — runs N test cases × M evaluators and aggregates a serializable report

Not implemented: rubric/multi-criteria scoring, additional provider backends, concurrent execution, HTML reports (see roadmap).

## Roadmap

- [x] Groundedness evaluator (RAG-aware: verifies answers stay anchored to retrieved context)
- [x] Safety evaluator (PII detection, refusal patterns, prompt-injection signals)
- [x] Cost & latency regression thresholds with CI-friendly exit codes
- [x] Markdown report rendering
- [x] GitHub Actions CI integration example
- [ ] Rubric / multi-criteria scoring for correctness
- [ ] Additional providers (OpenAI, Anthropic, Ollama)
- [ ] Concurrent execution (asyncio) for large test suites
- [ ] HTML report rendering

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

# Run tests (no network or API key needed)
pytest -v
```

## Regression gate (CLI / CI)

```bash
# Run the committed offline example suite (scripted clients, no API key needed)
llm-eval run   --suite examples/ci/suite.yaml   --baseline examples/ci/baseline.json   --thresholds examples/ci/thresholds.yaml   --pricing examples/ci/pricing.yaml   --report eval-report.md

# Intentionally promote new numbers (rewrites the baseline file)
llm-eval run --suite examples/ci/suite.yaml --pricing examples/ci/pricing.yaml   --baseline examples/ci/baseline.json --update-baseline
```

Exit codes: `0` gate passed, `1` threshold violation or regression vs baseline, `2` usage/config error.

`thresholds.yaml` (every field optional):

```yaml
max_p50_latency_ms: 500
max_p95_latency_ms: 1000
max_mean_cost_per_case: 0.001
max_total_cost: 0.01
min_pass_rate: {safety: 1.0, correctness: 0.9}
regression:               # "worse than baseline by more than..."
  latency_tolerance_pct: 20
  cost_tolerance_pct: 10
  pass_rate_tolerance: 0.0   # absolute drop allowed
```

Notes:

- Latency and cost cover the system-under-test calls (one per test case). Grader token usage is reported per result but not priced.
- `examples/ci/pricing.yaml` contains **illustrative placeholder prices**, not any vendor's rates. Supply your own.
- The example suite uses scripted clients, so it validates the harness and gate wiring, not the quality of any real model. Point `system_client` / `grader_client` at `type: groq` (needs `GROQ_API_KEY`) to evaluate a real model.
- Latency in CI with scripted clients is a constant; with real models, expect noise and set tolerances accordingly.

Using the safety evaluator directly:

```python
from llm_eval_harness.evaluators.safety import SafetyEvaluator, redact_pii
from llm_eval_harness.models import TestCase

tc = TestCase(id="t1", input="How do I reset my password?", metadata={"expect_refusal": False})
result = SafetyEvaluator().evaluate(tc, "Email jane@example.com")
print(result.verdict, result.details["findings"])   # masked previews only

print(redact_pii("Call (415) 555-0132 or mail jane@example.com"))
# Call [REDACTED_PHONE] or mail [REDACTED_EMAIL]
```

Per-case metadata: `expect_refusal` (true/false), `allow_injection` (true for deliberately adversarial cases).
Heuristics are signals, not guarantees: regexes miss obfuscated PII and novel injection phrasing.

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
├── models.py              # Pydantic models (TestCase, EvalResult, CaseMetrics, EvalReport)
├── llm_client.py          # LLMClient protocol + GroqClient + FakeLLMClient + ScriptedLLMClient
├── evaluators/
│   ├── base.py            # Abstract Evaluator base class (timing + error handling)
│   ├── correctness.py     # LLM-as-judge correctness scoring
│   ├── groundedness.py    # LLM-as-judge groundedness against retrieved context
│   └── safety.py          # Deterministic PII / refusal / prompt-injection checks
├── runner.py              # EvalRunner: evaluator × test-case grid, records latency + tokens
├── pricing.py             # Per-token pricing table
├── metrics.py             # Percentiles, RunSummary, Baseline load/save
├── gate.py                # Thresholds + baseline regression comparison
├── report.py              # Markdown report rendering
├── suite.py               # Declarative YAML/JSON suites
└── cli.py                 # `llm-eval run` entry point (CI gate)

tests/                     # unit + end-to-end CLI tests, no network calls
examples/basic_eval.py     # Demo against real Groq models
examples/ci/               # Offline suite, thresholds, pricing, committed baseline
.github/workflows/ci.yml   # test matrix + eval-gate job
```

### Design notes

- **Two distinct LLM clients per eval run.** The *system-under-test* client is the LLM being evaluated; each evaluator carries its own *grader* client. In production you'd typically use a cheaper, faster grader.
- **`Protocol` over inheritance.** `LLMClient` is a structural type — any class with the right `complete(...)` signature works. This is what makes `FakeLLMClient` and `GroqClient` interchangeable.
- **Errors don't kill the run.** If an evaluator raises, the base class catches it and emits an `ERROR` verdict so the rest of the run continues. The error is surfaced in the report.
- **All scores normalized to [0, 1].** Different evaluators may use native 1–5, 1–10, or boolean scales internally; reports aggregate cleanly because everything's on the same axis.

## License

MIT