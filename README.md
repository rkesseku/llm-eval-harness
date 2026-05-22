# LLM Evaluation & Guardrails Harness

A reusable, production-grade evaluation framework for LLM applications. Test correctness, groundedness, safety, and cost/latency before promoting prompt or model changes to production.

## Why this exists

In production LLM systems, every prompt tweak or model upgrade is a potential regression. This harness gives engineering teams a defensible answer to *"did this change make things better or worse?"* — with CI-integrated checks that block bad releases.

## Features

- **Correctness checks** — LLM-as-judge scoring against a held-out reference set
- **Groundedness checks** — verifies answers stay anchored to retrieved context (RAG-safe)
- **Safety filters** — PII detection, refusal patterns, prompt-injection signals
- **Cost & latency tracking** — per-test-case metrics with regression thresholds
- **Pluggable providers** — works with Groq, OpenAI, Anthropic, and local Ollama models

## Quick start

```bash
# Clone and set up
git clone git@github.com:rkesseku/llm-eval-harness.git
cd llm-eval-harness
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate    # Mac/Linux

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Project status

🚧 **Active development.** See [issues](https://github.com/rkesseku/llm-eval-harness/issues) for roadmap.

## License

MIT