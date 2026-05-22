"""
hello_llm.py

A minimal sanity check that our Groq client is wired up correctly.
Loads the API key from .env, sends a single prompt, prints the response.

Run from project root:
    python -m llm_eval_harness.hello_llm
"""

from __future__ import annotations

import os
import sys
from typing import Final

from dotenv import load_dotenv
from groq import Groq


MODEL: Final[str] = "llama-3.3-70b-versatile"


def main() -> int:
    """Send a single prompt to Groq and print the response. Returns exit code."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY not found. "
            "Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    client = Groq(api_key=api_key)

    print(f"Calling {MODEL}...\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise technical assistant. Respond in one short paragraph.",
            },
            {
                "role": "user",
                "content": "What is an LLM evaluation harness, and why does a production team need one?",
            },
        ],
        temperature=0.2,
        max_tokens=200,
    )

    answer = response.choices[0].message.content
    usage = response.usage

    print("--- Response ---")
    print(answer)
    print("\n--- Usage ---")
    print(f"Prompt tokens:     {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens:      {usage.total_tokens}")

    return 0


if __name__ == "__main__":
    sys.exit(main())