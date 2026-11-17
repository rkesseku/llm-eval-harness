"""
cli.py

Command-line gate for CI.

    llm-eval run --suite suite.yaml --baseline baseline.json \
                 --thresholds thresholds.yaml --pricing pricing.yaml \
                 --report report.md

Exit codes:
    0  run completed and the gate passed
    1  gate failed (threshold violation or regression vs baseline)
    2  usage / configuration error (bad file, unknown model price, ...)

``--update-baseline`` writes the current run's summary to the ``--baseline`` path
(intentional promotion of new numbers) and exits 0 unless the run itself errored.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from llm_eval_harness.gate import Thresholds, evaluate_gate, load_thresholds
from llm_eval_harness.metrics import Baseline, load_baseline, save_baseline, summarize
from llm_eval_harness.pricing import PricingTable, load_pricing
from llm_eval_harness.report import render_markdown
from llm_eval_harness.suite import build_runner, load_suite

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval", description="Run an LLM evaluation suite and gate on regressions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a suite and apply the gate.")
    run.add_argument("--suite", required=True, type=Path, help="Suite YAML/JSON file.")
    run.add_argument("--baseline", type=Path, help="Baseline JSON to compare against.")
    run.add_argument("--thresholds", type=Path, help="Thresholds YAML/JSON file.")
    run.add_argument("--pricing", type=Path, help="Pricing YAML/JSON (default: all prices zero).")
    run.add_argument("--report", type=Path, help="Write a Markdown report here.")
    run.add_argument("--json-out", type=Path, help="Write the full EvalReport JSON here.")
    run.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write this run's summary to --baseline instead of comparing against it.",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.update_baseline and args.baseline is None:
        print("error: --update-baseline requires --baseline PATH", file=sys.stderr)
        return EXIT_USAGE

    suite = load_suite(args.suite)
    pricing = load_pricing(args.pricing) if args.pricing else _free_pricing()
    thresholds = load_thresholds(args.thresholds) if args.thresholds else Thresholds()

    baseline_summary = None
    if args.baseline is not None and not args.update_baseline:
        if not args.baseline.exists():
            print(
                f"error: baseline {args.baseline} not found "
                "(create it with --update-baseline)",
                file=sys.stderr,
            )
            return EXIT_USAGE
        baseline_summary = load_baseline(args.baseline).summary

    runner, cases = build_runner(suite)
    report = runner.run(cases)
    summary = summarize(report, pricing)
    gate = evaluate_gate(summary, thresholds, baseline_summary)

    markdown = render_markdown(
        report,
        summary,
        gate=gate,
        baseline=baseline_summary,
        title=f"LLM Evaluation Report: {suite.name}",
    )
    if args.report:
        args.report.write_text(markdown, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(markdown)

    if args.update_baseline:
        save_baseline(Baseline(suite=suite.name, summary=summary), args.baseline)
        print(f"Baseline written to {args.baseline}")
        return EXIT_OK

    if not gate.passed:
        print(f"GATE FAILED: {len(gate.violations)} violation(s)", file=sys.stderr)
        for v in gate.violations:
            print(f"  - [{v.kind}] {v.message}", file=sys.stderr)
        return EXIT_GATE_FAILED
    print("GATE PASSED")
    return EXIT_OK


def _free_pricing() -> PricingTable:
    from llm_eval_harness.pricing import ModelPrice

    return PricingTable(default=ModelPrice(prompt_per_million=0.0, completion_per_million=0.0))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, KeyError, ValidationError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
