"""End-to-end tests for the ``llm-eval`` CLI gate using offline scripted clients."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_eval_harness.cli import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, main
from llm_eval_harness.metrics import load_baseline

REPO = Path(__file__).resolve().parent.parent
CI_DIR = REPO / "examples" / "ci"

PRICING = {
    "models": {"sut": {"prompt_per_million": 1.0, "completion_per_million": 1.0}},
}


def _suite(latency_ms: float = 100.0, answer: str = "Paris", prompt_tokens: int = 100) -> dict:
    return {
        "name": "t",
        "system_client": {
            "type": "scripted",
            "model": "sut",
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 10,
            "default": answer,
        },
        "grader_client": {"type": "scripted", "default": '{"score": 5, "reasoning": "ok"}'},
        "evaluators": [{"type": "safety"}, {"type": "correctness"}],
        "test_cases": [
            {"id": f"c{i}", "input": "Capital of France?", "expected_output": "Paris"}
            for i in range(4)
        ],
    }


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "pricing.yaml").write_text(yaml.safe_dump(PRICING))
    return tmp_path


def _write_suite(workdir: Path, **kw: object) -> Path:
    path = workdir / "suite.yaml"
    path.write_text(yaml.safe_dump(_suite(**kw)))  # type: ignore[arg-type]
    return path


def _args(workdir: Path, suite: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--suite",
        str(suite),
        "--pricing",
        str(workdir / "pricing.yaml"),
        *extra,
    ]


def test_update_baseline_then_pass(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = _write_suite(workdir)
    baseline = workdir / "baseline.json"
    assert main(_args(workdir, suite, "--baseline", str(baseline), "--update-baseline")) == EXIT_OK
    assert load_baseline(baseline).summary.latency_p50_ms == 100.0

    assert main(_args(workdir, suite, "--baseline", str(baseline))) == EXIT_OK
    assert "GATE PASSED" in capsys.readouterr().out


def test_latency_regression_exits_nonzero(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = workdir / "baseline.json"
    main(_args(workdir, _write_suite(workdir), "--baseline", str(baseline), "--update-baseline"))

    slow = _write_suite(workdir, latency_ms=150.0)  # +50% vs 20% default tolerance
    code = main(_args(workdir, slow, "--baseline", str(baseline)))
    assert code == EXIT_GATE_FAILED
    err = capsys.readouterr().err
    assert "GATE FAILED" in err and "latency_p50_ms regressed" in err


def test_cost_regression_exits_nonzero(workdir: Path) -> None:
    baseline = workdir / "baseline.json"
    main(_args(workdir, _write_suite(workdir), "--baseline", str(baseline), "--update-baseline"))
    pricey = _write_suite(workdir, prompt_tokens=400)
    assert main(_args(workdir, pricey, "--baseline", str(baseline))) == EXIT_GATE_FAILED


def test_pass_rate_regression_exits_nonzero(workdir: Path) -> None:
    baseline = workdir / "baseline.json"
    main(_args(workdir, _write_suite(workdir), "--baseline", str(baseline), "--update-baseline"))
    leaky = _write_suite(workdir, answer="Paris, mail jane@example.com")
    assert main(_args(workdir, leaky, "--baseline", str(baseline))) == EXIT_GATE_FAILED


def test_threshold_failure_without_baseline(workdir: Path) -> None:
    thresholds = workdir / "t.yaml"
    thresholds.write_text("max_p95_latency_ms: 50\n")
    code = main(_args(workdir, _write_suite(workdir), "--thresholds", str(thresholds)))
    assert code == EXIT_GATE_FAILED


def test_report_and_json_outputs(workdir: Path) -> None:
    report, js = workdir / "r.md", workdir / "r.json"
    code = main(
        _args(workdir, _write_suite(workdir), "--report", str(report), "--json-out", str(js))
    )
    assert code == EXIT_OK
    assert report.read_text(encoding="utf-8").startswith("# LLM Evaluation Report")
    assert '"case_metrics"' in js.read_text(encoding="utf-8")


def test_usage_errors(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = _write_suite(workdir)
    assert main(_args(workdir, suite, "--baseline", str(workdir / "nope.json"))) == EXIT_USAGE
    assert main(_args(workdir, suite, "--update-baseline")) == EXIT_USAGE
    assert main(["run", "--suite", str(workdir / "missing.yaml")]) == EXIT_USAGE
    # unknown model with no default price
    (workdir / "pricing.yaml").write_text(yaml.safe_dump({"models": {}}))
    assert main(_args(workdir, suite)) == EXIT_USAGE
    assert "No price configured" in capsys.readouterr().err


def test_committed_ci_example_passes_against_committed_baseline() -> None:
    """The exact invocation CI runs: must stay green on a clean checkout."""
    code = main(
        [
            "run",
            "--suite",
            str(CI_DIR / "suite.yaml"),
            "--baseline",
            str(CI_DIR / "baseline.json"),
            "--thresholds",
            str(CI_DIR / "thresholds.yaml"),
            "--pricing",
            str(CI_DIR / "pricing.yaml"),
        ]
    )
    assert code == EXIT_OK
