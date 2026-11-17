"""Command adapter templating and metrics trailer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from skillcheck.adapters.command import CommandAdapter, render_command
from skillcheck.errors import SkillCheckError
from skillcheck.runner import RunConfig, run_evaluation


def test_placeholders_are_shell_quoted(tmp_path: Path) -> None:
    rendered = render_command(
        "echo {prompt} {prompt_file} {workspace}",
        prompt="hello 'world'",
        prompt_file=tmp_path / "prompt.txt",
        workspace=tmp_path,
    )
    assert "hello 'world'" not in rendered
    assert "'" in rendered
    assert str(tmp_path) in rendered


def test_metrics_trailer_and_failed_command(tmp_path: Path) -> None:
    payload = {"tokens_in": 3, "tokens_out": 4, "turns": 2, "tool_calls": 1, "cost_usd": 0.25}
    script = tmp_path / "ok.py"
    script.write_text(
        "import json\nprint('SKILLCHECK_METRICS ' + json.dumps(" + repr(payload) + "))\n",
        encoding="utf-8",
    )
    result = CommandAdapter(f"{sys.executable} {script}").run(workspace=tmp_path, prompt="go", task_id="t", timeout=30)
    assert result.error is None
    assert result.tokens_in == 3
    assert result.tokens_out == 4
    assert result.turns == 2
    assert result.cost_usd == 0.25
    failing = CommandAdapter(f"{sys.executable} -c \"import sys; sys.stderr.write('boom'); sys.exit(3)\"")
    failed = failing.run(workspace=tmp_path, prompt="go", task_id="t", timeout=30)
    assert failed.error is not None
    assert "status 3" in failed.error
    assert "boom" in failed.error


def test_invalid_metrics_and_bool_token(tmp_path: Path) -> None:
    bad_script = tmp_path / "bad.py"
    bad_script.write_text("print('SKILLCHECK_METRICS not-json')\n", encoding="utf-8")
    with pytest.raises(SkillCheckError, match="JSON"):
        CommandAdapter(f"{sys.executable} {bad_script}").run(workspace=tmp_path, prompt="go", task_id="t", timeout=30)
    bool_script = tmp_path / "bool.py"
    bool_script.write_text(
        "print('SKILLCHECK_METRICS {\"tokens_in\": true}')\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="integer"):
        CommandAdapter(f"{sys.executable} {bool_script}").run(workspace=tmp_path, prompt="go", task_id="t", timeout=30)


def test_command_timeout(tmp_path: Path) -> None:
    result = CommandAdapter(f'{sys.executable} -c "import time; time.sleep(30)"').run(
        workspace=tmp_path, prompt="go", task_id="t", timeout=1
    )
    assert result.error is not None
    assert "timed out" in result.error


def test_command_adapter_changes_outcome_when_skill_is_present(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "src" / "skillcheck" / "fixtures" / "fix-inclusive-sum"
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        f"""
name: command-demo
tasks:
  - id: fix-inclusive-sum
    workspace: {fixture / "workspace"}
    hidden: {fixture / "hidden"}
    prompt: Fix the inclusive sum.
    verify: "{{python}} -m unittest discover -s tests -q"
    timeout_seconds: 30
""",
        encoding="utf-8",
    )
    skill = tmp_path / "AGENTS.md"
    skill.write_text("fix the range\n", encoding="utf-8")
    script = tmp_path / "agent.py"
    script.write_text(
        """
from pathlib import Path
import json
if Path("AGENTS.md").exists():
    path = Path("ranges.py")
    path.write_text(path.read_text().replace("range(start, end)", "range(start, end + 1)"))
print("SKILLCHECK_METRICS " + json.dumps({
    "tokens_in": 11, "tokens_out": 7, "turns": 2, "tool_calls": 1, "cost_usd": 0.25
}))
""",
        encoding="utf-8",
    )
    document = run_evaluation(
        RunConfig(
            tasks=str(suite),
            skills=[str(skill)],
            agent="command",
            trials=2,
            bootstrap_samples=200,
            agent_options={"command": f"{sys.executable} {script}"},
            output_dir=None,
        )
    )
    runs = document["runs"]
    assert isinstance(runs, list)
    baseline = [run for run in runs if run["condition"] == "baseline"]
    treatment = [run for run in runs if run["condition"] == "treatment"]
    assert baseline and all(run["passed"] is False for run in baseline)
    assert treatment and all(run["passed"] is True for run in treatment)
    assert document["verdict"]["label"] == "helps"
    assert document["overall"]["cost_difference"]["available"] is True
    assert document["overall"]["cost_difference"]["estimate"] == 0
