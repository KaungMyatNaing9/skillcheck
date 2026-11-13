"""End-to-end studies on the built-in suite with the fake adapter.

These tests prove the pipeline, not a live model. The suite-fix skill contains
scripted edits the fake adapter applies. The null skill does not. A real
provider is not contacted.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from skillcheck.cli import main
from skillcheck.runner import RunConfig, run_evaluation

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "examples" / "skills" / "suite-fix"
NULL = ROOT / "examples" / "skills" / "null-style"
SUM_ONLY = ROOT / "examples" / "skills" / "sum-only"


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _study(skill: Path, out: Path | None = None) -> dict[str, object]:
    return run_evaluation(
        RunConfig(
            tasks="builtin",
            skills=[str(skill)],
            agent="fake",
            trials=4,
            seed=0,
            bootstrap_samples=1000,
            output_dir=out,
        )
    )


def test_tasks_list_and_cli_usage() -> None:
    assert main(["tasks", "list"]) == 0
    assert main(["run", "--tasks", "builtin", "--agent", "fake", "--trials", "2"]) == 2
    assert main(["--version"]) == 0
    code = main(
        [
            "run",
            "--tasks",
            "builtin",
            "--skill",
            str(NULL),
            "--agent",
            "not-an-agent",
            "--trials",
            "2",
            "--bootstrap",
            "200",
            "--out",
            "/tmp/skillcheck-should-not-run",
        ]
    )
    assert code == 1


def test_suite_fix_is_detected_as_helps(tmp_path: Path) -> None:
    fixtures = ROOT / "src" / "skillcheck" / "fixtures"
    examples = ROOT / "examples"
    before = (_hash_tree(fixtures), _hash_tree(examples))
    document = _study(FIX, tmp_path / "out")
    assert _hash_tree(fixtures) == before[0]
    assert _hash_tree(examples) == before[1]
    assert document["verdict"]["label"] == "helps"
    assert document["verdict"]["outlier_driven"] is False
    assert document["overall"]["pass_rate_difference"]["estimate"] == 1
    assert document["overall"]["pass_rate_difference"]["ci_low"] == 1
    assert document["overall"]["cost_difference"]["available"] is False
    assert document["badge"]["color"] == "brightgreen"
    assert "pp pass rate" in document["badge"]["message"]
    runs = document["runs"]
    assert isinstance(runs, list)
    assert len(runs) == 3 * 4 * 2
    baseline = [run for run in runs if run["condition"] == "baseline"]
    treatment = [run for run in runs if run["condition"] == "treatment"]
    assert all(run["passed"] is False for run in baseline)
    assert all(run["passed"] is True for run in treatment)
    assert all(run["diff_lines_added"] + run["diff_lines_removed"] == 0 for run in baseline)
    assert all(run["diff_lines_added"] + run["diff_lines_removed"] > 0 for run in treatment)
    assert any("zero width" in warning for warning in document["warnings"])
    written = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert written["verdict"]["label"] == "helps"
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8").startswith("# SkillCheck report")
    badge = json.loads((tmp_path / "out" / "badge.json").read_text(encoding="utf-8"))
    assert badge == document["badge"]


def test_null_skill_has_no_detectable_effect(tmp_path: Path) -> None:
    out = tmp_path / "null"
    code = main(
        [
            "run",
            "--tasks",
            "builtin",
            "--skill",
            str(NULL),
            "--agent",
            "fake",
            "--trials",
            "4",
            "--bootstrap",
            "1000",
            "--seed",
            "0",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    document = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert document["verdict"]["label"] == "no detectable effect"
    assert document["verdict"]["outlier_driven"] is False
    assert document["overall"]["pass_rate_difference"]["estimate"] == 0
    assert document["badge"]["color"] == "blue"
    assert str(document["badge"]["message"]).startswith("no detectable effect")
    runs = document["runs"]
    assert all(run["passed"] is False for run in runs)
    assert all(run["diff_lines_added"] + run["diff_lines_removed"] == 0 for run in runs)
    report = main(["report", str(out / "results.json"), "--badge", str(tmp_path / "badge.json")])
    assert report == 0
    recomputed = json.loads((tmp_path / "badge.json").read_text(encoding="utf-8"))
    assert recomputed == document["badge"]


def test_single_task_skill_is_outlier_driven() -> None:
    document = _study(SUM_ONLY)
    assert document["verdict"]["label"] == "inconclusive"
    assert document["verdict"]["unadjusted_label"] == "helps"
    assert document["verdict"]["outlier_driven"] is True
    assert document["verdict"]["extreme_task_id"] == "fix-inclusive-sum"
    assert str(document["badge"]["message"]).startswith("inconclusive")
    assert document["badge"]["color"] == "yellow"
    by_task: dict[str, list[bool]] = {}
    runs = document["runs"]
    assert isinstance(runs, list)
    for run in runs:
        if run["condition"] == "treatment":
            by_task.setdefault(run["task_id"], []).append(run["passed"])
    assert by_task["fix-inclusive-sum"] == [True, True, True, True]
    assert by_task["fix-median"] == [False, False, False, False]
    assert by_task["fix-slugify"] == [False, False, False, False]


def test_user_suite_and_module_entrypoint(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        """
```skillcheck-fake
{"tasks": {"fix-add": {"actions": [
  {"op": "replace", "file": "app.py", "find": "return left - right", "replace": "return left + right"}
]}}}
```
""",
        encoding="utf-8",
    )
    out = tmp_path / "user"
    code = main(
        [
            "run",
            "--tasks",
            str(ROOT / "examples" / "minimal-task"),
            "--skill",
            str(skill),
            "--agent",
            "fake",
            "--trials",
            "2",
            "--bootstrap",
            "200",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    document = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert document["verdict"]["label"] == "helps"
    assert all(run["passed"] is False for run in document["runs"] if run["condition"] == "baseline")
    assert all(run["passed"] is True for run in document["runs"] if run["condition"] == "treatment")
    completed = subprocess.run(
        [sys.executable, "-m", "skillcheck", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "skillcheck 0.1.0" in completed.stdout
