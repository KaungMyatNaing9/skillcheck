"""Task suite loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillcheck.errors import SkillCheckError
from skillcheck.tasks import load_suite

ROOT = Path(__file__).resolve().parents[1]


def test_builtin_suite_has_hidden_tests() -> None:
    suite = load_suite("builtin")
    assert suite.name == "builtin-tiny"
    assert [task.id for task in suite.tasks] == [
        "fix-inclusive-sum",
        "fix-median",
        "fix-slugify",
    ]
    for task in suite.tasks:
        assert task.workspace is not None and task.workspace.is_dir()
        assert task.hidden is not None and task.hidden.is_dir()
        assert "{python}" in task.verify
        assert task.hide == ()
        hidden_tests = list(task.hidden.rglob("test_*.py"))
        assert hidden_tests
        assert not list(task.workspace.rglob("test_*.py"))


def test_example_suite_loads() -> None:
    suite = load_suite(str(ROOT / "examples" / "minimal-task"))
    assert suite.name == "minimal-example"
    assert suite.tasks[0].id == "fix-add"
    assert suite.tasks[0].workspace is not None
    assert (suite.tasks[0].workspace / "app.py").is_file()


def test_json_suite_round_trip(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    document = {
        "name": "json-suite",
        "description": "from json",
        "tasks": [
            {
                "id": "only-task",
                "workspace": "work",
                "prompt": "Do the thing.",
                "verify": "true",
                "timeout_seconds": 15,
            }
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    suite = load_suite(str(path))
    assert suite.name == "json-suite"
    assert suite.tasks[0].timeout_seconds == 15
    assert suite.tasks[0].prompt == "Do the thing."


def test_directory_without_suite_file_errors(tmp_path: Path) -> None:
    with pytest.raises(SkillCheckError, match=r"suite\.yaml"):
        load_suite(str(tmp_path))


def test_duplicate_ids_error(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: dup
tasks:
  - id: same
    workspace: work
    prompt: one
    verify: "true"
  - id: same
    workspace: work
    prompt: two
    verify: "true"
""",
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="duplicate"):
        load_suite(str(path))


def test_missing_prompt_error(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: bad
tasks:
  - id: missing
    workspace: work
    verify: "true"
""",
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="prompt"):
        load_suite(str(path))


def test_bad_task_id(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: bad
tasks:
  - id: "has space"
    workspace: work
    prompt: go
    verify: "true"
""",
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="invalid id"):
        load_suite(str(path))


def test_hide_path_cannot_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: bad
tasks:
  - id: escape
    workspace: work
    prompt: go
    verify: "true"
    hide: ["../secret"]
""",
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="hide"):
        load_suite(str(path))


def test_workspace_and_repo_together_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: bad
tasks:
  - id: both
    workspace: work
    repo: work
    prompt: go
    verify: "true"
""",
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="not both"):
        load_suite(str(path))


def test_timeout_must_be_positive(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: bad
tasks:
  - id: slow
    workspace: work
    prompt: go
    verify: "true"
    timeout_seconds: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="timeout"):
        load_suite(str(path))


def test_missing_suite_path() -> None:
    with pytest.raises(SkillCheckError, match="not found"):
        load_suite("/no/such/suite.yaml")
