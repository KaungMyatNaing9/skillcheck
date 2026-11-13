"""Sandbox isolation, hidden tests, and skill installation."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from skillcheck.errors import SkillCheckError
from skillcheck.sandbox import copy_tree_into, prepare_workspace, run_verify
from skillcheck.skills import SkillFile, resolve_skills
from skillcheck.tasks import load_suite


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_source_is_unchanged_and_sandboxes_are_separate() -> None:
    task = load_suite("builtin").tasks[0]
    assert task.workspace is not None
    before = _hash_tree(task.workspace)
    first = prepare_workspace(task, strip=[], install=[], prompt="one")
    second = prepare_workspace(task, strip=[], install=[], prompt="two")
    try:
        assert first.path != second.path
        (first.path / "ranges.py").write_text("broken\n", encoding="utf-8")
        assert "def sum_inclusive" in (second.path / "ranges.py").read_text(encoding="utf-8")
    finally:
        first.close()
        second.close()
    assert _hash_tree(task.workspace) == before
    assert not first.path.exists()


def test_hidden_tests_stay_out_until_verify() -> None:
    task = load_suite("builtin").tasks[0]
    prepared = prepare_workspace(task, strip=[], install=[], prompt="fix it")
    try:
        assert not (prepared.path / "tests").exists()
        prepared.restore_for_verify()
        assert (prepared.path / "tests" / "test_ranges.py").is_file()
    finally:
        prepared.close()


def test_skill_is_stripped_from_baseline_and_installed_for_treatment(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("old instructions\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: local
tasks:
  - id: local-task
    workspace: work
    prompt: Do nothing.
    verify: "true"
""",
        encoding="utf-8",
    )
    skill = tmp_path / "AGENTS.md"
    skill.write_text("new instructions\n", encoding="utf-8")
    task = load_suite(str(suite)).tasks[0]
    files = resolve_skills([str(skill)])
    baseline = prepare_workspace(task, strip=files, install=[], prompt="x")
    treatment = prepare_workspace(task, strip=files, install=files, prompt="x")
    try:
        assert not (baseline.path / "AGENTS.md").exists()
        assert (treatment.path / "AGENTS.md").read_text(encoding="utf-8") == "new instructions\n"
        (treatment.path / "app.py").write_text("print('changed')\n", encoding="utf-8")
        diff = treatment.diff()
        assert diff.files == 1
        assert "AGENTS.md" not in treatment.snapshot or treatment.snapshot["AGENTS.md"] == b"new instructions\n"
    finally:
        baseline.close()
        treatment.close()


def test_hide_list_round_trip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text("assert True\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
name: hidden
tasks:
  - id: hide-tests
    workspace: repo
    prompt: Work.
    verify: "true"
    hide: ["tests"]
""",
        encoding="utf-8",
    )
    task = load_suite(str(suite)).tasks[0]
    prepared = prepare_workspace(task, strip=[], install=[], prompt="x")
    try:
        assert not (prepared.path / "tests").exists()
        assert (prepared.path / "pkg" / "app.py").is_file()
        prepared.restore_for_verify()
        assert (prepared.path / "tests" / "test_app.py").is_file()
    finally:
        prepared.close()


def test_git_archive_honors_revision_and_hide(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "secret_test.py").write_text("assert False\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py", "secret_test.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=repo, check=True, capture_output=True)
    first = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "second"], cwd=repo, check=True, capture_output=True)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        f"""
name: git
tasks:
  - id: old-rev
    repo: {repo}
    rev: {first}
    prompt: Look at the code.
    verify: "true"
    hide: ["secret_test.py"]
""",
        encoding="utf-8",
    )
    task = load_suite(str(suite)).tasks[0]
    prepared = prepare_workspace(task, strip=[], install=[], prompt="x")
    try:
        assert prepared.path.joinpath("app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert not (prepared.path / "secret_test.py").exists()
        assert not (prepared.path / ".git").exists()
        prepared.restore_for_verify()
        assert (prepared.path / "secret_test.py").is_file()
    finally:
        prepared.close()


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (source / "link.txt").symlink_to(outside)
    destination = tmp_path / "dest"
    destination.mkdir()
    with pytest.raises(SkillCheckError, match="symlink"):
        copy_tree_into(source, destination)
    assert not (destination / "link.txt").exists()


def test_builtin_task_fails_before_the_fix() -> None:
    task = load_suite("builtin").tasks[0]
    prepared = prepare_workspace(task, strip=[], install=[], prompt=task.prompt)
    try:
        prepared.restore_for_verify()
        result = run_verify(prepared.path, task.verify, timeout=30)
        assert result.passed is False
        assert result.exit_code not in (None, 0)
    finally:
        prepared.close()


def test_verify_python_placeholder(tmp_path: Path) -> None:
    result = run_verify(tmp_path, '{python} -c "print(123)"', timeout=30)
    assert result.passed is True
    assert "123" in result.output_tail


def test_workspace_copy_skips_git_dir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    (source / ".git" / "config").write_text("git\n", encoding="utf-8")
    (source / "app.py").write_text("print(1)\n", encoding="utf-8")
    destination = tmp_path / "dest"
    copy_tree_into(source, destination)
    assert (destination / "app.py").is_file()
    assert not (destination / ".git").exists()


def test_install_rejects_dest_escape_via_skillfile(tmp_path: Path) -> None:
    source = tmp_path / "AGENTS.md"
    source.write_text("x\n", encoding="utf-8")
    with pytest.raises(SkillCheckError):
        SkillFile(source, "../AGENTS.md")
        resolve_skills([f"{source}::../AGENTS.md"])
