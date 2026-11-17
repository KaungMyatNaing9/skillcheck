"""Fake and replay adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillcheck.adapters.fake import (
    FakeAdapter,
    ReplayAdapter,
    approximate_tokens,
    parse_fake_actions,
)
from skillcheck.errors import SkillCheckError
from skillcheck.tasks import load_suite

ROOT = Path(__file__).resolve().parents[1]


def test_suite_fix_covers_every_builtin_task_and_null_covers_none() -> None:
    suite = load_suite("builtin")
    fix = (ROOT / "examples" / "skills" / "suite-fix" / "SKILL.md").read_text(encoding="utf-8")
    null = (ROOT / "examples" / "skills" / "null-style" / "SKILL.md").read_text(encoding="utf-8")
    only = (ROOT / "examples" / "skills" / "sum-only" / "SKILL.md").read_text(encoding="utf-8")
    for task in suite.tasks:
        assert parse_fake_actions(fix, task.id), task.id
        assert parse_fake_actions(null, task.id) == []
    assert parse_fake_actions(only, "fix-inclusive-sum")
    assert parse_fake_actions(only, "fix-median") == []
    assert parse_fake_actions(only, "fix-slugify") == []


def test_fake_adapter_applies_only_the_matching_task(tmp_path: Path) -> None:
    (tmp_path / "ranges.py").write_text("for i in range(start, end):\n", encoding="utf-8")
    skill = tmp_path / ".claude" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """
```skillcheck-fake
{"tasks": {"fix-inclusive-sum": {"actions": [
  {"op": "replace", "file": "ranges.py", "find": "range(start, end)", "replace": "range(start, end + 1)"}
]}}}
```
""",
        encoding="utf-8",
    )
    result = FakeAdapter().run(workspace=tmp_path, prompt="fix", task_id="fix-inclusive-sum", timeout=10)
    assert result.error is None
    assert "end + 1" in (tmp_path / "ranges.py").read_text(encoding="utf-8")
    assert result.tool_calls == 1
    assert result.tokens_in is not None and result.tokens_in > approximate_tokens("fix")
    other = tmp_path / "ranges.py"
    other.write_text("for i in range(start, end):\n", encoding="utf-8")
    skipped = FakeAdapter().run(workspace=tmp_path, prompt="fix", task_id="other", timeout=10)
    assert skipped.tool_calls == 0
    assert "end + 1" not in other.read_text(encoding="utf-8")


def test_null_skill_makes_no_edits(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Be tidy.\n", encoding="utf-8")
    result = FakeAdapter().run(workspace=tmp_path, prompt="go", task_id="anything", timeout=10)
    assert result.error is None
    assert result.tool_calls == 0
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    assert result.cost_usd is None


def test_fake_adapter_rejects_shell_and_ignores_files_outside(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("not installed here\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-skill.md"
    # The adapter must not scan the parent. A fence written beside the workspace is ignored.
    outside.write_text(
        '```skillcheck-fake\n{"actions": [{"op": "write", "file": "pwned.txt", "content": "no"}]}\n```\n',
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text(
        """
```skillcheck-fake
{"actions": [{"op": "run", "command": "echo hi"}]}
```
""",
        encoding="utf-8",
    )
    result = FakeAdapter().run(workspace=tmp_path, prompt="go", task_id="t", timeout=10)
    assert result.error is not None
    assert "does not run shell" in result.error
    assert not (tmp_path / "pwned.txt").exists()
    outside.unlink(missing_ok=True)


def test_invalid_fake_fence(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("```skillcheck-fake\n{not json}\n```\n", encoding="utf-8")
    result = FakeAdapter().run(workspace=tmp_path, prompt="go", task_id="t", timeout=10)
    assert result.error is not None
    assert "JSON" in result.error


def test_replay_write_run_cost_and_escape(tmp_path: Path) -> None:
    script = tmp_path / "replay.json"
    script.write_text(
        json.dumps(
            {
                "tokens_in": 3,
                "tokens_out": 4,
                "cost_usd": 0.5,
                "turns": 2,
                "tool_calls": 2,
                "actions": [
                    {"op": "write", "file": "out.txt", "content": "ok\n"},
                    {"op": "run", "command": "python3 -c \"open('ran.txt','w').write('yes')\""},
                ],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "work"
    workspace.mkdir()
    result = ReplayAdapter(script).run(workspace=workspace, prompt="go", task_id="t", timeout=10)
    assert result.error is None
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "ok\n"
    assert (workspace / "ran.txt").read_text(encoding="utf-8") == "yes"
    assert result.tokens_in == 3
    assert result.cost_usd == 0.5
    escape = tmp_path / "escape.json"
    escape.write_text(
        json.dumps({"actions": [{"op": "write", "file": "../escaped.txt", "content": "no"}]}),
        encoding="utf-8",
    )
    with pytest.raises(SkillCheckError, match="escapes"):
        ReplayAdapter(escape).run(workspace=workspace, prompt="go", task_id="t", timeout=10)
    assert not (tmp_path / "escaped.txt").exists()


def test_replay_can_select_a_task(tmp_path: Path) -> None:
    script = tmp_path / "replay.json"
    script.write_text(
        json.dumps(
            {
                "tasks": {
                    "wanted": {"actions": [{"op": "write", "file": "yes.txt", "content": "y"}]},
                    "default": {"actions": []},
                }
            }
        ),
        encoding="utf-8",
    )
    ReplayAdapter(script).run(workspace=tmp_path, prompt="go", task_id="wanted", timeout=10)
    assert (tmp_path / "yes.txt").is_file()


def test_missing_replay_script(tmp_path: Path) -> None:
    with pytest.raises(SkillCheckError, match="replay"):
        ReplayAdapter(tmp_path / "missing.json")


def test_approximate_tokens_is_a_documented_heuristic() -> None:
    assert approximate_tokens("") == 0
    assert approximate_tokens("abcd") == 1
    assert approximate_tokens("abcde") == 2
