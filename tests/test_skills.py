"""Skill install-path inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillcheck.errors import SkillCheckError
from skillcheck.skills import resolve_skills


def test_agents_claude_and_skill_md(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("agents\n", encoding="utf-8")
    claude = tmp_path / "CLAUDE.md"
    claude.write_text("claude\n", encoding="utf-8")
    skill_dir = tmp_path / "careful"
    skill_dir.mkdir()
    skill = skill_dir / "SKILL.md"
    skill.write_text("skill\n", encoding="utf-8")
    resolved = {item.dest: item.source for item in resolve_skills([str(agents), str(claude), str(skill)])}
    assert resolved["AGENTS.md"] == agents.resolve()
    assert resolved["CLAUDE.md"] == claude.resolve()
    assert resolved[".claude/skills/careful/SKILL.md"] == skill.resolve()


def test_directory_installs_skill_and_sibling(tmp_path: Path) -> None:
    skill_dir = tmp_path / "guide"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("body\n", encoding="utf-8")
    (skill_dir / "NOTES.md").write_text("notes\n", encoding="utf-8")
    dests = {item.dest for item in resolve_skills([str(skill_dir)])}
    assert dests == {".claude/skills/guide/SKILL.md", ".claude/skills/guide/NOTES.md"}


def test_directory_keeps_root_files_and_cursor_rules(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / ".cursor" / "rules").mkdir(parents=True)
    (root / "AGENTS.md").write_text("a\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("c\n", encoding="utf-8")
    (root / "SKILL.md").write_text("s\n", encoding="utf-8")
    (root / ".cursor" / "rules" / "style.mdc").write_text("rule\n", encoding="utf-8")
    dests = {item.dest for item in resolve_skills([str(root)])}
    assert "AGENTS.md" in dests
    assert "CLAUDE.md" in dests
    assert ".claude/skills/bundle/SKILL.md" in dests
    assert ".cursor/rules/style.mdc" in dests
    assert not any(dest.startswith(".claude/skills/bundle/.cursor") for dest in dests)


def test_manifest_and_explicit_dest(tmp_path: Path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text("hello\n", encoding="utf-8")
    manifest = tmp_path / "skillcheck.yaml"
    manifest.write_text(
        "install:\n  - source: guide.md\n    dest: AGENTS.md\n",
        encoding="utf-8",
    )
    from_manifest = resolve_skills([str(manifest)])
    assert [(item.dest, item.source.name) for item in from_manifest] == [("AGENTS.md", "guide.md")]
    explicit = resolve_skills([f"{guide}::.cursor/rules/guide.md"])
    assert explicit[0].dest == ".cursor/rules/guide.md"


def test_mdc_and_cursorrules(tmp_path: Path) -> None:
    rule = tmp_path / "style.mdc"
    rule.write_text("rule\n", encoding="utf-8")
    cursorrules = tmp_path / ".cursorrules"
    cursorrules.write_text("old\n", encoding="utf-8")
    dests = {item.dest for item in resolve_skills([str(rule), str(cursorrules)])}
    assert dests == {".cursor/rules/style.mdc", ".cursorrules"}


def test_unknown_file_and_traversal(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("nope\n", encoding="utf-8")
    with pytest.raises(SkillCheckError, match="Cannot infer"):
        resolve_skills([str(notes)])
    with pytest.raises(SkillCheckError, match="escapes"):
        resolve_skills([f"{notes}::../outside.md"])
    with pytest.raises(SkillCheckError, match="escapes"):
        resolve_skills([f"{notes}::/etc/passwd"])


def test_conflicting_destinations(tmp_path: Path) -> None:
    first = tmp_path / "one.md"
    second = tmp_path / "two.md"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    with pytest.raises(SkillCheckError, match="Two skill files"):
        resolve_skills([f"{first}::AGENTS.md", f"{second}::AGENTS.md"])


def test_directory_explicit_prefix(tmp_path: Path) -> None:
    folder = tmp_path / "rules"
    folder.mkdir()
    (folder / "a.mdc").write_text("a\n", encoding="utf-8")
    (folder / "b.mdc").write_text("b\n", encoding="utf-8")
    dests = sorted(item.dest for item in resolve_skills([f"{folder}::.cursor/rules"]))
    assert dests == [".cursor/rules/a.mdc", ".cursor/rules/b.mdc"]


def test_sha256_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("one\n", encoding="utf-8")
    first = resolve_skills([str(path)])[0].sha256
    path.write_text("two\n", encoding="utf-8")
    second = resolve_skills([str(path)])[0].sha256
    assert first != second
