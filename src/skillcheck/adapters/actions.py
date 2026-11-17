"""Apply a small set of file actions inside a workspace.

The fake adapter refuses `run`. A skill file must not be able to start a shell
just by being installed. Replay transcripts are local files the user chose, so
they may include `run`.
"""

from __future__ import annotations

from pathlib import Path

from skillcheck.errors import SkillCheckError
from skillcheck.paths import resolve_inside
from skillcheck.process import run_shell


def apply_actions(
    workspace: Path,
    actions: list[object],
    *,
    allow_run: bool,
    timeout: float,
) -> list[str]:
    """Apply actions in order. Return a short description of each one."""
    if not isinstance(actions, list):
        raise SkillCheckError("actions must be a list.")
    notes: list[str] = []
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            raise SkillCheckError(f"Action {index} must be an object.")
        op = action.get("op")
        if op == "replace":
            notes.append(_replace(workspace, action, index))
        elif op == "write":
            notes.append(_write(workspace, action, index))
        elif op == "run":
            if not allow_run:
                raise SkillCheckError(
                    "The fake adapter does not run shell commands from a skill. Use write or replace."
                )
            notes.append(_run(workspace, action, index, timeout))
        else:
            raise SkillCheckError(f"Action {index} has unknown op {op!r}. Use write, replace, or run.")
    return notes


def _replace(workspace: Path, action: dict[str, object], index: int) -> str:
    relative = _require_text(action, "file", index)
    find = _require_text(action, "find", index)
    if "replace" not in action or not isinstance(action["replace"], str):
        raise SkillCheckError(f"Action {index} needs a string 'replace' value.")
    replacement = action["replace"]
    path = resolve_inside(workspace, relative)
    if not path.is_file():
        raise SkillCheckError(f"Action {index} could not find file {relative}.")
    text = path.read_text(encoding="utf-8")
    if find not in text:
        raise SkillCheckError(f"Action {index} did not find the expected text in {relative}.")
    path.write_text(text.replace(find, replacement), encoding="utf-8")
    return f"replace {relative}"


def _write(workspace: Path, action: dict[str, object], index: int) -> str:
    relative = _require_text(action, "file", index)
    content = action.get("content")
    if not isinstance(content, str):
        raise SkillCheckError(f"Action {index} needs a string 'content' value.")
    path = resolve_inside(workspace, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"write {relative}"


def _run(workspace: Path, action: dict[str, object], index: int, timeout: float) -> str:
    command = _require_text(action, "command", index)
    completed = run_shell(command, workspace, timeout)
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[:500]
        raise SkillCheckError(f"Action {index} command exited {completed.returncode}: {detail}")
    return f"run {command}"


def _require_text(action: dict[str, object], key: str, index: int) -> str:
    value = action.get(key)
    if not isinstance(value, str) or value == "":
        raise SkillCheckError(f"Action {index} needs a non-empty string '{key}'.")
    return value
