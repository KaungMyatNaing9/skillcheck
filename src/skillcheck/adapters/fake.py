"""Deterministic adapters for tests and dry runs.

The fake adapter is not a model. It reads instruction files in the workspace
and applies JSON edits from fenced `skillcheck-fake` blocks. A skill with no
such block leaves the workspace unchanged. That is how the test suite shows
both a detectable effect and a true zero.

The replay adapter plays a JSON transcript the user supplies. It can run
shell commands. The fake adapter cannot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from skillcheck.adapters.actions import apply_actions
from skillcheck.adapters.base import AgentRunResult
from skillcheck.errors import SkillCheckError

_FENCE = re.compile(r"```skillcheck-fake[ \t]*\n(.*?)```", re.DOTALL)
_INSTRUCTION_ROOTS = ("AGENTS.md", "CLAUDE.md", "GEMINI.md", ".cursorrules")


def approximate_tokens(text: str) -> int:
    """Rough character heuristic: about 4 characters per token.

    Used only when a fake or replay transcript does not report usage. It is
    not a model tokenizer and must not be compared to provider invoices.
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


def instruction_files(workspace: Path) -> list[Path]:
    """Instruction files the fake adapter is willing to read, inside the workspace only."""
    found: list[Path] = []
    for name in _INSTRUCTION_ROOTS:
        path = workspace / name
        if path.is_file() and not path.is_symlink():
            found.append(path)
    rules = workspace / ".cursor" / "rules"
    if rules.is_dir():
        found.extend(_files_under(rules))
    skills = workspace / ".claude" / "skills"
    if skills.is_dir():
        found.extend(path for path in _files_under(skills) if path.name == "SKILL.md")
    return found


def parse_fake_actions(text: str, task_id: str) -> list[dict[str, object]]:
    """Collect actions for `task_id` from skillcheck-fake fences.

    A fence with a `tasks` object contributes only that task's actions.
    A fence with a top-level `actions` list applies to every task. When both
    keys are present, `tasks` wins and the top-level list is ignored.
    """
    actions: list[dict[str, object]] = []
    for match in _FENCE.finditer(text):
        body = match.group(1).strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SkillCheckError(f"skillcheck-fake block is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SkillCheckError("skillcheck-fake block must be a JSON object.")
        selected = _select_actions(payload, task_id)
        actions.extend(selected)
    return actions


class FakeAdapter:
    """Apply scripted edits found in the workspace's instruction files."""

    name = "fake"

    def run(
        self,
        *,
        workspace: Path,
        prompt: str,
        task_id: str,
        timeout: float,
    ) -> AgentRunResult:
        files = instruction_files(workspace)
        texts = [path.read_text(encoding="utf-8") for path in files]
        actions: list[dict[str, object]] = []
        notes: list[str] = []
        error: str | None = None
        try:
            for text in texts:
                actions.extend(parse_fake_actions(text, task_id))
            notes = apply_actions(workspace, actions, allow_run=False, timeout=timeout)
        except SkillCheckError as exc:
            notes = []
            error = str(exc)
        skill_text = "\n".join(texts)
        return AgentRunResult(
            tokens_in=approximate_tokens(prompt) + approximate_tokens(skill_text),
            tokens_out=approximate_tokens("\n".join(notes)),
            cost_usd=None,
            turns=1 + len(notes),
            tool_calls=len(notes),
            error=error,
        )


class ReplayAdapter:
    """Replay a JSON transcript of write, replace, and run actions."""

    name = "replay"

    def __init__(self, script_path: Path) -> None:
        try:
            payload = json.loads(script_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillCheckError(f"Could not read replay script {script_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SkillCheckError("Replay script must be a JSON object.")
        self.script_path = script_path
        self.payload = payload

    def run(
        self,
        *,
        workspace: Path,
        prompt: str,
        task_id: str,
        timeout: float,
    ) -> AgentRunResult:
        spec = self._spec_for(task_id)
        actions = spec.get("actions", [])
        if not isinstance(actions, list):
            raise SkillCheckError("Replay actions must be a list.")
        notes = apply_actions(workspace, actions, allow_run=True, timeout=timeout)
        tokens_in = _optional_int(spec.get("tokens_in"), "tokens_in")
        tokens_out = _optional_int(spec.get("tokens_out"), "tokens_out")
        if tokens_in is None and tokens_out is None:
            tokens_in = approximate_tokens(prompt)
            tokens_out = approximate_tokens("\n".join(notes))
        cost = spec.get("cost_usd")
        if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float))):
            raise SkillCheckError("Replay cost_usd must be a number or null.")
        turns = spec.get("turns", 1 + len(notes))
        tool_calls = spec.get("tool_calls", len(notes))
        return AgentRunResult(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=None if cost is None else float(cost),
            turns=_required_int(turns, "turns"),
            tool_calls=_required_int(tool_calls, "tool_calls"),
            error=None,
        )

    def _spec_for(self, task_id: str) -> dict[str, object]:
        tasks = self.payload.get("tasks")
        if isinstance(tasks, dict):
            selected = tasks.get(task_id, self.payload.get("default", {"actions": []}))
            if not isinstance(selected, dict):
                raise SkillCheckError(f"Replay entry for task {task_id} must be an object.")
            return selected
        return self.payload


def _select_actions(payload: dict[str, object], task_id: str) -> list[dict[str, object]]:
    if "tasks" in payload:
        tasks = payload["tasks"]
        if not isinstance(tasks, dict):
            raise SkillCheckError("'tasks' in a skillcheck-fake block must be an object.")
        selected = tasks.get(task_id, {})
        if selected in (None, {}):
            return []
        if not isinstance(selected, dict):
            raise SkillCheckError(f"skillcheck-fake entry for {task_id} must be an object.")
        raw = selected.get("actions", [])
    else:
        raw = payload.get("actions", [])
    if not isinstance(raw, list):
        raise SkillCheckError("'actions' must be a list.")
    actions: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SkillCheckError("Each action must be an object.")
        actions.append(item)
    return actions


def _files_under(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            found.append(path)
    return found


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, field)


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SkillCheckError(f"Replay field {field} must be a non-negative integer.")
    return value
