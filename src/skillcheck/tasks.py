"""Load a task suite from YAML or JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from skillcheck.errors import SkillCheckError
from skillcheck.paths import ensure_workspace_relative

_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_SUITE_FILENAMES = ("suite.yaml", "suite.yml", "suite.json")
_DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class Task:
    """One evaluation task.

    `workspace` plus optional `hidden` is a fixture directory. `repo` plus
    `rev` is a git snapshot. `hide` lists paths inside that snapshot that are
    removed before the agent runs and restored before verification.
    """

    id: str
    prompt: str
    verify: str
    timeout_seconds: float
    workspace: Path | None
    hidden: Path | None
    repo: Path | None
    rev: str | None
    hide: tuple[str, ...]


@dataclass(frozen=True)
class Suite:
    """A named list of tasks."""

    name: str
    description: str
    source: Path
    tasks: tuple[Task, ...]


def builtin_suite_path() -> Path:
    """Path to the packaged fixture suite."""
    return Path(__file__).resolve().parent / "fixtures" / "suite.yaml"


def load_suite(spec: str) -> Suite:
    """Load `builtin`, a suite file, or a directory that contains one."""
    if spec == "builtin":
        path = builtin_suite_path()
    else:
        path = Path(spec)
        if not path.exists():
            raise SkillCheckError(f"Task suite not found: {spec}")
        if path.is_dir():
            path = _find_suite_file(path)
    if not path.is_file():
        raise SkillCheckError(f"Task suite is not a file: {path}")
    return _load_suite_file(path)


def _find_suite_file(directory: Path) -> Path:
    for name in _SUITE_FILENAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise SkillCheckError(f"No suite.yaml, suite.yml, or suite.json in {directory}.")


def _load_suite_file(path: Path) -> Suite:
    document = _read_document(path)
    name = document.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise SkillCheckError(f"{path}: suite name must be a non-empty string.")
    description = document.get("description", "")
    if not isinstance(description, str):
        raise SkillCheckError(f"{path}: description must be a string.")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SkillCheckError(f"{path}: 'tasks' must be a non-empty list.")
    tasks = tuple(_load_task(path, item, index) for index, item in enumerate(raw_tasks, start=1))
    ids = [task.id for task in tasks]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if duplicates:
        raise SkillCheckError(f"{path}: duplicate task ids: {', '.join(duplicates)}.")
    return Suite(name=name.strip(), description=description.strip(), source=path.resolve(), tasks=tasks)


def _load_task(suite_path: Path, item: object, index: int) -> Task:
    where = f"{suite_path}: task {index}"
    if not isinstance(item, dict):
        raise SkillCheckError(f"{where} must be an object.")
    task_id = item.get("id")
    if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
        raise SkillCheckError(f"{where} has an invalid id. Use 1-81 characters: letters, digits, '.', '_', '-'.")
    where = f"{suite_path}: task {task_id}"
    prompt = _require_text(item, "prompt", where)
    verify = _require_text(item, "verify", where)
    timeout = _timeout(item.get("timeout_seconds", _DEFAULT_TIMEOUT), where)
    workspace = _optional_dir(suite_path.parent, item.get("workspace"), where, "workspace")
    hidden = _optional_dir(suite_path.parent, item.get("hidden"), where, "hidden")
    repo = _optional_dir(suite_path.parent, item.get("repo"), where, "repo")
    rev = item.get("rev")
    if rev is not None and not isinstance(rev, str):
        raise SkillCheckError(f"{where}: rev must be a string.")
    if rev is not None and not rev.strip():
        raise SkillCheckError(f"{where}: rev is empty.")
    if workspace is not None and repo is not None:
        raise SkillCheckError(f"{where}: set workspace or repo, not both.")
    if workspace is None and repo is None:
        raise SkillCheckError(f"{where}: set workspace or repo.")
    if rev is not None and repo is None:
        raise SkillCheckError(f"{where}: rev requires repo.")
    hide = _hide_list(item.get("hide", []), where)
    return Task(
        id=task_id,
        prompt=prompt,
        verify=verify,
        timeout_seconds=timeout,
        workspace=workspace,
        hidden=hidden,
        repo=repo,
        rev=None if rev is None else rev.strip(),
        hide=hide,
    )


def _require_text(item: dict[str, object], key: str, where: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillCheckError(f"{where}: {key} must be a non-empty string.")
    return value.strip()


def _timeout(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SkillCheckError(f"{where}: timeout_seconds must be a positive number.")
    return float(value)


def _optional_dir(base: Path, value: object, where: str, key: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SkillCheckError(f"{where}: {key} must be a path string.")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    if not resolved.exists():
        raise SkillCheckError(f"{where}: {key} does not exist: {resolved}")
    if not resolved.is_dir():
        raise SkillCheckError(f"{where}: {key} is not a directory: {resolved}")
    return resolved


def _hide_list(value: object, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SkillCheckError(f"{where}: hide must be a list of relative paths.")
    hidden: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise SkillCheckError(f"{where}: hide entries must be strings.")
        try:
            hidden.append(ensure_workspace_relative(entry))
        except SkillCheckError as exc:
            raise SkillCheckError(f"{where}: hide entry {entry!r} is not allowed. {exc}") from exc
    return tuple(hidden)


def _read_document(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            loaded = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(text)
        else:
            raise SkillCheckError(f"{path}: suite file must be .yaml, .yml, or .json.")
    except json.JSONDecodeError as exc:
        raise SkillCheckError(f"{path}: invalid JSON: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SkillCheckError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SkillCheckError(f"{path}: the suite file must contain an object.")
    return loaded
