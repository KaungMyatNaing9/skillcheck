"""Path checks shared by install, sandbox, and agent tools."""

from __future__ import annotations

from pathlib import Path

from skillcheck.errors import SkillCheckError

# Directories that are not part of the task the agent should edit.
# `.skillcheck` holds the harness prompt. VCS and tool caches are not the work.
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".skillcheck",
    }
)


def ensure_workspace_relative(dest: str) -> str:
    """Return a posix path that stays inside a workspace.

    Absolute paths and `..` segments are rejected. The result is used as a
    destination for skill files and as a `hide` entry.
    """
    raw = dest.strip()
    if not raw:
        raise SkillCheckError("A workspace path is empty.")
    if "\\" in raw:
        raise SkillCheckError(f"Workspace path must use '/' separators: {dest!r}")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise SkillCheckError(f"Workspace path escapes the sandbox: {dest!r}")
    normalized = path.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized == ".":
        raise SkillCheckError(f"Workspace path is empty: {dest!r}")
    return normalized


def resolve_inside(workspace: Path, raw: str) -> Path:
    """Resolve `raw` and require the result to stay inside `workspace`.

    Symlinks are followed. A link that points outside the workspace is rejected
    after resolution, so file tools cannot use a link as an escape hatch.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise SkillCheckError("Path is empty.")
    root = workspace.resolve()
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise SkillCheckError(f"Path escapes the workspace: {raw}")
    return resolved
