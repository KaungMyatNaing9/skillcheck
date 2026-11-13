"""Isolated copies of a task, with skill files installed only for treatment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from skillcheck.errors import SkillCheckError
from skillcheck.metrics import DiffStats, diff_trees, snapshot_tree
from skillcheck.paths import SKIP_DIR_NAMES, resolve_inside
from skillcheck.process import run_shell
from skillcheck.skills import SkillFile
from skillcheck.tasks import Task

_OUTPUT_TAIL = 2000


@dataclass
class VerifyResult:
    """Outcome of the task's verification command."""

    passed: bool
    exit_code: int | None
    seconds: float
    output_tail: str
    error: str | None


@dataclass
class PreparedWorkspace:
    """A temp copy the agent can edit without touching the source task."""

    root: Path
    path: Path
    vault: Path
    hidden_dir: Path | None
    snapshot: dict[str, bytes]
    keep: bool

    def diff(self) -> DiffStats:
        """Diff agent edits against the snapshot taken after skill install."""
        return diff_trees(self.snapshot, snapshot_tree(self.path))

    def restore_for_verify(self) -> None:
        """Put hidden tests back after the agent has finished."""
        _restore_tree(self.vault, self.path)
        if self.hidden_dir is not None:
            copy_tree_into(self.hidden_dir, self.path)

    def close(self) -> None:
        if self.keep:
            return
        shutil.rmtree(self.root, ignore_errors=True)


def prepare_workspace(
    task: Task,
    *,
    strip: list[SkillFile],
    install: list[SkillFile],
    prompt: str,
    keep: bool = False,
) -> PreparedWorkspace:
    """Copy the task, remove hidden files and skill destinations, then install.

    Baseline and treatment both strip the skill destinations, so the comparison
    is "these files absent" versus "these files present", not "whatever the
    repo already contained" versus an overwrite. Treatment then copies `install`.
    The snapshot used for diff size is taken after that install, so the skill
    files themselves are not counted as agent edits.
    """
    root = Path(tempfile.mkdtemp(prefix="skillcheck-"))
    path = root / "work"
    vault = root / "vault"
    path.mkdir()
    vault.mkdir()
    try:
        if task.workspace is not None:
            copy_tree_into(task.workspace, path)
        elif task.repo is not None:
            git_archive(task.repo, task.rev or "HEAD", path)
        else:
            raise SkillCheckError(f"Task {task.id} has neither workspace nor repo.")
        _move_aside(path, vault, task.hide)
        _strip(path, strip)
        _install(path, install)
        prompt_path = path / ".skillcheck" / "prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        snapshot = snapshot_tree(path)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return PreparedWorkspace(
        root=root,
        path=path,
        vault=vault,
        hidden_dir=task.hidden,
        snapshot=snapshot,
        keep=keep,
    )


def copy_tree_into(source: Path, destination: Path) -> None:
    """Copy regular files from `source` into `destination`, overwriting names.

    Directory symlinks are rejected. A file symlink is copied as bytes only
    when its target stays inside `source`.
    """
    root = source.resolve()
    if not root.is_dir():
        raise SkillCheckError(f"Not a directory: {source}")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        kept: list[str] = []
        for name in sorted(dirnames):
            if name in SKIP_DIR_NAMES:
                continue
            child = current / name
            if child.is_symlink():
                raise SkillCheckError(f"Refusing to copy symlink directory: {child}")
            kept.append(name)
        dirnames[:] = kept
        relative_dir = current.relative_to(root)
        (destination / relative_dir).mkdir(parents=True, exist_ok=True)
        for name in sorted(filenames):
            src = current / name
            target = destination / relative_dir / name
            target.write_bytes(_read_copy_bytes(src, root))


def git_archive(repo: Path, rev: str, destination: Path) -> None:
    """Extract `rev` from a local git repository. Uncommitted files are not included."""
    if shutil.which("git") is None or shutil.which("tar") is None:
        raise SkillCheckError("git and tar are required to materialize a repo task.")
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", rev],
        check=False,
        capture_output=True,
    )
    if archive.returncode != 0:
        detail = archive.stderr.decode("utf-8", "replace").strip()
        raise SkillCheckError(f"git archive {rev} failed for {repo}: {detail}")
    extract = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        check=False,
        capture_output=True,
    )
    if extract.returncode != 0:
        detail = extract.stderr.decode("utf-8", "replace").strip()
        raise SkillCheckError(f"Could not extract git archive: {detail}")


def run_verify(workspace: Path, command: str, timeout: float) -> VerifyResult:
    """Run the task's verification command.

    `{python}` is replaced with this interpreter. Any other text is passed to
    the shell unchanged. The command is trusted: it comes from the task file.
    """
    rendered = command.replace("{python}", _shell_quote(sys.executable))
    started = time.perf_counter()
    try:
        completed = run_shell(rendered, workspace, timeout)
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - started
        tail = _tail(_text(exc.output) + "\n" + _text(exc.stderr))
        return VerifyResult(
            passed=False,
            exit_code=None,
            seconds=seconds,
            output_tail=tail,
            error=f"Verification timed out after {timeout:.0f}s.",
        )
    seconds = time.perf_counter() - started
    tail = _tail(completed.stdout + "\n" + completed.stderr)
    return VerifyResult(
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        seconds=seconds,
        output_tail=tail,
        error=None,
    )


def _read_copy_bytes(src: Path, root: Path) -> bytes:
    if src.is_symlink():
        resolved = src.resolve()
        inside = resolved == root or root in resolved.parents
        if not inside or not resolved.is_file():
            raise SkillCheckError(
                f"Refusing to copy symlink '{src}'. The target must be a file inside the source tree."
            )
        return resolved.read_bytes()
    if not src.is_file():
        raise SkillCheckError(f"Refusing to copy non-file '{src}'.")
    return src.read_bytes()


def _move_aside(workspace: Path, vault: Path, relative_paths: tuple[str, ...]) -> None:
    for relative in relative_paths:
        src = resolve_inside(workspace, relative)
        if not src.exists():
            raise SkillCheckError(f"hide path does not exist in the task workspace: {relative}")
        dest = vault / relative
        if dest.exists():
            raise SkillCheckError(f"hide path collides in the vault: {relative}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


def _restore_tree(vault: Path, workspace: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(vault):
        dirnames.sort()
        for name in sorted(filenames):
            src = Path(dirpath) / name
            if src.is_symlink() or not src.is_file():
                continue
            relative = src.relative_to(vault)
            dest = workspace / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            elif dest.exists() or dest.is_symlink():
                dest.unlink()
            shutil.move(str(src), str(dest))


def _strip(workspace: Path, files: list[SkillFile]) -> None:
    for item in files:
        target = resolve_inside(workspace, item.dest)
        if target.is_symlink() or target.is_file():
            target.unlink()


def _install(workspace: Path, files: list[SkillFile]) -> None:
    for item in files:
        target = resolve_inside(workspace, item.dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.source.read_bytes())


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _tail(text: str) -> str:
    if len(text) <= _OUTPUT_TAIL:
        return text
    return text[-_OUTPUT_TAIL:]
