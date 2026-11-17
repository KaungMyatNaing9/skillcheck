"""Workspace snapshots and diff sizes."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path

from skillcheck.paths import SKIP_DIR_NAMES

# ndiff on multi-megabyte files is wasted work for a line-count summary.
_LARGE_FILE_BYTES = 200_000


@dataclass(frozen=True)
class DiffStats:
    """Line changes between two snapshots of regular files."""

    files: int
    lines_added: int
    lines_removed: int


def snapshot_tree(root: Path) -> dict[str, bytes]:
    """Map relative posix paths to file bytes.

    Skips the same cache and VCS directories the sandbox copier skips, plus
    `.skillcheck`, so the harness prompt is not counted as an agent edit.
    Symlinks are not followed.
    """
    files: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink() or not path.is_file():
                continue
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def line_delta(before: bytes | None, after: bytes | None) -> tuple[int, int]:
    """Return (added, removed) lines.

    Files larger than 200 KB use a newline-count approximation instead of an
    aligned diff. The report does not treat diff size as a verdict.
    """
    if before == after:
        return 0, 0
    if (before is not None and len(before) > _LARGE_FILE_BYTES) or (
        after is not None and len(after) > _LARGE_FILE_BYTES
    ):
        before_lines = 0 if before is None else before.count(b"\n")
        after_lines = 0 if after is None else after.count(b"\n")
        if after_lines >= before_lines:
            return after_lines - before_lines, 0
        return 0, before_lines - after_lines
    before_text = [] if before is None else before.decode("utf-8", "replace").splitlines()
    after_text = [] if after is None else after.decode("utf-8", "replace").splitlines()
    added = 0
    removed = 0
    for line in difflib.ndiff(before_text, after_text):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return added, removed


def diff_trees(before: dict[str, bytes], after: dict[str, bytes]) -> DiffStats:
    """Compare two snapshots. Identical files contribute nothing."""
    files = 0
    added = 0
    removed = 0
    for key in set(before) | set(after):
        left = before.get(key)
        right = after.get(key)
        if left == right:
            continue
        file_added, file_removed = line_delta(left, right)
        files += 1
        added += file_added
        removed += file_removed
    return DiffStats(files=files, lines_added=added, lines_removed=removed)
