"""Decide where a skill file is installed inside a task workspace."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from skillcheck.errors import SkillCheckError
from skillcheck.paths import SKIP_DIR_NAMES, ensure_workspace_relative

_ROOT_FILES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md", ".cursorrules")
_MANIFEST_NAMES = ("skillcheck.yaml", "skillcheck.yml", "skillcheck.json")


@dataclass(frozen=True)
class SkillFile:
    """One file copied into the treatment workspace."""

    source: Path
    dest: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source.read_bytes()).hexdigest()


def resolve_skills(specs: list[str]) -> list[SkillFile]:
    """Resolve CLI `--skill` values into concrete file copies.

    A spec is a path, or `path::relative/dest`. Destinations are inferred for
    AGENTS.md, CLAUDE.md, GEMINI.md, .cursorrules, SKILL.md, `.mdc` rule files,
    skill directories, and `skillcheck.yaml` manifests.
    """
    if not specs:
        raise SkillCheckError("Pass at least one --skill.")
    resolved: list[SkillFile] = []
    for spec in specs:
        resolved.extend(_resolve_one(spec))
    return _dedupe(resolved)


def _resolve_one(spec: str) -> list[SkillFile]:
    explicit_dest: str | None = None
    raw_path = spec
    if "::" in spec:
        raw_path, explicit_dest = spec.split("::", 1)
        if not raw_path.strip() or not explicit_dest.strip():
            raise SkillCheckError(f"Skill spec must be PATH or PATH::DEST, got {spec!r}.")
        explicit_dest = ensure_workspace_relative(explicit_dest)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise SkillCheckError(f"Skill path does not exist: {path}")
    if explicit_dest is not None:
        return _resolve_explicit(path, explicit_dest)
    if path.is_file():
        return _resolve_file(path)
    if path.is_dir():
        return _resolve_directory(path)
    raise SkillCheckError(f"Skill path is not a file or directory: {path}")


def _resolve_explicit(path: Path, dest: str) -> list[SkillFile]:
    if path.is_file():
        return [SkillFile(path, dest)]
    if not path.is_dir():
        raise SkillCheckError(f"Skill path is not a file or directory: {path}")
    files = _iter_files(path)
    if not files:
        raise SkillCheckError(f"Skill directory is empty: {path}")
    return [
        SkillFile(source, ensure_workspace_relative(f"{dest}/{source.relative_to(path).as_posix()}"))
        for source in files
    ]


def _resolve_file(path: Path) -> list[SkillFile]:
    name = path.name
    if name in _ROOT_FILES:
        return [SkillFile(path, name)]
    if name == "SKILL.md":
        parent = path.parent.name or "skill"
        return [SkillFile(path, f".claude/skills/{parent}/SKILL.md")]
    if name.endswith(".mdc"):
        return [SkillFile(path, f".cursor/rules/{name}")]
    if name in _MANIFEST_NAMES:
        return _parse_manifest(path)
    raise SkillCheckError(
        f"Cannot infer where to install {name}. "
        "Pass PATH::DEST, or use AGENTS.md, CLAUDE.md, GEMINI.md, .cursorrules, "
        "SKILL.md, a .mdc rule, a skill directory, or a skillcheck.yaml manifest."
    )


def _resolve_directory(path: Path) -> list[SkillFile]:
    for name in _MANIFEST_NAMES:
        manifest = path / name
        if manifest.is_file():
            return _parse_manifest(manifest)
    found: list[SkillFile] = []
    skill_members: list[Path] = []
    for source in _iter_files(path):
        relative = source.relative_to(path).as_posix()
        parts = Path(relative).parts
        if len(parts) == 1 and parts[0] in _ROOT_FILES:
            found.append(SkillFile(source, parts[0]))
            continue
        if len(parts) == 1 and parts[0] in _MANIFEST_NAMES:
            continue
        if len(parts) >= 3 and parts[0] == ".cursor" and parts[1] == "rules":
            found.append(SkillFile(source, relative))
            continue
        skill_members.append(source)
    if (path / "SKILL.md").is_file():
        prefix = f".claude/skills/{path.name}"
        for source in skill_members:
            relative = source.relative_to(path).as_posix()
            found.append(SkillFile(source, f"{prefix}/{relative}"))
    elif skill_members:
        names = ", ".join(source.relative_to(path).as_posix() for source in skill_members[:5])
        raise SkillCheckError(
            f"Cannot infer install paths for {path}. Unrecognized files include: {names}. "
            "Add SKILL.md, a known instruction filename, or a skillcheck.yaml manifest."
        )
    if not found:
        raise SkillCheckError(
            f"No installable skill files in {path}. "
            "Expected SKILL.md, AGENTS.md, CLAUDE.md, GEMINI.md, .cursorrules, "
            ".cursor/rules, or a skillcheck.yaml manifest."
        )
    return found


def _parse_manifest(path: Path) -> list[SkillFile]:
    document = _read_manifest(path)
    installs = document.get("install")
    if not isinstance(installs, list) or not installs:
        raise SkillCheckError(f"{path}: 'install' must be a non-empty list.")
    files: list[SkillFile] = []
    base = path.parent
    for index, item in enumerate(installs, start=1):
        if not isinstance(item, dict):
            raise SkillCheckError(f"{path}: install item {index} must be an object.")
        source_raw = item.get("source")
        dest_raw = item.get("dest")
        if not isinstance(source_raw, str) or not source_raw.strip():
            raise SkillCheckError(f"{path}: install item {index} needs a source path.")
        if not isinstance(dest_raw, str) or not dest_raw.strip():
            raise SkillCheckError(f"{path}: install item {index} needs a dest path.")
        source = Path(source_raw)
        if not source.is_absolute():
            source = base / source
        source = source.resolve()
        if not source.is_file():
            raise SkillCheckError(f"{path}: source is not a file: {source}")
        files.append(SkillFile(source, ensure_workspace_relative(dest_raw)))
    return files


def _read_manifest(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SkillCheckError(f"{path}: cannot parse manifest: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SkillCheckError(f"{path}: manifest must be an object.")
    return loaded


def _iter_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        current = Path(dirpath)
        for name in dirnames:
            if (current / name).is_symlink():
                raise SkillCheckError(f"Refusing to install symlink directory: {current / name}")
        for name in sorted(filenames):
            path = current / name
            if path.is_symlink():
                raise SkillCheckError(f"Refusing to install symlink: {path}")
            if path.is_file():
                found.append(path)
    return found


def _dedupe(files: list[SkillFile]) -> list[SkillFile]:
    by_dest: dict[str, SkillFile] = {}
    for item in files:
        previous = by_dest.get(item.dest)
        if previous is None:
            by_dest[item.dest] = item
            continue
        if previous.source.read_bytes() != item.source.read_bytes():
            raise SkillCheckError(f"Two skill files install to {item.dest}: {previous.source} and {item.source}.")
    return [by_dest[key] for key in sorted(by_dest)]
