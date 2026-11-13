"""Run a paired baseline-versus-treatment evaluation."""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from skillcheck import __version__
from skillcheck.adapters import build_adapter
from skillcheck.adapters.base import AgentAdapter, AgentRunResult
from skillcheck.adapters.openai_agent import OpenAICompatibleAdapter
from skillcheck.errors import SkillCheckError
from skillcheck.models import RunRecord
from skillcheck.report import finalize_document, render_markdown
from skillcheck.sandbox import prepare_workspace, run_verify
from skillcheck.skills import resolve_skills
from skillcheck.tasks import load_suite

_REDACTED_OPTIONS = {"api_key"}


@dataclass
class RunConfig:
    """Inputs for one evaluation.

    `adapter` is a test hook. The CLI leaves it empty and builds one from
    `agent` plus `agent_options`.
    """

    tasks: str
    skills: list[str]
    agent: str
    trials: int
    seed: int = 0
    confidence: float = 0.95
    bootstrap_samples: int = 4000
    negligible_margin: float = 0.05
    timeout: float | None = None
    agent_options: dict[str, str] = field(default_factory=dict)
    verbose: bool = False
    keep: bool = False
    output_dir: Path | None = None
    adapter: AgentAdapter | None = None


def run_evaluation(config: RunConfig) -> dict[str, object]:
    """Evaluate every task under both conditions and return a results document.

    When `output_dir` is set, also write `results.json`, `report.md`, and
    `badge.json`. The badge is computed from the pass-rate interval.
    """
    _validate(config)
    suite = load_suite(config.tasks)
    skills = resolve_skills(config.skills)
    adapter = config.adapter if config.adapter is not None else build_adapter(config.agent, config.agent_options)
    runs: list[RunRecord] = []
    started = time.perf_counter()
    for task in suite.tasks:
        timeout = config.timeout if config.timeout is not None else task.timeout_seconds
        for trial in range(1, config.trials + 1):
            for condition in ("baseline", "treatment"):
                install = skills if condition == "treatment" else []
                prepared = prepare_workspace(
                    task,
                    strip=skills,
                    install=install,
                    prompt=task.prompt,
                    keep=config.keep,
                )
                try:
                    agent_started = time.perf_counter()
                    try:
                        agent_result = adapter.run(
                            workspace=prepared.path,
                            prompt=task.prompt,
                            task_id=task.id,
                            timeout=timeout,
                        )
                    except Exception as exc:
                        if config.verbose:
                            traceback.print_exc()
                        agent_result = AgentRunResult(error=f"{type(exc).__name__}: {exc}")
                    agent_seconds = time.perf_counter() - agent_started
                    diff = prepared.diff()
                    prepared.restore_for_verify()
                    verify = run_verify(prepared.path, task.verify, timeout)
                finally:
                    if config.keep:
                        print(f"kept sandbox {prepared.path}", file=sys.stderr)
                    prepared.close()
                record = RunRecord(
                    task_id=task.id,
                    condition=condition,
                    trial=trial,
                    passed=verify.passed,
                    wall_time_seconds=agent_seconds + verify.seconds,
                    agent_time_seconds=agent_seconds,
                    verify_time_seconds=verify.seconds,
                    tokens_in=agent_result.tokens_in,
                    tokens_out=agent_result.tokens_out,
                    cost_usd=agent_result.cost_usd,
                    turns=agent_result.turns,
                    tool_calls=agent_result.tool_calls,
                    diff_files=diff.files,
                    diff_lines_added=diff.lines_added,
                    diff_lines_removed=diff.lines_removed,
                    agent_error=agent_result.error,
                    verify_exit_code=verify.exit_code,
                    verify_output_tail=verify.output_tail if not verify.passed else "",
                )
                runs.append(record)
                if config.verbose:
                    status = "pass" if verify.passed else "fail"
                    print(
                        f"{task.id} trial {trial} {condition}: {status}",
                        file=sys.stderr,
                    )
    document = _metadata(config, suite, skills, adapter)
    document["runs"] = [asdict(record) for record in runs]
    document["eval_wall_time_seconds"] = time.perf_counter() - started
    finalized = finalize_document(document)
    if config.output_dir is not None:
        write_outputs(finalized, config.output_dir)
    return finalized


def write_outputs(document: dict[str, object], output_dir: Path) -> dict[str, Path]:
    """Write the three result files. The badge is the one inside the document."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "results.json",
        "report": output_dir / "report.md",
        "badge": output_dir / "badge.json",
    }
    paths["results"].write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(render_markdown(document), encoding="utf-8")
    badge = document.get("badge")
    if not isinstance(badge, dict):
        raise SkillCheckError("Results are missing a badge.")
    paths["badge"].write_text(
        json.dumps(badge, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _validate(config: RunConfig) -> None:
    if config.trials < 1:
        raise SkillCheckError("--trials must be at least 1.")
    if not 0 < config.confidence < 1:
        raise SkillCheckError("--confidence must be between 0 and 1.")
    if not 0 < config.negligible_margin < 1:
        raise SkillCheckError("--margin must be between 0 and 1.")
    if config.bootstrap_samples < 200:
        raise SkillCheckError("--bootstrap must be at least 200.")
    if config.timeout is not None and config.timeout <= 0:
        raise SkillCheckError("--timeout must be a positive number of seconds.")


def _metadata(config: RunConfig, suite: object, skills: list[object], adapter: AgentAdapter) -> dict[str, object]:
    from skillcheck.skills import SkillFile
    from skillcheck.tasks import Suite

    if not isinstance(suite, Suite):
        raise SkillCheckError("Internal error: expected a suite.")
    skill_rows = []
    for item in skills:
        if not isinstance(item, SkillFile):
            raise SkillCheckError("Internal error: expected a skill file.")
        skill_rows.append({"source": str(item.source), "dest": item.dest, "sha256": item.sha256})
    agent_info: dict[str, object] = {
        "name": adapter.name,
        "options": {key: value for key, value in config.agent_options.items() if key not in _REDACTED_OPTIONS},
    }
    if isinstance(adapter, OpenAICompatibleAdapter):
        agent_info["model"] = adapter.model
        if isinstance(adapter.base_url, str):
            agent_info["base_url"] = adapter.base_url
    source = "builtin" if config.tasks == "builtin" else str(suite.source)
    return {
        "schema_version": 1,
        "skillcheck_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "seed": config.seed,
        "confidence": config.confidence,
        "bootstrap_samples": config.bootstrap_samples,
        "negligible_margin": config.negligible_margin,
        "trials": config.trials,
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "source": source,
            "path": str(suite.source),
            "tasks": [task.id for task in suite.tasks],
        },
        "agent": agent_info,
        "skills": skill_rows,
    }
