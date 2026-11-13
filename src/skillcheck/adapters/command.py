"""Shell out to an external agent CLI.

The template can use `{prompt}`, `{prompt_file}`, and `{workspace}`. Each
placeholder is shell-quoted. The prompt is also written to
`.skillcheck/prompt.txt` before the snapshot, so a command can read it from
disk without putting a huge prompt on the command line.

If the command prints a line starting with `SKILLCHECK_METRICS ` followed by
one JSON object, those fields are recorded. Anything else leaves tokens, cost,
turns, and tool calls unreported rather than guessed.

This adapter is tested with a local command. It has not been run against
Claude, Codex, Aider, or any other third-party CLI in this environment.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from skillcheck.adapters.base import AgentRunResult
from skillcheck.errors import SkillCheckError
from skillcheck.process import run_shell

_METRICS_PREFIX = "SKILLCHECK_METRICS "


class CommandAdapter:
    """Run a user-supplied command template in the workspace."""

    name = "command"

    def __init__(self, command: str) -> None:
        if not command.strip():
            raise SkillCheckError("The command adapter needs a non-empty command template.")
        self.command = command

    def run(
        self,
        *,
        workspace: Path,
        prompt: str,
        task_id: str,
        timeout: float,
    ) -> AgentRunResult:
        del task_id
        prompt_file = workspace / ".skillcheck" / "prompt.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt, encoding="utf-8")
        rendered = render_command(
            self.command,
            prompt=prompt,
            prompt_file=prompt_file,
            workspace=workspace,
        )
        try:
            completed = run_shell(rendered, workspace, timeout)
        except subprocess.TimeoutExpired:
            return AgentRunResult(error=f"Command timed out after {timeout:.0f}s.")
        metrics = parse_metrics_trailer(completed.stdout)
        error = None
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()[-500:]
            error = f"Command exited with status {completed.returncode}."
            if detail:
                error = f"{error} {detail}"
        return AgentRunResult(
            tokens_in=_metric_int(metrics, "tokens_in"),
            tokens_out=_metric_int(metrics, "tokens_out"),
            cost_usd=_metric_float(metrics, "cost_usd"),
            turns=_metric_int(metrics, "turns"),
            tool_calls=_metric_int(metrics, "tool_calls"),
            error=error,
        )


def render_command(template: str, *, prompt: str, prompt_file: Path, workspace: Path) -> str:
    """Substitute known placeholders. Unknown `{name}` text is left untouched."""
    replacements = {
        "{prompt}": shlex.quote(prompt),
        "{prompt_file}": shlex.quote(str(prompt_file)),
        "{workspace}": shlex.quote(str(workspace)),
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def parse_metrics_trailer(stdout: str) -> dict[str, object] | None:
    """Return the last SKILLCHECK_METRICS object, or None if the command printed none."""
    found: dict[str, object] | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_METRICS_PREFIX):
            continue
        raw = stripped[len(_METRICS_PREFIX) :]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SkillCheckError(f"SKILLCHECK_METRICS line is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SkillCheckError("SKILLCHECK_METRICS payload must be a JSON object.")
        found = payload
    return found


def _metric_int(metrics: dict[str, object] | None, key: str) -> int | None:
    if metrics is None or key not in metrics or metrics[key] is None:
        return None
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SkillCheckError(f"SKILLCHECK_METRICS field {key} must be a non-negative integer.")
    return value


def _metric_float(metrics: dict[str, object] | None, key: str) -> float | None:
    if metrics is None or key not in metrics or metrics[key] is None:
        return None
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillCheckError(f"SKILLCHECK_METRICS field {key} must be a number.")
    return float(value)
