"""Run records collected by the harness."""

from __future__ import annotations

from dataclasses import dataclass

from skillcheck.errors import SkillCheckError


@dataclass(frozen=True)
class RunRecord:
    """One condition of one trial of one task."""

    task_id: str
    condition: str
    trial: int
    passed: bool
    wall_time_seconds: float
    agent_time_seconds: float
    verify_time_seconds: float
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    turns: int | None
    tool_calls: int | None
    diff_files: int
    diff_lines_added: int
    diff_lines_removed: int
    agent_error: str | None
    verify_exit_code: int | None
    verify_output_tail: str


def _as_optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SkillCheckError(f"Run field {field} must be an integer or null.")
    return value


def _as_optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillCheckError(f"Run field {field} must be a number or null.")
    return float(value)


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillCheckError(f"Run field {field} must be a number.")
    return float(value)


def run_from_dict(data: object) -> RunRecord:
    """Parse one run object from a results document."""
    if not isinstance(data, dict):
        raise SkillCheckError("Each run must be a JSON object.")
    required = (
        "task_id",
        "condition",
        "trial",
        "passed",
        "wall_time_seconds",
        "agent_time_seconds",
        "verify_time_seconds",
        "diff_files",
        "diff_lines_added",
        "diff_lines_removed",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise SkillCheckError(f"Run is missing fields: {', '.join(missing)}.")
    task_id = data["task_id"]
    condition = data["condition"]
    if not isinstance(task_id, str) or not task_id:
        raise SkillCheckError("Run task_id must be a non-empty string.")
    if condition not in {"baseline", "treatment"}:
        raise SkillCheckError(f"Run condition must be baseline or treatment, got {condition!r}.")
    trial = data["trial"]
    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 1:
        raise SkillCheckError("Run trial must be an integer starting at 1.")
    passed = data["passed"]
    if not isinstance(passed, bool):
        raise SkillCheckError("Run passed must be true or false.")
    agent_error = data.get("agent_error")
    if agent_error is not None and not isinstance(agent_error, str):
        raise SkillCheckError("Run agent_error must be a string or null.")
    verify_exit = data.get("verify_exit_code")
    if verify_exit is not None and (isinstance(verify_exit, bool) or not isinstance(verify_exit, int)):
        raise SkillCheckError("Run verify_exit_code must be an integer or null.")
    tail = data.get("verify_output_tail", "")
    if not isinstance(tail, str):
        raise SkillCheckError("Run verify_output_tail must be a string.")
    return RunRecord(
        task_id=task_id,
        condition=condition,
        trial=trial,
        passed=passed,
        wall_time_seconds=_as_float(data["wall_time_seconds"], "wall_time_seconds"),
        agent_time_seconds=_as_float(data["agent_time_seconds"], "agent_time_seconds"),
        verify_time_seconds=_as_float(data["verify_time_seconds"], "verify_time_seconds"),
        tokens_in=_as_optional_int(data.get("tokens_in"), "tokens_in"),
        tokens_out=_as_optional_int(data.get("tokens_out"), "tokens_out"),
        cost_usd=_as_optional_float(data.get("cost_usd"), "cost_usd"),
        turns=_as_optional_int(data.get("turns"), "turns"),
        tool_calls=_as_optional_int(data.get("tool_calls"), "tool_calls"),
        diff_files=_as_optional_int(data["diff_files"], "diff_files") or 0,
        diff_lines_added=_as_optional_int(data["diff_lines_added"], "diff_lines_added") or 0,
        diff_lines_removed=_as_optional_int(data["diff_lines_removed"], "diff_lines_removed") or 0,
        agent_error=agent_error,
        verify_exit_code=verify_exit,
        verify_output_tail=tail,
    )
