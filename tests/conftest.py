"""Shared constructors for synthetic runs."""

from __future__ import annotations

from skillcheck.models import RunRecord


def make_run(**overrides: object) -> RunRecord:
    base: dict[str, object] = {
        "task_id": "task-a",
        "condition": "baseline",
        "trial": 1,
        "passed": False,
        "wall_time_seconds": 0.2,
        "agent_time_seconds": 0.2,
        "verify_time_seconds": 0.0,
        "tokens_in": 8,
        "tokens_out": 2,
        "cost_usd": None,
        "turns": 1,
        "tool_calls": 0,
        "diff_files": 0,
        "diff_lines_added": 0,
        "diff_lines_removed": 0,
        "agent_error": None,
        "verify_exit_code": 1,
        "verify_output_tail": "",
    }
    base.update(overrides)
    return RunRecord(**base)  # type: ignore[arg-type]


def runs_for(
    deltas: dict[str, list[int]],
    *,
    tokens: tuple[int, int] | None = (8, 8),
    cost: tuple[float, float] | None = None,
    wall: tuple[float, float] = (0.2, 0.2),
) -> list[RunRecord]:
    """Build paired runs from per-trial pass deltas of -1, 0, or 1.

    A delta of 0 means both conditions failed. Token and cost values, when
    given, are constant for each condition.
    """
    runs: list[RunRecord] = []
    for task_id, task_deltas in deltas.items():
        for index, delta in enumerate(task_deltas, start=1):
            if delta == 1:
                base_pass, treat_pass = False, True
            elif delta == -1:
                base_pass, treat_pass = True, False
            elif delta == 0:
                base_pass, treat_pass = False, False
            else:
                raise AssertionError(f"unsupported delta {delta}")
            for condition, passed, token_index, cost_index, wall_index in (
                ("baseline", base_pass, 0, 0, 0),
                ("treatment", treat_pass, 1, 1, 1),
            ):
                runs.append(
                    make_run(
                        task_id=task_id,
                        condition=condition,
                        trial=index,
                        passed=passed,
                        tokens_in=None if tokens is None else tokens[token_index],
                        tokens_out=None if tokens is None else 0,
                        cost_usd=None if cost is None else cost[cost_index],
                        wall_time_seconds=wall[wall_index],
                        verify_exit_code=0 if passed else 1,
                    )
                )
    return runs
