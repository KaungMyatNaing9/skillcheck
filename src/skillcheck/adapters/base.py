"""Adapter result shared by every agent runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class AgentRunResult:
    """What the harness needs from one agent invocation.

    Token and cost fields stay None when the adapter did not measure them.
    Zero is a real measurement. None means "not reported".
    """

    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    turns: int | None = None
    tool_calls: int | None = None
    error: str | None = None


class AgentAdapter(Protocol):
    """Run one agent against one workspace.

    Implementations must not mutate anything outside `workspace`. The harness
    compares conditions by giving each one its own copy.
    """

    name: str

    def run(
        self,
        *,
        workspace: Path,
        prompt: str,
        task_id: str,
        timeout: float,
    ) -> AgentRunResult:
        """Edit `workspace` according to `prompt`. Do not raise for a bad edit.

        Raise only when the adapter itself is misconfigured. A model failure
        or a non-zero command should come back as `AgentRunResult.error`.
        """
