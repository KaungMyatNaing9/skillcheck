"""Agent adapters and the factory the CLI uses."""

from __future__ import annotations

from pathlib import Path

from skillcheck.adapters.base import AgentAdapter
from skillcheck.adapters.command import CommandAdapter
from skillcheck.adapters.fake import FakeAdapter, ReplayAdapter
from skillcheck.adapters.openai_agent import build_openai_adapter
from skillcheck.errors import SkillCheckError

_OPENAI_OPTIONS = {
    "model",
    "base_url",
    "api_key",
    "max_turns",
    "temperature",
    "request_timeout",
    "input_usd_per_million",
    "output_usd_per_million",
}


def build_adapter(name: str, options: dict[str, str]) -> AgentAdapter:
    """Construct an adapter. Unknown options are errors so typos do not disappear."""
    key = name.strip().lower()
    if key == "fake":
        _reject_unknown(options, set(), "fake")
        return FakeAdapter()
    if key == "replay":
        _reject_unknown(options, {"script"}, "replay")
        script = options.get("script")
        if not script:
            raise SkillCheckError("The replay adapter needs --agent-opt script=PATH.")
        return ReplayAdapter(Path(script).expanduser().resolve())
    if key == "command":
        _reject_unknown(options, {"command"}, "command")
        command = options.get("command")
        if not command:
            raise SkillCheckError("The command adapter needs --agent-opt command=TEMPLATE.")
        return CommandAdapter(command)
    if key in {"openai", "openai-compatible"}:
        _reject_unknown(options, _OPENAI_OPTIONS, "openai")
        return build_openai_adapter(options)
    raise SkillCheckError(f"Unknown agent adapter {name!r}. Choose fake, replay, command, or openai.")


def _reject_unknown(options: dict[str, str], allowed: set[str], adapter: str) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise SkillCheckError(f"The {adapter} adapter does not accept option(s): {', '.join(unknown)}.")
