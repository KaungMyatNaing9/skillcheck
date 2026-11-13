"""Minimal tool loop for any OpenAI-compatible chat completions API.

Environment variables, in order:

- API key: `SKILLCHECK_API_KEY`, then `OPENAI_API_KEY`
- Model: `SKILLCHECK_MODEL`, then `OPENAI_MODEL` (required, no default model)
- Base URL: `SKILLCHECK_BASE_URL`, then `OPENAI_BASE_URL`, then
  `https://api.openai.com/v1`

Optional prices, USD per million tokens:

- `SKILLCHECK_INPUT_USD_PER_MILLION`
- `SKILLCHECK_OUTPUT_USD_PER_MILLION`

Cost is reported only when both prices are set and the response includes
usage. Token totals are the sum of the usage object on each response. A
provider that reports cumulative usage would be over-counted. This adapter's
HTTP and tool loop are tested against a local stub. A live provider has not
been called from this environment.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from skillcheck.adapters.base import AgentRunResult
from skillcheck.errors import SkillCheckError
from skillcheck.paths import SKIP_DIR_NAMES, resolve_inside
from skillcheck.process import run_shell

_READ_LIMIT = 20_000
_LIST_LIMIT = 200
_COMMAND_OUTPUT_LIMIT = 8_000
_COMMAND_TIMEOUT_CAP = 30.0

_SYSTEM_PROMPT = (
    "You are a coding agent inside a workspace. "
    "Follow AGENTS.md, CLAUDE.md, GEMINI.md, .cursorrules, .cursor/rules, and "
    ".claude/skills/**/SKILL.md when those files exist. "
    "Use the tools to read and edit files and to run commands. "
    "Call done when the task is finished."
)


class ChatClient(Protocol):
    """One chat-completions call. Tests substitute a scripted client."""

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the decoded JSON response body."""


class UrllibChatClient:
    """POST JSON to `{base_url}/chat/completions` with a bearer token."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.url = chat_completions_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise SkillCheckError(f"Model request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SkillCheckError(f"Model request failed: {exc.reason}") from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SkillCheckError("Model response was not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise SkillCheckError("Model response was not a JSON object.")
        return parsed


class OpenAICompatibleAdapter:
    """A short read/write/run loop over an injected chat client."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        client: ChatClient,
        max_turns: int = 8,
        temperature: float = 0.0,
        input_usd_per_million: float | None = None,
        output_usd_per_million: float | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        if max_turns < 1:
            raise SkillCheckError("max_turns must be at least 1.")
        self.model = model
        self.client = client
        self.max_turns = max_turns
        self.temperature = temperature
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million
        self.request_timeout = request_timeout
        self.base_url = getattr(client, "url", None)

    def run(
        self,
        *,
        workspace: Path,
        prompt: str,
        task_id: str,
        timeout: float,
    ) -> AgentRunResult:
        del task_id
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tokens_in = 0
        tokens_out = 0
        saw_usage = False
        tool_calls = 0
        turns = 0
        deadline = time.perf_counter() + timeout
        error: str | None = None
        for _ in range(self.max_turns):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                error = "Agent timed out."
                break
            turns += 1
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": messages,
                "tools": tool_schemas(),
            }
            response = self.client.complete(payload)
            usage = response.get("usage")
            if isinstance(usage, dict):
                saw_usage = True
                tokens_in += _usage_int(usage, "prompt_tokens")
                tokens_out += _usage_int(usage, "completion_tokens")
            message = assistant_message(response)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or not calls:
                break
            stop = False
            for call in calls:
                if not isinstance(call, dict):
                    continue
                tool_calls += 1
                name, arguments, call_id = parse_tool_call(call)
                if name == "done":
                    stop = True
                    break
                output = execute_tool(
                    workspace,
                    name,
                    arguments,
                    timeout=min(_COMMAND_TIMEOUT_CAP, remaining),
                )
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
            if stop:
                break
        else:
            error = f"Stopped after {self.max_turns} turns."
        return self._result(
            tokens_in=tokens_in if saw_usage else None,
            tokens_out=tokens_out if saw_usage else None,
            turns=turns,
            tool_calls=tool_calls,
            error=error,
        )

    def _result(
        self,
        *,
        tokens_in: int | None,
        tokens_out: int | None,
        turns: int,
        tool_calls: int,
        error: str | None,
    ) -> AgentRunResult:
        cost = None
        if (
            tokens_in is not None
            and tokens_out is not None
            and self.input_usd_per_million is not None
            and self.output_usd_per_million is not None
        ):
            cost = (
                tokens_in / 1_000_000 * self.input_usd_per_million
                + tokens_out / 1_000_000 * self.output_usd_per_million
            )
        return AgentRunResult(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            turns=turns,
            tool_calls=tool_calls,
            error=error,
        )


def build_openai_adapter(options: dict[str, str]) -> OpenAICompatibleAdapter:
    """Build the adapter from CLI options and the environment.

    The API key is read here and is not returned. Callers must not write
    `options` that contain a key into the results file.
    """
    api_key = _first(options.get("api_key"), "SKILLCHECK_API_KEY", "OPENAI_API_KEY")
    model = _first(options.get("model"), "SKILLCHECK_MODEL", "OPENAI_MODEL")
    base_url = _first(options.get("base_url"), "SKILLCHECK_BASE_URL", "OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not api_key:
        raise SkillCheckError("Set SKILLCHECK_API_KEY or OPENAI_API_KEY to use the openai adapter.")
    if not model:
        raise SkillCheckError("Set SKILLCHECK_MODEL or OPENAI_MODEL to use the openai adapter.")
    max_turns = _option_int(options, "max_turns", 8)
    temperature = _option_float(options, "temperature", 0.0)
    request_timeout = _option_float(options, "request_timeout", 60.0)
    return OpenAICompatibleAdapter(
        model=model,
        client=UrllibChatClient(base_url, api_key, request_timeout),
        max_turns=max_turns,
        temperature=temperature,
        input_usd_per_million=_price(options, "input_usd_per_million", "SKILLCHECK_INPUT_USD_PER_MILLION"),
        output_usd_per_million=_price(options, "output_usd_per_million", "SKILLCHECK_OUTPUT_USD_PER_MILLION"),
        request_timeout=request_timeout,
    )


def chat_completions_url(base_url: str) -> str:
    """Append `/chat/completions` unless the caller already included it."""
    trimmed = base_url.strip().rstrip("/")
    if not trimmed.startswith(("http://", "https://")):
        raise SkillCheckError("Model base URL must start with http:// or https://.")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return trimmed + "/chat/completions"


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI tool definitions for the built-in loop."""
    return [
        _function(
            "list_files",
            "List files under a directory inside the workspace.",
            {"path": {"type": "string", "description": "Relative directory. Defaults to '.'"}},
            [],
        ),
        _function(
            "read_file",
            "Read a text file inside the workspace.",
            {"path": {"type": "string"}},
            ["path"],
        ),
        _function(
            "write_file",
            "Create or overwrite a text file inside the workspace.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        _function(
            "edit_file",
            "Replace every occurrence of find with replace in a workspace file.",
            {
                "path": {"type": "string"},
                "find": {"type": "string"},
                "replace": {"type": "string"},
            },
            ["path", "find", "replace"],
        ),
        _function(
            "run_command",
            "Run a shell command with the workspace as the working directory.",
            {"command": {"type": "string"}},
            ["command"],
        ),
        _function(
            "done",
            "Stop after the task is finished.",
            {"summary": {"type": "string"}},
            [],
        ),
    ]


def assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """Pull the assistant message off a chat-completions response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise SkillCheckError("Model response did not include choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise SkillCheckError("Model response did not include a message.")
    cleaned: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    tool_calls = message.get("tool_calls")
    if tool_calls:
        cleaned["tool_calls"] = tool_calls
    return cleaned


def parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    """Return (name, arguments, id). Invalid argument JSON becomes an error field."""
    call_id = str(call.get("id") or "call")
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or "")
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, dict):
        arguments = raw_args
    elif isinstance(raw_args, str):
        if not raw_args.strip():
            arguments = {}
        else:
            try:
                loaded = json.loads(raw_args)
            except json.JSONDecodeError:
                return name, {"_error": "Tool arguments were not valid JSON."}, call_id
            arguments = loaded if isinstance(loaded, dict) else {}
    else:
        arguments = {}
    return name, arguments, call_id


def execute_tool(workspace: Path, name: str, arguments: dict[str, Any], *, timeout: float) -> str:
    """Run one tool. Errors are returned to the model instead of crashing the loop."""
    if arguments.get("_error"):
        return str(arguments["_error"])
    try:
        if name == "list_files":
            return _list_files(workspace, str(arguments.get("path") or "."))
        if name == "read_file":
            return _read_file(workspace, _argument(arguments, "path"))
        if name == "write_file":
            _write_file(workspace, _argument(arguments, "path"), _argument(arguments, "content", allow_empty=True))
            return "Wrote file."
        if name == "edit_file":
            return _edit_file(
                workspace,
                _argument(arguments, "path"),
                _argument(arguments, "find"),
                _argument(arguments, "replace", allow_empty=True),
            )
        if name == "run_command":
            return _run_command(workspace, _argument(arguments, "command"), timeout)
        return f"Unknown tool {name}."
    except SkillCheckError as exc:
        return f"error: {exc}"


def _list_files(workspace: Path, raw: str) -> str:
    path = resolve_inside(workspace, raw)
    root = workspace.resolve()
    if path.is_file():
        return path.relative_to(root).as_posix()
    if not path.is_dir():
        raise SkillCheckError(f"Not a directory: {raw}")
    lines: list[str] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            if candidate.is_symlink() or not candidate.is_file():
                continue
            lines.append(candidate.relative_to(root).as_posix())
            if len(lines) >= _LIST_LIMIT:
                lines.append(f"... truncated at {_LIST_LIMIT} files")
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(empty)"


def _read_file(workspace: Path, raw: str) -> str:
    path = resolve_inside(workspace, raw)
    if not path.is_file():
        raise SkillCheckError(f"File not found: {raw}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= _READ_LIMIT:
        return text
    return text[:_READ_LIMIT] + f"\n... truncated at {_READ_LIMIT} characters"


def _write_file(workspace: Path, raw: str, content: str) -> None:
    path = resolve_inside(workspace, raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _edit_file(workspace: Path, raw: str, find: str, replacement: str) -> str:
    path = resolve_inside(workspace, raw)
    if not path.is_file():
        raise SkillCheckError(f"File not found: {raw}")
    text = path.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise SkillCheckError(f"Text to replace was not found in {raw}.")
    path.write_text(text.replace(find, replacement), encoding="utf-8")
    return f"Replaced {count} occurrence(s) in {raw}."


def _run_command(workspace: Path, command: str, timeout: float) -> str:
    import subprocess

    try:
        completed = run_shell(command, workspace, timeout)
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout:.0f}s."
    output = (completed.stdout or "") + (completed.stderr or "")
    if len(output) > _COMMAND_OUTPUT_LIMIT:
        output = output[-_COMMAND_OUTPUT_LIMIT:]
    return f"exit {completed.returncode}\n{output}"


def _function(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _argument(arguments: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or (value == "" and not allow_empty):
        raise SkillCheckError(f"Tool argument '{key}' must be a string.")
    return value


def _usage_int(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SkillCheckError(f"Usage field {key} must be a non-negative integer.")
    return value


def _first(explicit: str | None, *env_names: str) -> str | None:
    if explicit:
        return explicit
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _option_int(options: dict[str, str], key: str, default: int) -> int:
    if key not in options:
        return default
    try:
        return int(options[key])
    except ValueError as exc:
        raise SkillCheckError(f"{key} must be an integer.") from exc


def _option_float(options: dict[str, str], key: str, default: float) -> float:
    if key not in options:
        return default
    try:
        return float(options[key])
    except ValueError as exc:
        raise SkillCheckError(f"{key} must be a number.") from exc


def _price(options: dict[str, str], option: str, env_name: str) -> float | None:
    raw = options.get(option) or os.environ.get(env_name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise SkillCheckError(f"{env_name} must be a number (USD per million tokens).") from exc
    if value < 0:
        raise SkillCheckError(f"{env_name} cannot be negative.")
    return value
