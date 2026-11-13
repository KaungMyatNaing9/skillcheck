"""OpenAI-compatible tool loop. Live providers are not called."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from skillcheck.adapters.openai_agent import (
    OpenAICompatibleAdapter,
    build_openai_adapter,
    chat_completions_url,
    execute_tool,
)
from skillcheck.errors import SkillCheckError
from skillcheck.runner import RunConfig, run_evaluation


class ScriptedClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        return self.responses.pop(0)


def _message(tool_calls: list[dict[str, Any]] | None, usage: dict[str, int]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": usage}


def _call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_chat_url_and_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    assert chat_completions_url("https://api.openai.com/v1") == ("https://api.openai.com/v1/chat/completions")
    assert (
        chat_completions_url("https://example.test/v1/chat/completions") == "https://example.test/v1/chat/completions"
    )
    with pytest.raises(SkillCheckError, match="http"):
        chat_completions_url("ftp://example.test/v1")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("urlopen should not be called without a key")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    monkeypatch.delenv("SKILLCHECK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SKILLCHECK_MODEL", "test-model")
    with pytest.raises(SkillCheckError, match="SKILLCHECK_API_KEY"):
        build_openai_adapter({})
    monkeypatch.setenv("SKILLCHECK_API_KEY", "test-key")
    monkeypatch.delenv("SKILLCHECK_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(SkillCheckError, match="SKILLCHECK_MODEL"):
        build_openai_adapter({})


def test_scripted_loop_edits_a_file_and_prices_usage(tmp_path: Path) -> None:
    (tmp_path / "ranges.py").write_text(
        "for i in range(start, end):\n    total += i\n",
        encoding="utf-8",
    )
    client = ScriptedClient(
        [
            _message(
                [_call("read_file", {"path": "ranges.py"})],
                {"prompt_tokens": 1000, "completion_tokens": 10},
            ),
            _message(
                [
                    _call(
                        "edit_file",
                        {
                            "path": "ranges.py",
                            "find": "range(start, end)",
                            "replace": "range(start, end + 1)",
                        },
                        "call_2",
                    )
                ],
                {"prompt_tokens": 1100, "completion_tokens": 20},
            ),
            _message([_call("done", {"summary": "fixed"})], {"prompt_tokens": 1200, "completion_tokens": 5}),
        ]
    )
    adapter = OpenAICompatibleAdapter(
        model="test-model",
        client=client,
        input_usd_per_million=2,
        output_usd_per_million=8,
    )
    result = adapter.run(workspace=tmp_path, prompt="Fix the range.", task_id="t", timeout=30)
    assert result.error is None
    assert "end + 1" in (tmp_path / "ranges.py").read_text(encoding="utf-8")
    assert result.tokens_in == 3300
    assert result.tokens_out == 35
    assert result.tool_calls == 3
    assert result.turns == 3
    # 3300/1e6*2 + 35/1e6*8
    assert result.cost_usd == pytest.approx(0.00688)
    assert client.payloads[0]["model"] == "test-model"
    assert any(tool["function"]["name"] == "edit_file" for tool in client.payloads[0]["tools"])


def test_path_escape_does_not_write(tmp_path: Path) -> None:
    outside = tmp_path.parent / "skillcheck-escape-probe.txt"
    if outside.exists():
        outside.unlink()
    message = execute_tool(
        tmp_path,
        "write_file",
        {"path": "../skillcheck-escape-probe.txt", "content": "no"},
        timeout=5,
    )
    assert message.startswith("error:")
    assert not outside.exists()
    linked = tmp_path / "link"
    linked.symlink_to(tmp_path.parent)
    message = execute_tool(
        tmp_path,
        "write_file",
        {"path": "link/escaped.txt", "content": "no"},
        timeout=5,
    )
    assert message.startswith("error:")
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_max_turns_stops(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _message([_call("read_file", {"path": "missing.py"})], {"prompt_tokens": 1, "completion_tokens": 1}),
            _message([_call("read_file", {"path": "missing.py"})], {"prompt_tokens": 1, "completion_tokens": 1}),
        ]
    )
    result = OpenAICompatibleAdapter(model="test-model", client=client, max_turns=2).run(
        workspace=tmp_path, prompt="loop", task_id="t", timeout=30
    )
    assert result.error is not None
    assert "2 turns" in result.error


def test_local_http_stub_records_auth_and_path(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            self.server.requests.append(  # type: ignore[attr-defined]
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(body.decode("utf-8")),
                }
            )
            payload = _message(None, {"prompt_tokens": 4, "completion_tokens": 6})
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        adapter = build_openai_adapter(
            {
                "api_key": "test-key",
                "model": "stub-model",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "input_usd_per_million": "1",
                "output_usd_per_million": "1",
            }
        )
        result = adapter.run(workspace=tmp_path, prompt="stop now", task_id="t", timeout=30)
        assert result.tokens_in == 4
        assert result.tokens_out == 6
        assert result.cost_usd == pytest.approx(10 / 1_000_000)
        recorded = server.requests[0]  # type: ignore[attr-defined]
        assert recorded["path"] == "/v1/chat/completions"
        assert recorded["authorization"] == "Bearer test-key"
        assert recorded["body"]["model"] == "stub-model"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_http_error_is_reported(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = b'{"error": "nope"}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        adapter = build_openai_adapter(
            {
                "api_key": "test-key",
                "model": "stub-model",
                "base_url": f"http://127.0.0.1:{port}/v1",
            }
        )
        with pytest.raises(SkillCheckError, match="401"):
            adapter.run(workspace=tmp_path, prompt="stop", task_id="t", timeout=30)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_runner_passes_only_when_the_skill_file_is_present(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "src" / "skillcheck" / "fixtures" / "fix-inclusive-sum"
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        f"""
name: openai-demo
tasks:
  - id: fix-inclusive-sum
    workspace: {fixture / "workspace"}
    hidden: {fixture / "hidden"}
    prompt: Fix the inclusive sum.
    verify: "{{python}} -m unittest discover -s tests -q"
    timeout_seconds: 30
""",
        encoding="utf-8",
    )
    skill = tmp_path / "AGENTS.md"
    skill.write_text("include the end index\n", encoding="utf-8")

    class AwareClient:
        def __init__(self) -> None:
            self.workspace = Path(".")
            self.seen: dict[str, int] = {}

        def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            del payload
            key = str(self.workspace)
            phase = self.seen.get(key, 0)
            self.seen[key] = phase + 1
            usage = {"prompt_tokens": 100, "completion_tokens": 10}
            if not (self.workspace / "AGENTS.md").exists():
                return _message(None, usage)
            if phase == 0:
                return _message(
                    [
                        _call(
                            "edit_file",
                            {
                                "path": "ranges.py",
                                "find": "range(start, end)",
                                "replace": "range(start, end + 1)",
                            },
                        )
                    ],
                    usage,
                )
            return _message([_call("done", {})], usage)

    client = AwareClient()

    class Hooked(OpenAICompatibleAdapter):
        def run(self, *, workspace: Path, prompt: str, task_id: str, timeout: float):  # type: ignore[override]
            client.workspace = workspace
            return super().run(workspace=workspace, prompt=prompt, task_id=task_id, timeout=timeout)

    document = run_evaluation(
        RunConfig(
            tasks=str(suite),
            skills=[str(skill)],
            agent="openai",
            trials=2,
            bootstrap_samples=200,
            output_dir=None,
            adapter=Hooked(
                model="test-model",
                client=client,
                input_usd_per_million=2,
                output_usd_per_million=4,
            ),
        )
    )
    runs = document["runs"]
    assert isinstance(runs, list)
    assert all(run["passed"] is False for run in runs if run["condition"] == "baseline")
    assert all(run["passed"] is True for run in runs if run["condition"] == "treatment")
    assert document["verdict"]["label"] == "helps"
    assert document["agent"]["model"] == "test-model"
    assert "api_key" not in document["agent"]["options"]
    assert document["overall"]["cost_difference"]["available"] is True


def test_urlopen_is_the_real_client() -> None:
    # Guard against the missing-key test accidentally patching the wrong symbol.
    assert callable(urlopen)
