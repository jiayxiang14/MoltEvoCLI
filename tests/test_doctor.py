"""moltevo --doctor：本地检查与最小请求的错误分类。"""

import asyncio
import json
import sys
from pathlib import Path

import anthropic
import httpx
import openai
import pytest

from agents import doctor
from agents.ui import print_doctor
from fakemodel import make_agent

FAKE_MCP = str(Path(__file__).with_name("fake_mcp_server.py"))


def run(coro):
    return asyncio.run(coro)


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "https://api.example/v1"))


def by_name(checks, name):
    return next(c for c in checks if c.name == name)


def test_local_checks_flag_unknown_price_and_window(workdir):
    checks = doctor.local_checks("deepseek-chat")
    assert by_name(checks, "price known").status == "warn"
    assert by_name(checks, "session dir writable").status == "ok"
    known = doctor.local_checks("claude-sonnet-4-6")
    assert by_name(known, "price known").status == "ok"
    assert by_name(known, "context window known").status == "ok"


def _agent_with_create(create, openai_mode=False):
    a = make_agent(openai_mode=openai_mode)
    if openai_mode:
        a._openai_client.chat.completions.create = create
    else:
        a._anthropic_client.messages.create = create
    return a


@pytest.mark.parametrize("err,status,needle", [
    (anthropic.NotFoundError("model: deepseek-chat", response=_resp(404), body=None), "fail", "model or endpoint not found"),
    (anthropic.AuthenticationError("bad key", response=_resp(401), body=None), "fail", "API key rejected"),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "fail", "cannot reach"),
    (anthropic.RateLimitError("slow", response=_resp(429), body=None), "warn", "rate limited"),
    (anthropic.BadRequestError("unknown model", response=_resp(400), body=None), "fail", "bad request"),
])
def test_model_request_errors_are_classified(workdir, err, status, needle):
    async def create(**kw):
        raise err
    checks = run(doctor.remote_checks(_agent_with_create(create)))
    c = by_name(checks, "model request")
    assert c.status == status and needle in c.detail
    assert doctor.exit_code(checks) == (1 if status == "fail" else 0)


def test_model_request_ok_sends_minimal_request(workdir):
    sent = {}

    async def create(**kw):
        sent.update(kw)
        return object()
    checks = run(doctor.remote_checks(_agent_with_create(create, openai_mode=True)))
    assert by_name(checks, "model request").status == "ok"
    assert sent["max_tokens"] == 1
    assert by_name(checks, "MCP servers").detail == "none configured"


def test_mcp_servers_are_checked(workdir):
    (workdir / ".mcp.json").write_text(json.dumps({"mcpServers": {
        "good": {"command": sys.executable, "args": ["-I", FAKE_MCP]},
        "broken": {"command": "/nonexistent/binary"},
    }}))

    async def create(**kw):
        return object()
    checks = run(doctor.remote_checks(_agent_with_create(create)))
    assert by_name(checks, "MCP server good").status == "ok"
    assert by_name(checks, "MCP server broken").status == "fail"
    print_doctor(checks)
