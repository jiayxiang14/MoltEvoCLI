"""
启动自检（moltevo --doctor）：把"看起来像模型出错、其实是环境没配好"的问题提前查出来。
思路参照 anthropics/claude-quickstarts 的 computer-use-best-practices/computer_use/preflight.py。

- 本地检查（不花钱，启动时自动跑，只在有问题时提示）：grep / git 是否存在、会话目录是否可写、
  模型是否在价格表和上下文窗口表里。
- 远程检查（只在 --doctor 时跑）：发一个 max_tokens=1 的最小请求，确认密钥、地址和模型名都被接受；
  逐个连接配置的 MCP 服务。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import anthropic
import openai

from agents.pricing import get_price
from agents.session import SESSION_DIR


@dataclass
class Check:
    name: str
    status: str          # "ok" | "warn" | "fail"
    detail: str
    fix: str = ""


def local_checks(model: str) -> list[Check]:
    from agents.agent import MODEL_CONTEXT
    checks = []
    for tool, why in (("grep", "grep_search falls back to a slower Python search"),
                      ("git", "git context and git commands are unavailable")):
        path = shutil.which(tool)
        checks.append(Check(f"{tool} available", "ok" if path else "warn",
                            path or f"not found: {why}", "" if path else f"install {tool}"))
    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=SESSION_DIR, delete=True):
            pass
        checks.append(Check("session dir writable", "ok", str(SESSION_DIR)))
    except OSError as e:
        checks.append(Check("session dir writable", "fail", f"{SESSION_DIR}: {e}",
                            "sessions, /undo and /rewind need this directory"))
    price = get_price(model)
    checks.append(Check("price known", "ok" if price else "warn",
                        f"{model}: ${price.input}/${price.output} per MTok" if price
                        else f"{model} is not in the price table; cost shows as unknown and --max-cost can't be enforced",
                        "" if price else 'add it under "pricing" in .moltevo/settings.json'))
    known_window = model in MODEL_CONTEXT
    checks.append(Check("context window known", "ok" if known_window else "warn",
                        f"{model}: {MODEL_CONTEXT.get(model, 200000)} tokens"
                        + ("" if known_window else " (assumed default)"),
                        "" if known_window else "compaction thresholds may be off for this model"))
    return checks


def _classify_error(e: Exception, model: str) -> Check:
    name = "model request"
    if isinstance(e, (anthropic.AuthenticationError, openai.AuthenticationError)):
        return Check(name, "fail", "API key rejected", "check APIKEY / ANTHROPIC_API_KEY / OPENAI_API_KEY")
    if isinstance(e, (anthropic.PermissionDeniedError, openai.PermissionDeniedError)):
        return Check(name, "fail", f"no access to {model}", "check the account's model access")
    if isinstance(e, (anthropic.NotFoundError, openai.NotFoundError)):
        return Check(name, "fail", f"model or endpoint not found: {model}", "check --model / MODEL and the base URL")
    if isinstance(e, (anthropic.BadRequestError, openai.BadRequestError)):
        hint = "check the model name" if "model" in str(e).lower() else "the endpoint rejected a minimal request"
        return Check(name, "fail", f"bad request: {e}", hint)
    if isinstance(e, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return Check(name, "fail", f"cannot reach the API: {e}", "check the base URL and network / proxy")
    if isinstance(e, (anthropic.RateLimitError, openai.RateLimitError)):
        return Check(name, "warn", "rate limited (the key and model are valid)", "")
    return Check(name, "fail", f"{type(e).__name__}: {e}", "")


async def remote_checks(agent: Any) -> list[Check]:
    checks: list[Check] = []
    started = time.monotonic()
    try:
        if agent.use_openai:
            await agent._openai_client.chat.completions.create(
                model=agent.model, max_tokens=1, messages=[{"role": "user", "content": "ping"}])
            endpoint = str(agent._openai_client.base_url)
        else:
            await agent._anthropic_client.messages.create(
                model=agent.model, max_tokens=1, messages=[{"role": "user", "content": "ping"}])
            endpoint = str(agent._anthropic_client.base_url)
        protocol = "OpenAI" if agent.use_openai else "Anthropic"
        checks.append(Check("model request", "ok",
                            f"{agent.model} via {protocol} protocol at {endpoint} "
                            f"({(time.monotonic() - started) * 1000:.0f} ms)"))
    except Exception as e:  # noqa: BLE001 - every failure is reported, not raised
        checks.append(_classify_error(e, agent.model))

    from agents.mcp_client import McpManager
    manager = McpManager()
    configs = manager._load_configs()
    if not configs:
        checks.append(Check("MCP servers", "ok", "none configured"))
    else:
        await manager.load_and_connect()
        for name in configs:
            connected = name in manager._connections
            tools = sum(1 for t in manager._tools if t.get("serverName") == name)
            checks.append(Check(f"MCP server {name}", "ok" if connected else "fail",
                                f"{tools} tools" if connected else "failed to connect",
                                "" if connected else "check its command / args in .mcp.json or settings.json"))
        await manager.disconnect_all()
    return checks


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "fail" for c in checks) else 0


def env_summary() -> str:
    keys = [k for k in ("APIKEY", "API", "MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
                        "OPENAI_API_KEY", "OPENAI_BASE_URL") if os.environ.get(k)]
    return ", ".join(keys) or "none"
