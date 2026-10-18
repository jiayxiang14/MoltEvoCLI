"""剧本式假模型：替换 Agent 的流式调用，按顺序返回预设的回复，不联网跑通两套主循环。"""

import json
from types import SimpleNamespace as NS

from agents.agent import Agent


def make_agent(*, openai_mode: bool = False, **kw) -> Agent:
    kw.setdefault("api_key", "sk-test")
    kw.setdefault("permission_mode", "bypassPermissions")
    if openai_mode:
        kw.setdefault("api_base", "http://fake/v1")
    else:
        kw.setdefault("anthropic_base_url", "https://fake.example/anthropic")
    a = Agent(**kw)
    a._mcp_initialized = True
    a._auto_save = lambda: None
    return a


def oai(text=None, tools=None, usage=None):
    calls = None
    if tools:
        calls = [{"id": f"call_{i}_{name}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(args)}}
                 for i, (name, args) in enumerate(tools)]
    return {
        "choices": [{"message": {"role": "assistant", "content": text, "tool_calls": calls},
                     "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 0},
    }


def anth(text=None, tools=None, usage=None):
    content = []
    if text:
        content.append(NS(type="text", text=text))
    for i, (name, args) in enumerate(tools or []):
        content.append(NS(type="tool_use", id=f"toolu_{i}_{name}", name=name, input=args))
    u = usage or {}
    return NS(
        content=content,
        stop_reason="tool_use" if tools else "end_turn",
        usage=NS(input_tokens=u.get("input_tokens", 1000), output_tokens=u.get("output_tokens", 100),
                 cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
                 cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0)),
    )


class Script:
    """按 agent 标签分发剧本：主 agent 用 "main"，子 agent 用 "sub:<type>" 等。"""

    def __init__(self, scripts: dict[str, list]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests: list[tuple[str, list]] = []

    def next_for(self, agent: Agent):
        label = agent._agent_label
        key = label if label in self.scripts else label.split(":")[0]
        self.requests.append((label, list(agent._openai_messages if agent.use_openai else agent._anthropic_messages)))
        return self.scripts[key].pop(0)


def install(monkeypatch, scripts: dict[str, list] | list, *, early: bool = True) -> Script:
    """在 Agent 类上替换流式调用，这样子 agent 也会走假模型。"""
    script = Script(scripts if isinstance(scripts, dict) else {"main": scripts})

    async def fake_anth(self, on_tool_block_complete=None, **k):
        resp = script.next_for(self)
        if early and on_tool_block_complete:
            for b in resp.content:
                if b.type == "tool_use":
                    on_tool_block_complete({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return resp

    async def fake_oai(self, *a, **k):
        return script.next_for(self)

    monkeypatch.setattr(Agent, "_call_anthropic_stream", fake_anth)
    monkeypatch.setattr(Agent, "_call_openai_stream", fake_oai)
    return script


def read_events(agent: Agent) -> list[dict]:
    path = agent._recorder.path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
