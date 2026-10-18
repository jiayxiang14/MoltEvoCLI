"""第四批：统一主循环、稳定的系统提示词与提示词缓存断点。"""

import asyncio

import pytest

import agents.agent as agent_mod
from agents.memory import get_memory_dir
from agents.tool_result import ToolResult
from fakemodel import anth, install, make_agent, oai, read_events


def run(coro):
    return asyncio.run(coro)


def count_prompt_builds(monkeypatch):
    calls = {"n": 0}
    real = agent_mod.build_system_prompt

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(agent_mod, "build_system_prompt", counting)
    return calls


# ── 稳定的系统提示词 ──────────────────────────────────────

def test_system_prompt_not_rebuilt_between_tool_rounds(workdir, monkeypatch):
    a = make_agent()
    builds = count_prompt_builds(monkeypatch)
    systems = []
    real_build = a._build_anthropic_request

    def spy():
        req = real_build()
        systems.append(req["system"])
        return req

    a._build_anthropic_request = spy
    script = [anth(tools=[("write_file", {"file_path": "a.txt", "content": "1"})]),
              anth(tools=[("run_shell", {"command": "echo hi"})]),
              anth(text="done")]
    seq = iter(script)

    async def fake_stream(on_tool_block_complete=None, **k):
        a._build_anthropic_request()          # 真实流式调用里构造请求的那一步
        return next(seq)

    a._call_anthropic_stream = fake_stream
    run(a.chat("go"))
    assert builds["n"] == 0                   # 改了文件、跑了 shell，系统提示词也不重建
    assert len(systems) == 3 and len(set(map(str, systems))) == 1


def test_memory_write_refreshes_system_prompt(workdir, monkeypatch):
    a = make_agent()
    builds = count_prompt_builds(monkeypatch)
    mem_file = get_memory_dir() / "note.md"
    install(monkeypatch, [anth(tools=[("write_file", {"file_path": str(mem_file), "content": "---\nname: n\n---\nx"})]),
                          anth(text="ok")])
    run(a.chat("remember this"))
    assert builds["n"] == 1


# ── 缓存断点与运行时提示 ──────────────────────────────────

def test_anthropic_cache_breakpoints_for_claude(workdir, monkeypatch):
    monkeypatch.delenv("MOLTEVO_PROMPT_CACHE", raising=False)
    a = make_agent(model="claude-sonnet-4-6")
    a._anthropic_messages = [{"role": "user", "content": "hello"},
                             {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]}]
    req = a._build_anthropic_request()
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in req["tools"][0]
    last_blocks = req["messages"][-1]["content"]
    assert last_blocks[0]["cache_control"] == {"type": "ephemeral"}      # 断点在真实内容上
    assert last_blocks[-1]["text"].startswith("<system-reminder>")        # 运行时提示在断点之后
    assert "cache_control" not in last_blocks[-1]
    # 历史本身不被修改
    assert a._anthropic_messages[-1]["content"] == [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]
    assert "cache_control" not in str(a._anthropic_messages)


def test_no_breakpoints_for_non_claude_unless_forced(workdir, monkeypatch):
    monkeypatch.delenv("MOLTEVO_PROMPT_CACHE", raising=False)
    a = make_agent(model="deepseek-chat")
    a._anthropic_messages = [{"role": "user", "content": "hello"}]
    req = a._build_anthropic_request()
    assert isinstance(req["system"], str) and "cache_control" not in str(req)
    assert req["messages"][-1]["content"][-1]["text"].startswith("<system-reminder>")
    monkeypatch.setenv("MOLTEVO_PROMPT_CACHE", "1")
    assert "cache_control" in str(a._build_anthropic_request())


def test_guidance_not_in_system_prompt():
    a = make_agent()
    assert "Runtime Fold Guidance" not in a._system_prompt
    a._refresh_runtime_system_prompt()
    assert "Runtime Fold Guidance" not in a._system_prompt


def test_openai_guidance_only_in_request_copy(workdir):
    a = make_agent(openai_mode=True)
    a._openai_messages += [{"role": "user", "content": "q"},
                           {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]},
                           {"role": "tool", "tool_call_id": "c", "content": "result"}]
    msgs = a._build_openai_request_messages()
    assert msgs[-1]["content"].startswith("result") and "Runtime Fold Guidance" in msgs[-1]["content"]
    assert a._openai_messages[-1]["content"] == "result"
    assert msgs[0]["content"] == a._openai_messages[0]["content"]       # 系统提示词前缀不变


# ── 统一主循环 ───────────────────────────────────────────

def test_consecutive_read_tools_batched_and_ordered_without_early(workdir, monkeypatch):
    for n in "abc":
        (workdir / f"{n}.txt").write_text(n * 3)
    a = make_agent()
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "a.txt"}), ("read_file", {"file_path": "b.txt"}),
                                      ("run_shell", {"command": "echo mid"}),
                                      ("read_file", {"file_path": "c.txt"})]),
                          anth(text="ok")], early=False)
    run(a.chat("go"))
    results = a._anthropic_messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_0_read_file", "toolu_1_read_file",
                                                   "toolu_2_run_shell", "toolu_3_read_file"]
    assert "aaa" in results[0]["content"] and "bbb" in results[1]["content"] and "ccc" in results[3]["content"]
    tools = [e for e in read_events(a) if e["type"] == "tool_call"]
    assert all(not e["early"] for e in tools)


def test_denied_call_inside_read_batch_keeps_others(workdir, monkeypatch):
    (workdir / "ok.txt").write_text("fine")
    a = make_agent(openai_mode=True)
    real = agent_mod.check_permission

    def fake_check(name, inp, mode, plan_path=None):
        if inp.get("file_path") == "secret.txt":
            return {"action": "deny", "message": "no secrets"}
        return real(name, inp, mode, plan_path)

    monkeypatch.setattr(agent_mod, "check_permission", fake_check)
    install(monkeypatch, [oai(tools=[("read_file", {"file_path": "secret.txt"}),
                                     ("read_file", {"file_path": "ok.txt"})]),
                          oai(text="ok")])
    run(a.chat("go"))
    tool_msgs = [m for m in a._openai_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_msgs[0]["content"].startswith("[tool error] Action denied: no secrets")
    assert "fine" in tool_msgs[1]["content"]


def test_openai_unsupported_call_type_gets_error_and_model_continues(workdir, monkeypatch):
    a = make_agent(openai_mode=True)
    resp = oai(text="x")
    resp["choices"][0]["message"]["tool_calls"] = [{"id": "w1", "type": "web_search", "function": {}}]
    script = install(monkeypatch, [resp, oai(text="ok, without search")])
    run(a.chat("go"))
    tool_msgs = [m for m in a._openai_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["w1"]
    assert tool_msgs[0]["content"].startswith("[tool error] Unknown tool")
    assert len(script.requests) == 2
