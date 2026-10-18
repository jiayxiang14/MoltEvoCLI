"""第三批 3a：结构化工具结果、事件日志、按模型计价、两套主循环的消息完整性。"""

import asyncio
import json

import pytest

from agents.agent import Agent, _with_retry
from agents.events import RunRecorder, redact
from agents.pricing import call_cost_usd, get_price
from agents.tool_result import ToolResult
from fakemodel import anth, install, make_agent, oai, read_events


def run(coro):
    return asyncio.run(coro)


def tool_results_of(agent):
    """Anthropic 历史里所有 tool_result block。"""
    out = []
    for m in agent._anthropic_messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            out += [b for b in m["content"] if b.get("type") == "tool_result"]
    return out


def tool_messages_of(agent):
    return [m for m in agent._openai_messages if isinstance(m, dict) and m.get("role") == "tool"]


# ── is_error ────────────────────────────────────────────

def test_anthropic_tool_result_carries_is_error(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, [anth(tools=[("run_shell", {"command": "exit 2"}),
                                      ("run_shell", {"command": "echo '0 errors'"})]),
                          anth(text="done")])
    run(a.chat("go"))
    res = tool_results_of(a)
    assert res[0].get("is_error") is True
    assert "is_error" not in res[1]          # 输出里有 "errors" 字样也不算失败


def test_openai_error_prefix(workdir, monkeypatch):
    a = make_agent(openai_mode=True)
    install(monkeypatch, [oai(tools=[("read_file", {"file_path": "missing.txt"})]), oai(text="done")])
    run(a.chat("go"))
    assert tool_messages_of(a)[0]["content"].startswith("[tool error] ")


# ── OpenAI 主循环的结构性 bug ───────────────────────────

def test_openai_multiple_tool_calls_execute_once_each(workdir, monkeypatch):
    a = make_agent(openai_mode=True)
    install(monkeypatch, [oai(tools=[("run_shell", {"command": "echo A >> log"}),
                                     ("run_shell", {"command": "echo B >> log"})]),
                          oai(text="done")])
    run(a.chat("go"))
    assert (workdir / "log").read_text().split() == ["A", "B"]
    ids = [m["tool_call_id"] for m in tool_messages_of(a)]
    assert ids == ["call_0_run_shell", "call_1_run_shell"]


def test_openai_concurrent_batch_survives_one_failure(workdir, monkeypatch):
    (workdir / "ok.txt").write_text("fine")
    a = make_agent(openai_mode=True)

    real = a._execute_tool_call

    async def flaky(name, inp):
        if inp.get("file_path") == "boom.txt":
            raise RuntimeError("disk on fire")
        return await real(name, inp)

    a._execute_tool_call = flaky
    install(monkeypatch, [oai(tools=[("read_file", {"file_path": "boom.txt"}),
                                     ("read_file", {"file_path": "ok.txt"})]),
                          oai(text="done")])
    run(a.chat("go"))
    msgs = tool_messages_of(a)
    assert msgs[0]["content"].startswith("[tool error] Error executing tool: disk on fire")
    assert "fine" in msgs[1]["content"]


def test_openai_budget_exceeded_still_answers_every_tool_call(workdir, monkeypatch):
    a = make_agent(openai_mode=True, max_turns=0)
    install(monkeypatch, [oai(tools=[("list_files", {"pattern": "*"}), ("list_files", {"pattern": "*.py"})])])
    run(a.chat("go"))
    msgs = tool_messages_of(a)
    assert [m["tool_call_id"] for m in msgs] == ["call_0_list_files", "call_1_list_files"]
    assert all("Tool execution skipped" in m["content"] for m in msgs)


def test_openai_dangling_tool_calls_are_repaired():
    msgs = [{"role": "system", "content": "s"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "tool_call_id": "a", "content": "x"},
            {"role": "user", "content": "next"}]
    out = Agent._normalize_openai_messages(msgs)
    assert [m.get("tool_call_id") for m in out if m.get("role") == "tool"] == ["a", "b"]
    assert out[-1] == {"role": "user", "content": "next"}


def test_denied_tool_is_error_and_logged(workdir, monkeypatch):
    a = make_agent(permission_mode="plan")
    install(monkeypatch, [anth(tools=[("run_shell", {"command": "ls"})]), anth(text="ok")])
    run(a.chat("go"))
    res = tool_results_of(a)[0]
    assert res["is_error"] and "Action denied" in res["content"]
    perms = [e for e in read_events(a) if e["type"] == "permission"]
    assert perms[0]["decision"] == "deny"


# ── 事件日志 ─────────────────────────────────────────────

def test_event_log_fields_and_trace(workdir, monkeypatch):
    a = make_agent(model="claude-sonnet-4-6")
    install(monkeypatch, [anth(tools=[("run_shell", {"command": "echo key=sk-abcdefghijklmnop"})]),
                          anth(text="done")])
    run(a.chat("go"))
    events = read_events(a)
    types = [e["type"] for e in events]
    assert types[0] == "turn_start" and types[-1] == "turn_end"
    assert types.count("model_call") == 2 and "tool_call" in types and "permission" in types
    trace_ids = {e["trace_id"] for e in events if "trace_id" in e}
    assert len(trace_ids) == 1
    tool = next(e for e in events if e["type"] == "tool_call")
    assert tool["tool"] == "run_shell" and tool["is_error"] is False and tool["exit_code"] == 0
    assert tool["duration_ms"] is not None and tool["early"] is False
    assert "sk-abcdefghijklmnop" not in tool["args"] and "sk-***" in tool["args"]
    call = next(e for e in events if e["type"] == "model_call")
    assert call["model"] == "claude-sonnet-4-6" and call["input_tokens"] == 1000
    assert call["latency_ms"] is not None


def test_early_execution_is_logged(workdir, monkeypatch):
    (workdir / "x.txt").write_text("x")
    a = make_agent()
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "x.txt"})]), anth(text="ok")])
    run(a.chat("go"))
    tool = next(e for e in read_events(a) if e["type"] == "tool_call")
    assert tool["early"] is True


def test_sub_agent_events_link_to_parent_span(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, {
        "main": [anth(tools=[("agent", {"type": "general", "description": "d", "prompt": "list files"})]),
                 anth(text="all done")],
        "sub": [anth(tools=[("list_files", {"pattern": "*"})]), anth(text="found nothing")],
    })
    run(a.chat("go"))
    events = read_events(a)
    agent_call = next(e for e in events if e["type"] == "tool_call" and e["tool"] == "agent")
    sub_tool = next(e for e in events if e["type"] == "tool_call" and e["tool"] == "list_files")
    assert sub_tool["agent"] == "sub:general"
    assert sub_tool["parent_span_id"] == agent_call["span_id"]
    assert sub_tool["trace_id"] == agent_call["trace_id"]


def test_redaction():
    assert redact("curl -H 'Authorization: Bearer abcdefghijkl'") == "curl -H 'Authorization: Bearer ***'"
    assert redact("API_KEY=hunter2 run") == "API_KEY=*** run"
    assert redact('{"password": "x"}') == '{"password": ***}'


# ── 计价 ─────────────────────────────────────────────────

def test_price_lookup_longest_prefix():
    assert get_price("claude-opus-4-1-20250805").input == 15
    assert get_price("claude-opus-4-20250514").input == 15
    assert get_price("claude-opus-4-6").input == 5
    assert get_price("gpt-4o-mini").input == 0.15
    assert get_price("gpt-4o-2024-08-06").input == 2.50
    assert get_price("deepseek-chat") is None


def test_call_cost_includes_cache_and_tiers():
    cost = call_cost_usd("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000,
                         cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert cost == pytest.approx(3 + 15 + 0.30 + 3.75)
    low = call_cost_usd("claude-haiku-5-5", input_tokens=50_000, output_tokens=0)
    high = call_cost_usd("claude-haiku-5-5", input_tokens=150_000, output_tokens=0)
    assert low == pytest.approx(50_000 * 0.10 / 1e6)
    assert high == pytest.approx(150_000 * 0.50 / 1e6)
    assert call_cost_usd("unknown-model", input_tokens=1, output_tokens=1) is None


def test_agent_cost_per_call_with_cache(workdir, monkeypatch):
    a = make_agent(model="claude-sonnet-4-6")
    install(monkeypatch, [anth(text="hi", usage={"input_tokens": 1000, "output_tokens": 500,
                                                  "cache_read_input_tokens": 9000})])
    run(a.chat("go"))
    expected = (1000 * 3 + 500 * 15 + 9000 * 0.30) / 1e6
    assert a._get_current_cost_usd() == pytest.approx(expected)
    assert a.total_input_tokens == 10_000              # 完整提示词 = 未缓存 + 缓存读
    assert a.last_input_token_count == 10_000
    assert a.get_run_stats()["cache_read_tokens"] == 9000


def test_openai_cached_tokens_split(workdir, monkeypatch):
    a = make_agent(openai_mode=True, model="gpt-4o")
    install(monkeypatch, [oai(text="hi", usage={"prompt_tokens": 1000, "completion_tokens": 10,
                                                 "cached_tokens": 800})])
    run(a.chat("go"))
    assert a._get_current_cost_usd() == pytest.approx((200 * 2.5 + 800 * 1.25 + 10 * 10) / 1e6)


def test_unknown_price_warns_and_reports_unknown(workdir, monkeypatch, capsys):
    a = make_agent(model="deepseek-chat", max_cost_usd=1.0, is_sub_agent=False)
    out = capsys.readouterr().out
    assert "不在价格表中" in out and "--max-cost" in out
    install(monkeypatch, [anth(text="hi")])
    run(a.chat("go"))
    assert a.get_run_stats()["cost_usd"] is None
    assert a._format_cost().startswith("unknown")


def test_sub_agent_spend_counts_against_parent_budget(workdir, monkeypatch):
    # 每次调用 1000 输入 + 100 输出，sonnet 价格下约 $0.0045；上限设 $0.008：
    # 主 agent 一次 + 子 agent 一次就超了，子 agent 应在自己的第一轮工具前停下。
    a = make_agent(model="claude-sonnet-4-6", max_cost_usd=0.008)
    script = install(monkeypatch, {
        "main": [anth(tools=[("agent", {"type": "general", "description": "d", "prompt": "p"})]),
                 anth(text="done")],
        "sub": [anth(tools=[("run_shell", {"command": "echo SHOULD_NOT_RUN > ran"})]), anth(text="never")],
    })
    run(a.chat("go"))
    assert not (workdir / "ran").exists()
    assert [label for label, _ in script.requests].count("sub:general") == 1


def test_retry_counter():
    calls = {"n": 0}
    retries = []

    class E(Exception):
        status_code = 529

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise E()
        return "ok"

    async def go():
        import agents.agent as ag
        orig = ag.asyncio.sleep

        async def fast_sleep(_):
            await orig(0)
        ag.asyncio.sleep = fast_sleep
        try:
            return await _with_retry(flaky, on_retry=lambda: retries.append(1))
        finally:
            ag.asyncio.sleep = orig
    assert run(go()) == "ok" and len(retries) == 2


# ── 统计输出 ─────────────────────────────────────────────

def test_run_stats_and_print(workdir, monkeypatch):
    from agents.ui import print_stats
    a = make_agent(model="claude-sonnet-4-6")
    install(monkeypatch, [anth(tools=[("run_shell", {"command": "exit 1"}),
                                      ("run_shell", {"command": "true"})]),
                          anth(text="done")])
    run(a.chat("go"))
    st = a.get_run_stats()
    assert st["tool_calls"] == 2 and st["tool_errors"] == 1 and st["tool_error_rate"] == 0.5
    assert st["model_calls"] == 2 and st["tools"]["run_shell"]["calls"] == 2
    print_stats(st)


def test_eval_runner_stats_fields(workdir, monkeypatch):
    # agents/eval/ 目前还没进仓库；干净检出里没有它时跳过。
    runner = pytest.importorskip("agents.eval.runner")
    a = make_agent(model="claude-sonnet-4-6")
    install(monkeypatch, [anth(text="hi")])
    run(a.chat("go"))
    fields = runner._stats_fields(a)
    assert fields["model"] == "claude-sonnet-4-6" and fields["model_calls"] == 1
    assert fields["cost_usd"] is not None and "tool_errors" in fields


def test_compaction_event(workdir, monkeypatch):
    a = make_agent()

    async def fake_compact(*, trigger):
        a._anthropic_messages = a._anthropic_messages[-1:]
        return True

    a._compact_anthropic = fake_compact
    a._anthropic_messages = [{"role": "user", "content": str(i)} for i in range(6)]
    a._trace_id = "t1"
    assert run(a._compact_conversation(trigger="tool"))
    ev = next(e for e in read_events(a) if e["type"] == "compaction")
    assert ev["trigger"] == "tool" and ev["messages_before"] == 6 and ev["messages_after"] == 1
