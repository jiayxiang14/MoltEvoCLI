"""第三批审查发现的问题的回归测试。"""

import asyncio
import builtins
from types import SimpleNamespace as NS

from agents import main as main_mod
from agents.tools import tool_definitions
from fakemodel import anth, install, make_agent, oai, read_events


def run(coro):
    return asyncio.run(coro)


def test_extract_now_confirm_respects_no_via_real_repl_confirm(workdir, monkeypatch):
    # 回归：REPL 的 confirm_fn 返回字符串，原来 bool("n") 为真，输入 n 也会写入 Skill。
    a = make_agent(permission_mode="default")
    a.set_confirm_fn(main_mod.repl_confirm)
    answers = iter(["n", "y"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert run(a._confirm_online_skill_write("create skill X")) is False
    assert run(a._confirm_online_skill_write("create skill X")) is True


def test_openai_missing_usage_is_unknown_cost_not_zero(workdir, monkeypatch):
    a = make_agent(openai_mode=True, model="gpt-4o")
    a.last_input_token_count = 12345
    resp = oai(text="hi")
    resp["usage"] = None
    install(monkeypatch, [resp])
    run(a.chat("go"))
    assert a.get_run_stats()["cost_usd"] is None
    assert a.last_input_token_count == 12345        # 保留上一次的值，压缩判断不受影响
    call = next(e for e in read_events(a) if e["type"] == "model_call")
    assert call["usage_missing"] is True and call["cost_usd"] is None


def test_accept_all_in_sub_agent_applies_to_parent(workdir, monkeypatch):
    (workdir / "a.txt").write_text("1\n")
    (workdir / "b.txt").write_text("1\n")
    a = make_agent(permission_mode="default")
    asked = []

    async def cf(m, *, preview=None, offer_all=False):
        asked.append(m)
        return "a"

    a.set_confirm_fn(cf)
    install(monkeypatch, {
        "main": [anth(tools=[("agent", {"type": "general", "description": "d", "prompt": "p"})]),
                 anth(tools=[("read_file", {"file_path": "b.txt"})]),
                 anth(tools=[("edit_file", {"file_path": "b.txt", "old_string": "1", "new_string": "2"})]),
                 anth(text="ok")],
        "sub": [anth(tools=[("read_file", {"file_path": "a.txt"})]),
                anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "1", "new_string": "2"})]),
                anth(text="done")],
    })
    run(a.chat("go"))
    assert asked == ["edit file: a.txt"]
    assert a.permission_mode == "acceptEdits"
    assert (workdir / "b.txt").read_text() == "2\n"


def test_permission_reason_is_redacted(workdir, monkeypatch):
    a = make_agent(permission_mode="default")

    async def cf(m, **kw):
        return "y"

    a.set_confirm_fn(cf)
    install(monkeypatch, [anth(tools=[("run_shell", {"command": "sudo curl -H 'Authorization: Bearer abcdefghijklmn' x"})]),
                          anth(text="ok")])
    run(a.chat("go"))
    perm = next(e for e in read_events(a) if e["type"] == "permission")
    assert "abcdefghijklmn" not in perm["reason"] and "Bearer ***" in perm["reason"]


def test_side_queries_are_billed_but_kept_out_of_main_tokens(workdir):
    a = make_agent(model="claude-sonnet-4-6")

    async def fake_create(**kw):
        return NS(content=[NS(type="text", text="[]")], stop_reason="end_turn",
                  usage=NS(input_tokens=2000, output_tokens=50, cache_read_input_tokens=0,
                           cache_creation_input_tokens=0))

    a._anthropic_client.messages.create = fake_create
    sq = a._build_side_query(purpose="memory_select")
    run(sq("system", "user"))
    st = a.get_run_stats()
    assert st["cost_usd"] > 0 and st["side_queries"] == 1
    assert st["side_query_tokens"] == {"input": 2000, "output": 50}
    assert a.total_input_tokens == 0 and st["model_calls"] == 0
    ev = next(e for e in read_events(a) if e["type"] == "side_query")
    assert ev["purpose"] == "memory_select"


def test_side_query_openai_usage_and_budget(workdir):
    a = make_agent(openai_mode=True, model="gpt-4o", max_cost_usd=0.0001)

    async def fake_create(**kw):
        return NS(choices=[NS(message=NS(content="ok"), finish_reason="stop")],
                  usage=NS(prompt_tokens=1000, completion_tokens=10, prompt_tokens_details=None))

    a._openai_client.chat.completions.create = fake_create
    run(a._build_side_query(purpose="fold")("s", "u"))
    assert a._check_budget()["exceeded"]           # 侧查询的花费计入 --max-cost


def test_sub_agent_parent_span_with_concurrent_early_tool(workdir, monkeypatch):
    # agent 工具排在一个提前执行的只读工具前面，两者并发：子 agent 仍要挂到 agent 调用的 span 上。
    (workdir / "x.txt").write_text("x")
    a = make_agent()
    install(monkeypatch, {
        "main": [anth(tools=[("agent", {"type": "general", "description": "d", "prompt": "p"}),
                             ("read_file", {"file_path": "x.txt"})]),
                 anth(text="ok")],
        "sub": [anth(tools=[("list_files", {"pattern": "*"})]), anth(text="done")],
    })
    run(a.chat("go"))
    events = read_events(a)
    agent_span = next(e for e in events if e["type"] == "tool_call" and e["tool"] == "agent")["span_id"]
    sub_events = [e for e in events if e.get("agent") == "sub:general"]
    assert sub_events and all(e["parent_span_id"] == agent_span for e in sub_events)


def test_run_shell_description_mentions_background_redirect():
    desc = next(t for t in tool_definitions if t["name"] == "run_shell")["description"]
    assert "2>&1 &" in desc and "killed" in desc
