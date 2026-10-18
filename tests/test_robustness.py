"""借鉴 anthropics/claude-quickstarts 的四处改进：按区间裁剪、按异常类型重试、空回复恢复、角色交替。"""

import asyncio
import json

import anthropic
import httpx
import openai
import pytest

from agents.agent import _is_retryable
from fakemodel import anth, install, make_agent, oai


def run(coro):
    return asyncio.run(coro)


# ── 1. 按区间裁剪：前缀每个区间只变一次 ─────────────────────────────

def _add_result(a, i):
    a._anthropic_messages.append({"role": "assistant", "content": [
        {"type": "tool_use", "id": f"t{i}", "name": "run_shell", "input": {"command": f"echo {i}"}}]})
    a._anthropic_messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"output {i} " + "x" * 50}]})


def test_snip_changes_prefix_only_once_per_interval():
    a = make_agent()
    a.last_input_token_count = int(a.effective_window * 0.65)      # 超过 60% 裁剪阈值
    a._anthropic_messages = [{"role": "user", "content": "start"}]
    prefix_changes = 0
    for i in range(30):
        before = json.dumps(a._anthropic_messages)
        a._snip_stale_results_anthropic()
        if json.dumps(a._anthropic_messages) != before:
            prefix_changes += 1
        _add_result(a, i)
    # 原来超过 3 条后每轮都裁一条（约 27 次改写）；按区间裁剪只在每个区间回落时改写一次
    assert prefix_changes <= 4, prefix_changes


def test_snip_still_bounds_unsnipped_results():
    a = make_agent()
    a.last_input_token_count = int(a.effective_window * 0.65)
    a._anthropic_messages = [{"role": "user", "content": "start"}]
    for i in range(30):
        _add_result(a, i)
        a._snip_stale_results_anthropic()
    kept = [b for m in a._anthropic_messages if isinstance(m["content"], list) for b in m["content"]
            if b.get("type") == "tool_result" and "output" in b["content"]]
    assert 3 <= len(kept) <= 3 + 8


def test_openai_snip_interval():
    a = make_agent(openai_mode=True)
    a.last_input_token_count = int(a.effective_window * 0.65)
    changes = 0
    for i in range(30):
        before = json.dumps(a._openai_messages)
        a._snip_stale_results_openai()
        changes += json.dumps(a._openai_messages) != before
        a._openai_messages += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]},
                               {"role": "tool", "tool_call_id": f"c{i}", "content": f"out {i}"}]
    assert changes <= 4, changes


# ── 2. 按异常类型判断能否重试 ─────────────────────────────────────

def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "https://api.example/v1"))


@pytest.mark.parametrize("err,expected", [
    (anthropic.InternalServerError("boom", response=_resp(500), body=None), True),
    (anthropic.APIStatusError("bad gateway", response=_resp(502), body=None), True),
    (anthropic.RateLimitError("slow down", response=_resp(429), body=None), True),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), True),
    (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), True),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), True),
    (openai.InternalServerError("boom", response=_resp(500), body=None), True),
    (httpx.RemoteProtocolError("peer closed connection"), True),
    (anthropic.BadRequestError("bad", response=_resp(400), body=None), False),
    (anthropic.AuthenticationError("key", response=_resp(401), body=None), False),
    (openai.NotFoundError("no such model", response=_resp(404), body=None), False),
    (ValueError("bug in our code"), False),
])
def test_retry_classification(err, expected):
    assert _is_retryable(err) is expected


# ── 3. 空回复：不写进历史，提示模型继续，超过上限才放弃 ──────────────────

def test_empty_response_is_retried_with_nudge(workdir, monkeypatch):
    a = make_agent()
    script = install(monkeypatch, [anth(), anth(text="real answer")])
    run(a.chat("hello"))
    assert len(script.requests) == 2
    assistants = [m for m in a._anthropic_messages if m["role"] == "assistant"]
    assert all(m["content"] for m in assistants)                     # 没有空的助手消息
    assert "real answer" in str(assistants[-1]["content"])


def test_openai_empty_response_retried(workdir, monkeypatch):
    a = make_agent(openai_mode=True)
    script = install(monkeypatch, [oai(), oai(text="ok")])
    run(a.chat("hello"))
    assert len(script.requests) == 2
    assert not any(m.get("role") == "assistant" and not m.get("content") and not m.get("tool_calls")
                   for m in a._openai_messages if isinstance(m, dict))


def test_repeated_empty_responses_give_up(workdir, monkeypatch, capsys):
    a = make_agent()
    script = install(monkeypatch, [anth(), anth(), anth(), anth()])
    run(a.chat("hello"))
    assert len(script.requests) == 3                                   # 首次 + 重试 2 次
    assert "empty" in capsys.readouterr().out.lower()


# ── 4. 角色交替：不出现相邻的用户消息 ───────────────────────────────

def _no_adjacent_users(messages):
    roles = [m["role"] for m in messages]
    return all(not (x == "user" and y == "user") for x, y in zip(roles, roles[1:]))


def test_no_adjacent_user_messages_after_budget_stop(workdir, monkeypatch):
    a = make_agent(max_turns=0)
    script = install(monkeypatch, [anth(tools=[("list_files", {"pattern": "*"})]), anth(text="second")])
    run(a.chat("first"))                     # 预算超限：本轮以工具结果（用户消息）结束
    run(a.chat("again"))
    assert all(_no_adjacent_users(req) for _, req in script.requests)
    assert _no_adjacent_users(a._anthropic_messages)


def test_no_adjacent_user_messages_after_compaction(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, [anth(text="one"), anth(text="after compact")])
    run(a.chat("first"))

    async def fake_compact(*, trigger):
        a._anthropic_messages = [{"role": "user", "content": "folded memory"}]
        return True

    a._compact_anthropic = fake_compact
    run(a.compact())
    run(a.chat("second"))
    assert _no_adjacent_users(a._anthropic_messages)


def test_compact_tool_result_merged_not_adjacent(workdir, monkeypatch):
    a = make_agent()
    a._anthropic_messages = [{"role": "user", "content": str(i)} if i % 2 == 0 else
                             {"role": "assistant", "content": [{"type": "text", "text": str(i)}]} for i in range(6)]

    async def fake_compact(*, trigger):
        a._anthropic_messages = [{"role": "user", "content": "folded memory"}]
        return True

    a._compact_anthropic = fake_compact
    script = install(monkeypatch, [anth(tools=[("compact_context", {"reason": "long"})]), anth(text="continuing")])
    run(a.chat("go on"))
    assert all(_no_adjacent_users(req) for _, req in script.requests)
