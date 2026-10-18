"""被中断的轮次：结果日志补结果、消息修复只补不删、检查点位置、快照缺失时整体拒绝。"""

import asyncio

import pytest

import agents.agent as agent_mod
from agents import session as session_mod
from agents.agent import Agent
from fakemodel import anth, install, make_agent, oai


def run(coro):
    return asyncio.run(coro)


ROUND = [("write_file", {"file_path": "a.txt", "content": "A"}),
         ("run_shell", {"command": "sleep 30"}),
         ("write_file", {"file_path": "b.txt", "content": "B"})]


def interrupt_during_shell(a):
    """开始执行后 1 秒按下 Ctrl+C：a.txt 已写完，shell 执行到一半，b.txt 还没开始。"""
    async def go():
        asyncio.get_running_loop().call_later(1.0, a.abort)
        await a.chat("do three things")
    run(go())


def results_by_id_anthropic(a):
    out = {}
    for m in a._anthropic_messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    out[b["tool_use_id"]] = b
    return out


def test_anthropic_interrupted_round_filled_from_journal(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, [anth(tools=ROUND), anth(text="next")])
    interrupt_during_shell(a)
    run(a.chat("continue"))
    res = results_by_id_anthropic(a)
    assert res["toolu_0_write_file"]["content"].startswith("Successfully wrote")       # 真实结果
    assert "is_error" not in res["toolu_0_write_file"]
    assert "may have partially run" in res["toolu_1_run_shell"]["content"]
    assert "safe to retry" in res["toolu_2_write_file"]["content"]
    assert (workdir / "a.txt").exists() and not (workdir / "b.txt").exists()


def test_openai_interrupted_round_filled_from_journal(workdir, monkeypatch):
    a = make_agent(openai_mode=True)
    install(monkeypatch, [oai(tools=ROUND), oai(text="next")])
    interrupt_during_shell(a)
    run(a.chat("continue"))
    msgs = {m["tool_call_id"]: m["content"] for m in a._openai_messages
            if isinstance(m, dict) and m.get("role") == "tool"}
    assert msgs["call_0_write_file"].startswith("Successfully wrote")
    assert "may have partially run" in msgs["call_1_run_shell"]
    assert "safe to retry" in msgs["call_2_write_file"]


def test_crash_during_tools_then_resume_uses_real_results(workdir, monkeypatch):
    a = make_agent()
    a._auto_save = Agent._auto_save.__get__(a)
    real_exec = a._execute_tool_call

    async def crash_on_shell(name, inp):
        if name == "run_shell":
            raise SystemExit("process killed")         # 模拟进程在执行 shell 时被杀
        return await real_exec(name, inp)

    a._execute_tool_call = crash_on_shell
    install(monkeypatch, [anth(tools=ROUND)])
    with pytest.raises(SystemExit):
        run(a.chat("do three things"))
    saved = session_mod.load_session(a.session_id)
    assert any(m["role"] == "assistant" for m in saved["anthropicMessages"])   # 执行前已存档

    b = make_agent()
    b.restore_session(saved)
    install(monkeypatch, [anth(text="resumed")])
    run(b.chat("continue"))
    res = results_by_id_anthropic(b)
    assert res["toolu_0_write_file"]["content"].startswith("Successfully wrote")
    assert "may have partially run" in res["toolu_1_run_shell"]["content"]
    assert "safe to retry" in res["toolu_2_write_file"]["content"]


def test_checkpoint_correct_after_interrupted_turn(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, [anth(tools=[("run_shell", {"command": "sleep 30"})]), anth(text="three")])

    async def go():
        asyncio.get_running_loop().call_later(0.5, a.abort)
        await a.chat("turn one")
    run(go())
    run(a.chat("turn two"))
    ok, _ = a.rewind(2, "conversation")
    assert ok
    assert all(m.get("content") != "turn two" for m in a._anthropic_messages)
    assert a._anthropic_messages[0]["content"] == "turn one"


def test_partial_results_are_completed_not_dropped():
    a = make_agent()
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "n", "input": {}},
                                              {"type": "tool_use", "id": "y", "name": "n", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"},
                                         {"type": "text", "text": "note"}]}]
    out = a._normalize_anthropic_messages(msgs)
    assert len(out) == 3
    blocks = out[2]["content"]
    assert [b.get("tool_use_id") for b in blocks[:2]] == ["x", "y"] and blocks[2]["type"] == "text"
    assert blocks[1]["is_error"]


def test_orphan_results_removed_without_changing_length():
    a = make_agent()
    msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "ghost", "content": "?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
    out = a._normalize_anthropic_messages(msgs)
    assert len(out) == 2 and "tool_result" not in str(out[0])


def test_dangling_assistant_string_user_merged(workdir):
    a = make_agent()
    msgs = [{"role": "assistant", "content": [{"type": "tool_use", "id": "z", "name": "n", "input": {}}]},
            {"role": "user", "content": "new question"}]
    out = a._normalize_anthropic_messages(msgs)
    assert len(out) == 2
    assert out[1]["content"][0]["tool_use_id"] == "z" and out[1]["content"][1]["text"] == "new question"


def _two_turns_two_files(workdir, monkeypatch, a):
    for n in "bc":
        (workdir / f"{n}.txt").write_text("orig")
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "b.txt"})]),
                          anth(tools=[("write_file", {"file_path": "b.txt", "content": "B2"})]), anth(text="t1"),
                          anth(tools=[("read_file", {"file_path": "c.txt"})]),
                          anth(tools=[("write_file", {"file_path": "c.txt", "content": "C2"})]), anth(text="t2")])
    run(a.chat("one"))
    run(a.chat("two"))


def test_missing_snapshot_refuses_whole_rewind(workdir, monkeypatch):
    a = make_agent()
    _two_turns_two_files(workdir, monkeypatch, a)
    a._edit_history._stack[0].snapshot.unlink()
    ok, msg = a.rewind(1, "code", force=True)
    assert not ok and "Snapshots are missing" in msg
    assert (workdir / "b.txt").read_text() == "B2" and (workdir / "c.txt").read_text() == "C2"   # 一个都没动
    assert a.rewind(2, "code")[0] and (workdir / "c.txt").read_text() == "orig"                 # 不涉及缺失快照的照常回退
    ok, msg = a.undo_last_edit()
    assert not ok and "missing" in msg


def test_resume_keeps_all_records_aligned_with_checkpoints(workdir, monkeypatch):
    a = make_agent()
    a._auto_save = Agent._auto_save.__get__(a)
    _two_turns_two_files(workdir, monkeypatch, a)
    a._edit_history._stack[0].snapshot.unlink()
    a._auto_save()
    b = make_agent()
    b.restore_session(session_mod.load_session(a.session_id))
    assert len(b._edit_history) == 2 and [c["edits"] for c in b._checkpoints] == [0, 1]
    assert b.rewind(2, "code")[0] and (workdir / "c.txt").read_text() == "orig"


def test_tool_search_activation_refreshes_system_prompt(workdir, monkeypatch):
    a = make_agent()
    calls = {"n": 0}
    real = agent_mod.build_system_prompt

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(agent_mod, "build_system_prompt", counting)
    install(monkeypatch, [anth(tools=[("tool_search", {"query": "plan"})]), anth(text="ok")])
    run(a.chat("find tools"))
    assert calls["n"] == 1
