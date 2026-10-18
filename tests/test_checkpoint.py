"""检查点与 /rewind、撤销栈持久化、--resume 按项目过滤。"""

import asyncio
import builtins
import json
import signal

from agents import main as main_mod
from agents import session as session_mod
from agents.agent import Agent
from fakemodel import anth, install, make_agent, oai, read_events


def run(coro):
    return asyncio.run(coro)


def two_turns(monkeypatch, a, *, shell_in_turn2=False):
    """第 1 轮写 v1，第 2 轮改成 v2（可选再跑一条 shell）。"""
    turn2 = [anth(tools=[("read_file", {"file_path": "f.txt"})]),
             anth(tools=[("write_file", {"file_path": "f.txt", "content": "v2"})])]
    if shell_in_turn2:
        turn2.append(anth(tools=[("run_shell", {"command": "echo side-effect > shell.out"})]))
    install(monkeypatch, [anth(tools=[("write_file", {"file_path": "f.txt", "content": "v1"})]), anth(text="t1"),
                          *turn2, anth(text="t2")])
    run(a.chat("first: create f"))
    run(a.chat("second: change f"))


def test_rewind_both_restores_files_and_conversation(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a)
    before_turn2 = a._checkpoints[1]["anthropic_len"]
    ok, msg = a.rewind(2, "both")
    assert ok, msg
    assert (workdir / "f.txt").read_text() == "v1"
    assert len(a._anthropic_messages) == before_turn2
    assert [c["turn"] for c in a._checkpoints] == [1]
    assert any(e["type"] == "rewind" for e in read_events(a))


def test_rewind_to_first_turn_deletes_new_file(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a)
    ok, _ = a.rewind(1, "both")
    assert ok and not (workdir / "f.txt").exists() and a._anthropic_messages == []


def test_rewind_code_only_keeps_conversation_and_notifies(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a)
    n_messages = len(a._anthropic_messages)
    ok, _ = a.rewind(2, "code")
    assert ok and (workdir / "f.txt").read_text() == "v1"
    assert len(a._anthropic_messages) == n_messages
    assert a._pending_notices and "before turn 2" in a._pending_notices[0]
    assert str((workdir / "f.txt").resolve()) not in a._read_file_state


def test_rewind_conversation_only_keeps_files_and_notifies(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a)
    ok, _ = a.rewind(2, "conversation")
    assert ok and (workdir / "f.txt").read_text() == "v2"
    assert "kept on disk" in a._pending_notices[0]


def test_rewind_refuses_on_conflict_without_partial_changes(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a)
    (workdir / "f.txt").write_text("edited by user")
    n_messages = len(a._anthropic_messages)
    ok, msg = a.rewind(1, "both")
    assert not ok and "Nothing was reverted" in msg
    assert (workdir / "f.txt").read_text() == "edited by user"
    assert len(a._anthropic_messages) == n_messages          # 对话也没动
    ok, _ = a.rewind(1, "both", force=True)
    assert ok and not (workdir / "f.txt").exists()


def test_rewind_warns_about_shell(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a, shell_in_turn2=True)
    cps = a.list_checkpoints()
    assert cps[1]["shell_since"] and cps[1]["file_changes_since"] == 1
    ok, msg = a.rewind(2, "both")
    assert ok and "NOT reverted" in msg
    assert (workdir / "shell.out").exists()                 # shell 的副作用确实不会回退


def test_after_compaction_only_code_can_be_rewound(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a)

    async def fake_compact(*, trigger):
        a._anthropic_messages = a._anthropic_messages[-1:]
        return True

    a._compact_anthropic = fake_compact
    assert run(a._compact_conversation(trigger="manual"))
    ok, msg = a.rewind(2, "both")
    assert not ok and "/rewind 2 code" in msg
    assert a.rewind(2, "code")[0] and (workdir / "f.txt").read_text() == "v1"


def test_rewind_reverts_sub_agent_edits(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, {
        "main": [anth(text="hello"),
                 anth(tools=[("agent", {"type": "general", "description": "d", "prompt": "p"})]), anth(text="ok")],
        "sub": [anth(tools=[("write_file", {"file_path": "s.txt", "content": "x"})]), anth(text="done")],
    })
    run(a.chat("turn one"))
    run(a.chat("turn two"))
    assert (workdir / "s.txt").exists()
    assert a.rewind(2, "code")[0] and not (workdir / "s.txt").exists()


def test_openai_conversation_rewind(workdir, monkeypatch):
    a = make_agent(openai_mode=True)
    install(monkeypatch, [oai(text="one"), oai(text="two")])
    run(a.chat("first"))
    n = len(a._openai_messages)
    run(a.chat("second"))
    assert a.rewind(2, "conversation")[0]
    assert len(a._openai_messages) == n and a._openai_messages[0]["role"] == "system"


def test_checkpoints_and_undo_survive_resume(workdir, monkeypatch):
    a = make_agent()
    a._auto_save = Agent._auto_save.__get__(a)                # 用真实的存档
    two_turns(monkeypatch, a)
    saved = session_mod.load_session(a.session_id)
    assert saved["checkpoints"] and saved["editHistory"]["records"]

    b = make_agent()
    b.restore_session(saved)
    assert [c["turn"] for c in b.list_checkpoints()] == [1, 2]
    assert b.undo_last_edit()[0] and (workdir / "f.txt").read_text() == "v1"
    assert b.rewind(1, "both")[0] and not (workdir / "f.txt").exists()


def test_resume_only_considers_current_project(workdir, monkeypatch):
    d = session_mod.SESSION_DIR
    d.mkdir(parents=True, exist_ok=True)
    for sid, cwd, start in [("aaaa1111", "/proj/a", "2026-10-19T10:00:00Z"),
                            ("bbbb2222", "/proj/b", "2026-10-19T11:00:00Z"),
                            ("aaaa3333", "/proj/a", "2026-10-19T09:00:00Z")]:
        (d / f"{sid}.json").write_text(json.dumps({"metadata": {"id": sid, "cwd": cwd, "startTime": start}}))
    assert session_mod.get_latest_session_id("/proj/a") == "aaaa1111"
    assert session_mod.get_latest_session_id("/proj/b") == "bbbb2222"
    assert session_mod.find_session_id("aaaa3", "/proj/a") == "aaaa3333"
    assert session_mod.find_session_id("aaaa", "/proj/a") is None        # 有歧义
    assert session_mod.find_session_id("bbbb", "/proj/a") is None        # 别的项目的会话


def test_repl_rewind_interactive(workdir, monkeypatch):
    a = make_agent()
    two_turns(monkeypatch, a)
    typed = iter(["/rewind", "2", "f", "/rewind 9", "/rewind x y"])

    def fake_input(prompt=""):
        try:
            return next(typed)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    previous = signal.getsignal(signal.SIGINT)
    try:
        run(main_mod.run_repl(a))
    finally:
        signal.signal(signal.SIGINT, previous)
    assert (workdir / "f.txt").read_text() == "v1"               # 只回退了文件
    assert [c["turn"] for c in a._checkpoints] == [1, 2]
