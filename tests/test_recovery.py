"""第三批 3b：改文件前快照与 /undo、确认时展示 diff、完成后验证命令。"""

import asyncio

import pytest

from agents import snapshots
from agents.tools import check_permission
from fakemodel import anth, install, make_agent, oai, read_events


def run(coro):
    return asyncio.run(coro)


def user_texts(agent):
    return [m["content"] for m in agent._anthropic_messages
            if m["role"] == "user" and isinstance(m["content"], str)]


# ── 快照与 /undo ─────────────────────────────────────────

def test_undo_restores_edited_file_and_notifies_model(workdir, monkeypatch):
    f = workdir / "a.py"
    f.write_text("x = 1\n")
    a = make_agent()
    script = install(monkeypatch, [
        anth(tools=[("read_file", {"file_path": "a.py"})]),
        anth(tools=[("edit_file", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})]),
        anth(text="changed"),
        anth(text="ok"),
    ])
    run(a.chat("change it"))
    assert f.read_text() == "x = 2\n"
    ok, msg = a.undo_last_edit()
    assert ok and f.read_text() == "x = 1\n"
    assert str(f.resolve()) not in a._read_file_state          # 必须重新读
    run(a.chat("next"))
    last_request_user = [m for m in script.requests[-1][1] if m["role"] == "user"][-1]["content"]
    assert "/undo" in last_request_user and "Re-read" in last_request_user
    assert any(e["type"] == "undo" and e["ok"] for e in read_events(a))


def test_undo_new_file_deletes_it(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, [anth(tools=[("write_file", {"file_path": "new.txt", "content": "hi"})]), anth(text="ok")])
    run(a.chat("go"))
    assert (workdir / "new.txt").exists()
    ok, _ = a.undo_last_edit()
    assert ok and not (workdir / "new.txt").exists()


def test_undo_refuses_after_external_change_unless_forced(workdir, monkeypatch):
    f = workdir / "a.txt"
    f.write_text("one\n")
    a = make_agent()
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "a.txt"})]),
                          anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "one", "new_string": "two"})]),
                          anth(text="ok")])
    run(a.chat("go"))
    f.write_text("user edited\n")
    ok, msg = a.undo_last_edit()
    assert not ok and "--force" in msg and f.read_text() == "user edited\n"
    ok, _ = a.undo_last_edit(force=True)
    assert ok and f.read_text() == "one\n"


def test_undo_stack_order_and_empty(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, [anth(tools=[("write_file", {"file_path": "f.txt", "content": "v1"})]),
                          anth(tools=[("read_file", {"file_path": "f.txt"})]),
                          anth(tools=[("write_file", {"file_path": "f.txt", "content": "v2"})]),
                          anth(text="ok")])
    run(a.chat("go"))
    assert len(a.list_edits()) == 2
    assert a.undo_last_edit()[0] and (workdir / "f.txt").read_text() == "v1"
    assert a.undo_last_edit()[0] and not (workdir / "f.txt").exists()
    assert a.undo_last_edit() == (False, "Nothing to undo.")


def test_failed_edit_leaves_no_undo_entry(workdir, monkeypatch):
    (workdir / "a.txt").write_text("abc")
    a = make_agent()
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "a.txt"})]),
                          anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "zzz", "new_string": "y"})]),
                          anth(text="ok")])
    run(a.chat("go"))
    assert a.list_edits() == []


def test_large_file_not_snapshotted_is_reported(workdir, monkeypatch):
    monkeypatch.setattr(snapshots, "MAX_SNAPSHOT_BYTES", 10)
    (workdir / "big.txt").write_text("0123456789abcdef")
    a = make_agent()
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "big.txt"})]),
                          anth(tools=[("write_file", {"file_path": "big.txt", "content": "small"})]),
                          anth(text="ok")])
    run(a.chat("go"))
    results = [b for m in a._anthropic_messages if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"]]
    assert "cannot be undone" in results[-1]["content"]
    assert a.list_edits() == []


def test_sub_agent_edits_are_undoable_from_parent(workdir, monkeypatch):
    a = make_agent()
    install(monkeypatch, {
        "main": [anth(tools=[("agent", {"type": "general", "description": "d", "prompt": "p"})]), anth(text="ok")],
        "sub": [anth(tools=[("write_file", {"file_path": "s.txt", "content": "from sub"})]), anth(text="done")],
    })
    run(a.chat("go"))
    assert (workdir / "s.txt").exists()
    assert a.undo_last_edit()[0] and not (workdir / "s.txt").exists()


# ── 确认时展示 diff ───────────────────────────────────────

def test_default_mode_existing_file_edit_needs_confirm_with_diff(workdir):
    (workdir / "a.py").write_text("x = 1\ny = 2\n")
    d = check_permission("edit_file", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 3"}, "default")
    assert d["action"] == "confirm" and d["cacheable"] is False and d["offer_all"] is True
    assert "-x = 1" in d["preview"] and "+x = 3" in d["preview"]
    d = check_permission("write_file", {"file_path": "a.py", "content": "z = 0\n"}, "default")
    assert d["action"] == "confirm" and "+z = 0" in d["preview"] and "-y = 2" in d["preview"]


@pytest.mark.parametrize("mode", ["acceptEdits", "bypassPermissions", "dontAsk"])
def test_other_modes_do_not_ask_for_existing_file_edits(workdir, mode):
    (workdir / "a.py").write_text("x = 1\n")
    d = check_permission("edit_file", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 3"}, mode)
    assert d["action"] == "allow"


def test_edit_with_missing_old_string_is_not_asked(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    d = check_permission("edit_file", {"file_path": "a.py", "old_string": "nope", "new_string": "x"}, "default")
    assert d["action"] == "allow"            # 工具执行时自己会报错，不必打扰用户


def test_new_file_confirm_has_content_preview(workdir):
    d = check_permission("write_file", {"file_path": "n.txt", "content": "hello\nworld"}, "default")
    assert d["action"] == "confirm" and d["preview"] == "+hello\n+world"


def _agent_with_answers(answers, **kw):
    a = make_agent(permission_mode="default", **kw)
    asked = []

    async def cf(message, *, preview=None, offer_all=False):
        asked.append((message, preview, offer_all))
        return answers.pop(0)

    a.set_confirm_fn(cf)
    return a, asked


def test_each_edit_is_asked_and_all_switches_to_accept_edits(workdir, monkeypatch):
    f = workdir / "a.txt"
    f.write_text("1\n")
    a, asked = _agent_with_answers(["y", "a"])
    install(monkeypatch, [
        anth(tools=[("read_file", {"file_path": "a.txt"})]),
        anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "1", "new_string": "2"})]),
        anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "2", "new_string": "3"})]),
        anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "3", "new_string": "4"})]),
        anth(text="ok"),
    ])
    run(a.chat("go"))
    assert f.read_text() == "4\n"
    assert len(asked) == 2                                   # 第三次在 acceptEdits 下不再问
    assert "+2" in asked[0][1] and asked[0][2] is True
    assert a.permission_mode == "acceptEdits"
    decisions = [e["decision"] for e in read_events(a) if e["type"] == "permission" and e["tool"] == "edit_file"]
    assert decisions[:2] == ["user_approved", "user_approved_all"]


def test_denied_edit_is_reported_to_model(workdir, monkeypatch):
    (workdir / "a.txt").write_text("1\n")
    a, _ = _agent_with_answers(["n"])
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "a.txt"})]),
                          anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "1", "new_string": "2"})]),
                          anth(text="ok")])
    run(a.chat("go"))
    assert (workdir / "a.txt").read_text() == "1\n"


def test_print_confirmation_with_preview():
    from agents.ui import print_confirmation
    print_confirmation("edit file: a.py", preview="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y")
    print_confirmation("rm -rf build")


# ── 完成后验证 ───────────────────────────────────────────

def test_verify_not_run_without_edits(workdir, monkeypatch):
    a = make_agent(verify_command="echo RAN > verify.log")
    install(monkeypatch, [anth(text="just answering")])
    run(a.chat("question"))
    assert not (workdir / "verify.log").exists()


def test_verify_failure_feeds_back_until_fixed(workdir, monkeypatch):
    a = make_agent(verify_command="test -f fixed || (echo 'TEST FAILED: missing fixed' && exit 1)")
    script = install(monkeypatch, [
        anth(tools=[("write_file", {"file_path": "code.py", "content": "pass"})]),
        anth(text="done!"),                                           # 验证失败 → 继续
        anth(tools=[("write_file", {"file_path": "fixed", "content": "ok"})]),
        anth(text="really done"),                                     # 验证通过 → 结束
    ])
    run(a.chat("implement"))
    feedback = [t for t in user_texts(a) if t.startswith("[Automatic verification failed]")]
    assert len(feedback) == 1 and "TEST FAILED: missing fixed" in feedback[0]
    verifies = [e for e in read_events(a) if e["type"] == "verify"]
    assert [v["passed"] for v in verifies] == [False, True]
    assert len(script.requests) == 4


def test_verify_gives_up_after_max_retries(workdir, monkeypatch):
    a = make_agent(verify_command="exit 1")
    steps = []
    for i in range(3):
        steps += [anth(tools=[("write_file", {"file_path": f"try{i}.py", "content": "x"})]), anth(text=f"done {i}")]
    install(monkeypatch, steps)
    run(a.chat("go"))
    verifies = [e for e in read_events(a) if e["type"] == "verify"]
    assert len(verifies) == 3 and not any(v["passed"] for v in verifies)


def test_verify_counts_sub_agent_edits_openai_path(workdir, monkeypatch):
    a = make_agent(openai_mode=True, verify_command="echo checked >> verify.log")
    install(monkeypatch, {
        "main": [oai(tools=[("agent", {"type": "general", "description": "d", "prompt": "p"})]), oai(text="ok")],
        "sub": [oai(tools=[("write_file", {"file_path": "s.txt", "content": "x"})]), oai(text="done")],
    })
    run(a.chat("go"))
    assert (workdir / "verify.log").read_text().split() == ["checked"]


def test_repl_end_to_end_confirm_all_then_undo(workdir, monkeypatch):
    import builtins
    import signal
    from agents import main as main_mod
    (workdir / "a.txt").write_text("1\n")
    a = make_agent(permission_mode="default")
    install(monkeypatch, [anth(tools=[("read_file", {"file_path": "a.txt"})]),
                          anth(tools=[("edit_file", {"file_path": "a.txt", "old_string": "1", "new_string": "2"})]),
                          anth(text="edited")])
    typed = iter(["please edit", "a", "/undo list", "/undo"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
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
    assert any("y/n/a" in p for p in prompts)
    assert a.permission_mode == "acceptEdits"
    assert (workdir / "a.txt").read_text() == "1\n"       # /undo 恢复了原内容
