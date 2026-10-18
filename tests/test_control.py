"""第一批：Ctrl+C 中断、计划审批、子 agent 权限与连接配置、异常时存盘。"""

import asyncio
import builtins
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from agents import main as main_mod
from fakemodel import make_agent


def run(coro):
    return asyncio.run(coro)


def test_plan_approval_execute_switches_mode_and_shows_content(workdir):
    a = make_agent(permission_mode="default")
    run(a._execute_plan_mode_tool("enter_plan_mode"))
    assert a.permission_mode == "plan"
    Path(a._plan_file_path).parent.mkdir(parents=True, exist_ok=True)
    Path(a._plan_file_path).write_text("STEP 1: do it", encoding="utf-8")
    seen = {}

    async def approve(content):
        seen["content"] = content
        return {"choice": "execute"}

    a.set_plan_approval_fn(approve)
    run(a._execute_plan_mode_tool("exit_plan_mode"))
    assert seen["content"] == "STEP 1: do it"
    assert a.permission_mode == "acceptEdits"
    assert a._plan_file_path is None


def test_plan_keep_planning_stays_in_plan(workdir):
    a = make_agent(permission_mode="default")
    run(a._execute_plan_mode_tool("enter_plan_mode"))

    async def keep(content):
        return {"choice": "keep-planning", "feedback": "more detail"}

    a.set_plan_approval_fn(keep)
    out = run(a._execute_plan_mode_tool("exit_plan_mode"))
    assert a.permission_mode == "plan" and "more detail" in out


def test_plan_manual_execute_restores_previous_mode(workdir):
    a = make_agent(permission_mode="acceptEdits")
    run(a._execute_plan_mode_tool("enter_plan_mode"))

    async def manual(content):
        return {"choice": "manual-execute"}

    a.set_plan_approval_fn(manual)
    run(a._execute_plan_mode_tool("exit_plan_mode"))
    assert a.permission_mode == "acceptEdits"


def test_plan_toggle_round_trip(workdir):
    a = make_agent(permission_mode="default")
    a.toggle_plan_mode()
    a.toggle_plan_mode()
    assert a.permission_mode == "default" and a._plan_file_path is None


@pytest.mark.parametrize("mode", ["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"])
def test_sub_agent_inherits_permission_mode(workdir, mode):
    a = make_agent(permission_mode=mode)
    assert a._spawn_sub_agent("sys", []).permission_mode == mode


def test_sub_agent_inherits_connection_confirm_and_recorder(workdir):
    a = make_agent(permission_mode="default")

    async def cf(m):
        return True

    a.set_confirm_fn(cf)
    s = a._spawn_sub_agent("sys", [], label="sub:general")
    assert str(s._anthropic_client.base_url).startswith("https://fake.example/anthropic")
    assert s._anthropic_client.api_key == "sk-test"
    assert s.confirm_fn is cf
    a._confirmed_paths.add("x")
    assert "x" in s._confirmed_paths
    assert s._recorder is a._recorder
    assert s.is_sub_agent and s._agent_label == "sub:general"


def test_chat_saves_before_reraising(workdir):
    a = make_agent()
    saved = []
    a._auto_save = lambda: saved.append(1)

    async def boom(msg):
        raise RuntimeError("model down")

    a._chat_anthropic = boom
    with pytest.raises(RuntimeError):
        run(a.chat("hi there"))
    assert saved == [1]


def test_ctrl_c_aborts_running_turn(workdir, monkeypatch):
    a = make_agent()

    async def slow(msg):
        await asyncio.sleep(10)

    a._chat_anthropic = slow
    lines = iter(["hello world"])

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    previous = signal.getsignal(signal.SIGINT)
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    t0 = time.time()
    try:
        run(main_mod.run_repl(a))
    finally:
        signal.signal(signal.SIGINT, previous)
    assert time.time() - t0 < 3
    assert a._aborted
