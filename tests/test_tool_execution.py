"""第二批：shell / grep 真异步、超时与进程组清理、MCP 客户端的容错。"""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from agents import tools
from agents.agent import Agent
from agents.mcp_client import McpConnection
from agents.tools import _grep_python, _parse_timeout_ms, _run_shell, execute_tool
from conftest import ROOT, pid_alive
from fakemodel import make_agent

FAKE_MCP = str(Path(__file__).with_name("fake_mcp_server.py"))


def run(coro):
    return asyncio.run(coro)


# ── shell ──────────────────────────────────────────────

def test_timeout_fires_on_time_and_keeps_output(workdir):
    t = time.time()
    r = run(_run_shell({"command": "echo start; sleep 100", "timeout": 1500}))
    assert time.time() - t < 4
    assert r.is_error and r.meta["timed_out"]
    assert "start" in r.content


def test_timeout_kills_child_processes(workdir):
    run(_run_shell({"command": "sleep 101 & echo $! > child.pid; wait", "timeout": 1000}))
    time.sleep(0.2)
    assert not pid_alive(int(Path("child.pid").read_text()))


def test_background_child_holding_pipe_is_bounded_and_killed(workdir):
    t = time.time()
    r = run(_run_shell({"command": "(sleep 102 & echo $! > bg.pid); echo hi", "timeout": 1500}))
    assert time.time() - t < 4 and "hi" in r.content
    time.sleep(0.2)
    assert not pid_alive(int(Path("bg.pid").read_text()))


def test_cancel_kills_process(workdir):
    async def go():
        task = asyncio.create_task(_run_shell({"command": "echo $$ > c.pid; exec sleep 103", "timeout": 60000}))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(go())
    time.sleep(0.2)
    assert not pid_alive(int(Path("c.pid").read_text()))


def test_stdin_is_devnull(workdir):
    t = time.time()
    r = run(_run_shell({"command": "read x; echo got:$x", "timeout": 5000}))
    assert "got:" in r.content and time.time() - t < 2


def test_nonzero_exit_is_error_with_both_streams(workdir):
    r = run(_run_shell({"command": "echo out; echo err >&2; exit 3"}))
    assert r.is_error and r.meta["exit_code"] == 3
    assert r.content.startswith("Command failed (exit code 3)") and "out" in r.content and "err" in r.content


def test_success_is_not_error_even_if_output_says_error(workdir):
    r = run(_run_shell({"command": "echo '0 errors, all good'"}))
    assert not r.is_error and r.meta["exit_code"] == 0


def test_huge_output_capped(workdir):
    r = run(_run_shell({"command": "head -c 5000000 /dev/zero | tr '\\0' 'a'"}))
    assert "bytes omitted" in r.content and len(r.content) < 1_100_000


def test_timeout_parse():
    assert _parse_timeout_ms("abc") == 30000
    assert _parse_timeout_ms(1e12) == 600000
    assert _parse_timeout_ms(5) == 1000


def test_shells_run_concurrently_without_blocking_loop(workdir):
    ticks = 0

    async def go():
        nonlocal ticks

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1
        tk = asyncio.create_task(ticker())
        t = time.time()
        await asyncio.gather(_run_shell({"command": "sleep 1"}), _run_shell({"command": "sleep 1"}))
        tk.cancel()
        return time.time() - t
    assert run(go()) < 1.6
    assert ticks >= 15


# ── grep / 其他工具 ─────────────────────────────────────

def test_grep_finds_and_misses():
    r = run(execute_tool("grep_search", {"pattern": "def _run_process", "path": str(ROOT / "agents")}))
    assert "tools.py" in r.content and not r.is_error
    r = run(execute_tool("grep_search", {"pattern": "zzz_no_such_token_qq", "path": str(ROOT / "agents")}))
    assert r.content == "No matches found." and not r.is_error


def test_grep_timeout_returns_partial(monkeypatch):
    monkeypatch.setattr(tools, "GREP_TIMEOUT_S", 0.05)
    t = time.time()
    r = run(execute_tool("grep_search", {"pattern": "zzz_qq_nomatch", "path": "/usr"}))
    assert "timed out" in r.content and time.time() - t < 4


def test_python_grep_deadline_and_invalid_regex(monkeypatch):
    monkeypatch.setattr(tools, "GREP_TIMEOUT_S", 0.3)
    t = time.time()
    r = _grep_python("zzz_qq_nomatch", "/usr", None)
    assert "timed out" in r.content and time.time() - t < 1.5
    r = _grep_python("(", str(ROOT), None)
    assert r.is_error and r.content.startswith("Error: invalid regex")


def test_read_list_edit(workdir):
    f = workdir / "a.txt"
    f.write_text("hello\nworld\n")
    state = {}
    r = run(execute_tool("read_file", {"file_path": str(f)}, state))
    assert "hello" in r.content and str(f.resolve()) in state
    assert "a.txt" in run(execute_tool("list_files", {"pattern": "*.txt", "path": str(workdir)})).content
    r = run(execute_tool("edit_file", {"file_path": str(f), "old_string": "world", "new_string": "there"}, state))
    assert not r.is_error and "there" in f.read_text()
    r = run(execute_tool("edit_file", {"file_path": str(f), "old_string": "nope", "new_string": "x"}, state))
    assert r.is_error
    r = run(execute_tool("read_file", {"file_path": str(workdir / "missing.txt")}))
    assert r.is_error


def test_leftover_early_task_is_cancelled_and_process_killed(workdir):
    a = make_agent()

    async def go():
        task = asyncio.create_task(a._timed_tool_call(
            "run_shell", {"command": "echo $$ > e.pid; exec sleep 104", "timeout": 60000}, "span"))
        a._inflight_tool_tasks.add(task)
        task.add_done_callback(a._inflight_tool_tasks.discard)
        await asyncio.sleep(0.5)
        await a._cancel_inflight_tool_tasks()
        return task
    task = run(go())
    time.sleep(0.2)
    assert task.done() and not pid_alive(int(Path("e.pid").read_text()))


# ── MCP ────────────────────────────────────────────────

def test_mcp_resilience():
    async def go():
        c = McpConnection("fake", sys.executable, ["-I", FAKE_MCP])
        await c.connect()
        await asyncio.wait_for(c.initialize(), 5)
        assert (await asyncio.wait_for(c.list_tools(), 5))[0]["name"] == "echo"
        r = await c.call_tool("echo", {"a": 1}, timeout=5)
        assert r.content.startswith("echo:") and not r.is_error
        r = await c.call_tool("fail", {}, timeout=5)
        assert r.is_error and r.content == "tool failed"
        t = time.time()
        with pytest.raises(RuntimeError, match="larger than"):
            await c.call_tool("big", {}, timeout=30)
        assert time.time() - t < 10
        assert (await c.call_tool("echo", {"b": 2}, timeout=5)).content.startswith("echo:")
        with pytest.raises(TimeoutError):
            await c.call_tool("hang", {}, timeout=1)
        assert len(c._pending) == 0
        await c.close()

        c = McpConnection("fake2", sys.executable, ["-I", FAKE_MCP])
        await c.connect()
        await asyncio.wait_for(c.initialize(), 5)
        t = time.time()
        with pytest.raises(RuntimeError, match="something broke"):
            await c.call_tool("die", {}, timeout=30)
        assert time.time() - t < 3
        t = time.time()
        with pytest.raises(RuntimeError, match="unavailable"):
            await c.call_tool("echo", {}, timeout=30)
        assert time.time() - t < 0.5
        proc = c._process
        await c.close()
        assert proc.returncode is not None
    run(go())
