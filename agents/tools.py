#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from tqdm.utils import IS_WIN

from agents.memory import get_memory_dir
from agents import shell_policy
from agents.snapshots import EditHistory
from agents.tool_result import ToolResult

ToolDef = dict  # Anthropic tool schema dict
#权限模式
PermissionMode = str  # "default" | "plan" | "acceptEdits" | "bypassPermissions" | "dontAsk"

READ_TOOLS = {"read_file", "list_files", "grep_search", "compact_context"}
EDIT_TOOLS = {"write_file", "edit_file", "skill_evolve", "skill_create"}


#并发安全的工具可以并行运行（只读，无副作用）
CONCURRENCY_SAFE_TOOLS = {"read_file", "list_files", "grep_search"}







def get_active_tool_definitions(all_tools: list[ToolDef] | None = None) -> list[ToolDef]:
    """过滤并返回当前可用的工具定义列表，主要用于 API 调用前剔除尚未激活的“延迟工具”（deferred tools），并删除无关的元数据字段。"""
    tools = all_tools if all_tools is not None else tool_definitions
    return [
        {k: v for k, v in t.items() if k != "deferred"}
        for t in tools
        if not t.get("deferred") or t["name"] in _activated_tools
    ]



#工具定义
tool_definitions: list[ToolDef] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to read"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to write"},
                "content": {"type": "string", "description": "The content to write to the file"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing an exact string match with new content. The old_string must match exactly (including whitespace and indentation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to edit"},
                "old_string": {"type": "string", "description": "The exact string to find and replace"},
                "new_string": {"type": "string", "description": "The string to replace it with"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern. Returns matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": 'Glob pattern to match files (e.g., "**/*.ts", "src/**/*")'},
                "path": {"type": "string", "description": "Base directory to search from. Defaults to current directory."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in. Defaults to current directory."},
                "include": {"type": "string", "description": 'File glob pattern to include (e.g., "*.ts", "*.py")'},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Execute a shell command and return its output. Use this for running tests, installing packages, git operations, etc. "
            "Commands get no stdin, and on timeout the command and all its child processes are killed. "
            "To start a long-running background process (e.g. a dev server), redirect its output so the call can return: "
            "`cmd > server.log 2>&1 &`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "number", "description": "Timeout in milliseconds (default: 30000)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "skill",
        "description": "Invoke a registered skill by name. Skills are prompt templates loaded from .moltevo/skills/. Returns the skill's resolved prompt to follow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "The name of the skill to invoke"},
                "args": {"type": "string", "description": "Optional arguments to pass to the skill"},
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "compact_context",
        "description": "Compact the current conversation context into structured session memory when the context is long, tool results are noisy, or a strategy reset is useful. This preserves task progress, current next steps, and tool-use experience, then continues from the folded memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief reason for compacting now, such as long context, many tool calls, repeated failures, or strategy change.",
                },
            },
        },
    },
    {
        "name": "skill_evolve",
        "description": "Persist an explicit reusable user correction or workflow preference into an existing skill. Creates a version snapshot before editing the skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "The registered skill name to evolve"},
                "lesson": {"type": "string", "description": "Durable reusable rule to add to the skill"},
                "rationale": {"type": "string", "description": "Why this lesson should affect future similar tasks"},
                "target": {
                    "type": "string",
                    "enum": ["active", "project", "user"],
                    "description": "Which skill file to update. Defaults to active.",
                },
            },
            "required": ["skill_name", "lesson"],
        },
    },
    {
        "name": "skill_create",
        "description": "Create a new reusable skill from explicit durable workflow guidance when no suitable existing skill exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Concise reusable skill name"},
                "description": {"type": "string", "description": "One-sentence description of what the skill does and when to use it"},
                "instructions": {"type": "string", "description": "Reusable SKILL.md body. Focus on durable method, constraints, and workflow, not one-off task content."},
                "when_to_use": {"type": "string", "description": "Trigger condition for auto-invocation"},
                "target": {
                    "type": "string",
                    "enum": ["project", "user"],
                    "description": "Where to create the skill. Defaults to project.",
                },
                "context": {
                    "type": "string",
                    "enum": ["inline", "fork"],
                    "description": "Skill execution mode. Defaults to inline.",
                },
                "user_invocable": {"type": "boolean", "description": "Whether users can invoke it manually with /<skill>. Defaults to false."},
                "allowed_tools": {"type": "string", "description": "Optional comma-separated allowed tools for fork mode"},
                "evidence": {"type": "string", "description": "Short user-provided evidence showing why this is reusable"},
            },
            "required": ["name", "description", "instructions"],
        },
    },
    {
        "name": "enter_plan_mode",
        "description": "Enter plan mode to switch to a read-only planning phase. In plan mode, you can only read files and write to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "exit_plan_mode",
        "description": "Exit plan mode after you have finished writing your plan to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "agent",
        "description": "Launch a sub-agent to handle a task autonomously. Sub-agents have isolated context and return their result. Types: 'explore' (read-only), 'plan' (read-only, structured planning), 'general' (full tools).",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Short (3-5 word) description of the sub-agent's task"},
                "prompt": {"type": "string", "description": "Detailed task instructions for the sub-agent"},
                "type": {"type": "string", "enum": ["explore", "plan", "general"], "description": "Agent type. Default: general"},
            },
            "required": ["description", "prompt"],
        },
    },
    # ─── Tool search (deferred tool loader) ─────────────────────
    {
        "name": "tool_search",
        "description": "Search for available tools by name or keyword. Returns full schema definitions for matching deferred tools so you can use them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Tool name or search keywords"},
            },
            "required": ["query"],
        },
    },
]





#----------------------工具调用----------------------------

def _resolve_tool_path(raw_path: str, *, must_exist: bool = True) -> Path:
    path = Path(raw_path)
    if path.exists() or not path.is_absolute():
        return path

    parts = path.parts
    cwd = Path.cwd()
    for i in range(1, len(parts)):
        candidate = cwd.joinpath(*parts[i:])
        if must_exist and candidate.exists():
            return candidate
        if not must_exist and candidate.parent.exists():
            return candidate

    return path


#读取文件并且在读取文件的基础上添加行号
def _read_file(inp:dict) -> ToolResult:
    try:
        path = _resolve_tool_path(inp["file_path"])
        content = path.read_text(errors="replace")
        lines = content.split("\n")
        numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
        return ToolResult.ok(numbered)
    except Exception as e:
        return ToolResult.error(f"Error reading file: {e}")

def _write_file(inp:dict) -> ToolResult:
    try:
        path = _resolve_tool_path(inp["file_path"], must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inp["content"])
        _auto_update_memory_index(str(path))
        lines = inp["content"].split("\n")
        line_count = len(lines)
        preview = "\n".join(f"{i + 1:4d} | {l}" for i, l in enumerate(lines[:30]))
        trunc = f"\n  ... ({line_count} lines total)" if line_count > 30 else ""
        return ToolResult.ok(f"Successfully wrote to {inp['file_path']} ({line_count} lines)\n\n{preview}{trunc}")
    except Exception as e:
        return ToolResult.error(f"Error writing file: {e}")

def _auto_update_memory_index(file_path:str) -> None:
    try:
        mem_dir = str(get_memory_dir())
        if file_path.startswith(mem_dir) and file_path.endswith(".md") and not file_path.endswith("MEMORY.md"):
            mem_path = Path(mem_dir)
            lines = ["# Memory Index", ""]
            for f in sorted(mem_path.glob("*.md")):
                if f.name == "MEMORY.md":
                    continue
                try:
                    raw = f.read_text()
                    name_match = re.search(r"^name:\s*(.+)$", raw, re.MULTILINE)
                    type_match = re.search(r"^type:\s*(.+)$", raw, re.MULTILINE)
                    desc_match = re.search(r"^description:\s*(.+)$", raw, re.MULTILINE)
                    if name_match and type_match:
                        n = name_match.group(1).strip()
                        t = type_match.group(1).strip()
                        d = desc_match.group(1).strip()
                        lines.append(f"- **[{n}]({f.name})** ({t}) — {d}")
                except Exception:
                    pass
            (mem_path / "MEMORY.md").write_text("\n".join(lines))
    except Exception:
        pass

#-------------------------编辑助手，符号规范化+差异化------------------------

#将各种Unicode引号字符统一转换为标准的ASCII直引号。
def _normalize_quotes(s: str) -> str:
    s = re.sub("[\u2018\u2019\u2032]", "'", s)
    s = re.sub('[\u201c\u201d\u2033]', '"', s)
    return s

#查询匹配到的字符串
def _find_actual_string(file_content: str, search_string: str) -> str | None:
    if search_string in file_content:
        return search_string
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx:idx + len(search_string)]
    return None

#生成一个简单的文本差异格式
def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")

    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for l in old_lines:
        parts.append(f"- {l}")
    for l in new_lines:
        parts.append(f"+ {l}")
    return "\n".join(parts)

#编辑文件
def _edit_file(inp: dict) -> ToolResult:
    try:
        path = _resolve_tool_path(inp["file_path"])
        content = path.read_text(errors="replace")

        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return ToolResult.error(f"Error: old_string not found in {inp['file_path']}")

        occurrences = content.count(inp["old_string"])
        if occurrences > 1:
            return ToolResult.error(f"Error: old_string found {occurrences} times in {inp['file_path']}. Must be unique.")

        new_content = content.replace(actual, inp["new_string"], 1)
        path.write_text(new_content)

        diff = _generate_diff(content, actual, inp["new_string"])

        quote_note = " (matched via quote normalization)" if actual != inp["old_string"] else ""

        return ToolResult.ok(f"Successfully edited {inp['file_path']}{quote_note}\n\n{diff}")
    except Exception as e:
        return ToolResult.error(f"Error editing file: {e}")


def _list_files(inp: dict) -> ToolResult:
    try:
        base = _resolve_tool_path(inp.get("path") or ".")
        pattern = inp["pattern"]
        files = []
        for p in base.glob(pattern):
            if p.is_file():
                rel = str(p.relative_to(base) if base != Path(".") else p)

                if "node_modules" in rel or ".git" in rel.split(os.sep):
                    continue
                files.append(rel)
                if len(files) >= 200:
                    break
        if not files:
            return ToolResult.ok("No files found matching the pattern.")
        result = "\n".join(files[:200])

        if len(files) > 200:
            result += f"\n...and {len(files) - 200} more files are found ..."
        return ToolResult.ok(result)
    except Exception as e:
        return ToolResult.error(f"Error listing files: {e}")


#--------子进程执行：shell 和 grep 共用，不阻塞事件循环---------------------

MAX_CAPTURE_BYTES = 1_000_000   # 单个输出流最多保留的字节数，超出只留开头和结尾
SHELL_DEFAULT_TIMEOUT_MS = 30_000
SHELL_MAX_TIMEOUT_MS = 600_000
GREP_TIMEOUT_S = 10
KILL_GRACE_S = 2                # SIGTERM 之后等这么久还不退出就 SIGKILL


class _CappedBuffer:
    """边读边存的输出缓冲：超过上限时保留开头和结尾，中间丢弃并计数，防止无限输出撑爆内存。"""

    def __init__(self, limit: int = MAX_CAPTURE_BYTES):
        self._half = limit // 2
        self._head = bytearray()
        self._tail = bytearray()
        self._dropped = 0

    def write(self, data: bytes) -> None:
        if len(self._head) < self._half:
            take = self._half - len(self._head)
            self._head += data[:take]
            data = data[take:]
        if data:
            self._tail += data
            if len(self._tail) > self._half:
                cut = len(self._tail) - self._half
                self._dropped += cut
                del self._tail[:cut]

    def text(self) -> str:
        out = self._head.decode("utf-8", errors="replace")
        if self._dropped:
            out += f"\n... [{self._dropped} bytes omitted] ...\n"
        return out + self._tail.decode("utf-8", errors="replace")


async def _pump(stream: asyncio.StreamReader, buf: _CappedBuffer) -> None:
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        buf.write(chunk)


def _signal_process_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    # 不看 proc.returncode：shell 本身退出后，它启动的后台子进程可能还拿着输出管道，
    # 只要进程组里还有成员，killpg 就能把它们一起带走。
    try:
        if IS_WIN:
            if proc.returncode is None:
                proc.kill()
        else:
            os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    _signal_process_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), KILL_GRACE_S)
    except asyncio.TimeoutError:
        pass
    _signal_process_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    await proc.wait()


async def _run_process(cmd: str | list[str], *, shell: bool, timeout_s: float) -> tuple[int | None, str, str, bool]:
    """
    运行子进程并返回 (退出码, stdout, stderr, 是否超时)。
    - 新建进程组：超时或被取消时整组终止，不留孤儿进程。
    - stdin 接 /dev/null：需要交互输入的命令立即读到 EOF，不会和 REPL 抢终端。
    - 自己读输出流：超时后已经产生的输出仍然保留。
    """
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if not IS_WIN:
        kwargs["start_new_session"] = True
    if shell:
        proc = await asyncio.create_subprocess_shell(cmd, **kwargs)
    else:
        proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)

    out, err = _CappedBuffer(), _CappedBuffer()
    finished = asyncio.ensure_future(
        asyncio.gather(_pump(proc.stdout, out), _pump(proc.stderr, err), proc.wait())
    )
    timed_out = False
    try:
        # shield：超时只停止等待，读任务继续跑，进程组被杀后它们会读完剩余输出并结束。
        await asyncio.wait_for(asyncio.shield(finished), timeout_s)
    except asyncio.TimeoutError:
        timed_out = True
    finally:
        # 超时、被取消（Ctrl+C）或出错都会走到这里：整组终止。
        if not finished.done():
            await _terminate_process_group(proc)
            try:
                await asyncio.wait_for(finished, KILL_GRACE_S)
            except asyncio.TimeoutError:
                finished.cancel()
    return proc.returncode, out.text(), err.text(), timed_out


def _format_grep_lines(lines: list[str]) -> str:
    output = "\n".join(lines[:100])
    if len(lines) > 100:
        output += f"\n... and {len(lines) - 100} more matches"
    return output


async def _grep_search(inp: dict) -> ToolResult:
    pattern = inp["pattern"]
    path = str(_resolve_tool_path(inp.get("path") or "."))
    include = inp.get("include")

    if not IS_WIN and shutil.which("grep"):
        args = ["grep", "--line-number", "--color=never", "-r",
                "--exclude-dir=.git", "--exclude-dir=node_modules"]
        if include:
            args.append(f"--include={include}")
        args.extend(["--", pattern, path])
        try:
            returncode, stdout, _stderr, timed_out = await _run_process(args, shell=False, timeout_s=GREP_TIMEOUT_S)
        except OSError:
            returncode, stdout, timed_out = None, "", False
        lines = [l for l in stdout.split("\n") if l]
        if timed_out:
            # 超时直接返回部分结果，不再退回 Python 全量重扫（那样更慢）。
            note = f"[grep timed out after {GREP_TIMEOUT_S}s; results are partial. Narrow the path or use include.]"
            text = f"{_format_grep_lines(lines)}\n{note}" if lines else f"No matches found before timeout. {note}"
            return ToolResult.ok(text, timed_out=True)
        if returncode == 1:
            return ToolResult.ok("No matches found.")
        if returncode == 0 or (returncode == 2 and lines):
            # 退出码 2 且有结果：通常是个别文件无权限读取，结果仍然可用。
            return ToolResult.ok(_format_grep_lines(lines))

    # 没有 grep（如 Windows）或 grep 不接受这个模式：用 Python 实现，放进线程并设截止时间。
    return await asyncio.to_thread(_grep_python, pattern, path, include)


def _grep_python(pattern: str, directory: str, include: str | None) -> ToolResult:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return ToolResult.error(f"Error: invalid regex pattern: {e}")
    include_pattern = include
    matches: list[str] = []
    deadline = time.monotonic() + GREP_TIMEOUT_S
    timed_out = False

    def walk(d: str) -> None:
        nonlocal timed_out
        if len(matches) >= 200:
            return
        if time.monotonic() > deadline:
            timed_out = True
            return
        try:
            entries = os.listdir(d)
        except Exception:
            return
        for name in entries:
            # 每个条目前都检查：子目录超时或结果已满后，不再继续读同级文件。
            if timed_out or len(matches) >= 200:
                return
            if time.monotonic() > deadline:
                timed_out = True
                return
            if name.startswith(".") or name == "node_modules":
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                walk(full)
                continue
            if include_pattern and not fnmatch.fnmatch(name, include_pattern):
                continue
            try:
                text = Path(full).read_text(errors="replace")
                for i, line in enumerate(text.split("\n")):
                    if regex.search(line):
                        matches.append(f"{full}:{i+1}:{line}")
                        if len(matches) >= 200:
                            return
            except Exception:
                pass

    walk(directory)
    note = f"\n[search timed out after {GREP_TIMEOUT_S}s; results are partial. Narrow the path or use include.]" if timed_out else ""
    if not matches:
        return ToolResult.ok("No matches found." + note, timed_out=timed_out)
    return ToolResult.ok(_format_grep_lines(matches) + note, timed_out=timed_out)



#--------截断过长的工具调用结果

MAX_RESULT_CHARS = 50000


def _truncate_result(result:str) -> str:
    if len(result) <= MAX_RESULT_CHARS:
        return result
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return (
        result[:keep_each]
        +f"\n\n[... truncated {len(result) - keep_each * 2} chars ...]\n\n"
        +result[-keep_each:]
    )






#----------延迟工具激活---------------------------
_activated_tools: set[str] = set()



def reset_activated_tools() -> None:
    _activated_tools.clear()

def get_active_tool_definitions(all_tools: list[ToolDef] | None = None) -> list[ToolDef]:
    tools = all_tools if all_tools is not None else tool_definitions
    return [
        {k: v for k, v in t.items() if k != "deferred"}
        for t in tools
        if not t.get("deferred") or t["name"] in _activated_tools
    ]

def get_deferred_tool_names(all_tools: list[ToolDef] | None = None) -> list[str]:
    tools = all_tools if all_tools is not None else tool_definitions
    return [t["name"] for t in tools if t.get("deferred") and t["name"] not in _activated_tools]

#执行shell命令
def _parse_timeout_ms(raw: object) -> float:
    """模型传来的 timeout 可能是字符串或离谱的值：转成数字并限制在 1 秒到 10 分钟之间。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = SHELL_DEFAULT_TIMEOUT_MS
    return min(max(value, 1000), SHELL_MAX_TIMEOUT_MS)


async def _run_shell(inp: dict) -> ToolResult:
    timeout_ms = _parse_timeout_ms(inp.get("timeout", SHELL_DEFAULT_TIMEOUT_MS))
    try:
        returncode, stdout, stderr, timed_out = await _run_process(
            inp["command"], shell=True, timeout_s=timeout_ms / 1000
        )
    except Exception as e:
        return ToolResult.error(f"Error: {e}")
    stdout_part = f"\nStdout: {stdout}" if stdout else ""
    stderr_part = f"\nStderr: {stderr}" if stderr else ""
    if timed_out:
        return ToolResult.error(
            f"Command timed out after {int(timeout_ms)}ms; the process group was terminated."
            f" Output before timeout:{stdout_part or ' (none)'}{stderr_part}",
            exit_code=returncode, timed_out=True,
        )
    if returncode != 0:
        return ToolResult.error(f"Command failed (exit code {returncode}){stdout_part}{stderr_part}",
                                exit_code=returncode)
    return ToolResult.ok(stdout or "(no output)", exit_code=0)


#危险命令检测模式列表

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    # 语义上等价于 rm -rf 的间接删除方式
    re.compile(r"\bfind\b.*-delete\b"),
    re.compile(r"\brsync\b.*--delete\b"),
    re.compile(r"\bshutil\.rmtree\b"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
]
def _is_user_written_skill(name: str) -> bool:
    from agents.skill_candidates import skill_author
    from agents.skills import get_skill_by_name
    skill = get_skill_by_name(str(name or ""))
    return bool(skill) and skill_author(skill.skill_dir) != "agent"


def _write_roots() -> list[Path]:
    """允许写入的根目录：项目目录、agent 的记忆目录和计划文件目录。"""
    return [Path.cwd().resolve(), get_memory_dir().resolve(), (Path.home() / ".moltevo" / "plans").resolve()]


def _allowed_commands() -> set[str]:
    """内置白名单 + settings.json 里的 allowed_commands（用户级和项目级）。"""
    allowed = set(shell_policy.DEFAULT_ALLOWED_COMMANDS)
    for path in (Path.home() / ".moltevo" / "settings.json", Path.cwd() / ".moltevo" / "settings.json"):
        extra = (_load_settings(path) or {}).get("allowed_commands")
        if isinstance(extra, list):
            allowed |= {str(c) for c in extra}
    return allowed


def is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)

#权限规则
def _parse_rule(rule: str) -> dict:
    m = re.match(r"^([a-z_]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}

def _load_settings(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return None

_cached_rules: dict | None = None



def load_permission_rules() -> dict:
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules

    allow: list[dict] = []
    deny: list[dict] = []

    user_settings = _load_settings(Path.home() / ".moltevo" / "settings.json")
    project_settings = _load_settings(Path.cwd() / ".moltevo" / "settings.json")

    for settings in [user_settings, project_settings]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        for r in perms.get("allow", []):
            allow.append(_parse_rule(r))
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))

    _cached_rules = {"allow": allow, "deny": deny}
    return _cached_rules


def _matches_rule(rule: dict, tool_name: str, inp: dict) -> bool:
    if rule["tool"] != tool_name:
        return False
    if rule["pattern"] is None:
        return True

    value = ""
    if tool_name == "run_shell":
        value = inp.get("command", "")
    elif "file_path" in inp:
        value = inp["file_path"]
    else:
        return True

    pattern = rule["pattern"]
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return value == pattern


def _check_permission_rules(tool_name: str, inp: dict) -> str | None:
    rules = load_permission_rules()
    for rule in rules["deny"]:
        if _matches_rule(rule, tool_name, inp):
            return "deny"
    for rule in rules["allow"]:
        if _matches_rule(rule, tool_name, inp):
            return "allow"
    return None


def check_permission(
    tool_name: str,
    inp: dict,
    mode: str = "default",
    plan_file_path: str | None = None,
) -> dict:
    """Returns {"action": "allow"|"deny"|"confirm", "message": ...}"""
    if mode == "bypassPermissions":
        return {"action": "allow"}

    rule_result = _check_permission_rules(tool_name, inp)
    if rule_result == "deny":
        return {"action": "deny", "message": f"Denied by permission rule for {tool_name}"}
    if rule_result == "allow":
        return {"action": "allow"}

    if tool_name in READ_TOOLS:
        # 读取敏感路径（~/.ssh、.env、私钥等）也要确认。
        target = inp.get("file_path") or inp.get("path") or ""
        if target and shell_policy.is_sensitive_path(str(target)):
            message = f"read a sensitive path: {target}"
            if mode == "dontAsk":
                return {"action": "deny", "message": f"Auto-denied (dontAsk mode): {message}"}
            return {"action": "confirm", "message": message}
        return {"action": "allow"}

    if mode == "plan":
        if tool_name in EDIT_TOOLS:
            file_path = inp.get("file_path") or inp.get("path")
            if plan_file_path and file_path == plan_file_path:
                return {"action": "allow"}
            return {"action": "deny", "message": f"Blocked in plan mode: {tool_name}"}
        if tool_name == "run_shell":
            return {"action": "deny", "message": "Shell commands blocked in plan mode"}

    if tool_name in ("enter_plan_mode", "exit_plan_mode"):
        return {"action": "allow"}

    # 项目边界：写到项目目录外（agent 自己的记忆和计划目录除外）即使在 acceptEdits 下也要确认。
    if tool_name in ("write_file", "edit_file") and inp.get("file_path"):
        if not shell_policy.is_within(str(inp["file_path"]), _write_roots()):
            message = f"write outside the project: {inp['file_path']}"
            if mode == "dontAsk":
                return {"action": "deny", "message": f"Auto-denied (dontAsk mode): {message}"}
            return {"action": "confirm", "message": message, "cacheable": False}

    # 用户手写的 Skill 对 agent 只读：模型想用 skill_evolve 改它，即使在 acceptEdits 下也要确认。
    if tool_name == "skill_evolve" and _is_user_written_skill(inp.get("skill_name", "")):
        message = f"modify user-written skill: {inp.get('skill_name', '')}"
        if mode == "dontAsk":
            return {"action": "deny", "message": f"Auto-denied (dontAsk mode): {message}"}
        return {"action": "confirm", "message": message, "cacheable": False}

    if mode == "acceptEdits" and tool_name in EDIT_TOOLS:
        return {"action": "allow"}

    # 默认模式下修改已有文件也要确认，并展示 diff。每次修改都单独询问（不缓存），
    # 用户可以选 a 切换到 acceptEdits。dontAsk 模式保持原行为，避免 CI 里所有编辑都被自动拒绝。
    if mode == "default" and tool_name in ("write_file", "edit_file"):
        path = _resolve_tool_path(inp.get("file_path", ""), must_exist=False)
        if path.is_file():
            preview = _edit_preview(tool_name, inp, path)
            if preview is not None:
                verb = "overwrite file" if tool_name == "write_file" else "edit file"
                return {"action": "confirm", "message": f"{verb}: {inp.get('file_path', '')}",
                        "preview": preview, "cacheable": False, "offer_all": True}

    needs_confirm = False
    confirm_message = ""

    if tool_name == "run_shell":
        command = inp.get("command", "")
        action, reason = shell_policy.evaluate(command, allowed=_allowed_commands(),
                                               write_roots=_write_roots(), dangerous=is_dangerous(command))
        if action == "confirm":
            needs_confirm = True
            confirm_message = reason
    elif tool_name == "write_file" and not _resolve_tool_path(inp.get("file_path", ""), must_exist=False).exists():
        needs_confirm = True
        confirm_message = f"write new file: {inp.get('file_path', '')}"
    elif tool_name == "edit_file" and not _resolve_tool_path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"edit non-existent file: {inp.get('file_path', '')}"
    elif tool_name == "skill_evolve":
        needs_confirm = True
        confirm_message = f"evolve skill: {inp.get('skill_name', '')}"
    elif tool_name == "skill_create":
        needs_confirm = True
        confirm_message = f"create skill: {inp.get('name', '')}"

    if needs_confirm:
        if mode == "dontAsk":
            return {"action": "deny", "message": f"Auto-denied (dontAsk mode): {confirm_message}"}
        decision = {"action": "confirm", "message": confirm_message}
        if tool_name == "write_file":
            decision.update(preview=_new_file_preview(inp.get("content", "")), offer_all=True)
        return decision

    return {"action": "allow"}






def _skill_op_result(result: dict) -> ToolResult:
    text = _truncate_result(json.dumps(result, ensure_ascii=False, indent=2))
    return ToolResult.ok(text) if result.get("ok") else ToolResult.error(text)


MAX_PREVIEW_LINES = 200


def _clip_lines(lines: list[str]) -> str:
    if len(lines) > MAX_PREVIEW_LINES:
        lines = lines[:MAX_PREVIEW_LINES] + [f"... ({len(lines) - MAX_PREVIEW_LINES} more lines)"]
    return "\n".join(lines)


def _edit_preview(tool_name: str, inp: dict, path: Path) -> str | None:
    """模拟这次修改并生成 unified diff。找不到要替换的文本时返回 None（工具执行时会报错，不必询问）。"""
    try:
        old = path.read_text(errors="replace")
    except OSError:
        return None
    if tool_name == "write_file":
        new = inp.get("content", "")
    else:
        actual = _find_actual_string(old, inp.get("old_string", ""))
        if not actual or old.count(inp.get("old_string", "")) > 1:
            return None
        new = old.replace(actual, inp.get("new_string", ""), 1)
    name = inp.get("file_path", str(path))
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                     fromfile=f"a/{name}", tofile=f"b/{name}", lineterm=""))
    return _clip_lines(diff) if diff else "(no changes)"


def _new_file_preview(content: str) -> str:
    return _clip_lines([f"+{line}" for line in content.splitlines()])


#----------------执行工具调用-----------------------
# 'agent' 和 'skill' 这两个工具在 agent.py 中处理，以避免循环依赖。"

async def execute_tool(
    name: str, inp: dict, read_file_state: dict[str, float] | None = None,
    edit_history: EditHistory | None = None,
) -> ToolResult:
    if name == "read_file":
        # 读大文件可能耗时，放进线程，不阻塞事件循环；读写共享状态的代码留在主线程。
        result = await asyncio.to_thread(_read_file, inp)
        if read_file_state is not None and not result.is_error:
            abs_path = str(_resolve_tool_path(inp["file_path"]).resolve())
            try:
                read_file_state[abs_path] =  os.path.getmtime(abs_path)
            except OSError:
                pass
        result.content = _truncate_result(result.content)
        return result

    if name in ("write_file", "edit_file") and read_file_state is not None:
        abs_path = str(_resolve_tool_path(inp["file_path"], must_exist=(name == "edit_file")).resolve())
        if os.path.exists(abs_path):
            if abs_path not in read_file_state:
                verb = "writing" if name == "write_file" else "editing"
                return ToolResult.error(f"Error: You must read this file before {verb}. Use read_file first to see its current contents.")
            if os.path.getmtime(abs_path) != read_file_state[abs_path]:
                verb = "writing" if name == "write_file" else "editing"
                return ToolResult.error(f"Warning: {inp['file_path']} was modified externally since your last read. Please read_file again before {verb}.")

    #搜索和激活延迟加载的工具。
    if name == "tool_search":
        query = (inp.get("query") or "").lower()
        deferred = [t for t in tool_definitions if t.get("deferred")]
        matches = [
            t for t in deferred
            if query in t["name"].lower() or query in (t.get("description") or "").lower()
        ]
        if not matches:
            return ToolResult.ok("No matching deferred tools found.")

        for m in matches:
            _activated_tools.add(m["name"])

        return ToolResult.ok(json.dumps(
            [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["input_schema"]} for t in
             matches],
            indent=2,
        ))
    if name == "skill_evolve":
        from .skills import evolve_skill

        result = evolve_skill(
            skill_name=inp.get("skill_name", ""),
            lesson=inp.get("lesson", ""),
            rationale=inp.get("rationale", ""),
            target=inp.get("target", "active"),
        )
        return _skill_op_result(result)

    if name == "skill_create":
        from .skills import create_skill

        result = create_skill(
            name=inp.get("name", ""),
            description=inp.get("description", ""),
            instructions=inp.get("instructions", ""),
            when_to_use=inp.get("when_to_use", "") or inp.get("when-to-use", ""),
            target=inp.get("target", "project"),
            context=inp.get("context", "inline"),
            user_invocable=bool(inp.get("user_invocable", False)),
            allowed_tools=inp.get("allowed_tools"),
            evidence=inp.get("evidence", ""),
        )
        return _skill_op_result(result)

    if name == "run_shell":
        result = await _run_shell(inp)
    elif name == "grep_search":
        result = await _grep_search(inp)
    elif name == "list_files":
        # 大目录上的 glob 可能很慢，放进线程。
        result = await asyncio.to_thread(_list_files, inp)
    elif name in ("write_file", "edit_file"):
        # 写之前存快照，供 /undo 恢复。
        record = None
        if edit_history is not None:
            record = edit_history.before_write(_resolve_tool_path(inp["file_path"], must_exist=False), name)
        # 写操作留在主线程同步执行：很快，而且被取消时不会出现"线程里写完了、模型却不知道"的情况。
        result = (_write_file if name == "write_file" else _edit_file)(inp)
        if edit_history is not None:
            if record is None:
                if not result.is_error:
                    result.content += "\n(Note: file too large to snapshot; this change cannot be undone with /undo.)"
            elif result.is_error:
                edit_history.discard(record)
            else:
                edit_history.commit(record)
    else:
        return ToolResult.error(f"Unknown tool: {name}")
    result.content = _truncate_result(result.content)

    # 更新时间
    if name in ("write_file", "edit_file") and read_file_state is not None and not result.is_error:
        abs_path = str(_resolve_tool_path(inp["file_path"], must_exist=False).resolve())
        try:
            read_file_state[abs_path] = os.path.getmtime(abs_path)
        except OSError:
            pass

    return result



def reset_permission_cache() -> None:
    global _cached_rules
    _cached_rules = None



