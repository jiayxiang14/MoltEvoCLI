#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable, Any

import anthropic
import httpx
import openai

from agents.events import RunRecorder, new_id, redact, summarize_args
from agents.mcp_client import McpManager
from agents.pricing import call_cost_usd, get_price
from agents.snapshots import EditHistory
from agents.tool_journal import ToolJournal
from agents.tool_result import ToolResult
from agents.memory import MemoryPrefetch, start_memory_prefetch, format_memories_for_injection, get_memory_dir
from agents.prompt import build_system_prompt
from agents.session_memory import (
    FOLD_SESSION_MEMORY_SYSTEM,
    build_anthropic_transcript,
    build_folding_user_prompt,
    build_openai_transcript,
    fallback_folded_memory,
    format_folded_memory,
    parse_folded_memory,
)
from agents.session import SESSION_DIR, save_folded_session_memory, save_session
from agents.subagent import get_sub_agent_config
from agents.tools import ToolDef, tool_definitions, execute_tool, CONCURRENCY_SAFE_TOOLS, check_permission, \
    get_active_tool_definitions, _run_shell
from agents.ui import print_info, print_divider, print_assistant_text, print_sub_agent_start, print_sub_agent_end, \
    start_spinner, stop_spinner, print_cost, print_tool_call, print_tool_result, print_confirmation, print_retry, \
    print_error, print_warning


# 指数退避重试


# 请求本身有问题的错误，重试也不会成功（参照 anthropics/claude-quickstarts 的 computer-use loop）。
_UNRECOVERABLE_ERRORS = (
    anthropic.BadRequestError, anthropic.AuthenticationError, anthropic.PermissionDeniedError,
    anthropic.NotFoundError, anthropic.UnprocessableEntityError,
    openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError,
    openai.NotFoundError, openai.UnprocessableEntityError,
)
# 限流、连接断开、超时（APITimeoutError 是 APIConnectionError 的子类）。
_RECOVERABLE_ERRORS = (
    anthropic.RateLimitError, anthropic.APIConnectionError,
    openai.RateLimitError, openai.APIConnectionError,
    httpx.TransportError,          # 流式读取中途断开等，SDK 不一定会包装
)


def _is_retryable(error: Exception) -> bool:
    """按异常类型判断：限流、连接问题、5xx 重试；参数、认证、权限、模型不存在等直接放弃。"""
    if isinstance(error, _UNRECOVERABLE_ERRORS):
        return False
    if isinstance(error, _RECOVERABLE_ERRORS):
        return True
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int) and (status in (408, 409, 429) or 500 <= status < 600):
        return True
    msg = str(error).lower()
    return any(s in msg for s in ("overloaded", "econnreset", "etimedout", "connection reset"))


def _safe_utf8_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_for_utf8(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_utf8_text(value)
    if isinstance(value, list):
        return [_sanitize_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_utf8(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_for_utf8(key): _sanitize_for_utf8(item)
            for key, item in value.items()
        }
    return value


async def _with_retry(fn, max_retries: int = 3, on_retry: Callable[[], None] | None = None):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            if on_retry:
                on_retry()
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "deepseek-chat":200000
}

def _get_context_windows(model:str)->int:
    return MODEL_CONTEXT.get(model, 200000)


#多层级压缩常数
SNIP_THRESHOLD = 0.60
AUTO_COMPACT_THRESHOLD = 0.70
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
MICROCOMPACT_IDLE_S = 5 * 60  # 5 minutes

KEEP_RECENT_RESULTS = 3
# 按区间裁剪：未裁剪的结果攒到 KEEP_RECENT_RESULTS + SNIP_INTERVAL 条才一次性裁回 KEEP_RECENT_RESULTS 条。
# 每轮都多裁一条会让请求前缀每轮都变、缓存每轮都失效；按区间裁剪每 SNIP_INTERVAL 条结果才改写一次前缀。
SNIP_INTERVAL = 8

# 当前正在执行的工具调用的 span。用 ContextVar 而不是实例字段：每个 asyncio 任务有自己的一份，
# 并发执行的工具不会互相覆盖；子 agent 派生时读取它作为 parent_span_id。
_CURRENT_SPAN: contextvars.ContextVar[str | None] = contextvars.ContextVar("moltevo_current_span", default=None)

# 模型连续返回空内容时，最多提示它继续的次数
MAX_EMPTY_RETRIES = 2

# 完成后验证
MAX_VERIFY_RETRIES = 2
VERIFY_TIMEOUT_MS = 300_000
VERIFY_FEEDBACK_CHARS = 4000



def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384

#转换tool的形式到openai
def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class Agent:
    def __init__(self,
                 *,
                 permission_mode:str="default",
                 model:str="deepseek-chat",
                 api_base: str | None=None,
                 anthropic_base_url: str | None=None,
                 api_key: str | None=None,
                 thinking: bool=False,
                 max_cost_usd: float | None=None,
                 max_turns: int | None=None,
                 confirm_fn:Callable[[str], Awaitable[bool]] | None=None,
                 custom_system_prompt: str | None=None,
                 custom_tools: list[ToolDef] | None=None,
                 is_sub_agent: bool=False,
                 recorder: RunRecorder | None=None,
                 trace_parent: dict[str, Any] | None=None,
                 events_path: Path | None=None,
                 edit_history: EditHistory | None=None,
                 verify_command: str | None=None,):
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.model = model
        self.use_openai = bool(api_base)
        self.is_sub_agent = is_sub_agent
        # 保留连接配置，派生子 agent 时原样继承。
        self._api_base = api_base
        self._anthropic_base_url = anthropic_base_url
        self._api_key = api_key
        self.tools = custom_tools or tool_definitions
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.confirm_fn = confirm_fn
        self._custom_system_prompt = custom_system_prompt
        self.effective_window=_get_context_windows(model) -20000
        self.session_id = uuid.uuid4().hex[:8]
        self.session_start_time= time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        self.current_turns = 0
        self.last_api_call_time = 0
        # 事件日志、汇总统计和累计费用；子 agent 与父 agent 共用同一个 recorder。
        self._recorder = recorder if recorder is not None else RunRecorder(self.session_id, path=events_path)
        trace_parent = trace_parent or {}
        self._trace_id: str | None = trace_parent.get("trace_id")
        self._parent_span_id: str | None = trace_parent.get("parent_span_id")
        self._agent_label: str = trace_parent.get("label") or ("sub" if is_sub_agent else "main")
        # 子 agent 里选 a（本会话编辑都自动同意）时，通知父 agent 一起切换。
        self._parent_accept_all: Callable[[], None] | None = None
        self._call_retries = 0
        self._call_first_token_at: float | None = None
        # 改文件前的快照（/undo）；子 agent 共用，子 agent 的修改同样可以撤销。
        # 注意用 is not None：EditHistory 定义了 __len__，空栈在布尔判断里为假。
        self._edit_history = edit_history if edit_history is not None else EditHistory(self.session_id)
        # 下一条用户消息前要附上的系统通知（例如用户撤销了某次修改）。
        self._pending_notices: list[str] = []
        # 完成后验证：本轮改过文件才跑，失败时把输出交还模型，最多重试 MAX_VERIFY_RETRIES 次。
        self.verify_command = verify_command
        self._turn_commit_mark = 0
        self._verify_attempts = 0
        self._verify_failed = False
        # 检查点：主 agent 每轮开始前记一个，/rewind 据此回退对话和/或代码。
        # 压缩、/clear 等整体改写历史时 _history_epoch 加一，之前的检查点就不能再回退对话了。
        self._checkpoints: list[dict[str, Any]] = []
        self._turn_counter = 0
        self._history_epoch = 0
        self._turn_preview = ""
        # 工具调用结果日志（按 tool_call_id）：一轮被打断后据此补结果，避免已执行的操作被重做。
        # 只有主 agent 的消息会存档和恢复，子 agent 只在内存里记。
        journal_path = None if is_sub_agent else SESSION_DIR / f"{self.session_id}.journal.jsonl"
        self._journal = ToolJournal(journal_path)
        self._journal_files: list[str] = [str(journal_path)] if journal_path else []
        if recorder is None and not is_sub_agent and get_price(model) is None:
            note = "费用将显示为未知" + ("，--max-cost 上限无法生效" if max_cost_usd is not None else "")
            print_warning(f"模型 {model} 不在价格表中，{note}。可在 settings.json 的 pricing 字段补充价格。")


        self._aborted = False
        #存储异步任务
        self._current_task:asyncio.Task | None = None
        #权限白名单
        self._confirmed_paths: set[str] = set()
        # 流式阶段提前启动的工具任务；本轮结束、中断或出错时统一取消，避免在后台继续跑。
        self._inflight_tool_tasks: set[asyncio.Task] = set()


        # 计划模式”（Plan Mode）状态的变量
        self._pre_plan_mode: str | None=None
        self._plan_file_path: str | None=None
        self._plan_approval_fn : Callable[[str], Awaitable[bool]] | None=None
        self._context_cleared : bool=False

        #思考模式
        self._thinking_mode = self._resolve_thinking_mode()

        #子agent的输出缓存
        self._output_buffer: list[str] | None=None
        self._turn_output_buffer: list[str] | None = None

        # 编辑前读取
        self._read_file_state: dict[str, float] ={}

        #MCP集成
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        #记忆回溯
        #记忆agent已经回答过的信息
        self._already_surfaced_memories: set[str] = set()
        #当前会话占用的字节数
        self._session_memory_bytes = 0

        #区分message的历史消息
        self._anthropic_messages: list[str] = []
        self._openai_messages: list[str] = []
        self._last_retrieved_skill_reference: dict[str, Any] | None = None
        self._last_retrieved_skill_hits: list[dict[str, Any]] = []
        self._pending_skill_extraction_window: dict[str, Any] | None = None
        self._background_skill_tasks: set[asyncio.Task] = set()
        self._folded_session_memories: list[dict[str, Any]] = []
        self._fold_last_time: float = 0.0
        self._fold_count: int = 0
        self._tool_error_streak: int = 0
        self._same_tool_repeat_count: int = 0
        self._last_tool_name: str = ""

        #构建系统提示词
        self._base_system_prompt = custom_system_prompt or build_system_prompt()

        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt

        #初始化大模型客户端
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._anthropic_client = None
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            kwargs : dict[str,Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

        self._refresh_runtime_system_prompt()

    #判断返回模型的思考模式
    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not self._model_supports_thinking():
            return "disabled"

        if self._mode_supports_adaptive_thinking(self.model):
            return "adaptive"
        return "enabled"

    def _model_supports_thinking(self) -> bool:
        m = self.model.lower()
        if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
            return False
        if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
            return True
        return False
    def _model_supports_adaptive_thinking(self) -> bool:
        m = self.model.lower()
        return "opus-4-6" in m or "sonnet-4-6" in m

    #生成一个用于保存 AI 计划（Plan）的 Markdown 文件的绝对路径。
    def _generate_plan_file_path(self) -> str:
        d = Path.home() / ".moltevo" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

    # Plan Mode Active

    Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

    ## Plan File: {self._plan_file_path}
    Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

    ## Workflow
    1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
    2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
    3. **Write Plan**: Write a structured plan to the plan file including:
       - **Context**: Why this change is needed
       - **Steps**: Implementation steps with critical file paths
       - **Verification**: How to test the changes
    4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

    IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    #判断当前的任务所有的任务是否完成
    @property
    def is_processing(self)->bool:
        return self._current_task is not None and not self._current_task.done()

    #大模型调用的工厂方法,构建一个用于记忆召回（memory recall）的 sideQuery 可调用对象，兼容anthropic, openai。
    def _build_side_query(self, *, max_tokens: int = 256, purpose: str = "side_query"):
        if self._anthropic_client:
            client = self._anthropic_client
            model = self.model
            async def _sq(system:str, user_message:str)->str:
                started = time.monotonic()
                resp = await client.messages.create(
                    model=model, max_tokens=max(1, int(max_tokens)), system=system,
                messages=[{"role": "user", "content": user_message}],
                )
                # 侧查询同样要计费并计入预算。
                self._record_model_call(self._anthropic_usage(resp.usage) if getattr(resp, "usage", None) else None,
                                        started=started, stop_reason=getattr(resp, "stop_reason", None),
                                        kind="side", purpose=purpose)
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                if not text.strip():
                    block_types = [str(getattr(b, "type", "")) for b in getattr(resp, "content", [])]
                    logging.warning(
                        "side_query returned empty Anthropic-compatible response: model=%s stop_reason=%s content_block_types=%s",
                        getattr(resp, "model", model),
                        getattr(resp, "stop_reason", ""),
                        block_types,
                    )
                return text
            return _sq
        if self._openai_client:
            client = self._openai_client
            model = self.model
            async def _sq_openai(system:str, user_message:str)->str:
                started = time.monotonic()
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max(1, int(max_tokens)),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],

                )
                self._record_model_call(self._openai_usage(getattr(resp, "usage", None)), started=started,
                                        stop_reason=resp.choices[0].finish_reason if resp.choices else None,
                                        kind="side", purpose=purpose)
                if not resp.choices:
                    logging.warning("side_query returned no OpenAI-compatible choices: model=%s", model)
                    return ""
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    logging.warning(
                        "side_query returned empty OpenAI-compatible response: model=%s finish_reason=%s message=%s",
                        model,
                        getattr(choice, "finish_reason", ""),
                        choice.message,
                    )
                return content
            return _sq_openai
        return None
    #异步任务取消（Abort）
    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn:Callable[[str], Awaitable[dict]]) -> None:
        self._plan_approval_fn = fn

    def _leave_plan_mode(self, target_mode: str) -> None:
        """离开计划模式的唯一出口：切换权限模式、清理计划文件路径、恢复系统提示词。"""
        self.permission_mode = target_mode
        self._pre_plan_mode = None
        self._plan_file_path = None
        self._system_prompt = self._base_system_prompt
        if self.use_openai and self._openai_messages:
            self._openai_messages[0]["content"] = self._system_prompt


    #计划模式开关（“状态切换与现场保护”机制）
    def toggle_plan_mode(self) -> str:
        """
               1. 退出计划模式（从 plan 切回原模式）
               当当前模式已经是 plan 时，执行 if 分支：
               恢复之前的状态：self.permission_mode = self._pre_plan_mode or "default"。
                   在进入计划模式时，程序会把原本的模式保存在 _pre_plan_mode 里。退出时，就把它重新拿出来赋值回去，恢复到切换前的状态。
               清理计划模式的痕迹：把 _pre_plan_mode 和 _plan_file_path（计划文件路径）清空，并将系统提示词 _system_prompt 恢复为最基础的 _base_system_prompt。
               同步 OpenAI 消息：如果底层使用的是 OpenAI 接口，它还会同步更新消息列表里的第一条系统提示词，确保 AI 的上下文也跟着切换回来。
               反馈返回：打印退出提示，并返回恢复后的模式名称。

               2. 进入计划模式（从其他模式切入 plan）
       当当前模式不是 plan 时，执行 else 分支：
       保护当前现场：self._pre_plan_mode = self.permission_mode。先把当前正在使用的模式（比如正常模式或自动接受模式）暂存起来，方便以后能原路返回。
       切换并初始化：将当前模式设为 "plan"，生成一个专属的计划文件路径，并扩展系统提示词。通过拼接 _build_plan_mode_prompt()，给 AI 注入“只动脑不动手、输出结构化计划”的专属指令。
       同步 OpenAI 消息：同样地，如果使用 OpenAI，也会实时更新上下文里的系统提示词。
       反馈与返回：打印进入提示（包含计划文件的路径），并返回 "plan"。
        """
        if self.permission_mode == "plan":
            self._leave_plan_mode(self._pre_plan_mode or "default")
            print_info(f"Exited plan mode -> {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> dict:
        return {"input":self.total_input_tokens, "output":self.total_output_tokens}

    #主入口

    async def  chat(self, user_message:str)->None:
        #懒加载MCP服务在第一次chat的时候
        if not self._mcp_initialized and not self.is_sub_agent:
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print_error(f"MCP init failed: {e}")

        original_user_message = _safe_utf8_text(user_message)
        ready_skill_extraction_window: dict[str, Any] | None = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        if not self.is_sub_agent:
            ready_skill_extraction_window = self._pop_pending_skill_extraction_window(original_user_message)
            user_message, self._last_retrieved_skill_reference = self._augment_user_message_with_skill_context(
                original_user_message
            )

        if self._pending_notices and not self.is_sub_agent:
            user_message = "\n".join(self._pending_notices) + "\n\n" + user_message
            self._pending_notices = []
        self._turn_commit_mark = self._edit_history.commits
        self._verify_attempts = 0
        self._verify_failed = False
        self._turn_preview = original_user_message

        self._aborted = False
        self._turn_output_buffer = []
        if not self._parent_span_id:
            # 主 agent 每轮用户输入一个 trace；子 agent 沿用父 agent 的 trace。
            self._trace_id = new_id()
        turn_started = time.monotonic()
        turns_before = self.current_turns
        turn_error: str | None = None
        self._recorder.emit("turn_start", input_chars=len(original_user_message), **self._trace_fields())
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        self._current_task = asyncio.create_task(coro)
        try:
            await self._current_task
        except asyncio.CancelledError:
            self._aborted = True
        except Exception as e:
            turn_error = f"{type(e).__name__}: {e}"
            self._recorder.emit("error", where="chat", error=redact(turn_error), **self._trace_fields())
            # 模型调用重试用尽等异常照常向上抛，但先把这一轮已有的输入和工具结果存盘。
            if not self.is_sub_agent:
                self._auto_save()
            raise
        finally:
            self._current_task = None
            await self._cancel_inflight_tool_tasks()
            self._recorder.emit(
                "turn_end", duration_ms=round((time.monotonic() - turn_started) * 1000),
                turns=self.current_turns - turns_before, aborted=self._aborted, error=turn_error,
                **self._trace_fields(),
            )
        assistant_text = "".join(self._turn_output_buffer or []).strip()
        self._turn_output_buffer = None
        if not self.is_sub_agent and not self._aborted:
            self._schedule_background_skill_task(self._run_skill_usage_tracking(original_user_message, assistant_text))
            if ready_skill_extraction_window:
                self._schedule_background_skill_task(self._run_online_skill_evolution(ready_skill_extraction_window))
            self._set_pending_skill_extraction_window(
                original_user_message=original_user_message,
                assistant_text=assistant_text,
                retrieved_reference=self._last_retrieved_skill_reference,
            )
        if not self.is_sub_agent:
            print_divider()
            self._auto_save()



   #执行一次对话，收集本轮模型输出文本，并返回本轮消耗的 token 数
    async def run_once(self, prompt:str)->None:
        self._output_buffer = []
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        await self.chat(prompt)
        text = "".join(self._output_buffer)
        self._output_buffer = None
        return {
            "text": text,
            "tokens":{
                "input":self.total_input_tokens-prev_in,
                "output":self.total_output_tokens-prev_out
            },
        }

    #输出工具：统一处理模型输出文本。根据当前是否处于“收集输出”的模式
    # 决定是把文本存进缓冲区，还是直接打印到终端。
    def _emit_text(self, text:str)->None:
        text = _safe_utf8_text(text)
        if self._turn_output_buffer is not None:
            self._turn_output_buffer.append(text)
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    def _build_fold_guidance_section(self) -> str:
        """
        运行时折叠提示（上下文用量、连续出错次数等）。这些数字每次请求都可能变，
        所以不放进系统提示词（会让系统提示词之后的缓存全部失效），而是在请求时临时附在最后一条消息末尾。
        """
        if self._custom_system_prompt is not None:
            return ""
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0.0
        last_fold = "never" if not self._fold_last_time else f"{int((time.time() - self._fold_last_time) / 60)}m ago"
        return (
            "<system-reminder>\n# Runtime Fold Guidance\n"
            f"- Current context utilization: {utilization:.0%}\n"
            f"- Recent tool error streak: {self._tool_error_streak}\n"
            f"- Same tool repeat count: {self._same_tool_repeat_count}\n"
            f"- Last fold: {last_fold}\n"
            "- If the context is getting long, the same tool is being retried without progress, or tool failures are accumulating, call `compact_context` before trying more tools.\n"
            "- If you folded very recently and the next step is clear, prefer continuing rather than folding again.\n"
            "</system-reminder>"
        )

    def _refresh_runtime_system_prompt(self) -> None:
        """
        重建系统提示词。为了让提示词缓存生效，系统提示词必须稳定：只在会话开始、Skill/记忆变化、
        压缩、切换 plan 模式时调用，不在每轮工具执行后调用（其中的 git 状态是重建时的快照）。
        """
        if self._custom_system_prompt is not None:
            return
        self._base_system_prompt = build_system_prompt()
        if self.permission_mode == "plan":
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt
        if self.use_openai and self._openai_messages:
            self._openai_messages[0]["content"] = self._system_prompt

    def _record_tool_outcome(self, tool_name: str, success: bool) -> None:
        if tool_name == self._last_tool_name:
            self._same_tool_repeat_count += 1
        else:
            self._same_tool_repeat_count = 1
        self._last_tool_name = tool_name
        if success:
            self._tool_error_streak = 0
        else:
            self._tool_error_streak += 1

    def _record_fold_event(self) -> None:
        self._fold_last_time = time.time()
        self._fold_count += 1
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    # ── 工具执行的公共收尾：两套主循环都走这里 ─────────────────────────
    def _trace_fields(self, span_id: str | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {"trace_id": self._trace_id, "agent": self._agent_label}
        if span_id:
            fields["span_id"] = span_id
        if self._parent_span_id:
            fields["parent_span_id"] = self._parent_span_id
        return fields

    async def _timed_tool_call(self, name: str, inp: dict, span_id: str, call_id: str | None = None) -> ToolResult:
        """执行一次工具调用并计时。除取消外不抛异常：异常转成错误结果，避免一个工具拖垮整轮。"""
        self._journal.started(call_id, name)
        started = time.monotonic()
        span_token = _CURRENT_SPAN.set(span_id)
        try:
            result = await self._execute_tool_call(name, inp)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            result = ToolResult.error(f"Error executing tool: {e}")
        finally:
            _CURRENT_SPAN.reset(span_token)
        if not isinstance(result, ToolResult):
            result = ToolResult.ok(str(result))
        result.meta.setdefault("duration_ms", (time.monotonic() - started) * 1000)
        return result

    def _finish_tool_call(self, name: str, inp: dict, result: ToolResult, *, span_id: str,
                          early: bool = False, call_id: str | None = None) -> ToolResult:
        """截断或落盘 → 打印 → 记录成败 → 写事件。"""
        if name == "run_shell":
            self._edit_history.note_shell()
        content = self._persist_large_result(name, _safe_utf8_text(result.content))
        print_tool_result(name, content)
        self._record_tool_outcome(name, not result.is_error)
        meta = {k: v for k, v in result.meta.items() if k in ("exit_code", "timed_out")}
        self._recorder.record_tool_call(
            tool=name, is_error=result.is_error, duration_ms=result.meta.get("duration_ms"),
            early=early, result_bytes=len(content.encode()), args=summarize_args(inp),
            **meta, **self._trace_fields(span_id),
        )
        result.content = content
        self._journal.done(call_id, name, result)
        if name in ("write_file", "edit_file") and not result.is_error and self._writes_memory(inp):
            # 记忆索引是系统提示词的一部分；系统提示词不再每轮重建，所以写记忆后单独刷新。
            self._refresh_runtime_system_prompt()
        return result

    @staticmethod
    def _writes_memory(inp: dict) -> bool:
        try:
            target = Path(str(inp.get("file_path", ""))).expanduser().resolve()
            return target.is_relative_to(get_memory_dir().resolve())
        except (OSError, ValueError):
            return False

    async def _authorize_tool(self, name: str, inp: dict, span_id: str) -> ToolResult | None:
        """权限判定 + 必要时询问用户。返回 None 表示放行，否则返回交给模型的拒绝结果。"""
        perm = check_permission(name, inp, self.permission_mode, self._plan_file_path)
        message = perm.get("message", "")
        trace = self._trace_fields(span_id)
        if perm["action"] == "deny":
            print_info(f"Denied: {message}")
            self._record_tool_outcome(name, False)
            self._recorder.record_permission(tool=name, decision="deny", reason=message, **trace)
            return ToolResult.error(f"Action denied: {message}")
        if perm["action"] == "confirm" and message:
            # 修改已有文件的确认每次都问（内容不同），其余同一条提示确认一次后本会话不再问。
            cacheable = perm.get("cacheable", True)
            if cacheable and message in self._confirmed_paths:
                self._recorder.record_permission(tool=name, decision="allow_cached", reason=message, **trace)
                return None
            answer = await self._confirm_dangerous(message, preview=perm.get("preview"),
                                                   offer_all=perm.get("offer_all", False))
            decision = {"yes": "user_approved", "all": "user_approved_all", "no": "user_denied"}[answer]
            self._recorder.record_permission(tool=name, decision=decision, reason=message, **trace)
            if answer == "no":
                self._record_tool_outcome(name, False)
                return ToolResult.error("User denied this action.")
            if answer == "all":
                self._accept_all_edits()
                print_info("File edits will be auto-approved for the rest of this session (acceptEdits mode).")
            if cacheable:
                self._confirmed_paths.add(message)
            return None
        self._recorder.record_permission(tool=name, decision="allow", **trace)
        return None

    def _accept_all_edits(self) -> None:
        """用户选了 a：本 agent 和它的所有上级 agent 都切换到 acceptEdits。"""
        if self.permission_mode == "default":
            self.permission_mode = "acceptEdits"
        if self._parent_accept_all:
            self._parent_accept_all()

    @staticmethod
    def _anthropic_tool_result(tool_use_id: str, result: ToolResult) -> dict:
        block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": result.content}
        if result.is_error:
            block["is_error"] = True
        return block

    @staticmethod
    def _openai_tool_message(tool_call_id: str, result: ToolResult) -> dict:
        # OpenAI 协议没有 is_error 字段，用统一前缀告诉模型这次调用失败了。
        content = f"[tool error] {result.content}" if result.is_error else result.content
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def _augment_user_message_with_skill_context(self, user_message: str) -> tuple[str, dict[str, Any] | None]:
        try:
            from .skills import format_retrieved_skill_context

            context, top_ref = format_retrieved_skill_context(user_message, limit=3)
        except Exception:
            return user_message, None
        if top_ref and isinstance(top_ref.get("all_hits"), list):
            self._last_retrieved_skill_hits = list(top_ref.get("all_hits") or [])
        if not context.strip():
            return user_message, top_ref
        return f"{user_message}\n\n{context}", top_ref

    def _strip_runtime_injections(self, text: str) -> str:
        return re.sub(r"\n*<retrieved_skills>.*?</retrieved_skills>\s*", "", str(text or ""), flags=re.DOTALL).strip()

    def _message_text(self, msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, str):
            return self._strip_runtime_injections(content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif "content" in block and block.get("type") not in {"tool_result", "tool_use"}:
                        parts.append(str(block.get("content") or ""))
            return self._strip_runtime_injections("\n".join(parts))
        return ""

    def _recent_dialog_messages(self, *, max_messages: int = 8) -> list[dict[str, str]]:
        raw_messages = self._openai_messages if self.use_openai else self._anthropic_messages
        out: list[dict[str, str]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = self._message_text(msg)
            if text:
                out.append({"role": role, "content": text})
        return out[-max(2, int(max_messages)) :]

    async def _confirm_online_skill_write(self, summary: str) -> bool:
        if self.permission_mode in {"bypassPermissions", "acceptEdits"}:
            return True
        if self.permission_mode in {"plan", "dontAsk"}:
            return False
        if self.confirm_fn is None:
            return False
        try:
            # 与工具确认走同一套答案解析：confirm_fn 返回的是用户输入的字符串，不能直接 bool()。
            return await self._confirm_dangerous(summary) == "yes"
        except Exception:
            return False

    async def _confirm_background_online_skill_write(self, summary: str) -> bool:
        return self.permission_mode in {"bypassPermissions", "acceptEdits"}

    def _online_evolution_enabled(self) -> bool:
        raw = os.environ.get("MOLTEVO_AUTO_SKILL_EVOLUTION", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _schedule_background_skill_task(self, coro) -> None:
        if self.permission_mode == "plan":
            try:
                coro.close()
            except Exception:
                pass
            return
        task = asyncio.create_task(coro)
        self._background_skill_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._background_skill_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception:
                pass

        task.add_done_callback(_done)

    async def drain_background_skill_tasks(self) -> None:
        tasks = [task for task in self._background_skill_tasks if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _pop_pending_skill_extraction_window(self, next_user_feedback: str) -> dict[str, Any] | None:
        pending = self._pending_skill_extraction_window
        self._pending_skill_extraction_window = None
        if not pending:
            return None
        messages = list(pending.get("messages") or [])
        feedback = _safe_utf8_text(next_user_feedback).strip()
        if feedback:
            messages.append({"role": "user", "content": feedback})
        pending["messages"] = messages[-10:]
        pending["next_user_feedback"] = feedback
        return pending

    def _set_pending_skill_extraction_window(
        self,
        *,
        original_user_message: str,
        assistant_text: str,
        retrieved_reference: dict[str, Any] | None,
    ) -> None:
        if not original_user_message.strip() or not assistant_text.strip():
            return
        self._pending_skill_extraction_window = {
            "messages": self._recent_dialog_messages(max_messages=8),
            "latest_user": original_user_message,
            "latest_assistant": assistant_text,
            "retrieved_reference": self._compact_retrieved_reference(retrieved_reference),
            "session_id": self.session_id,
        }

    def _compact_retrieved_reference(self, ref: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ref:
            return None
        return {k: v for k, v in ref.items() if k != "all_hits"}

    async def _run_online_skill_evolution(self, window: dict[str, Any], *, interactive_confirm: bool = False) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        messages = list(window.get("messages") or [])
        if not messages:
            return

        side_query = self._build_side_query(max_tokens=2200, purpose="skill_evolution")
        if side_query is None:
            return

        try:
            from .online_skill_evolution import online_ingest
        except Exception:
            return

        result = await online_ingest(
            messages=messages,
            side_query=side_query,
            retrieved_reference=window.get("retrieved_reference") or None,
            hint=str(window.get("hint") or ""),
            confirm_write=self._confirm_online_skill_write if interactive_confirm else self._confirm_background_online_skill_write,
            target=os.environ.get("MOLTEVO_AUTO_SKILL_TARGET", "project"),
        )
        if result.get("ok"):
            if result.get("action") in {"add", "merge"}:
                self._refresh_runtime_system_prompt()
                print_info(f"Online skill {result.get('action')}: {result.get('skill')}")
        elif result.get("action") not in {"add_denied", "merge_denied"}:
            print_error(f"Online skill evolution failed: {result.get('error') or result}")

    async def _run_skill_usage_tracking(self, original_user_message: str, assistant_text: str) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        hits = list(self._last_retrieved_skill_hits or [])
        if not hits or not assistant_text.strip():
            return
        side_query = self._build_side_query(max_tokens=700, purpose="skill_usage_judge")
        try:
            from .online_skill_evolution import judge_retrieved_skill_usage
            from .skills import record_usage_judgments

            judgments = await judge_retrieved_skill_usage(
                hits=hits,
                user_message=original_user_message,
                assistant_text=assistant_text,
                side_query=side_query,
            )
            result = record_usage_judgments(judgments)
            if result.get("pruned"):
                self._refresh_runtime_system_prompt()
        except Exception:
            return

    async def extract_now(self, hint: str = "") -> dict[str, Any]:
        pending = self._pending_skill_extraction_window
        if not pending:
            return {"ok": False, "error": "no pending online skill extraction window"}
        window = dict(pending)
        window["hint"] = hint
        await self._run_online_skill_evolution(window, interactive_confirm=True)
        self._pending_skill_extraction_window = None
        return {"ok": True}


    def clear_history(self)->None:
        self._anthropic_messages = []
        self._openai_messages = []
        self._history_epoch += 1
        self._pending_skill_extraction_window = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content":self._system_prompt})
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        print_info("Conversation cleared.")

    def show_cost(self):
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        cache_info = (f" | cache read {self._recorder.cache_read_tokens} / write {self._recorder.cache_write_tokens}"
                      if self._recorder.cache_read_tokens or self._recorder.cache_write_tokens else "")
        print_info(
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out{cache_info}\n"
            f"  Cost: {self._format_cost()}{budget_info}{turn_info}")

    def _format_cost(self) -> str:
        r = self._recorder
        if r.cost_unknown:
            return f"unknown (known part ${r.cost_usd:.4f})" if r.cost_usd else "unknown (model not in price table)"
        return f"${r.cost_usd:.4f}"

    def get_run_stats(self) -> dict[str, Any]:
        """本会话（含子 agent）的汇总统计，供 /stats 和评测 runner 使用。"""
        return {
            "model": self.model,
            "turns": self.current_turns,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            **self._recorder.summary(),
        }

    #获取当前的花费（按模型价格逐次累加，子 agent 的花费实时计入同一个 recorder）
    def _get_current_cost_usd(self) -> float:
        return self._recorder.cost_usd

    #检查预算
    def _check_budget(self) -> dict:
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    #压缩会话
    async def compact(self)->None:
        compacted = await self._compact_conversation(trigger="manual")
        if not compacted:
            print_info("Nothing to compact yet.")


    #恢复会话信息
    def restore_session(self, data:dict)->None:
        if data.get("anthropicMessages"):
            # 不在这里修复消息：下一轮开头会在读入结果日志之后再修复，补进去的才是真实结果。
            self._anthropic_messages = _sanitize_for_utf8(data["anthropicMessages"])
        if data.get("openaiMessages"):
            self._openai_messages = _sanitize_for_utf8(data["openaiMessages"])
        if isinstance(data.get("foldedSessionMemories"), list):
            self._folded_session_memories = _sanitize_for_utf8(data["foldedSessionMemories"])
        # 检查点和撤销栈一起恢复，--resume 之后仍可 /undo、/rewind。
        if isinstance(data.get("checkpoints"), list):
            self._checkpoints = data["checkpoints"]
            self._turn_counter = int(data.get("turnCounter", len(self._checkpoints)))
            self._history_epoch = int(data.get("historyEpoch", 0))
        if isinstance(data.get("editHistory"), dict):
            self._edit_history.load_dict(data["editHistory"])
        # 读入之前会话的结果日志（可能是多次 --resume 串起来的），恢复后的修复能用上真实结果。
        for path in data.get("journalFiles") or []:
            if path not in self._journal_files:
                self._journal.load(Path(path))
                self._journal_files.insert(0, path)
        print_info(f"Session restored ({self._get_message_count()} messages).")



#整理 Anthropic 的历史消息，修正部分角色错误，并丢弃不合法的工具调用消息。
    def _normalize_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        """
        修复消息，保证每个 tool_use 后面紧跟的用户消息里都有它的 tool_result。
        只"补"不"删"：缺的结果按结果日志补进下一条用户消息（没有就在末尾追加一条），
        这样不会改变已有消息的位置（检查点按消息条数定位）；找不到对应 tool_use 的孤立结果被去掉。
        """
        role_normalized = []
        for msg in messages:
            copied = dict(msg)
            content = copied.get("content")
            if copied.get("role") == "user" and isinstance(content, list):
                if any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content):
                    copied["role"] = "assistant"
            role_normalized.append(copied)

        normalized: list[dict] = []
        i = 0
        while i < len(role_normalized):
            msg = role_normalized[i]
            i += 1
            calls = []
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                calls = [b for b in msg["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
            if msg.get("role") == "user" and (not normalized or not self._anthropic_tool_use_ids(normalized[-1])):
                msg = self._drop_orphan_results(msg, set())
            normalized.append(msg)
            if not calls:
                continue
            ids = {b.get("id") for b in calls}
            nxt = role_normalized[i] if i < len(role_normalized) else None
            if nxt is not None and nxt.get("role") == "user":
                nxt = self._drop_orphan_results(nxt, ids)
                content = nxt.get("content")
                blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
                have = self._anthropic_tool_result_ids({"role": "user", "content": blocks})
                results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
                others = [b for b in blocks if not (isinstance(b, dict) and b.get("type") == "tool_result")]
                missing = [self._anthropic_tool_result(b["id"], self._journal.resolve(b["id"], b.get("name", "")))
                           for b in calls if b.get("id") not in have]
                if missing:
                    # tool_result 必须排在用户消息的最前面。
                    nxt = {**nxt, "content": results + missing + others}
                normalized.append(nxt)
                i += 1
            else:
                normalized.append({"role": "user", "content": [
                    self._anthropic_tool_result(b["id"], self._journal.resolve(b["id"], b.get("name", "")))
                    for b in calls]})
        return normalized

    @staticmethod
    def _drop_orphan_results(msg: dict, valid_ids: set) -> dict:
        """去掉找不到对应 tool_use 的 tool_result；消息变空时留一段说明文字，保持消息条数不变。"""
        content = msg.get("content")
        if not isinstance(content, list):
            return msg
        kept = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result"
                                           and b.get("tool_use_id") not in valid_ids)]
        if len(kept) == len(content):
            return msg
        return {**msg, "content": kept or [{"type": "text", "text": "(orphaned tool results removed)"}]}

    @staticmethod
    def _anthropic_tool_use_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        }

    @staticmethod
    def _anthropic_tool_result_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("tool_use_id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
        }

    def _get_message_count(self) -> int:
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self) -> None:
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self._get_message_count(),
                },
                "anthropicMessages": _sanitize_for_utf8(self._anthropic_messages) if not self.use_openai else None,
                "openaiMessages": _sanitize_for_utf8(self._openai_messages) if self.use_openai else None,
                "foldedSessionMemories": _sanitize_for_utf8(self._folded_session_memories),
                "checkpoints": self._checkpoints,
                "turnCounter": self._turn_counter,
                "historyEpoch": self._history_epoch,
                "editHistory": self._edit_history.to_dict(),
                "journalFiles": self._journal_files,
            })
        except Exception:
            pass

    #自动压缩
    async def _check_and_compact(self)->None:
        if self.last_input_token_count > self.effective_window * AUTO_COMPACT_THRESHOLD:
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation(trigger="auto")

    async def _compact_conversation(self, *, trigger: str = "manual")->bool:
        messages_before = self._get_message_count()
        tokens_before = self.last_input_token_count
        started = time.monotonic()
        if self.use_openai:
            compacted = await self._compact_openai(trigger=trigger)
        else:
            compacted = await self._compact_anthropic(trigger=trigger)
        self._recorder.emit(
            "compaction", trigger=trigger, compacted=compacted, messages_before=messages_before,
            messages_after=self._get_message_count(), context_tokens_before=tokens_before,
            duration_ms=round((time.monotonic() - started) * 1000), **self._trace_fields(),
        )
        if compacted:
            self._history_epoch += 1
            print_info("Conversation compacted.")
        return compacted

    async def _compact_anthropic(self, *, trigger: str)->bool:
        if len (self._anthropic_messages)<4:
            return False

        transcript = build_anthropic_transcript(_sanitize_for_utf8(self._anthropic_messages))
        if not transcript.strip():
            return False
        memory = await self._generate_folded_session_memory(transcript)
        self._record_folded_session_memory(trigger, memory)
        self._record_fold_event()
        self._anthropic_messages = [{"role": "user", "content": format_folded_memory(memory)}]
        self.last_input_token_count = 0
        self._refresh_runtime_system_prompt()
        return True

    async def _compact_openai(self, *, trigger: str)->bool:
        if len (self._openai_messages)<4:
            return False
        system_msg = self._openai_messages[0]
        transcript = build_openai_transcript(_sanitize_for_utf8(self._openai_messages))
        if not transcript.strip():
            return False
        memory = await self._generate_folded_session_memory(transcript)
        self._record_folded_session_memory(trigger, memory)
        self._record_fold_event()
        self._openai_messages=[
            system_msg,
            {"role": "user", "content": format_folded_memory(memory)},
        ]
        self.last_input_token_count=0
        self._refresh_runtime_system_prompt()
        return True

    async def _generate_folded_session_memory(self, transcript: str) -> dict[str, Any]:
        side_query = self._build_side_query(max_tokens=6000, purpose="fold")
        if side_query is None:
            return fallback_folded_memory(transcript)
        try:
            raw = await side_query(FOLD_SESSION_MEMORY_SYSTEM, build_folding_user_prompt(transcript))
            return parse_folded_memory(raw)
        except Exception:
            return fallback_folded_memory(transcript)

    def _record_folded_session_memory(self, trigger: str, memory: dict[str, Any]) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trigger": trigger,
            "session_id": self.session_id,
            **memory,
        }
        self._folded_session_memories.append(record)
        try:
            save_folded_session_memory(self.session_id, _sanitize_for_utf8(record))
        except Exception:
            pass

    #多层级压缩流水线
    def _run_compression_pipeline(self)->None:
        if self.use_openai:
            self._budget_tool_results_openai()
            self._snip_stale_results_openai()
            self._microcompact_openai()
        else:
            self._budget_tool_results_anthropic()
            self._snip_stale_results_anthropic()
            self._microcompact_anthropic()

    #第一层级压缩，预算压缩
    def _budget_tool_results_anthropic(self)->None:
        #计算利用率：utilization = 已用Token / 有效窗口大小。
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        #如果利用率低于 50%，说明空间还很充裕，直接返回，不做任何处理。
        if utilization < 0.5:
            return
        #动态预算（Budget）：危急状态（>70%）：如果利用率很高，允许单个工具结果保留 15,000 个字符。
        # 警戒状态（50%-70%）：如果利用率中等，只允许保留 30000 个字符。
        budget = 15000 if utilization > 0.7 else 30000

        for msg in self._anthropic_messages:

            #只处理 role 为 "user" 的消息。在工具调用流程中，工具的执行结果通常是以“用户”的身份反馈给模型的。

            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and len(block["content"]) > budget:
                    #计算保留长度 (keep)：keep = (budget - 80) // 2 这里预留了约 80 个字符的空间给中间的提示语，剩下的长度平分给开头和结尾。
                    keep = (budget - 80) // 2
                    #重组新内容 = 开头部分 + 提示语 + 结尾部分
                    block["content"] = block["content"][:keep] + f"\n\n[... budgeted: {len(block['content']) - keep * 2} chars truncated ...]\n\n" + block["content"][-keep:]

    def _budget_tool_results_openai(self)->None:
        #计算利用率：utilization = 已用Token / 有效窗口大小。
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        #如果利用率低于 50%，说明空间还很充裕，直接返回，不做任何处理。
        if utilization < 0.5:
            return
        #动态预算（Budget）：危急状态（>70%）：如果利用率很高，允许单个工具结果保留 15,000 个字符。
        # 警戒状态（50%-70%）：如果利用率中等，只允许保留 30000 个字符。
        budget = 15000 if utilization > 0.7 else 30000

        for msg in self._openai_messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > budget:
                keep = (budget - 80) // 2
                msg["content"] = msg["content"][:keep] + f"\n\n[... budgeted: {len(msg['content']) - keep * 2} chars truncated ...]\n\n" + msg["content"][-keep:]


    #第二级策略：修剪过期的工具执行结果
    def _snip_stale_results_anthropic(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        results = []
        for mindex,  msg in enumerate(self._anthropic_messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue

            for bindex, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] != SNIP_PLACEHOLDER:
                    tool_use_id = block.get("tool_use_id")
                    # 对每个 tool_result，通过 tool_use_id 反查它来自哪个工具
                    tool_info = self._find_tool_use_by_id(tool_use_id)
                    if tool_info and tool_info["name"] in SNIPPABLE_TOOLS:
                        results.append({"mindex": mindex, "bindex": bindex, "name": tool_info["name"], "file_path": tool_info.get("input", {}).get("file_path")})

        if len(results) < KEEP_RECENT_RESULTS + SNIP_INTERVAL:
            return

        to_snip =  set()
        seen_files: dict[str, list[int]] = {}

        for i, r in enumerate(results):
            if r["name"] == "read_file" and r.get("file_path"):
                seen_files.setdefault(r["file_path"], []).append(i)
        #如果一个文件被读取了多次，只保留最后一次读取的结果，把前面几次读取的内容全部标记为“修剪”（Snip）。
        for indices in seen_files.values():
            if len (indices) >1 :
                for j in indices[:-1]:
                    to_snip.add (j)

        snip_before = len(results) - KEEP_RECENT_RESULTS
        for i in range (snip_before):
            to_snip.add(i)

        for idx in to_snip:
            r = results[idx]
            self._anthropic_messages[r["mindex"]]["content"][r["bindex"]]["content"] = SNIP_PLACEHOLDER

    def _snip_stale_results_openai(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] != SNIP_PLACEHOLDER:
                tool_msgs.append(i)
        if len(tool_msgs) < KEEP_RECENT_RESULTS + SNIP_INTERVAL:
            return
        snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(snip_count):
            self._openai_messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER

    #微压缩

    #基于“时间”的上下文瘦身策略，
    #如果已经很久没说话了，说明之前的工具执行结果你已经看完了，那就把它们清理掉，腾出空间

    def _microcompact_anthropic(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return

        all_results = []
        for mindex, msg in enumerate(self._anthropic_messages):
            if msg.get("role")!="user" or not isinstance(msg.get("content"), list):
                continue
            for bindex, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                    all_results.append((mindex, bindex))

        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            self._anthropic_messages[mi]["content"][bi]["content"] = "[Old result cleared]"

    def _microcompact_openai(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                tool_msgs.append(i)
        clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            self._openai_messages[tool_msgs[i]]["content"] = "[Old result cleared]"

    def _find_tool_use_by_id(self, tool_use_id: int) -> dict | None:
        for msg in self._anthropic_messages:
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                continue

            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return {"name": block["name"], "input": block.get("input", {})}

    #大结果持久化
    #如果工具返回的结果太大（超过 30KB），不要硬塞进上下文里，而是把它存成一个临时文件。
    # 然后在对话里只留一个‘文件路径’和‘内容预览’。如果模型后面还需要看完整内容，它可以再次调用工具去读取这个文件

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        THRESHOLD = 30 * 1024  # 30 KB
        #转换成字节
        if (len (result.encode())) <= THRESHOLD:
            return result

        d = Path.home() / ".moltevo-cli" / "tool-results"
        d.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time() * 1000)}-{tool_name}.txt"
        filepath = d / filename
        filepath.write_text(result, encoding="utf-8")

        lines = result.split("\n")
        preview = "\n".join(lines[:200])
        size_kb = len(result.encode()) / 1024

        return (
            f"[Result too large ({size_kb:.1f} KB, {len(lines)} lines). "
            f"Full output saved to {filepath}. "
            f"You can use read_file to see the full result.]\n\n"
            f"Preview (first 200 lines):\n{preview}"
        )

    #执行工具入口

    async def _execute_tool_call(self, name: str, inp: dict) -> ToolResult:
        if name == "compact_context":
            return await self._execute_compact_context_tool(inp)
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return ToolResult.ok(await self._execute_plan_mode_tool(name))
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
            # Route MCP tool calls to the MCP manager
        if self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        result = await execute_tool(name, inp, self._read_file_state, self._edit_history)
        if name in {"skill_create", "skill_evolve"} and not result.is_error:
            self._refresh_runtime_system_prompt()
        if name == "tool_search" and not result.is_error and not result.content.startswith("No matching"):
            # 系统提示词里列着"可按需加载的工具"，激活后刷新一次，避免列表过时。
            self._refresh_runtime_system_prompt()
        return result

    async def _execute_compact_context_tool(self, inp: dict) -> ToolResult:
        reason = str(inp.get("reason") or "").strip()
        compacted = await self._compact_conversation(trigger="tool")
        if not compacted:
            return ToolResult.error("No context compaction was performed because there is not enough conversation history yet.")
        self._context_cleared = True
        suffix = f"\nReason: {reason}" if reason else ""
        return ToolResult.ok(
            "Context compacted into structured session memory. "
            "Continue from the folded memory now present in the conversation context."
            f"{suffix}"
        )


    async def _execute_skill_tool(self, inp: dict) -> ToolResult:
        from .skills import execute_skill
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))

        if not result:
            return ToolResult.error(f"Unknown skill: {inp.get('skill_name', '')}")

        #fork 表示这个 skill 不直接把 prompt 塞回当前对话，而是要启动一个子 Agent 单独完成任务。
        if result["context"] == "fork":
            # result["allowed_tools"] - 直接访问
            tools = (
                [t for t in self.tools if t["name"] in  result["allowed_tools"] ]
                #result.get("allowed_tools") - 安全访问
                # 存在key：返回对应的值（可能是 None、[]、["tool1"] 等）
                # 不存在key：返回 None（不会抛异常）
                if result.get("allowed_tools")
                else  [t for t in self.tools if t["name"] != "agent"]
            )

            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = self._spawn_sub_agent(result["prompt"], tools, label=f"skill:{inp.get('skill_name', '')}")
            try:
                sub_result = await sub_agent.run_once(inp.get("args") or "Execute this skill task.")
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return ToolResult.ok(sub_result["text"] or "(Skill produced no output)")
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return ToolResult.error(f"Skill fork error: {e}")

        return ToolResult.ok(f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}')

    async def _execute_plan_mode_tool(self, name):
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path =  self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Entered plan mode (read-only). Plan file: " + self._plan_file_path)
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."
        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = Path(self._plan_file_path).read_text(encoding="utf-8")
            # 交互式审批流程（如果有审批函数）
            if self._plan_approval_fn:
                result = await self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice =="keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                #离开计划模式
                saved_plan_path = self._plan_file_path
                self._leave_plan_mode(target_mode)

                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(f"Plan approved. Context cleared, executing in {target_mode} mode.")
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )
                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )
            # 没有审批函数时的回退（例如子代理）
            self._leave_plan_mode(self._pre_plan_mode or "default")

            print_info("Exited plan mode. Restored to " + self.permission_mode + " mode.")
            return f"Exited plan mode. Permission mode restored to: {self.permission_mode}\n\n## Your Plan:\n{plan_content}"

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """清空历史信息，但是保留系统prompt."""
        self._anthropic_messages = []
        self._openai_messages = []
        self._history_epoch += 1
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.last_input_token_count = 0
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    async def _cancel_inflight_tool_tasks(self) -> None:
        """取消没被用到的提前执行任务，并等它们清理完（例如终止子进程组）。"""
        pending = [t for t in self._inflight_tool_tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._inflight_tool_tasks.clear()

    def _spawn_sub_agent(self, system_prompt: str, tools: list[ToolDef], *, label: str = "sub") -> "Agent":
        """
        派生子 agent 的唯一入口。
        - 连接配置（协议、地址、密钥、模型）原样继承，否则子 agent 会连到 SDK 默认地址。
        - 权限模式继承父 agent：派出去的 agent 不能比派它的 agent 权限更大。
        - 共用确认函数和已确认集合，主 agent 批过的操作不重复询问。
        - 共用 recorder：事件串到派它出去的那次工具调用上，费用实时计入同一个 --max-cost。
        """
        sub_agent = Agent(
            model=self.model,
            api_base=self._api_base,
            anthropic_base_url=self._anthropic_base_url,
            api_key=self._api_key,
            custom_system_prompt=system_prompt,
            custom_tools=tools,
            is_sub_agent=True,
            permission_mode=self.permission_mode,
            confirm_fn=self.confirm_fn,
            max_cost_usd=self.max_cost_usd,
            recorder=self._recorder,
            edit_history=self._edit_history,
            trace_parent={"trace_id": self._trace_id, "parent_span_id": _CURRENT_SPAN.get(),
                          "label": label},
        )
        sub_agent._confirmed_paths = self._confirmed_paths
        sub_agent._parent_accept_all = self._accept_all_edits
        return sub_agent

    async def _execute_agent_tool(self, inp:dict) -> ToolResult:
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")
        print_sub_agent_start(agent_type, description)

        config = get_sub_agent_config(agent_type)

        sub_agent = self._spawn_sub_agent(config["system_prompt"], config["tools"], label=f"sub:{agent_type}")
        try:
            result = await sub_agent.run_once(prompt)
            self.total_input_tokens += result["tokens"]["input"]
            self.total_output_tokens += result["tokens"]["output"]
            print_sub_agent_end(agent_type, description)
            return ToolResult.ok(result["text"] or "(Sub-agent produced no output)")
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return ToolResult.error(f"Sub-agent error: {e}")

#--------------Anthropic 后端---------------
    async def  _chat_anthropic(self, user_message: str) -> None:
        await self._run_turn(_AnthropicAdapter(self), user_message)

    async def _chat_openai(self, user_message: str) -> None:
        await self._run_turn(_OpenAIAdapter(self), user_message)

    #--------------统一的主循环：协议相关的部分交给适配器---------------
    async def _run_turn(self, p: "_ProviderAdapter", user_message: str) -> None:
        # 上一轮被打断时可能留下不完整的消息（没有结果的工具调用），先按结果日志补齐。
        p.normalize()
        # 检查点在修复之后、追加本轮用户消息之前记录，记下的消息条数才对得上。
        if not self.is_sub_agent:
            self._record_checkpoint(self._turn_preview)
        user_message = _safe_utf8_text(user_message)
        # 先把本轮用户输入放入消息历史，后续每轮模型调用都会带上这段上下文。
        p.append_user(user_message)

        # 异步内存预取：主 agent 才需要查 memory，sub agent 不额外注入记忆。
        # 这里只启动后台任务，不阻塞当前模型调用流程。
        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query(purpose="memory_select")
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )
        empty_retries = 0
        while True:
            # 外部请求中止时，结束整个 agent loop。
            if self._aborted:
                break

            # 每轮调用模型前尝试压缩上下文，避免消息历史过长。
            self._run_compression_pipeline()

            # 记忆预取完成后，把取回来的 memory 追加到最后一条用户消息里；consumed 保证只注入一次。
            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        p.inject_into_last_user(_safe_utf8_text(format_memories_for_injection(memories)))
                        for m in memories:
                            # 记录本 session 已经注入过的 memory，后续检索时可避免重复。
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += m.size
                except Exception:
                    # memory 注入失败不应该中断主对话流程。
                    pass

            if not self.is_sub_agent:
                start_spinner()

            # 上一次迭代因 context_break 等提前退出时，可能留下没被等待的提前执行任务。
            await self._cancel_inflight_tool_tasks()
            # 流式阶段提前启动的工具任务。key 是工具调用 id。
            early_executions: dict[str, asyncio.Task] = {}
            early_spans: dict[str, str] = {}

            def _on_tool_block(block: dict):
                # 流式响应中一旦完整收到一个工具调用，如果工具是并发安全且权限允许，
                # 就提前开始执行，减少等待完整模型响应后的空档时间。
                if block["name"] in CONCURRENCY_SAFE_TOOLS:
                    perm = check_permission(block["name"], block["input"], self.permission_mode, self._plan_file_path)
                    if perm["action"] == "allow":
                        span_id = new_id()
                        self._recorder.record_permission(tool=block["name"], decision="allow", early=True,
                                                         **self._trace_fields(span_id))
                        task = asyncio.create_task(
                            self._timed_tool_call(block["name"], block["input"], span_id, block["id"]))
                        early_executions[block["id"]] = task
                        early_spans[block["id"]] = span_id
                        self._inflight_tool_tasks.add(task)
                        task.add_done_callback(self._inflight_tool_tasks.discard)

            started = time.monotonic()
            self._call_retries = 0
            turn = await p.call_model(_on_tool_block)
            if not self.is_sub_agent:
                stop_spinner()

            # 记录本次模型调用的耗时、token、缓存命中和费用，用于成本展示与预算控制。
            self._record_model_call(turn.usage, started=started, stop_reason=turn.stop_reason)

            if turn.empty:
                # 空回复不写进历史（空的助手消息会让下一次请求被拒），提示模型继续；连续多次才放弃。
                empty_retries += 1
                self._recorder.emit("empty_response", attempt=empty_retries, **self._trace_fields())
                if empty_retries > MAX_EMPTY_RETRIES:
                    print_warning(f"The model returned {empty_retries} empty responses in a row; stopping this turn.")
                    break
                p.append_user("Please continue; do not produce an empty response.")
                continue
            empty_retries = 0

            # 没有工具调用，说明模型已经给出最终回复；配置了验证命令且本轮改过文件时先验证。
            if not turn.tool_calls:
                feedback = await self._maybe_verify()
                if feedback:
                    p.append_user(feedback)
                    continue
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens, self._format_cost())
                break

            # 有工具调用时，进入工具执行。这里同时检查 turn/budget 限制。
            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                # 每个工具调用都要有对应的结果，否则下一次请求会因为消息不完整被拒绝。
                p.append_tool_results([(c, ToolResult.error(f"Tool execution skipped: {budget['reason']}"))
                                       for c in turn.tool_calls])
                break

            if not self.is_sub_agent:
                # 执行工具前先存档：进程若在执行中崩溃，--resume 后历史里有这些工具调用，
                # 再按结果日志补上真实结果，模型就不会把已经执行过的操作再做一遍。
                self._auto_save()
            results, cleared_result = await self._execute_tool_calls(turn.tool_calls, early_executions, early_spans)
            if cleared_result is not None:
                # 工具执行过程中清理了上下文（折叠、清空后执行计划），把结果作为新的用户消息写入。
                p.append_user(cleared_result.content)
            else:
                # 中途被中断时，没执行到的调用也要有结果（按结果日志区分"未开始"和"执行中"），保证消息完整。
                answered = {c.id for c, _ in results}
                results += [(c, self._journal.resolve(c.id, c.name)) for c in turn.tool_calls if c.id not in answered]
                p.append_tool_results(results)
                p.after_tool_round()

            self._context_cleared = False
            # 工具结果可能很长，每轮工具执行后检查是否需要压缩上下文。
            await self._check_and_compact()

    async def _execute_tool_calls(self, calls: list["ToolCall"], early_executions: dict[str, asyncio.Task],
                                  early_spans: dict[str, str]) -> tuple[list[tuple["ToolCall", ToolResult]], ToolResult | None]:
        """
        执行一轮工具调用。返回 (按原顺序的 (调用, 结果) 列表, 清空上下文的那次结果或 None)。
        - 流式阶段已提前执行的，等它完成；
        - 相邻的只读工具合成一批并发执行（只读工具不会弹确认）；
        - 其余按顺序逐个"判定权限 → 执行"，确认弹窗一个接一个出现。
        """
        pairs: list[tuple[ToolCall, ToolResult]] = []
        i = 0
        while i < len(calls) and not self._aborted:
            call = calls[i]
            if call.id in early_executions:
                print_tool_call(call.name, call.input)
                result = self._finish_tool_call(call.name, call.input, await early_executions[call.id],
                                                span_id=early_spans[call.id], early=True, call_id=call.id)
                pairs.append((call, result))
                i += 1
                continue

            if call.name in CONCURRENCY_SAFE_TOOLS:
                group: list[ToolCall] = []
                while (i < len(calls) and calls[i].name in CONCURRENCY_SAFE_TOOLS
                       and calls[i].id not in early_executions):
                    group.append(calls[i])
                    i += 1
                planned = []
                for g in group:
                    print_tool_call(g.name, g.input)
                    span_id = new_id()
                    planned.append((g, span_id, await self._authorize_tool(g.name, g.input, span_id)))
                runnable = [(g, span_id) for g, span_id, denial in planned if denial is None]
                # _timed_tool_call 不抛异常（取消除外），一个工具出错不会拖垮同批的其他工具。
                outcomes = await asyncio.gather(*[self._timed_tool_call(g.name, g.input, span_id, g.id)
                                                  for g, span_id in runnable])
                finished = {g.id: self._finish_tool_call(g.name, g.input, r, span_id=span_id, call_id=g.id)
                            for (g, span_id), r in zip(runnable, outcomes)}
                for g, _, denial in planned:
                    if denial is not None:
                        self._journal.done(g.id, g.name, denial)
                pairs += [(g, denial if denial is not None else finished[g.id]) for g, _, denial in planned]
                continue

            print_tool_call(call.name, call.input)
            span_id = new_id()
            i += 1
            denial = await self._authorize_tool(call.name, call.input, span_id)
            if denial is not None:
                self._journal.done(call.id, call.name, denial)
                pairs.append((call, denial))
                continue
            result = self._finish_tool_call(call.name, call.input,
                                            await self._timed_tool_call(call.name, call.input, span_id, call.id),
                                            span_id=span_id, call_id=call.id)
            if self._context_cleared:
                # 上下文已被清空（之前的结果也随之作废），停止处理本轮剩余工具。
                self._context_cleared = False
                return pairs, result
            pairs.append((call, result))
        return pairs, None

    @staticmethod
    def _block_to_dict(block) -> dict:
        if block.type == "text":
            return {"type": "text", "text": _safe_utf8_text(block.text)}
        if block.type == "tool_use":
            raw_input = dict(block.input) if hasattr(block.input, 'items') else block.input
            return {"type": "tool_use", "id": _safe_utf8_text(block.id), "name": _safe_utf8_text(block.name), "input": _sanitize_for_utf8(raw_input)}
        # Fallback
        return {"type": _safe_utf8_text(block.type)}

    async def _call_anthropic_stream(self, on_tool_block_complete=None):

        async def _do():
            max_output =  _get_max_output_tokens(self.model)

            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if self._thinking_mode != "disabled" else 16384,
                **self._build_anthropic_request(),
            }
            #如果开启了思考模式，就给 Anthropic 请求加上 thinking 参数。
            if self._thinking_mode  in ("adaptive", "enabled"):
                create_params["thinking"]={"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True
            self._call_first_token_at = None

            tool_blocks_by_index: dict[int, dict] = {}

            async with self._anthropic_client.messages.stream(**create_params)as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue
                    if self._call_first_token_at is None and event.type in ("content_block_start", "content_block_delta"):
                        self._call_first_token_at = time.monotonic()
                    # 当事件是工具调用开始：
                    if event.type == "content_block_start":
                        cb = getattr(event, 'content_block', None)
                        #如果 block 类型是 tool_use，就记录这个工具调用：
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            #因为工具参数 JSON 是流式分片返回的，所以先准备一个空字符串 input_json。
                            tool_blocks_by_index[event.index]= {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }
                    #当事件是内容增量，分三种情况。
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # 第一种，普通文本：模型输出正文时，
                        # 调用 _emit_text()。如果是普通交互，就打印；
                        # 如果是 run_once()，就写入 _output_buffer。
                        if hasattr(delta, "text"):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        #第二种，thinking 内容：
                        #如果模型返回思考内容，也输出出来，并在开头加：[thinking]
                        elif hasattr(delta, 'thinking'):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n  [thinking] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        #第三种，工具参数 JSON 片段：工具调用的参数不是一次性返回，
                        # 而是一段一段返回，所以这里不断拼接到 input_json。
                        elif hasattr(delta, 'partial_json'):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += _safe_utf8_text(delta.partial_json)
                    #当一个 content block 结束：
                    #如果结束的是之前记录的工具调用，就把拼好的 JSON 解析出来：
                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json
                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            #然后调用回调：
                            #这个回调的作用通常是：工具调用一完整，
                            # 就可以提前开始执行工具，不必等整条 assistant 消息全部结束。
                            on_tool_block_complete({
                                "type": "tool_use", "id": _safe_utf8_text(tb["id"]),
                                "name": _safe_utf8_text(tb["name"]), "input": _sanitize_for_utf8(parsed),
                            })
                final_message = await stream.get_final_message()

            #过滤思考的message（因为 thinking 内容一般不应该进入历史消息，否则后续上下文会变大，也可能不符合 API 消息格式要求。）
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message
#调用 _do()，如果遇到可重试错误，就由 _with_retry() 负责重试。
        return await _with_retry(_do, on_retry=self._note_retry)

    async def _call_openai_stream(self) -> dict:
        async def _do():
            stream = await self._openai_client.chat.completions.create(
                model=self.model,
                tools=_sanitize_for_utf8(_to_openai_tools(get_active_tool_definitions(self.tools))),
                messages=self._build_openai_request_messages(),
                stream=True,
                stream_options={"include_usage": True},
            )

            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None
            self._call_first_token_at = None

            async for chunk in stream:
                if chunk.usage:
                    # 缓存命中：OpenAI 在 prompt_tokens_details.cached_tokens，DeepSeek 在 prompt_cache_hit_tokens。
                    details = getattr(chunk.usage, "prompt_tokens_details", None)
                    cached = (getattr(details, "cached_tokens", 0) if details else 0) \
                        or getattr(chunk.usage, "prompt_cache_hit_tokens", 0) or 0
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "cached_tokens": cached,
                    }

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if self._call_first_token_at is None and delta and (delta.content or delta.tool_calls):
                    self._call_first_token_at = time.monotonic()

                if delta and delta.content:
                    if first_text:
                        stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += _safe_utf8_text(delta.content)

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += _safe_utf8_text(tc.function.arguments)
                        else:
                            tool_calls[tc.index] = {
                                "id": _safe_utf8_text(tc.id or ""),
                                "name": _safe_utf8_text((tc.function.name if tc.function else "") or ""),
                                "arguments": _safe_utf8_text((tc.function.arguments if tc.function else "") or ""),
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": assembled,
                    },
                    "finish_reason": finish_reason or "stop",
                }],
                # 有些 OpenAI 兼容服务不支持 stream_options，拿不到用量时如实返回 None。
                "usage": usage,
            }

        return await _with_retry(_do, on_retry=self._note_retry)

    # ── 请求构造：稳定前缀 + 缓存断点 + 每次请求临时附加的运行时提示 ───────────
    def _prompt_caching_enabled(self) -> bool:
        # 只对 claude 模型默认加 cache_control：第三方 Anthropic 兼容接口不一定认这个字段。
        # DeepSeek / OpenAI 有自动前缀缓存，不需要断点，只要前缀稳定就能命中。
        flag = os.environ.get("MOLTEVO_PROMPT_CACHE", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            return True
        if flag in {"0", "false", "no", "off"}:
            return False
        return self.model.lower().startswith("claude")

    def _build_anthropic_request(self) -> dict[str, Any]:
        """系统提示词、工具定义、最后一条消息各打一个缓存断点；运行时提示附在断点之后，不进入缓存也不写进历史。"""
        cache = self._prompt_caching_enabled()
        marker = {"type": "ephemeral"}
        system: Any = _safe_utf8_text(self._system_prompt)
        tools = _sanitize_for_utf8(get_active_tool_definitions(self.tools))
        messages = _sanitize_for_utf8(self._anthropic_messages)   # 返回的是新对象，修改不影响历史
        if cache:
            system = [{"type": "text", "text": system, "cache_control": marker}]
            if tools:
                tools = [*tools[:-1], {**tools[-1], "cache_control": marker}]
        guidance = self._build_fold_guidance_section()
        if messages and (cache or guidance):
            last = dict(messages[-1])
            content = last.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(b) for b in content or []]
            if cache and blocks:
                blocks[-1] = {**blocks[-1], "cache_control": marker}
            if guidance:
                blocks.append({"type": "text", "text": guidance})
            last["content"] = blocks
            messages = [*messages[:-1], last]
        return {"system": system, "tools": tools, "messages": messages}

    def _build_openai_request_messages(self) -> list:
        """运行时提示附在最后一条消息末尾（只改请求副本），保持前面的前缀不变，便于自动前缀缓存命中。"""
        messages = _sanitize_for_utf8(self._openai_messages)
        guidance = self._build_fold_guidance_section()
        if guidance and messages and isinstance(messages[-1], dict) and isinstance(messages[-1].get("content"), str) \
                and messages[-1].get("role") in ("user", "tool"):
            messages = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + "\n\n" + guidance}]
        return messages

    def _note_retry(self) -> None:
        self._call_retries += 1

    @staticmethod
    def _openai_usage(u: Any) -> dict[str, int] | None:
        """OpenAI 兼容用量对象 → 统一格式；没有用量返回 None。"""
        if u is None:
            return None
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) if details else 0) or getattr(u, "prompt_cache_hit_tokens", 0) or 0
        prompt = getattr(u, "prompt_tokens", 0) or 0
        return {"input": max(prompt - cached, 0), "output": getattr(u, "completion_tokens", 0) or 0,
                "cache_read": cached, "cache_write": 0}

    @staticmethod
    def _anthropic_usage(u: Any) -> dict[str, int]:
        # Anthropic 的 input_tokens 不含缓存读写部分，三者相加才是这次请求的完整提示词大小。
        return {
            "input": getattr(u, "input_tokens", 0) or 0,
            "output": getattr(u, "output_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }

    def _record_model_call(self, usage: dict[str, int] | None, *, started: float, stop_reason: str | None,
                           kind: str = "main", purpose: str | None = None) -> None:
        """
        更新 token 统计、按本次调用的模型价格计费，并写事件。
        usage 为 None 表示服务没有返回用量：这次调用的费用记为未知，而不是 0。
        kind="side" 是侧查询（记忆筛选、Skill 抽取、折叠等），只计费，不影响上下文大小统计。
        """
        missing = usage is None
        usage = usage or {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        prompt_total = usage["input"] + usage["cache_read"] + usage["cache_write"]
        if kind == "main":
            # 主对话的 token 总数不含侧查询（侧查询单独统计在 recorder 里，费用照常计入）。
            self.total_input_tokens += prompt_total
            self.total_output_tokens += usage["output"]
            self.last_api_call_time = time.time()
            if not missing:
                # 拿不到用量时保留上一次的值，否则压缩会以为上下文是空的。
                self.last_input_token_count = prompt_total
        cost = None if missing else call_cost_usd(
            self.model, input_tokens=usage["input"], output_tokens=usage["output"],
            cache_read_tokens=usage["cache_read"], cache_write_tokens=usage["cache_write"])
        ttft = (self._call_first_token_at - started) * 1000 if kind == "main" and self._call_first_token_at else None
        extra = {"usage_missing": True} if missing else {}
        if purpose:
            extra["purpose"] = purpose
        self._recorder.record_model_call(
            cost_usd=cost, latency_ms=(time.monotonic() - started) * 1000, ttft_ms=ttft,
            retries=self._call_retries if kind == "main" else 0,
            cache_read=usage["cache_read"], cache_write=usage["cache_write"],
            event_type="model_call" if kind == "main" else "side_query",
            model=self.model, input_tokens=usage["input"], output_tokens=usage["output"],
            stop_reason=stop_reason, **extra, **self._trace_fields(),
        )

    @staticmethod
    def _normalize_openai_messages(messages: list, resolve: Callable[[str, str], ToolResult] | None = None) -> list:
        """给没有结果的 tool_call 补结果（有结果日志时按日志补）：下一次请求不会因消息不完整被拒绝。"""
        out: list = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            out.append(msg)
            i += 1
            calls = msg.get("tool_calls") if isinstance(msg, dict) and msg.get("role") == "assistant" else None
            if not calls:
                continue
            answered = set()
            while i < len(messages) and isinstance(messages[i], dict) and messages[i].get("role") == "tool":
                answered.add(messages[i].get("tool_call_id"))
                out.append(messages[i])
                i += 1
            for call in calls:
                if call.get("id") not in answered:
                    name = (call.get("function") or {}).get("name", "")
                    result = resolve(call.get("id"), name) if resolve else \
                        ToolResult.error("Tool execution was interrupted before it produced a result.")
                    out.append(Agent._openai_tool_message(call.get("id"), result))
        return out

    async def _confirm_dangerous(self, command: str, *, preview: str | None = None,
                                 offer_all: bool = False) -> str:
        """询问用户。返回 "yes" / "no" / "all"（all 仅在 offer_all 时有效：本会话后续编辑自动同意）。"""
        print_confirmation(command, preview=preview)
        if self.confirm_fn:
            answer = await self.confirm_fn(command, preview=preview, offer_all=offer_all)
        else:
            # Fallback: blocking input
            try:
                answer = input("  Allow? (y/n/a): " if offer_all else "  Allow? (y/n): ")
            except EOFError:
                answer = "n"
        if answer is True or answer is False:
            return "yes" if answer else "no"
        answer = str(answer).strip().lower()
        if offer_all and answer.startswith("a"):
            return "all"
        return "yes" if answer.startswith("y") else "no"

    # ── 检查点与 /rewind ──────────────────────────────────
    def _record_checkpoint(self, user_message: str) -> None:
        self._turn_counter += 1
        self._checkpoints.append({
            "turn": self._turn_counter,
            "ts": time.time(),
            "preview": " ".join(user_message.split())[:60],
            "anthropic_len": len(self._anthropic_messages),
            "openai_len": len(self._openai_messages),
            "edits": len(self._edit_history),
            "shell_runs": self._edit_history.shell_runs,
            "epoch": self._history_epoch,
        })

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """每个检查点附上：之后改过几处文件、是否跑过 shell、对话还能不能回退。"""
        out = []
        for cp in self._checkpoints:
            out.append({
                **cp,
                "file_changes_since": max(len(self._edit_history) - cp["edits"], 0),
                "shell_since": self._edit_history.shell_runs > cp["shell_runs"],
                "conversation_rewindable": cp["epoch"] == self._history_epoch,
            })
        return out

    def rewind(self, turn: int, mode: str = "both", *, force: bool = False) -> tuple[bool, str]:
        """
        回到第 turn 轮开始之前。mode: both（对话和代码）/ conversation（只回退对话）/ code（只回退代码）。
        先回退代码：有冲突时整个操作取消，对话不动。
        """
        if mode not in ("both", "conversation", "code"):
            return False, f"Unknown rewind mode: {mode}"
        cp = next((c for c in self._checkpoints if c["turn"] == turn), None)
        if cp is None:
            return False, f"No checkpoint for turn {turn}."
        if mode != "code" and cp["epoch"] != self._history_epoch:
            return False, (f"The conversation was compacted or cleared after turn {turn}, so it can't be "
                           f"rewound to that point. You can still rewind code: /rewind {turn} code")
        notes: list[str] = []
        if self._edit_history.shell_runs > cp["shell_runs"]:
            notes.append("Shell commands ran after this point; their effects on disk were NOT reverted.")
        restored: list[Path] = []
        if mode in ("both", "code"):
            ok, message, restored = self._edit_history.undo_to(cp["edits"], force=force)
            if not ok:
                return False, message
            notes.insert(0, message)
            for path in restored:
                self._read_file_state.pop(str(path), None)
        if mode in ("both", "conversation"):
            self._anthropic_messages = self._anthropic_messages[:cp["anthropic_len"]]
            self._openai_messages = self._openai_messages[:cp["openai_len"]]
            self._checkpoints = [c for c in self._checkpoints if c["turn"] < turn]
            self._pending_skill_extraction_window = None
            notes.insert(0, f"Conversation rewound to before turn {turn}.")
            if mode == "conversation" and len(self._edit_history) > cp["edits"]:
                self._pending_notices.append(
                    "[System note: the user rewound the conversation, but files changed in the removed turns were "
                    "kept on disk. Re-read files before editing them.]")
        if mode == "code" and restored:
            self._pending_notices.append(
                f"[System note: the user reverted files to their state before turn {turn}: "
                + ", ".join(str(p) for p in restored) + ". Re-read them before editing.]")
        self._recorder.emit("rewind", turn=turn, mode=mode, forced=force,
                            files_restored=[str(p) for p in restored], **self._trace_fields())
        self._auto_save()
        return True, " ".join(notes)

    # ── /undo ────────────────────────────────────────────
    def undo_last_edit(self, *, force: bool = False) -> tuple[bool, str]:
        ok, message, path = self._edit_history.undo(force=force)
        if ok and path is not None:
            # 让模型重新读文件，并明确告诉它发生了什么，否则它会基于自己改过的内容继续编辑。
            self._read_file_state.pop(str(path), None)
            self._pending_notices.append(
                f"[System note: the user ran /undo and reverted your last change to {path}. "
                f"Re-read the file before editing it again.]")
        self._recorder.emit("undo", ok=ok, forced=force, path=str(path) if path else None)
        return ok, message

    def list_edits(self) -> list[str]:
        return [f"#{r.seq} {r.tool} {r.path}" + ("" if r.existed else " (new file)")
                for r in self._edit_history.records()]

    # ── 完成后验证 ────────────────────────────────────────
    async def _maybe_verify(self) -> str | None:
        """
        模型准备结束本轮时调用。本轮改过文件且配置了验证命令才运行。
        返回 None 表示可以结束；返回字符串表示验证失败，应作为用户消息交还模型继续修。
        """
        if not self.verify_command or self.is_sub_agent:
            return None
        if self._edit_history.commits == self._turn_commit_mark:
            if self._verify_failed:
                print_warning("Verification is still failing and no further file changes were made.")
                self._verify_failed = False
            return None
        self._turn_commit_mark = self._edit_history.commits
        print_info(f"Running verification: {self.verify_command}")
        started = time.monotonic()
        result = await _run_shell({"command": self.verify_command, "timeout": VERIFY_TIMEOUT_MS})
        self._recorder.emit(
            "verify", command=summarize_args(self.verify_command), passed=not result.is_error,
            attempt=self._verify_attempts + 1, exit_code=result.meta.get("exit_code"),
            timed_out=result.meta.get("timed_out", False),
            duration_ms=round((time.monotonic() - started) * 1000), **self._trace_fields(),
        )
        if not result.is_error:
            print_info("Verification passed.")
            self._verify_failed = False
            return None
        self._verify_failed = True
        self._verify_attempts += 1
        if self._verify_attempts > MAX_VERIFY_RETRIES:
            print_warning(f"Verification still failing after {MAX_VERIFY_RETRIES} retries; stopping.")
            self._verify_failed = False
            return None
        print_warning(f"Verification failed (retry {self._verify_attempts}/{MAX_VERIFY_RETRIES}); "
                      f"asking the model to fix it.")
        return (
            "[Automatic verification failed]\n"
            f"Command: {self.verify_command}\n"
            f"{result.content[-VERIFY_FEEDBACK_CHARS:]}\n\n"
            "Fix the cause of this failure, then finish. Do not report success until it passes."
        )


# ─── 协议适配器：主循环里和协议有关的部分 ─────────────────────────────────
# 消息仍按各协议的原生格式存放（存档、压缩、折叠、回退都直接操作它们），
# 适配器只负责：修复消息、追加用户消息、注入记忆、调用模型并转成统一的工具调用、写回工具结果。

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ModelTurn:
    tool_calls: list[ToolCall]
    usage: dict[str, int] | None
    stop_reason: str | None
    empty: bool = False         # 既没有文字也没有工具调用：适配器不会把它写进历史


class _ProviderAdapter:
    def __init__(self, agent: Agent):
        self.agent = agent

    def normalize(self) -> None: ...
    def append_user(self, text: str) -> None: ...
    def inject_into_last_user(self, text: str) -> None: ...
    async def call_model(self, on_tool_block) -> ModelTurn: ...
    def append_tool_results(self, pairs: list[tuple[ToolCall, ToolResult]]) -> None: ...

    def after_tool_round(self) -> None:
        pass


class _AnthropicAdapter(_ProviderAdapter):
    # Anthropic 协议要求用户和助手交替出现；相邻的用户消息在第三方兼容接口上可能被拒。
    STOPPED_PLACEHOLDER = "[previous turn ended here]"

    def normalize(self) -> None:
        a = self.agent
        a._anthropic_messages = a._normalize_anthropic_messages(_sanitize_for_utf8(a._anthropic_messages))
        # 上一轮以用户消息结束（预算超限、中断、/compact 之后）：补一条占位的助手消息再开始新一轮。
        # 在记录检查点之前补，检查点里的消息条数才对得上。
        if a._anthropic_messages and a._anthropic_messages[-1].get("role") == "user":
            a._anthropic_messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": self.STOPPED_PLACEHOLDER}]})

    def append_user(self, text: str) -> None:
        messages = self.agent._anthropic_messages
        last = messages[-1] if messages else None
        if last and last.get("role") == "user":
            # 轮次中途（例如折叠之后、空回复之后）最后一条已是用户消息：合并进去，不追加新的。
            content = last.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
            last["content"] = blocks + [{"type": "text", "text": text}]
            return
        messages.append({"role": "user", "content": text})

    def inject_into_last_user(self, text: str) -> None:
        messages = self.agent._anthropic_messages
        last = messages[-1] if messages else None
        if last and last.get("role") == "user":
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = content + "\n\n" + text
            elif isinstance(content, list):
                content.append({"type": "text", "text": text})
        else:
            # 最后一条不是 user message 时，单独追加一条用户消息承载 memory。
            messages.append({"role": "user", "content": text})

    async def call_model(self, on_tool_block) -> ModelTurn:
        a = self.agent
        response = await a._call_anthropic_stream(on_tool_block_complete=on_tool_block)
        usage = a._anthropic_usage(response.usage)
        if not any(b.type == "tool_use" or (b.type == "text" and (b.text or "").strip()) for b in response.content):
            return ModelTurn([], usage, getattr(response, "stop_reason", None), empty=True)
        # 把模型返回的所有 content block 写入历史，后续 tool_result 要与这些 tool_use 对应。
        a._anthropic_messages.append({"role": "assistant",
                                      "content": [a._block_to_dict(b) for b in response.content]})
        calls = [ToolCall(b.id, b.name, dict(b.input) if hasattr(b.input, "items") else b.input)
                 for b in response.content if b.type == "tool_use"]
        return ModelTurn(calls, usage, getattr(response, "stop_reason", None))

    def append_tool_results(self, pairs: list[tuple[ToolCall, ToolResult]]) -> None:
        # Anthropic 要求 assistant/tool_use 后面紧跟一条包含全部 tool_result 的 user 消息；
        # 中断时缺的结果由下一轮开头的 normalize 补齐。
        if pairs:
            self.agent._anthropic_messages.append(
                {"role": "user", "content": [self.agent._anthropic_tool_result(c.id, r) for c, r in pairs]})


class _OpenAIAdapter(_ProviderAdapter):
    def normalize(self) -> None:
        a = self.agent
        a._openai_messages = a._normalize_openai_messages(a._openai_messages, a._journal.resolve)

    def append_user(self, text: str) -> None:
        self.agent._openai_messages.append({"role": "user", "content": text})

    def inject_into_last_user(self, text: str) -> None:
        messages = self.agent._openai_messages
        last = messages[-1] if messages else None
        if last and last.get("role") == "user":
            last["content"] = (last.get("content") or "") + "\n\n" + text
        else:
            messages.append({"role": "user", "content": text})

    async def call_model(self, on_tool_block) -> ModelTurn:
        a = self.agent
        response = await a._call_openai_stream()
        choice = response.get("choices", [{}])[0] if response.get("choices") else {}
        message = choice.get("message", {})
        u = response.get("usage")
        usage = None
        if u:
            cached = u.get("cached_tokens", 0) or 0
            usage = {"input": max(u["prompt_tokens"] - cached, 0), "output": u["completion_tokens"],
                     "cache_read": cached, "cache_write": 0}
        if not message.get("tool_calls") and not (message.get("content") or "").strip():
            return ModelTurn([], usage, choice.get("finish_reason"), empty=True)
        a._openai_messages.append(message)
        calls = []
        for tc in message.get("tool_calls") or []:
            if tc.get("type") != "function":
                # 不支持的调用类型也照常走执行流程，得到 "Unknown tool" 错误结果，模型看到后可以继续。
                calls.append(ToolCall(tc.get("id", ""), f"unsupported:{tc.get('type')}", {}))
                continue
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args))
        return ModelTurn(calls, usage, choice.get("finish_reason"))

    def append_tool_results(self, pairs: list[tuple[ToolCall, ToolResult]]) -> None:
        for c, r in pairs:
            self.agent._openai_messages.append(self.agent._openai_tool_message(c.id, r))

    def after_tool_round(self) -> None:
        a = self.agent
        a._openai_messages = a._normalize_openai_messages(a._openai_messages, a._journal.resolve)
