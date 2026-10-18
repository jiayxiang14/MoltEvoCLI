"""
工具调用的结果日志（预写日志），以模型给出的 tool_call_id 作键。

每次工具调用：执行前记 started，执行完记 done（带交给模型的完整结果）。每条写完立即落盘。
当一轮被中断（Ctrl+C、进程崩溃后 --resume）、历史里留下没有结果的工具调用时，按日志补结果：
- 有 done：用真实结果补齐，避免模型把已经执行过的操作再做一遍；
- 只有 started：执行到一半被打断，副作用不确定，提示先检查状态再重试；
- 都没有：还没开始执行，可以安全重试。

为什么不是"请求 ID 去重"：重做是模型发起的，每次都会生成新的 tool_call_id，ID 去重识别不出来；
真正要避免的是"已经执行完、但结果没进历史"的调用被当成没执行。
日志和会话存档放在一起（会话存档本来就保存完整的工具结果）；事件日志仍只记元数据。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agents.tool_result import ToolResult


class ToolJournal:
    def __init__(self, path: Path | None):
        self.path = path
        self.entries: dict[str, dict] = {}

    def _append(self, record: dict) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # 日志写不进去不能影响工具执行本身；最坏情况退化为"状态未知"。
            pass

    def started(self, call_id: str | None, tool: str) -> None:
        if not call_id:
            return
        self.entries[call_id] = {"state": "started", "tool": tool}
        self._append({"id": call_id, "state": "started", "tool": tool, "ts": round(time.time(), 3)})

    def done(self, call_id: str | None, tool: str, result: ToolResult) -> None:
        if not call_id:
            return
        entry = {"state": "done", "tool": tool, "content": result.content, "is_error": result.is_error}
        self.entries[call_id] = entry
        self._append({"id": call_id, **entry, "ts": round(time.time(), 3)})

    def load(self, path: Path) -> None:
        """读入一个日志文件（例如 --resume 时上一个会话的日志），后写的记录覆盖先写的。"""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                rec = json.loads(line)
                call_id = rec.pop("id")
                rec.pop("ts", None)
                self.entries[call_id] = rec
            except (ValueError, KeyError, TypeError):
                continue      # 崩溃时写了一半的最后一行

    def resolve(self, call_id: str, tool: str) -> ToolResult:
        """给一个没有结果的工具调用补结果。"""
        entry = self.entries.get(call_id)
        if entry and entry.get("state") == "done":
            return ToolResult(entry.get("content", ""), bool(entry.get("is_error")), {"recovered": True})
        if entry:
            return ToolResult.error(
                f"Tool call '{tool}' was interrupted while running; it may have partially run. "
                f"Check the current state before retrying it.")
        return ToolResult.error(f"Tool call '{tool}' was interrupted before it started; it is safe to retry.")
