"""
事件日志与运行统计。

每个会话一个只追加的 JSONL 文件：~/.moltevo-cli/sessions/<会话ID>.events.jsonl。
每写一条就 flush，进程崩溃最多丢正在写的那一行。只记元数据（耗时、token、是否出错、大小），
不记工具输出和文件内容；参数只留前 500 字符并打码常见密钥格式。

RunRecorder 由主 agent 创建、子 agent 共用：
- 事件写进同一个文件，子 agent 的事件用 parent_span_id 指向派它出去的那次工具调用；
- 费用实时累加在同一处，子 agent 运行中也受同一个 --max-cost 约束；
- 内存里同时维护汇总，供 /stats 和评测 runner 读取。
"""

from __future__ import annotations

import json
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, IO

EVENTS_DIR = Path.home() / ".moltevo-cli" / "sessions"
MAX_ARG_CHARS = 500

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}"),
    # 键名后可能紧跟引号：参数是先 json.dumps 再打码的，JSON 里的键形如 "password": "x"。
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd)\b(\\?\"?\s*[=:]\s*)(\\?\"[^\"\\]*\\?\"|'[^']*'|[^\s,}]+)"),
]


def redact(text: str) -> str:
    text = _SECRET_PATTERNS[0].sub("sk-***", text)
    text = _SECRET_PATTERNS[1].sub("Bearer ***", text)
    return _SECRET_PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


def summarize_args(args: Any) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    text = redact(text)
    return text if len(text) <= MAX_ARG_CHARS else text[:MAX_ARG_CHARS] + "…"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


class RunRecorder:
    def __init__(self, session_id: str, *, path: Path | None = None, enabled: bool = True):
        self.session_id = session_id
        self.path = path or (EVENTS_DIR / f"{session_id}.events.jsonl")
        self.enabled = enabled
        self._fh: IO[str] | None = None
        # 费用：任一调用的价格未知，累计费用就不完整。
        self.cost_usd = 0.0
        self.cost_unknown = False
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        # 汇总统计
        self.tool_stats: dict[str, dict[str, Any]] = {}
        self.model_latency_ms: list[float] = []
        self.model_ttft_ms: list[float] = []
        self.model_calls = 0
        self.model_retries = 0
        self.permission_counts: dict[str, int] = {}

    # ── 写事件 ──────────────────────────────────────────
    def emit(self, event_type: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"ts": round(time.time(), 3), "session": self.session_id, "type": event_type, **fields}
        try:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except OSError:
            # 日志写不进去不能影响 agent 本身。
            self.enabled = False

    def close(self) -> None:
        if self._fh:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    # ── 记录 + 汇总 ─────────────────────────────────────
    def record_model_call(self, *, cost_usd: float | None, latency_ms: float, ttft_ms: float | None,
                          retries: int, cache_read: int, cache_write: int, **fields: Any) -> None:
        self.model_calls += 1
        self.model_retries += retries
        self.model_latency_ms.append(latency_ms)
        if ttft_ms is not None:
            self.model_ttft_ms.append(ttft_ms)
        self.cache_read_tokens += cache_read
        self.cache_write_tokens += cache_write
        if cost_usd is None:
            self.cost_unknown = True
        else:
            self.cost_usd += cost_usd
        self.emit("model_call", cost_usd=cost_usd, latency_ms=round(latency_ms),
                  ttft_ms=round(ttft_ms) if ttft_ms is not None else None, retries=retries,
                  cache_read_tokens=cache_read, cache_write_tokens=cache_write, **fields)

    def record_tool_call(self, *, tool: str, is_error: bool, duration_ms: float | None, **fields: Any) -> None:
        s = self.tool_stats.setdefault(tool, {"calls": 0, "errors": 0, "durations": []})
        s["calls"] += 1
        s["errors"] += int(is_error)
        if duration_ms is not None:
            s["durations"].append(duration_ms)
        self.emit("tool_call", tool=tool, is_error=is_error,
                  duration_ms=round(duration_ms) if duration_ms is not None else None, **fields)

    def record_permission(self, *, tool: str, decision: str, **fields: Any) -> None:
        self.permission_counts[decision] = self.permission_counts.get(decision, 0) + 1
        self.emit("permission", tool=tool, decision=decision, **fields)

    # ── 读统计 ──────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        tool_calls = sum(s["calls"] for s in self.tool_stats.values())
        tool_errors = sum(s["errors"] for s in self.tool_stats.values())
        all_durations = [d for s in self.tool_stats.values() for d in s["durations"]]
        return {
            "model_calls": self.model_calls,
            "model_retries": self.model_retries,
            "model_latency_ms_p50": _pct(self.model_latency_ms, 50),
            "model_latency_ms_total": round(sum(self.model_latency_ms)),
            "ttft_ms_p50": _pct(self.model_ttft_ms, 50),
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
            "tool_error_rate": round(tool_errors / tool_calls, 3) if tool_calls else None,
            "tool_ms_total": round(sum(all_durations)),
            "tools": {
                name: {"calls": s["calls"], "errors": s["errors"],
                       "p50_ms": _pct(s["durations"], 50)}
                for name, s in sorted(self.tool_stats.items())
            },
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": None if self.cost_unknown else round(self.cost_usd, 6),
            "cost_usd_known_part": round(self.cost_usd, 6),
            "permissions": dict(self.permission_counts),
            "events_file": str(self.path) if self.enabled else None,
        }
