"""工具执行结果：成功与否由工具自己判定，而不是事后匹配输出里的 "error" 字样。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # 结构化元数据：exit_code、timed_out、duration_ms 等，写进事件日志，不发给模型。
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, content: str, **meta: Any) -> "ToolResult":
        return cls(content, False, meta)

    @classmethod
    def error(cls, content: str, **meta: Any) -> "ToolResult":
        return cls(content, True, meta)
