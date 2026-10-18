"""
按模型计价。

每次模型调用单独算费用再累加，所以主 agent 和子 agent 用不同模型时也算得对。
查不到价格的模型返回 None，不猜：调用方据此显示"未知"，并提示费用上限无法生效。

价格单位：美元 / 百万 token。核对日期 2026-10-19，来源：
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing （cache_write 取 5 分钟缓存写入价）
- OpenAI:    https://developers.openai.com/api/docs/pricing （标准档）
- DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing （取高峰价作为上限，非高峰为一半）
  注意：DeepSeek 价格页已不再列出 deepseek-chat / deepseek-reasoner，因此不收录这两个名称。

可在 ~/.moltevo/settings.json 或 ./.moltevo/settings.json 的 "pricing" 字段覆盖或补充：
  {"pricing": {"deepseek-chat": {"input": 0.3, "output": 1.2, "cache_read": 0.006}}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float | None = None   # 缺省时按 input 价计
    cache_write: float | None = None  # 缺省时按 input 价计
    # 分档计价：提示词总 token 超过 threshold 时改用 above 的价格（如 Haiku 5.5）。
    threshold: int | None = None
    above: "Price | None" = None


def _p(i, o, cr=None, cw=None, **kw) -> Price:
    return Price(i, o, cr, cw, **kw)


# 按模型名前缀匹配，最长前缀优先（例如 claude-opus-4-1 先于 claude-opus-4-2025）。
BUILTIN_PRICES: dict[str, Price] = {
    # Anthropic
    "claude-fable-5-1": _p(10, 50, 0.25, 12.5),
    "claude-opus-5-5": _p(4, 20, 0.20, 5),
    "claude-opus-5": _p(5, 25, 0.50, 6.25),
    "claude-opus-4-8": _p(5, 25, 0.50, 6.25),
    "claude-opus-4-7": _p(5, 25, 0.50, 6.25),
    "claude-opus-4-6": _p(5, 25, 0.50, 6.25),
    "claude-opus-4-5": _p(5, 25, 0.50, 6.25),
    "claude-opus-4-1": _p(15, 75, 1.50, 18.75),
    "claude-opus-4-2025": _p(15, 75, 1.50, 18.75),        # Claude Opus 4（带日期的 ID）
    "claude-sonnet-5-5": _p(2, 10, 0.10, 2.5),
    "claude-sonnet-5": _p(2, 10, 0.20, 2.5),
    "claude-sonnet-4-6": _p(3, 15, 0.30, 3.75),
    "claude-sonnet-4-5": _p(3, 15, 0.30, 3.75),
    "claude-sonnet-4-2025": _p(3, 15, 0.30, 3.75),        # Claude Sonnet 4（带日期的 ID）
    "claude-haiku-5-5": _p(0.10, 0.50, 0.01, 0.125, threshold=100_000,
                           above=_p(0.50, 2.50, 0.05, 0.625)),
    "claude-haiku-4-5": _p(1, 5, 0.10, 1.25),
    "claude-3-5-haiku": _p(0.80, 4, 0.08, 1),
    # OpenAI
    "gpt-4o-mini": _p(0.15, 0.60, 0.075),
    "gpt-4o": _p(2.50, 10.00, 1.25),
    # DeepSeek（高峰价）
    "deepseek-flash": _p(0.3, 1.2, 0.006),
    "deepseek-v4-flash": _p(0.3, 1.2, 0.006),               # 旧名称，官方说明按 Flash 计费
    "deepseek-v4-pro": _p(1.32, 3.96, 0.044),
}

_overrides: dict[str, Price] | None = None


def _load_overrides() -> dict[str, Price]:
    global _overrides
    if _overrides is not None:
        return _overrides
    merged: dict[str, Price] = {}
    for path in (Path.home() / ".moltevo" / "settings.json", Path.cwd() / ".moltevo" / "settings.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8")).get("pricing") or {}
        except (OSError, ValueError, AttributeError):
            continue
        for name, v in raw.items():
            try:
                merged[name] = Price(float(v["input"]), float(v["output"]),
                                     float(v["cache_read"]) if v.get("cache_read") is not None else None,
                                     float(v["cache_write"]) if v.get("cache_write") is not None else None)
            except (KeyError, TypeError, ValueError):
                continue
    _overrides = merged
    return merged


def get_price(model: str) -> Price | None:
    m = (model or "").lower()
    table = {**BUILTIN_PRICES, **{k.lower(): v for k, v in _load_overrides().items()}}
    best = None
    for prefix in table:
        if m.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return table[best] if best else None


def call_cost_usd(model: str, *, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    """单次调用的费用。input_tokens 指未命中缓存的输入。查不到价格返回 None。"""
    price = get_price(model)
    if price is None:
        return None
    prompt_total = input_tokens + cache_read_tokens + cache_write_tokens
    if price.threshold is not None and price.above is not None and prompt_total > price.threshold:
        price = price.above
    cr = price.cache_read if price.cache_read is not None else price.input
    cw = price.cache_write if price.cache_write is not None else price.input
    return (input_tokens * price.input + output_tokens * price.output
            + cache_read_tokens * cr + cache_write_tokens * cw) / 1_000_000


def reset_pricing_cache() -> None:
    global _overrides
    _overrides = None
