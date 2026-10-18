"""
Skill 候选区与晋升门禁。

自进化抽出的新 Skill、合并后的新版本，不再直接写进生效中的 Skill，而是先进候选区（不参与检索，
不影响 agent 行为），过了门禁才晋升：
1. 静态检查（确定性，不花钱）：说明非空、长度在上限内、不含密钥；
2. 证据：同一个候选在不同轮次/会话出现 >= MIN_OCCURRENCES 次，或用户明确批准；
3. 用户手写的 Skill 对 agent 只读：想改它只能形成"覆盖候选"，必须由用户批准，永不自动晋升。

晋升方式：acceptEdits / 跳过确认模式下过门禁即自动晋升；默认模式下进入"待批准"，用 /skills review 处理。
登记表在 .moltevo/skill-evolution/candidates.json（随项目，不随会话）。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from agents.events import redact

MIN_OCCURRENCES = 2
MAX_INSTRUCTION_CHARS = 8000
STALE_CANDIDATE_DAYS = 30


def _registry_path() -> Path:
    return Path.cwd() / ".moltevo" / "skill-evolution" / "candidates.json"


def _load() -> dict[str, dict[str, Any]]:
    try:
        return json.loads(_registry_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(registry: dict[str, dict[str, Any]]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)          # 原子替换，进程中途退出不会留下写了一半的登记表


def _key(action: str, name: str, target: str) -> str:
    ident = target if action == "merge" else name
    return f"{action}:{re.sub(r'[^a-z0-9一-鿿]+', '-', ident.lower()).strip('-')}"


def skill_author(skill_dir: str | Path) -> str:
    """SKILL.md frontmatter 里的 author。没有这个字段的视为用户手写（保守处理：只读、不自动淘汰）。"""
    try:
        head = (Path(skill_dir) / "SKILL.md").read_text(encoding="utf-8")[:2000]
    except OSError:
        return "user"
    m = re.search(r"^author:\s*(\S+)\s*$", head, re.MULTILINE)
    return m.group(1).strip() if m else "user"


def static_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    instructions = str(record.get("instructions") or "").strip()
    if not instructions:
        problems.append("empty instructions")
    if len(instructions) > MAX_INSTRUCTION_CHARS:
        problems.append(f"instructions longer than {MAX_INSTRUCTION_CHARS} chars")
    text = " ".join(str(record.get(k) or "") for k in ("name", "description", "when_to_use", "instructions"))
    if redact(text) != text:
        problems.append("looks like it contains a secret")
    if record.get("action") == "add" and not str(record.get("description") or "").strip():
        problems.append("missing description")
    return problems


def stage(*, action: str, name: str, target_skill: str, description: str, when_to_use: str,
          instructions: str, tags: list[str] | None, evidence: str, reason: str,
          session_id: str, window_id: str, target_is_manual: bool) -> dict[str, Any]:
    """登记或更新一个候选；同一候选来自新的轮次/会话时计一次出现。返回最新记录。"""
    registry = _load()
    key = _key(action, name, target_skill)
    now = time.time()
    record = registry.get(key) or {"key": key, "status": "pending", "created": now, "occurrences": 0,
                                   "sources": [], "approved": False}
    source = f"{session_id}:{window_id}"
    if source not in record["sources"]:
        record["sources"] = (record["sources"] + [source])[-20:]
        record["occurrences"] = int(record.get("occurrences", 0)) + 1
    record.update({
        "action": action, "name": name, "target_skill": target_skill, "description": description,
        "when_to_use": when_to_use, "instructions": instructions, "tags": list(tags or []),
        "evidence": evidence, "reason": reason, "updated": now, "target_is_manual": target_is_manual,
    })
    if record["status"] in ("rejected", "promoted"):
        # 被拒绝或已晋升后再次出现：作为新一轮候选重新计数。
        record.update(status="pending", occurrences=1, sources=[source], approved=False)
    record["problems"] = static_problems(record)
    registry[key] = record
    _save(registry)
    return record


def gate(record: dict[str, Any], *, auto: bool) -> tuple[bool, str]:
    """返回 (能否晋升, 原因)。auto=True 表示不经用户批准的自动晋升。"""
    if record.get("problems"):
        return False, "static check failed: " + "; ".join(record["problems"])
    if record.get("approved"):
        return True, "approved by the user"
    if record.get("target_is_manual"):
        return False, "changes a user-written skill; needs your approval"
    if not auto:
        return False, "waiting for your approval"
    if int(record.get("occurrences", 0)) < MIN_OCCURRENCES:
        return False, f"seen {record.get('occurrences', 0)}/{MIN_OCCURRENCES} times"
    return True, f"seen {record['occurrences']} times"


def promote(record: dict[str, Any]) -> dict[str, Any]:
    """把候选写成生效中的 Skill（新增或合并），并在登记表里标记已晋升。"""
    from agents.skills import create_skill, evolve_skill
    if record["action"] == "merge":
        result = evolve_skill(
            skill_name=record["target_skill"], lesson=record.get("evidence") or record.get("description", ""),
            rationale=record.get("reason") or "promoted candidate", target="active",
            instructions=record.get("instructions", ""), description=record.get("description", ""),
            when_to_use=record.get("when_to_use", ""), tags=record.get("tags"))
    else:
        result = create_skill(
            name=record["name"], description=record.get("description", ""),
            instructions=record.get("instructions", ""), when_to_use=record.get("when_to_use", ""),
            target="project", context="inline", user_invocable=False,
            evidence=record.get("evidence", ""), actor="online", tags=record.get("tags"))
    registry = _load()
    if record["key"] in registry and result.get("ok"):
        registry[record["key"]].update(status="promoted", promoted=time.time())
        _save(registry)
    return result


def approve(key: str) -> dict[str, Any]:
    registry = _load()
    record = registry.get(key)
    if not record:
        return {"ok": False, "error": f"no candidate {key}"}
    record["approved"] = True
    _save(registry)
    ok, why = gate(record, auto=False)
    if not ok:
        return {"ok": False, "error": why}
    return promote(record)


def reject(key: str) -> bool:
    registry = _load()
    if key not in registry:
        return False
    registry[key].update(status="rejected", rejected=time.time())
    _save(registry)
    return True


def pending() -> list[dict[str, Any]]:
    return [r for r in _load().values() if r.get("status") == "pending"]


def cleanup(now: float | None = None) -> int:
    """清理超过 STALE_CANDIDATE_DAYS 天没再出现的待定候选。"""
    now = now or time.time()
    registry = _load()
    stale = [k for k, r in registry.items()
             if r.get("status") == "pending" and now - float(r.get("updated", now)) > STALE_CANDIDATE_DAYS * 86400]
    for k in stale:
        del registry[k]
    if stale:
        _save(registry)
    return len(stale)


def window_id(window: dict[str, Any]) -> str:
    """同一个对话窗口重复处理时不重复计数。"""
    basis = json.dumps([window.get("latest_user", ""), window.get("latest_assistant", "")], ensure_ascii=False)
    return hashlib.sha256(basis.encode()).hexdigest()[:12]
