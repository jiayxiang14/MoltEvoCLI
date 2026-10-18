
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from typing import Any
import json

SESSION_DIR = Path.home() / ".moltevo-cli" / "sessions"



def _ensure_dir() -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def get_project_session_dir() -> Path:
    d = Path.cwd() / ".moltevo" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(session_id: str, data: dict[str, Any]) -> None:
    _ensure_dir()
    (SESSION_DIR / f"{session_id}.json").write_text(json.dumps(data, indent=2, default=str))


def save_folded_session_memory(session_id: str, record: dict[str, Any]) -> None:
    d = get_project_session_dir()
    line = json.dumps(record, ensure_ascii=False, default=str)
    with (d / f"{session_id}.folded-memory.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    (d / f"{session_id}.folded-memory.latest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_session(session_id: str) -> dict[str, Any] | None:
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def list_sessions(cwd: str | None = None) -> list[dict[str, Any]]:
    """列出会话元数据。传入 cwd 时只返回在该目录下创建的会话（所有项目的会话存放在同一个目录里）。"""
    _ensure_dir()
    results = []
    for f in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            meta = data.get("metadata")
            if meta and (cwd is None or meta.get("cwd") == cwd):
                results.append(meta)
        except Exception:
            pass
    return results


def get_latest_session_id(cwd: str | None = None) -> str | None:
    sessions = list_sessions(cwd)
    if not sessions:
        return None
    # 同一时刻开始的会话按保存时间排序（同一会话的多次保存会更新文件修改时间）。
    sessions.sort(key=lambda s: (s.get("startTime", ""), _mtime(s.get("id"))), reverse=True)
    return sessions[0].get("id")


def find_session_id(prefix: str, cwd: str | None = None) -> str | None:
    """按会话 ID 前缀查找；有歧义或找不到返回 None。"""
    matches = [s.get("id") for s in list_sessions(cwd) if str(s.get("id", "")).startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def _mtime(session_id: str | None) -> float:
    try:
        return (SESSION_DIR / f"{session_id}.json").stat().st_mtime
    except OSError:
        return 0.0
