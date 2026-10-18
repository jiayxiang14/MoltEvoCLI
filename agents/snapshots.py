"""
改文件前的快照与撤销。

write_file / edit_file 执行前把原内容存到 ~/.moltevo-cli/snapshots/<会话ID>/，并在撤销栈里记一条：
路径、原来是否存在、快照位置、写入后内容的哈希。/undo 按栈逆序恢复。

- 不基于 git：非 git 目录也能用，也不碰用户的暂存区。
- 撤销前比对哈希：文件在 agent 写入之后又被改过（例如用户手动编辑），默认拒绝撤销，需 --force。
- 覆盖范围只有 write_file / edit_file；shell 命令造成的改动（sed -i、rm 等）不在其中。
- 超过 MAX_SNAPSHOT_BYTES 的文件不存快照，这次修改也就不能撤销（会在结果里提示）。
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT_ROOT = Path.home() / ".moltevo-cli" / "snapshots"
MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024


def _hash_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


@dataclass
class EditRecord:
    seq: int
    path: Path
    existed: bool
    snapshot: Path | None
    tool: str
    ts: float
    after_hash: str | None = None


class EditHistory:
    def __init__(self, session_id: str, root: Path | None = None):
        self.dir = (root or SNAPSHOT_ROOT) / session_id
        self._stack: list[EditRecord] = []
        self._seq = 0
        # 成功写入的累计次数（撤销不减）。主 agent 用它判断本轮是否改过文件，子 agent 的修改也算在内。
        self.commits = 0
        # run_shell 的累计执行次数（主 agent 和子 agent 共用）：回退时据此提示"这些轮次跑过 shell，改动无法回退"。
        self.shell_runs = 0

    def note_shell(self) -> None:
        self.shell_runs += 1

    def __len__(self) -> int:
        return len(self._stack)

    def before_write(self, path: Path, tool: str) -> EditRecord | None:
        """写入前调用。返回 None 表示文件过大没有存快照，这次修改不能撤销。"""
        path = path.resolve()
        existed = path.exists()
        snapshot = None
        if existed:
            try:
                if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                    return None
                self.dir.mkdir(parents=True, exist_ok=True)
                snapshot = self.dir / f"{self._seq + 1:04d}-{path.name}"
                shutil.copy2(path, snapshot)
            except OSError:
                return None
        self._seq += 1
        return EditRecord(self._seq, path, existed, snapshot, tool, time.time())

    def commit(self, record: EditRecord) -> None:
        """写入成功后调用：记下写入后的哈希，入栈。"""
        record.after_hash = _hash_file(record.path)
        self._stack.append(record)
        self.commits += 1

    def discard(self, record: EditRecord) -> None:
        """写入失败时调用：删掉没用上的快照。"""
        if record.snapshot:
            record.snapshot.unlink(missing_ok=True)

    def records(self) -> list[EditRecord]:
        return list(reversed(self._stack))

    def undo_to(self, length: int, *, force: bool = False) -> tuple[bool, str, list[Path]]:
        """
        批量撤销到撤销栈只剩 length 条。先整体检查冲突：每个文件只看它最后一次被改时的哈希，
        有冲突就一个都不撤（除非 force），避免回退到一半。返回 (是否成功, 说明, 被恢复的路径)。
        """
        pending = self._stack[length:]
        if not pending:
            return True, "No file changes to revert.", []
        latest: dict[Path, EditRecord] = {}
        for record in pending:
            latest[record.path] = record
        # 快照文件缺失时无法恢复，force 也不行：整体拒绝，避免回退到一半。
        missing = [r.path for r in pending if r.existed and not (r.snapshot and r.snapshot.exists())]
        if missing:
            names = ", ".join(sorted({str(p) for p in missing}))
            return False, f"Snapshots are missing for: {names}. Nothing was reverted.", []
        conflicts = [p for p, r in latest.items()
                     if (_hash_file(p) if p.exists() else None) != r.after_hash]
        if conflicts and not force:
            names = ", ".join(str(p) for p in conflicts)
            return False, (f"These files changed after the agent wrote them: {names}. "
                           f"Nothing was reverted; use force to overwrite anyway."), []
        restored: list[Path] = []
        while len(self._stack) > length:
            ok, message, path = self.undo(force=True)
            if not ok:
                return False, f"Stopped part-way: {message}", restored
            if path is not None and path not in restored:
                restored.append(path)
        return True, f"Reverted {len(pending)} file change(s) in {len(restored)} file(s).", restored

    # ── 持久化：跟着会话存档一起保存，--resume 之后仍可撤销 ─────────────
    def to_dict(self) -> dict:
        return {
            "seq": self._seq, "commits": self.commits, "shell_runs": self.shell_runs,
            "records": [{"seq": r.seq, "path": str(r.path), "existed": r.existed,
                         "snapshot": str(r.snapshot) if r.snapshot else None, "tool": r.tool,
                         "ts": r.ts, "after_hash": r.after_hash} for r in self._stack],
        }

    def load_dict(self, data: dict) -> None:
        self._seq = int(data.get("seq", 0))
        self.commits = int(data.get("commits", 0))
        self.shell_runs = int(data.get("shell_runs", 0))
        # 全部保留：检查点按撤销栈长度定位，丢掉任何一条都会让检查点错位。
        # 快照文件缺失的记录在撤销时会被拒绝并说明原因。
        self._stack = [
            EditRecord(int(r["seq"]), Path(r["path"]), bool(r["existed"]),
                       Path(r["snapshot"]) if r.get("snapshot") else None, r.get("tool", ""),
                       float(r.get("ts", 0)), r.get("after_hash"))
            for r in data.get("records", [])
        ]

    def undo(self, *, force: bool = False) -> tuple[bool, str, Path | None]:
        """撤销最近一次修改。返回 (是否成功, 说明, 被恢复的路径)。"""
        if not self._stack:
            return False, "Nothing to undo.", None
        record = self._stack[-1]
        if record.existed and not (record.snapshot and record.snapshot.exists()):
            return False, f"The snapshot for {record.path} is missing; this change can't be undone.", record.path
        current = _hash_file(record.path) if record.path.exists() else None
        if current != record.after_hash and not force:
            return False, (f"{record.path} has changed since the agent wrote it; "
                           f"refusing to undo. Use /undo --force to overwrite anyway."), record.path
        try:
            if record.existed:
                shutil.copy2(record.snapshot, record.path)
                action = "restored"
            else:
                record.path.unlink(missing_ok=True)
                action = "deleted (it did not exist before)"
        except OSError as e:
            return False, f"Undo failed: {e}", record.path
        self._stack.pop()
        return True, f"Undid {record.tool} on {record.path}: {action}.", record.path
