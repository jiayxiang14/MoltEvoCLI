"""
Skill 效果实验（问题清单第 44 条），v2。两个实验：

1. 开 / 关对照（ab）：同一套固定任务，每题在两种条件下各跑 K 次，同一个模型。
   - skills_on：装上全部任务 Skill（与本题矛盾的除外）再加干扰 Skill，模拟真实项目里几十个 Skill 并存，
     检验 agent 能否在干扰中找到并用上对的那个；
   - skills_off：不装任何 Skill；
   - skills_isolated（--isolated 时才跑）：只装本题的 Skill，用来量出"干扰"本身的代价。
   另有一类"不该触发 Skill"的副作用题（negative）：任务本身做对了没有、有没有误用 Skill，分开记录。
   每次运行记录漏斗：检索命中（目标排第几：hit@1、MRR）→ 模型调用 skill 工具 → 判分通过，失败时能看出卡在哪一步。
   调用了非目标 Skill 时区分两种：按那个 Skill 自身的规则本来就适用（任务的 also_applicable），还是真的误调用。
2. 学习曲线（learning）：教学会话里由脚本扮演用户（明说的纠正、隐含的纠正、一次性指令三类），
   之后在新会话里做测试题。三个条件：
   - control：不经过教学；
   - learned_approved：一段教学，最新的待定候选由程序模拟批准（"抽取 → 检索 → 照做"链路的上限）；
   - learned_gate：两段措辞不同的教学（两个会话），用真实门禁，不模拟批准（真实使用时能否学会）。
   一次性指令不应该被学成 Skill 或写进记忆，出现了就记为误学。明说的偏好通常走记忆、隐含的做法
   走 Skill，报告按"经由记忆 / 经由 Skill / 两者 / 都没有"拆分。

判分全部确定性（文件内容、正则、Python 结构、执行过的命令、新建的文件），不用大模型判分。
每次运行在独立临时目录、独立 HOME 里进行；完整轨迹（事件 JSONL + 最终消息）存在 runs/<时间>/trajectories/。
API 调用失败的运行记为出错，不计入成功率和配对检验；账户类错误（401/402/403）立即停止实验。
样本量小，只能检测出较大的差异，报告里会写明。

用法：
  python -m agents.skill_ab ab                    # 默认 3 次
  python -m agents.skill_ab ab --isolated         # 加上"只装本题 Skill"的条件
  python -m agents.skill_ab learning              # 默认 2 次
  python -m agents.skill_ab ab --dry-run          # 只列出任务和预计运行次数
  python -m agents.skill_ab report <运行目录>      # 重新生成 Markdown 报告
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import fnmatch
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

TASKS_FILE = Path(__file__).resolve().parents[1] / "data" / "skill_ab" / "tasks.json"
RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "skill_ab" / "runs"
TASK_TIMEOUT_S = 300
# 账户类错误（密钥无效、余额不足、无权限）：重试和继续跑都没用，立即停止实验。
FATAL_STATUS_CODES = {401, 402, 403}
# 出错（没跑起来）的运行占比超过这个值时，汇总标注结果不可靠。
MAX_ERROR_RATE = 0.2
DEFAULT_REPS = {"ab": 3, "learning": 2}
AB_CONDITIONS = ("skills_on", "skills_off")
LEARNING_CONDITIONS = ("learned_approved", "learned_gate", "control")
# 统计"新建了哪些文件"时忽略的目录：运行时自己产生的，不算 agent 的改动。
IGNORED_DIRS = {".moltevo", "_home", "__pycache__", ".pytest_cache", ".git", "node_modules", ".mypy_cache", ".ruff_cache"}
CATEGORY_ROLE = {
    "preference": "feasibility check (the baseline cannot know these preferences); not an effect estimate",
    "convention": "main effect",
    "failure-lesson": "main effect; compare cost and turns, the baseline can discover the trick by exploring",
    "negative": "side effects: pass rate should not drop with skills installed; see misuse rate",
}


# ── 任务与判分 ────────────────────────────────────────────

def load_suite(path: Path = TASKS_FILE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def install_skills(root: Path, skills: dict[str, dict[str, str]]) -> None:
    for name, spec in skills.items():
        d = root / ".moltevo" / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {spec['description']}\nwhen-to-use: {spec['when_to_use']}\n"
            f"user-invocable: false\n---\n{spec['instructions']}\n", encoding="utf-8")


def materialize(task: dict[str, Any], root: Path, *, skills: dict[str, dict[str, str]] | None) -> None:
    """在 root 下写出任务的仓库文件；skills 为要装进 .moltevo/skills/ 的 Skill（None 表示不装）。"""
    write_files(root, task.get("files", {}))
    if skills:
        install_skills(root, skills)


def skills_for(task: dict[str, Any], suite: dict[str, Any], condition: str) -> dict[str, dict[str, str]] | None:
    """某个条件下要装的 Skill。"""
    if condition == "skills_off":
        return None
    if condition == "skills_isolated":
        return {n: suite["skills"][n] for n in task.get("skills") or []}
    excluded = set(task.get("exclude_skills") or [])
    chosen = {n: spec for n, spec in suite["skills"].items() if n not in excluded}
    chosen.update(suite.get("distractors") or {})
    return chosen


def workspace_files(root: Path) -> set[str]:
    out = set()
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and not (set(rel.parts) & IGNORED_DIRS) and p.suffix != ".pyc":
            out.add(rel.as_posix())
    return out


def reset_workspace(root: Path) -> None:
    """清空工作区，只保留 .moltevo（学到的 Skill、候选登记表）和 _home（用户级数据）。"""
    for p in root.iterdir():
        if p.name in (".moltevo", "_home"):
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def grade(check: dict[str, Any], root: Path, answer: str, ctx: dict[str, Any] | None = None) -> tuple[bool, str]:
    """确定性判分。返回 (是否通过, 说明)。ctx 提供 files（任务初始文件）和 commands（执行过的 shell 命令）。"""
    ctx = ctx or {}
    kind = check["type"]
    flags = re.MULTILINE | (re.IGNORECASE if check.get("ignore_case") else 0)
    if kind == "answer_regex":
        ok = re.search(check["pattern"], answer or "", flags) is not None
        return ok, f"answer {'matches' if ok else 'does not match'} /{check['pattern']}/"
    if kind in ("file_regex", "file_not_regex"):
        p = root / check["path"]
        if not p.exists():
            return kind == "file_not_regex", f"{check['path']} missing"
        found = re.search(check["pattern"], p.read_text(encoding="utf-8", errors="replace"), flags) is not None
        ok = found if kind == "file_regex" else not found
        return ok, f"{check['path']} {'matches' if found else 'does not match'} /{check['pattern']}/"
    if kind == "glob_regex":
        files = sorted(root.glob(check["glob"]))
        hits = [f for f in files if re.search(check["pattern"], f.read_text(encoding="utf-8", errors="replace"), flags)]
        return bool(hits), f"{len(hits)} of {len(files)} files matching {check['glob']} contain /{check['pattern']}/"
    if kind == "docstring_contains":
        p = root / check["path"]
        if not p.exists():
            return False, f"{check['path']} missing"
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            return False, f"syntax error: {e}"
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == check["function"]:
                doc = ast.get_docstring(node) or ""
                ok = check["contains"] in doc
                return ok, f"docstring of {check['function']} {'contains' if ok else 'lacks'} {check['contains']!r}"
        return False, f"function {check['function']} not found"
    if kind == "no_new_files":
        new = sorted(f for f in workspace_files(root) - set(ctx.get("files") or ())
                     if not any(fnmatch.fnmatch(f, pat) for pat in check.get("allow") or []))
        return not new, f"new files: {', '.join(new)}" if new else "no unexpected new files"
    if kind == "no_command_regex":
        hits = [c for c in ctx.get("commands") or [] if re.search(check["pattern"], c, flags)]
        return not hits, f"ran {hits[0]!r}" if hits else f"no command matching /{check['pattern']}/"
    if kind == "not":
        ok, detail = grade(check["check"], root, answer, ctx)
        return not ok, f"not ({detail})"
    if kind == "all":
        results = [grade(c, root, answer, ctx) for c in check["checks"]]
        return all(r[0] for r in results), "; ".join(r[1] for r in results)
    raise ValueError(f"unknown check type: {kind}")


# ── 统计 ─────────────────────────────────────────────────

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """配对二值结果的精确 McNemar 检验（双侧）。b = 只有实验组成功，c = 只有对照组成功。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _avg(rs: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rs if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return round(sum(vals) / len(vals), 4) if vals else None


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def condition_stats(records: list[dict[str, Any]], cond: str) -> dict[str, Any]:
    """一个条件的成功率和效率（只用有效运行；出错的单独计数）。"""
    rs = [r for r in records if r["condition"] == cond and not r.get("error")]
    s = sum(bool(r["passed"]) for r in rs)
    lo, hi = wilson(s, len(rs))
    costs = [r.get("cost_usd") for r in rs]
    total_cost = None if any(c is None for c in costs) else sum(costs)
    prompt = sum(r.get("prompt_tokens") or 0 for r in rs)
    return {"n": len(rs), "passed": s, "rate": round(s / len(rs), 3) if rs else None,
            "ci95": [round(lo, 3), round(hi, 3)],
            "errored": sum(1 for r in records if r["condition"] == cond and r.get("error")),
            "avg_turns": _avg(rs, "turns"), "avg_cost_usd": _avg(rs, "cost_usd"),
            "cost_per_success_usd": round(total_cost / s, 6) if total_cost is not None and s else None,
            "avg_tool_errors": _avg(rs, "tool_errors"), "avg_prompt_tokens": _avg(rs, "prompt_tokens"),
            "cache_hit_rate": round(sum(r.get("cache_read_tokens") or 0 for r in rs) / prompt, 3) if prompt else None}


def paired(records: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """同一题、同一次序号的 a、b 两次运行配成一对；只有两边都有效运行的才算。"""
    pairs: dict[tuple[str, int], dict[str, bool]] = {}
    for r in records:
        if not r.get("error"):
            pairs.setdefault((r.get("item_id") or r["task_id"], r["rep"]), {})[r["condition"]] = bool(r["passed"])
    pairs = {k: v for k, v in pairs.items() if a in v and b in v}
    only_a = sum(1 for v in pairs.values() if v[a] and not v[b])
    only_b = sum(1 for v in pairs.values() if v[b] and not v[a])
    return {"pairs": len(pairs), f"only_{a}": only_a, f"only_{b}": only_b,
            "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4)}


def summarize(records: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """比较两个条件（a 为实验组，b 为对照组）。

    出错的运行（API 调用失败、超时等，任务根本没跑完）不是"模型没做对"，不计入成功率、置信区间和配对检验，
    只单独计数。"""
    errored = sum(1 for r in records if r.get("error"))
    out: dict[str, Any] = {a: condition_stats(records, a), b: condition_stats(records, b)}
    out["errored_runs"] = errored
    out["reliable"] = not records or errored / len(records) <= MAX_ERROR_RATE
    out["paired"] = paired(records, a, b)
    per_task: dict[str, dict[str, int]] = {}
    for r in records:
        if not r.get("error"):
            key = r.get("item_id") or r["task_id"]
            per_task.setdefault(key, {}).setdefault(r["condition"], 0)
            per_task[key][r["condition"]] += int(bool(r["passed"]))
    out["per_task"] = per_task
    return out


def summarize_ab(records: list[dict[str, Any]], conditions: tuple[str, ...]) -> dict[str, Any]:
    out = summarize(records, "skills_on", "skills_off")
    if "skills_isolated" in conditions:
        out["skills_isolated"] = condition_stats(records, "skills_isolated")
        out["paired_isolated_vs_on"] = paired(records, "skills_isolated", "skills_on")
    out["by_category"] = {}
    for cat in dict.fromkeys(r.get("category") for r in records):
        sub = [r for r in records if r.get("category") == cat]
        entry: dict[str, Any] = {c: condition_stats(sub, c) for c in conditions}
        entry["paired"] = paired(sub, "skills_on", "skills_off")
        entry["role"] = CATEGORY_ROLE.get(cat, "")
        if cat == "negative":
            entry["misuse_rate"] = {c: _rate([bool(r.get("misused")) for r in sub
                                              if r["condition"] == c and not r.get("error")]) for c in conditions}
        out["by_category"][cat] = entry
    # 漏斗：只看装了目标 Skill 的有效运行
    funnel: dict[str, Any] = {}
    for cond in [c for c in conditions if c != "skills_off"]:
        rs = [r for r in records if r["condition"] == cond and not r.get("error") and r.get("target_skills")]
        invoked = [r for r in rs if r.get("invoked_target")]
        ranks = [r.get("target_rank") for r in rs]
        funnel[cond] = {
            "runs": len(rs),
            "retrieval_recall_at_3": _rate([bool(r.get("retrieved_target")) for r in rs]),
            "retrieval_hit_at_1": _rate([rank == 1 for rank in ranks]),
            # 平均倒数排名：排第 1 记 1，第 2 记 1/2，没检索到记 0
            "retrieval_mrr": round(sum(1 / k for k in ranks if k) / len(ranks), 3) if ranks else None,
            "invocation_rate": _rate([bool(r.get("invoked_target")) for r in rs]),
            "pass_rate_when_invoked": _rate([bool(r["passed"]) for r in invoked]),
            "pass_rate_when_not_invoked": _rate([bool(r["passed"]) for r in rs if not r.get("invoked_target")]),
            "avg_applicable_extra_invocations": _avg(rs, "applicable_extra_invocations"),
            "avg_wrong_invocations": _avg(rs, "wrong_invocations"),
        }
    out["funnel"] = funnel
    return out


def summarize_learning(records: list[dict[str, Any]]) -> dict[str, Any]:
    errored = sum(1 for r in records if r.get("error"))
    out: dict[str, Any] = {c: condition_stats(records, c) for c in LEARNING_CONDITIONS}
    out["errored_runs"] = errored
    out["reliable"] = not records or errored / len(records) <= MAX_ERROR_RATE
    should_learn = [r for r in records if r.get("kind") != "oneoff"]
    oneoff = [r for r in records if r.get("kind") == "oneoff"]
    out["vs_control"] = {c: paired(should_learn, c, "control") for c in ("learned_approved", "learned_gate")}
    out["by_kind"] = {}
    for kind in dict.fromkeys(r.get("kind") for r in records):
        sub = [r for r in records if r.get("kind") == kind]
        out["by_kind"][kind] = {c: condition_stats(sub, c) for c in LEARNING_CONDITIONS}

    def promoted_rate(rs: list[dict[str, Any]], cond: str) -> float | None:
        return _rate([bool(r.get("skills_promoted")) for r in rs if r["condition"] == cond and not r.get("error")])
    out["learning"] = {
        # 该学的（明说 + 隐含）：有没有晋升出 Skill
        "promotion_rate": {c: promoted_rate(should_learn, c) for c in ("learned_approved", "learned_gate")},
        # 不该学的（一次性指令）：晋升了就是误学
        "false_learning_rate": {c: promoted_rate(oneoff, c) for c in ("learned_approved", "learned_gate")},
        # 一次性指令被写进记忆，同样是误学
        "false_memory_rate": {c: _rate([bool(r.get("memories_written")) for r in oneoff
                                        if r["condition"] == c and not r.get("error")])
                              for c in ("learned_approved", "learned_gate")},
        "avg_pending_candidates_gate": _avg([r for r in records if r["condition"] == "learned_gate" and not r.get("error")],
                                            "pending_candidates"),
        # 该学的题经由哪条通路学到：记忆 / Skill / 两者 / 都没有，以及各自的测试通过率
        "learned_via": {
            c: {via: {"n": len(rs), "passed": sum(1 for r in rs if r["passed"])}
                for via in ("memory", "skill", "both", "neither")
                for rs in [[r for r in should_learn if r["condition"] == c and not r.get("error")
                            and _learned_via(r) == via]]}
            for c in ("learned_approved", "learned_gate")},
    }
    out["per_item"] = summarize(records, "learned_approved", "control")["per_task"]
    return out


# ── 运行 ─────────────────────────────────────────────────

def _describe_error(e: BaseException) -> tuple[str, bool]:
    """返回 (错误说明, 是否为账户类致命错误)。"""
    status = getattr(e, "status_code", None)
    if isinstance(e, asyncio.TimeoutError):
        return f"timeout after {TASK_TIMEOUT_S}s", False
    return f"{type(e).__name__}: {e}", status in FATAL_STATUS_CODES


@contextlib.contextmanager
def sandbox(prefix: str):
    """独立工作目录 + 独立 HOME：用户级 Skill、记忆不会混进实验。"""
    root = Path(tempfile.mkdtemp(prefix=f"moltevo-ab-{prefix}-"))
    home = root / "_home"
    home.mkdir()
    saved = {k: os.environ.get(k) for k in ("HOME", "MOLTEVO_AUTO_SKILL_EVOLUTION")}
    old_cwd = os.getcwd()
    os.chdir(root)
    os.environ["HOME"] = str(home)
    try:
        yield root
    finally:
        # 恢复工作目录和环境变量，不影响同一进程里之后的运行。
        os.chdir(old_cwd)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(root, ignore_errors=True)


def make_agent(agent_kwargs: dict[str, Any], *, evolution: bool, events_path: Path | None = None):
    """events_path 不为空时把事件（模型调用、工具调用、检索结果）写进这次运行的轨迹文件。"""
    from agents.agent import Agent
    from agents.skills import reset_skill_cache
    reset_skill_cache()
    os.environ["MOLTEVO_AUTO_SKILL_EVOLUTION"] = "1" if evolution else "0"
    agent = Agent(permission_mode="bypassPermissions", events_path=events_path, **agent_kwargs)
    agent._mcp_initialized = True        # 实验不连 MCP，减少无关变量
    agent._auto_save = lambda: None       # 不往真实会话目录写
    from agents.tool_journal import ToolJournal
    agent._journal = ToolJournal(None)
    if events_path is None:
        agent._recorder.enabled = False
    return agent


def tool_calls(agent: Any) -> list[tuple[str, dict[str, Any]]]:
    """从主 agent 的消息历史里取出全部工具调用（名称, 参数），两种协议都支持。"""
    calls: list[tuple[str, dict[str, Any]]] = []
    if agent.use_openai:
        for m in agent._openai_messages:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls.append((fn.get("name", ""), args if isinstance(args, dict) else {}))
    else:
        for m in agent._anthropic_messages:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        calls.append((b.get("name", ""), b.get("input") or {}))
    return calls


def _retrieved(events_path: Path | None) -> list[str]:
    """轨迹里第一次检索（即测试题那条用户消息）返回的 Skill。"""
    if not events_path or not events_path.exists():
        return []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "skill_retrieved":
            return [s.get("name") for s in ev.get("skills") or []]
    return []


def _usage_fields(agent: Any) -> dict[str, Any]:
    st = agent.get_run_stats()
    side_in = (st.get("side_query_tokens") or {}).get("input", 0)
    prompt = (st.get("input_tokens") or 0) + side_in
    return {"turns": st["turns"], "tool_errors": st["tool_errors"], "cost_usd": st["cost_usd"], "model": st["model"],
            "prompt_tokens": prompt, "output_tokens": st.get("output_tokens"),
            "cache_read_tokens": st.get("cache_read_tokens"), "cache_write_tokens": st.get("cache_write_tokens"),
            "cache_hit_rate": round(st.get("cache_read_tokens", 0) / prompt, 3) if prompt else None}


def _finish_trajectory(agent: Any) -> None:
    """把最终消息历史也写进轨迹，回放时能看到模型实际看到的全部内容。"""
    messages = agent._openai_messages if agent.use_openai else agent._anthropic_messages
    agent._recorder.emit("final_messages", messages=messages)
    agent._recorder.close()


def _traj(traj_dir: Path | None, name: str) -> Path | None:
    return traj_dir / f"{name}.jsonl" if traj_dir else None


def _shell_commands(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [str(a.get("command", "")) for n, a in calls if n == "run_shell"]


async def run_task(task: dict[str, Any], suite: dict[str, Any], *, condition: str, rep: int,
                   agent_kwargs: dict[str, Any], traj_dir: Path | None = None) -> dict[str, Any]:
    targets = list(task.get("skills") or [])
    with sandbox(f"{task['id']}-{condition}") as root:
        materialize(task, root, skills=skills_for(task, suite, condition))
        events_path = _traj(traj_dir, f"{task['id']}-{condition}-{rep}")
        agent = make_agent(agent_kwargs, evolution=False, events_path=events_path)
        started, answer, error, fatal = time.time(), "", "", False
        try:
            result = await asyncio.wait_for(agent.run_once(task["prompt"]), timeout=TASK_TIMEOUT_S)
            answer = result["text"]
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the experiment
            error, fatal = _describe_error(e)
        calls = tool_calls(agent)
        ctx = {"files": set(task.get("files", {})), "commands": _shell_commands(calls)}
        correct, detail = grade(task["check"], root, answer, ctx)
        misuse = [d for ok, d in (grade(m, root, answer, ctx) for m in task.get("misuse") or []) if not ok]
        usage = _usage_fields(agent)
        if events_path:
            _finish_trajectory(agent)
        retrieved = _retrieved(events_path)
        invoked = [str(a.get("skill_name", "")) for n, a in calls if n == "skill"]
        applicable = set(task.get("also_applicable") or [])
        rank = next((i for i, name in enumerate(retrieved, start=1) if name in targets), None)
        return {"experiment": "ab", "task_id": task["id"], "category": task.get("category"), "condition": condition,
                "rep": rep, "passed": correct and not misuse and not error, "correct": correct,
                "misused": bool(misuse), "detail": "; ".join([detail] + [f"misuse: {d}" for d in misuse]),
                "error": error, "fatal": fatal, "elapsed_s": round(time.time() - started, 2),
                "target_skills": targets, "retrieved": retrieved,
                "retrieved_target": rank is not None if targets else None,
                "target_rank": rank if targets else None,
                "invoked_skills": invoked,
                "invoked_target": bool(set(targets) & set(invoked)) if targets else None,
                "applicable_extra_invocations": sum(1 for s in invoked if s not in targets and s in applicable),
                "wrong_invocations": sum(1 for s in invoked if s not in targets and s not in applicable),
                "trajectory": events_path.name if events_path else None,
                **usage}


async def _teach(turns: list[str], agent_kwargs: dict[str, Any], events_path: Path | None) -> tuple[str, bool, float | None]:
    """跑一段教学会话。返回 (错误, 是否致命, 花费)。"""
    teacher = make_agent(agent_kwargs, evolution=True, events_path=events_path)
    error, fatal = "", False
    for turn in turns:
        try:
            await asyncio.wait_for(teacher.chat(turn), timeout=TASK_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            # 教学会话没跑完，这次"学习"就不成立，整次运行记为出错
            msg, fatal = _describe_error(e)
            error = f"teaching: {msg}"
            break
        await teacher.drain_background_skill_tasks()
    cost = teacher.get_run_stats()["cost_usd"]
    if events_path:
        _finish_trajectory(teacher)
    return error, fatal, cost


def _memory_files() -> set[str]:
    from agents.memory import get_memory_dir
    return {p.name for p in get_memory_dir().glob("*.md") if p.name != "MEMORY.md"}


def _learned_via(r: dict[str, Any]) -> str:
    skill, memory = bool(r.get("skills_promoted")), bool(r.get("memories_written"))
    return {(True, True): "both", (True, False): "skill", (False, True): "memory"}.get((skill, memory), "neither")


async def run_learning(item: dict[str, Any], suite: dict[str, Any], *, condition: str, rep: int,
                       agent_kwargs: dict[str, Any], traj_dir: Path | None = None) -> dict[str, Any]:
    """learned_*：先跑教学会话（gate 条件跑两段）再在新会话里做测试题；control：直接做测试题。"""
    from agents import skill_candidates
    task = next(t for t in suite["tasks"] if t["id"] == item["test_task"])
    item_id = item.get("id") or item["test_task"]
    with sandbox(f"learn-{item_id}-{condition}") as root:
        promoted: list[str] = []
        memories: list[str] = []
        pending_left = 0
        error, fatal = "", False
        teach_cost: float | None = 0.0
        if condition != "control":
            from agents.skills import discover_skills, reset_skill_cache
            reset_skill_cache()
            before = {s.name for s in discover_skills()}
            memories_before = _memory_files()
            sessions = [item["teaching_turns"]]
            if condition == "learned_gate":
                sessions.append(item.get("teaching_turns_alt") or item["teaching_turns"])
            for i, turns in enumerate(sessions, start=1):
                # 每段教学都从干净的仓库开始，只保留 .moltevo（候选登记表、已晋升的 Skill）
                reset_workspace(root)
                write_files(root, item.get("teaching_files") or task.get("files", {}))
                error, fatal, cost = await _teach(turns, agent_kwargs,
                                                  _traj(traj_dir, f"{item_id}-{condition}-{rep}-teach{i}"))
                teach_cost = None if cost is None or teach_cost is None else teach_cost + cost
                if error:
                    break
            if not error and condition == "learned_approved":
                # 模拟用户在 /skills review 里批准最新的那条（静态检查仍然生效）。全部批准会把
                # 同一经验的几个中间版本都晋升出来，夸大重复。
                latest = max(skill_candidates.pending(), key=lambda r: r.get("updated", 0), default=None)
                if latest:
                    skill_candidates.approve(latest["key"])
            pending_left = len(skill_candidates.pending())
            # 统计生效中 Skill 的增量：既包括批准的，也包括教学中满足门禁、自动晋升的。
            reset_skill_cache()
            promoted = sorted({s.name for s in discover_skills()} - before)
            # 明说的偏好通常由 agent 自己写进记忆（抽取器会跳过记忆已覆盖的内容），两条通路都要记。
            memories = sorted(_memory_files() - memories_before)
        # 测试题在干净的仓库里做，教学留下的文件不会让测试题"白送"
        reset_workspace(root)
        materialize(task, root, skills=None)
        answer = ""
        events_path = _traj(traj_dir, f"{item_id}-{condition}-{rep}")
        agent = make_agent(agent_kwargs, evolution=False, events_path=events_path)
        if not error:
            try:
                answer = (await asyncio.wait_for(agent.run_once(task["prompt"]), timeout=TASK_TIMEOUT_S))["text"]
            except Exception as e:  # noqa: BLE001
                error, fatal = _describe_error(e)
        ctx = {"files": set(task.get("files", {})), "commands": _shell_commands(tool_calls(agent))}
        passed, detail = grade(task["check"], root, answer, ctx)
        usage = _usage_fields(agent)
        if events_path:
            _finish_trajectory(agent)
        test_cost = usage["cost_usd"]
        usage["cost_usd"] = None if test_cost is None or teach_cost is None else round(test_cost + teach_cost, 6)
        return {"experiment": "learning", "item_id": item_id, "kind": item.get("kind", "explicit"),
                "task_id": task["id"], "condition": condition, "rep": rep,
                "passed": passed and not error, "detail": detail, "error": error, "fatal": fatal,
                "skills_promoted": promoted, "memories_written": memories, "pending_candidates": pending_left,
                "teaching_cost_usd": round(teach_cost, 6) if teach_cost is not None else None,
                "trajectory": events_path.name if events_path else None,
                **usage}


def plan(kind: str, suite: dict[str, Any], *, isolated: bool = False) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    if kind == "ab":
        return AB_CONDITIONS + (("skills_isolated",) if isolated else ()), suite["tasks"]
    return LEARNING_CONDITIONS, suite["learning"]


async def run_experiment(kind: str, *, reps: int, agent_kwargs: dict[str, Any],
                         suite: dict[str, Any] | None = None, out_dir: Path | None = None,
                         isolated: bool = False) -> dict[str, Any]:
    suite = suite or load_suite()
    records: list[dict[str, Any]] = []
    out_dir = out_dir or RUNS_DIR / (time.strftime("%Y%m%dT%H%M%S") + f"-{kind}")
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    conditions, items = plan(kind, suite, isolated=isolated)
    runner = run_task if kind == "ab" else run_learning
    aborted = ""
    for item in items:
        for rep in range(1, reps + 1):
            for cond in conditions:
                rec = await runner(item, suite, condition=cond, rep=rep, agent_kwargs=agent_kwargs, traj_dir=traj_dir)
                records.append(rec)
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                status = "ERROR" if rec["error"] else ("PASS" if rec["passed"] else "FAIL")
                print(f"[{len(records)}] {rec.get('item_id') or rec['task_id']} {cond} rep{rep}: {status}"
                      f" ({rec['error'] or rec['detail']})")
                if rec.get("fatal"):
                    aborted = rec["error"]
                    break
            if aborted:
                break
        if aborted:
            break
    summary = summarize_ab(records, conditions) if kind == "ab" else summarize_learning(records)
    planned = len(items) * reps * len(conditions)
    summary.update(experiment=kind, conditions=list(conditions), reps=reps,
                   planned_runs=planned, completed_runs=len(records))
    if aborted:
        summary["aborted"] = f"stopped after an account error (retrying will not help): {aborted}"
    notes = [f"{len(items)} items x {reps} reps per condition; small samples only detect large effects. "
             "Graders are deterministic (no LLM judge). Errored runs are excluded from rates and pairing."]
    if not summary["reliable"]:
        notes.append(f"UNRELIABLE: {summary['errored_runs']} of {len(records)} runs errored "
                     f"(more than {MAX_ERROR_RATE:.0%}); fix the cause and rerun.")
    if aborted:
        notes.append(f"INCOMPLETE: {len(records)} of {planned} planned runs.")
    summary["note"] = " ".join(notes)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(summary, records), encoding="utf-8")
    return summary


# ── 报告 ─────────────────────────────────────────────────

def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{v * 100:.0f}%"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _stats_rows(stats: dict[str, Any], conds: list[str]) -> list[str]:
    rows = ["| condition | n | pass rate | 95% CI | errored | avg turns | cost/success | cache hit |",
            "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        s = stats.get(c)
        if not s:
            continue
        rows.append(f"| {c} | {s['n']} | {_fmt(s['rate'], True)} | {_fmt(s['ci95'][0], True)}–{_fmt(s['ci95'][1], True)} "
                    f"| {s['errored']} | {_fmt(s['avg_turns'])} | {_fmt(s['cost_per_success_usd'])} "
                    f"| {_fmt(s['cache_hit_rate'], True)} |")
    return rows


def render_report(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    conds = summary.get("conditions") or []
    lines = [f"# Skill experiment: {summary.get('experiment', '')}", "", summary.get("note", ""), ""]
    if summary.get("aborted"):
        lines += [f"**Aborted:** {summary['aborted']}", ""]
    lines += ["## Overall", ""] + _stats_rows(summary, conds) + [""]
    if summary.get("experiment") == "ab":
        p = summary["paired"]
        lines += [f"Paired skills_on vs skills_off: {p['pairs']} pairs, only on {p['only_skills_on']}, "
                  f"only off {p['only_skills_off']}, McNemar p = {p['mcnemar_p']}", ""]
        for cat, entry in summary.get("by_category", {}).items():
            pp = entry["paired"]
            lines += [f"## {cat}", "", f"_{entry.get('role', '')}_", ""] + _stats_rows(entry, conds)
            lines += ["", f"Paired: {pp['pairs']} pairs, only on {pp['only_skills_on']}, only off {pp['only_skills_off']}, "
                          f"McNemar p = {pp['mcnemar_p']}"]
            if "misuse_rate" in entry:
                lines.append("Misuse rate: " + ", ".join(f"{c} {_fmt(v, True)}" for c, v in entry["misuse_rate"].items()))
            lines.append("")
        lines += ["## Funnel (runs with a target skill installed)", "",
                  "| condition | runs | recall@3 | hit@1 | MRR | invoked | pass if invoked | pass if not "
                  "| extra (applicable) | wrong invocations |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c, f in summary.get("funnel", {}).items():
            lines.append(f"| {c} | {f['runs']} | {_fmt(f['retrieval_recall_at_3'], True)} | {_fmt(f['retrieval_hit_at_1'], True)} "
                         f"| {_fmt(f['retrieval_mrr'])} | {_fmt(f['invocation_rate'], True)} "
                         f"| {_fmt(f['pass_rate_when_invoked'], True)} | {_fmt(f['pass_rate_when_not_invoked'], True)} "
                         f"| {_fmt(f['avg_applicable_extra_invocations'])} | {_fmt(f['avg_wrong_invocations'])} |")
        lines.append("")
    else:
        for c, pp in summary.get("vs_control", {}).items():
            lines.append(f"{c} vs control (explicit + implicit items): {pp['pairs']} pairs, only {c} {pp[f'only_{c}']}, "
                         f"only control {pp['only_control']}, McNemar p = {pp['mcnemar_p']}")
        lr = summary.get("learning", {})
        lines += ["", "## Learning quality", "",
                  "| condition | promotion rate (should learn) | false learning: skill | false learning: memory |",
                  "|---|---|---|---|"]
        for c in ("learned_approved", "learned_gate"):
            lines.append(f"| {c} | {_fmt(lr.get('promotion_rate', {}).get(c), True)} "
                         f"| {_fmt(lr.get('false_learning_rate', {}).get(c), True)} "
                         f"| {_fmt(lr.get('false_memory_rate', {}).get(c), True)} |")
        lines += ["", "Learned via (should-learn items; passed / runs):", "",
                  "| condition | memory | skill | both | neither |", "|---|---|---|---|---|"]
        for c, via in lr.get("learned_via", {}).items():
            lines.append(f"| {c} | " + " | ".join(f"{via[v]['passed']}/{via[v]['n']}"
                                                  for v in ("memory", "skill", "both", "neither")) + " |")
        lines.append("")
        for kind, entry in summary.get("by_kind", {}).items():
            lines += [f"## {kind}", ""] + _stats_rows(entry, conds) + [""]
    bad = [r for r in records if r.get("error") or not r.get("passed")]
    if bad:
        lines += ["## Failed or errored runs", "",
                  "| item | condition | rep | result | detail | trajectory |", "|---|---|---|---|---|---|"]
        for r in bad:
            detail = (r.get("error") or r.get("detail") or "").replace("|", "\\|")[:160]
            lines.append(f"| {r.get('item_id') or r['task_id']} | {r['condition']} | {r['rep']} "
                         f"| {'ERROR' if r.get('error') else 'FAIL'} | {detail} | {r.get('trajectory') or '-'} |")
        lines.append("")
    return "\n".join(lines)


def report(run_dir: Path) -> str:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    text = render_report(summary, records)
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m agents.skill_ab")
    parser.add_argument("kind", choices=["ab", "learning", "report"])
    parser.add_argument("run_dir", nargs="?", help="report: the run directory")
    parser.add_argument("--reps", type=int, default=None, help="default: 3 for ab, 2 for learning")
    parser.add_argument("--isolated", action="store_true", help="ab: also run with only the task's own skill installed")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.kind == "report":
        if not args.run_dir:
            parser.error("report needs a run directory")
        print(report(Path(args.run_dir)))
        return
    suite = load_suite()
    reps = args.reps or DEFAULT_REPS[args.kind]
    conditions, items = plan(args.kind, suite, isolated=args.isolated)
    runs = len(items) * reps * len(conditions)
    if args.dry_run:
        for it in items:
            print(f"- {it.get('id') or it['test_task']} ({it.get('category') or it.get('kind')})")
        print(f"{len(items)} items x {reps} reps x {len(conditions)} conditions ({', '.join(conditions)}) = {runs} runs")
        return
    from agents.main import _load_env_file, _resolve_api_config
    _load_env_file()
    base, key, use_openai = _resolve_api_config(args.api_base)
    model = args.model or os.environ.get("MODEL") or "deepseek-chat"
    kwargs = {"model": model, "api_key": key,
              "api_base": base if use_openai else None, "anthropic_base_url": None if use_openai else base}
    summary = asyncio.run(run_experiment(args.kind, reps=reps, agent_kwargs=kwargs, isolated=args.isolated))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_task", "per_item")}, ensure_ascii=False, indent=2))
    if summary.get("aborted") or not summary["reliable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
