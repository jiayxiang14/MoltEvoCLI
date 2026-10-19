"""
Skill 效果实验（问题清单第 44 条）。两个实验：

1. 开 / 关对照（ab）：同一套固定任务，在"装了相关 Skill"和"没装"两种条件下各跑 K 次，同一个模型。
   每题有确定性判分器（文件内容、正则、Python 结构检查），不用大模型判分。
   同一题、同一次序号的两次运行配成一对，用精确 McNemar 检验比较；成功率附 Wilson 95% 置信区间。
2. 学习曲线（learning）：第一段会话里由脚本扮演用户纠正 agent 的做法，经候选区晋升出 Skill
   （为了结果可复现，用程序模拟"用户批准"）；第二段会话做一道类似任务，与"没经历第一段"的对照组比较。

每次运行都在独立的临时目录里进行，HOME 也指向临时目录，用户级 Skill 和记忆不会混进实验。
样本量小（默认 12 题 × 3 次），只能检测出较大的差异，报告里会写明。

用法：
  python -m agents.skill_ab ab --reps 3
  python -m agents.skill_ab learning --reps 3
  python -m agents.skill_ab ab --dry-run          # 只列出任务和预计运行次数
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
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


# ── 任务与判分 ────────────────────────────────────────────

def load_suite(path: Path = TASKS_FILE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(task: dict[str, Any], root: Path, *, skills: dict[str, dict[str, str]] | None) -> None:
    """在 root 下写出任务的仓库文件；skills 不为 None 时装进 .moltevo/skills/。"""
    for rel, content in task.get("files", {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    for name in (task.get("skills") or []) if skills is not None else []:
        spec = skills[name]
        d = root / ".moltevo" / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {spec['description']}\nwhen-to-use: {spec['when_to_use']}\n"
            f"user-invocable: false\n---\n{spec['instructions']}\n", encoding="utf-8")


def grade(check: dict[str, Any], root: Path, answer: str) -> tuple[bool, str]:
    """确定性判分。返回 (是否通过, 说明)。"""
    kind = check["type"]
    flags = re.MULTILINE | (re.IGNORECASE if check.get("ignore_case") else 0)
    if kind == "answer_regex":
        ok = re.search(check["pattern"], answer or "", flags) is not None
        return ok, f"answer {'matches' if ok else 'does not match'} /{check['pattern']}/"
    if kind in ("file_regex", "file_not_regex"):
        p = root / check["path"]
        if not p.exists():
            return False, f"{check['path']} missing"
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
    if kind == "all":
        results = [grade(c, root, answer) for c in check["checks"]]
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
    """配对二值结果的精确 McNemar 检验（双侧）。b = 只有开 Skill 成功，c = 只有关 Skill 成功。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(records: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """比较两个条件（a 为实验组，b 为对照组）。

    出错的运行（API 调用失败、超时等，任务根本没跑完）不是"模型没做对"，不计入成功率、置信区间和配对检验，
    只单独计数。"""
    valid = [r for r in records if not r.get("error")]
    errored = len(records) - len(valid)
    out: dict[str, Any] = {}
    for cond in (a, b):
        rs = [r for r in valid if r["condition"] == cond]
        s = sum(r["passed"] for r in rs)
        lo, hi = wilson(s, len(rs))
        out[cond] = {"n": len(rs), "passed": s, "rate": round(s / len(rs), 3) if rs else None,
                     "ci95": [round(lo, 3), round(hi, 3)],
                     "errored": sum(1 for r in records if r["condition"] == cond and r.get("error")),
                     "avg_turns": _avg(rs, "turns"), "avg_cost_usd": _avg(rs, "cost_usd"),
                     "avg_tool_errors": _avg(rs, "tool_errors")}
    out["errored_runs"] = errored
    out["reliable"] = not records or errored / len(records) <= MAX_ERROR_RATE
    pairs = {}
    for r in valid:
        pairs.setdefault((r["task_id"], r["rep"]), {})[r["condition"]] = r["passed"]
    # 只有两个条件都有效运行的才算一对
    pairs = {k: v for k, v in pairs.items() if a in v and b in v}
    only_a = sum(1 for v in pairs.values() if v.get(a) and v.get(b) is False)
    only_b = sum(1 for v in pairs.values() if v.get(b) and v.get(a) is False)
    out["paired"] = {"pairs": len(pairs), f"only_{a}": only_a, f"only_{b}": only_b,
                     "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4)}
    per_task: dict[str, dict[str, int]] = {}
    for r in valid:
        per_task.setdefault(r["task_id"], {}).setdefault(r["condition"], 0)
        per_task[r["task_id"]][r["condition"]] += int(r["passed"])
    out["per_task"] = per_task
    return out


def _avg(rs: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


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


def make_agent(agent_kwargs: dict[str, Any], *, evolution: bool):
    from agents.agent import Agent
    from agents.skills import reset_skill_cache
    reset_skill_cache()
    os.environ["MOLTEVO_AUTO_SKILL_EVOLUTION"] = "1" if evolution else "0"
    agent = Agent(permission_mode="bypassPermissions", **agent_kwargs)
    agent._mcp_initialized = True        # 实验不连 MCP，减少无关变量
    agent._auto_save = lambda: None       # 不往真实会话目录写
    from agents.tool_journal import ToolJournal
    agent._journal = ToolJournal(None)
    agent._recorder.enabled = False
    return agent


async def run_task(task: dict[str, Any], suite: dict[str, Any], *, condition: str, rep: int,
                   agent_kwargs: dict[str, Any]) -> dict[str, Any]:
    with sandbox(f"{task['id']}-{condition}") as root:
        materialize(task, root, skills=suite["skills"] if condition == "skills_on" else None)
        agent = make_agent(agent_kwargs, evolution=False)
        started, answer, error, fatal = time.time(), "", "", False
        try:
            result = await asyncio.wait_for(agent.run_once(task["prompt"]), timeout=TASK_TIMEOUT_S)
            answer = result["text"]
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the experiment
            error, fatal = _describe_error(e)
        passed, detail = grade(task["check"], root, answer)
        st = agent.get_run_stats()
        return {"experiment": "ab", "task_id": task["id"], "category": task.get("category"), "condition": condition,
                "rep": rep, "passed": passed and not error, "detail": detail, "error": error, "fatal": fatal,
                "elapsed_s": round(time.time() - started, 2), "turns": st["turns"],
                "tool_errors": st["tool_errors"], "cost_usd": st["cost_usd"], "model": st["model"]}


async def run_learning(item: dict[str, Any], suite: dict[str, Any], *, condition: str, rep: int,
                       agent_kwargs: dict[str, Any]) -> dict[str, Any]:
    """learned：先跑教学会话并晋升学到的 Skill，再在新会话里做测试题；control：直接做测试题。"""
    from agents import skill_candidates
    task = next(t for t in suite["tasks"] if t["id"] == item["test_task"])
    with sandbox(f"learn-{task['id']}-{condition}") as root:
        materialize(task, root, skills=None)
        promoted: list[str] = []
        error, fatal = "", False
        if condition == "learned":
            from agents.skills import discover_skills, reset_skill_cache
            before = {s.name for s in discover_skills()}
            teacher = make_agent(agent_kwargs, evolution=True)
            for turn in item["teaching_turns"]:
                try:
                    await asyncio.wait_for(teacher.chat(turn), timeout=TASK_TIMEOUT_S)
                except Exception as e:  # noqa: BLE001
                    # 教学会话没跑完，这次"学习"就不成立，整次运行记为出错
                    msg, fatal = _describe_error(e)
                    error = f"teaching: {msg}"
                    break
                await teacher.drain_background_skill_tasks()
            for record in skill_candidates.pending():
                # 还没过门禁的候选：模拟用户在 /skills review 里批准，保证结果可复现（静态检查仍然生效）
                skill_candidates.approve(record["key"])
            # 统计生效中 Skill 的增量：既包括上面批准的，也包括教学中已满足门禁、自动晋升的。
            reset_skill_cache()
            promoted = sorted({s.name for s in discover_skills()} - before)
        answer = ""
        if not error:
            agent = make_agent(agent_kwargs, evolution=False)
            try:
                answer = (await asyncio.wait_for(agent.run_once(task["prompt"]), timeout=TASK_TIMEOUT_S))["text"]
            except Exception as e:  # noqa: BLE001
                error, fatal = _describe_error(e)
        passed, detail = grade(task["check"], root, answer)
        return {"experiment": "learning", "task_id": task["id"], "condition": condition, "rep": rep,
                "passed": passed and not error, "detail": detail, "error": error, "fatal": fatal,
                "skills_promoted": promoted}


async def run_experiment(kind: str, *, reps: int, agent_kwargs: dict[str, Any],
                         suite: dict[str, Any] | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    suite = suite or load_suite()
    records: list[dict[str, Any]] = []
    out_dir = out_dir or RUNS_DIR / (time.strftime("%Y%m%dT%H%M%S") + f"-{kind}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if kind == "ab":
        conditions, items, runner = ("skills_on", "skills_off"), suite["tasks"], run_task
    else:
        conditions, items, runner = ("learned", "control"), suite["learning"], run_learning
    aborted = ""
    for item in items:
        for rep in range(1, reps + 1):
            for cond in conditions:
                rec = await runner(item, suite, condition=cond, rep=rep, agent_kwargs=agent_kwargs)
                records.append(rec)
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                status = "ERROR" if rec["error"] else ("PASS" if rec["passed"] else "FAIL")
                print(f"[{len(records)}] {rec['task_id']} {cond} rep{rep}: {status}"
                      f" ({rec['error'] or rec['detail']})")
                if rec.get("fatal"):
                    aborted = rec["error"]
                    break
            if aborted:
                break
        if aborted:
            break
    summary = summarize(records, *conditions)
    planned = len(items) * reps * len(conditions)
    summary["planned_runs"] = planned
    summary["completed_runs"] = len(records)
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m agents.skill_ab")
    parser.add_argument("kind", choices=["ab", "learning"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = load_suite()
    items = suite["tasks"] if args.kind == "ab" else suite["learning"]
    runs = len(items) * args.reps * 2
    if args.dry_run:
        for it in items:
            print(f"- {it['id'] if 'id' in it else it['test_task']}")
        print(f"{len(items)} items x {args.reps} reps x 2 conditions = {runs} agent runs")
        return
    from agents.main import _load_env_file, _resolve_api_config
    _load_env_file()
    base, key, use_openai = _resolve_api_config(args.api_base)
    model = args.model or os.environ.get("MODEL") or "deepseek-chat"
    kwargs = {"model": model, "api_key": key,
              "api_base": base if use_openai else None, "anthropic_base_url": None if use_openai else base}
    summary = asyncio.run(run_experiment(args.kind, reps=args.reps, agent_kwargs=kwargs))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("aborted") or not summary["reliable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
