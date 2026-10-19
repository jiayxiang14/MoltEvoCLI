"""第五批 5c：效果实验工具。重点验证判分器（标准解法通过、朴素解法不通过）和实验流程本身。"""

import asyncio
import os
import subprocess
import sys

import pytest

from agents import skill_ab
from fakemodel import anth, install

SUITE = skill_ab.load_suite()
TASKS = {t["id"]: t for t in SUITE["tasks"]}

# 每题：遵循 Skill 的标准解法（改动的文件, 最终回答）和没遵循 Skill 的朴素解法
GOOD = {
    "pref-commit-msg": ({"COMMIT_MSG.txt": "feat(calc): add sub\n"}, ""),
    "pref-tldr": ({}, "add returns the sum.\nTL;DR: adds two numbers."),
    "pref-docstring": ({"calc.py": 'def mul(a, b):\n    """Multiply.\n\n    Args:\n        a: x\n\n    Returns:\n        product\n    """\n    return a * b\n'}, ""),
    "pref-spdx": ({"strutil.py": "# SPDX-License-Identifier: MIT\ndef slugify(t):\n    return t.lower().replace(' ', '-')\n"}, ""),
    "conv-test-count": ({"RESULT.txt": "3\n"}, ""),
    "conv-module-location": ({"src/app/dates.py": "def parse_date(s):\n    pass\n"}, ""),
    "conv-logging": ({"src/app/config.py": "from app.log import log\n\ndef load_config(path):\n    try:\n        return open(path).read()\n    except FileNotFoundError:\n        log.error('missing %s', path)\n        return ''\n"}, ""),
    "conv-changelog": ({"CHANGELOG.md": "# Changelog\n\n## Unreleased\n- Add neg()\n\n## 1.0.0\n- Initial release\n"}, ""),
    "fail-gen-env": ({"data.json": '{"ok": true}'}, ""),
    "fail-tz": ({"RESULT.txt": "TZ=UTC python -m pytest -q\n"}, ""),
    "fail-slow-marker": ({"RESULT.txt": "2\n"}, ""),
    "fail-module-entry": ({"out.txt": "done"}, ""),
}
NAIVE = {
    "pref-commit-msg": ({"COMMIT_MSG.txt": "Add sub function\n"}, ""),
    "pref-tldr": ({}, "add returns the sum of a and b."),
    "pref-docstring": ({"calc.py": 'def mul(a, b):\n    """Multiply two numbers."""\n    return a * b\n'}, ""),
    "pref-spdx": ({"strutil.py": "def slugify(t):\n    return t.lower().replace(' ', '-')\n"}, ""),
    "conv-test-count": ({"RESULT.txt": "0\n"}, ""),
    "conv-module-location": ({"dates.py": "def parse_date(s):\n    pass\n"}, ""),
    "conv-logging": ({"src/app/config.py": "def load_config(path):\n    try:\n        return open(path).read()\n    except FileNotFoundError:\n        print('missing')\n        return ''\n"}, ""),
    "conv-changelog": ({}, ""),
    "fail-gen-env": ({}, ""),
    "fail-tz": ({"RESULT.txt": "python -m pytest -q\n"}, ""),
    "fail-slow-marker": ({"RESULT.txt": "3\n"}, ""),
    "fail-module-entry": ({}, ""),
}


def _apply(task, root, files):
    skill_ab.materialize(task, root, skills=None)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_grader_accepts_good_and_rejects_naive(tmp_path, task_id):
    task = TASKS[task_id]
    good, naive = tmp_path / "good", tmp_path / "naive"
    _apply(task, good, GOOD[task_id][0])
    _apply(task, naive, NAIVE[task_id][0])
    assert skill_ab.grade(task["check"], good, GOOD[task_id][1])[0], task_id
    assert not skill_ab.grade(task["check"], naive, NAIVE[task_id][1])[0], task_id


def test_failure_lesson_fixtures_really_fail_without_the_trick(tmp_path):
    # 确认"失败经验"类任务的坑是真的：不照 Skill 做就会失败，照做就成功
    task = TASKS["fail-gen-env"]
    skill_ab.materialize(task, tmp_path, skills=None)
    env = {k: v for k, v in os.environ.items() if k != "GEN_MODE"}
    assert subprocess.run([sys.executable, "tools/gen.py"], cwd=tmp_path, env=env).returncode == 3
    assert subprocess.run([sys.executable, "tools/gen.py"], cwd=tmp_path, env={**env, "GEN_MODE": "fast"}).returncode == 0
    task = TASKS["conv-test-count"]
    skill_ab.materialize(task, tmp_path / "t", skills=None)
    plain = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tmp_path / "t",
                           capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert plain.returncode != 0
    fixed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tmp_path / "t",
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert fixed.returncode == 0 and "3 passed" in fixed.stdout


def test_statistics():
    lo, hi = skill_ab.wilson(25, 60)
    assert 0.29 < lo < 0.31 and 0.54 < hi < 0.55
    assert skill_ab.mcnemar_exact(0, 0) == 1.0
    assert skill_ab.mcnemar_exact(8, 0) == pytest.approx(2 / 2 ** 8)
    recs = [{"task_id": "t", "rep": i, "condition": c, "passed": p, "turns": 1, "cost_usd": None, "tool_errors": 0}
            for i, (a, b) in enumerate([(True, False), (True, False), (True, True)]) for c, p in (("on", a), ("off", b))]
    s = skill_ab.summarize(recs, "on", "off")
    assert s["on"]["passed"] == 3 and s["off"]["passed"] == 1 and s["paired"]["only_on"] == 2


def test_ab_experiment_end_to_end_with_fake_model(tmp_path, monkeypatch):
    # 假模型：只有在用户消息里看到被注入的 Skill 时才按约定写提交信息 —— 验证开/关两组的 Skill 隔离是否正确
    suite = {"skills": SUITE["skills"], "tasks": [TASKS["pref-commit-msg"]], "learning": []}
    from agents.agent import Agent

    async def fake_stream(self, on_tool_block_complete=None, **k):
        first = self._anthropic_messages[0]["content"]
        text = first if isinstance(first, str) else str(first)
        if any(m["role"] == "assistant" for m in self._anthropic_messages):
            return anth(text="done")
        msg = "feat(calc): add sub" if "commit-style" in text else "Add sub"
        return anth(tools=[("write_file", {"file_path": "COMMIT_MSG.txt", "content": msg})])

    monkeypatch.setattr(Agent, "_call_anthropic_stream", fake_stream)
    cwd = os.getcwd()
    summary = asyncio.run(skill_ab.run_experiment(
        "ab", reps=2, suite=suite, out_dir=tmp_path,
        agent_kwargs={"model": "claude-sonnet-4-6", "api_key": "sk-test",
                      "anthropic_base_url": "https://fake.example/anthropic"}))
    assert os.getcwd() == cwd
    assert "MOLTEVO_AUTO_SKILL_EVOLUTION" not in os.environ          # 实验不改动进程环境
    assert summary["skills_on"]["passed"] == 2 and summary["skills_off"]["passed"] == 0
    assert summary["paired"]["only_skills_on"] == 2
    assert (tmp_path / "results.jsonl").read_text().count("\n") == 4


def test_learning_experiment_end_to_end_with_fake_model(tmp_path, monkeypatch):
    import json as _json
    from agents.agent import Agent
    suite = {"skills": SUITE["skills"], "tasks": [TASKS["pref-spdx"]],
             "learning": [{"test_task": "pref-spdx", "teaching_turns": ["make textutil", "add SPDX header please", "thanks"]}]}
    candidate = {"name": "spdx-header", "description": "Create a new module or Python file with an SPDX license header",
                 "when_to_use": "create a new module python file", "instructions": "First line: # SPDX-License-Identifier: MIT",
                 "evidence": "user asked", "tags": []}

    def fake_side_query(self, **kw):
        async def sq(system, user):
            if "Skill Extractor" in system:
                return _json.dumps({"skills": [candidate], "feedback": "corrected"})
            if "Skill Set Manager" in system:
                return _json.dumps({"action": "add"})
            return "[]"
        return sq

    async def fake_stream(self, on_tool_block_complete=None, **k):
        if self._anthropic_messages[-1]["role"] == "user" and any(
                m["role"] == "assistant" and "tool_use" in str(m["content"]) for m in self._anthropic_messages[-2:-1]):
            self._emit_text("done")
            return anth(text="done")
        text = str(self._anthropic_messages[0]["content"])
        if "strutil" in text:
            header = "# SPDX-License-Identifier: MIT\n" if "spdx-header" in text else ""
            return anth(tools=[("write_file", {"file_path": "strutil.py", "content": header + "def slugify(t):\n    return t\n"})])
        self._emit_text("ok")              # 真实流式调用会走这条路径，抽取窗口需要回复文本
        return anth(text="ok")

    monkeypatch.setattr(Agent, "_build_side_query", fake_side_query)
    monkeypatch.setattr(Agent, "_call_anthropic_stream", fake_stream)
    summary = asyncio.run(skill_ab.run_experiment(
        "learning", reps=1, suite=suite, out_dir=tmp_path,
        agent_kwargs={"model": "claude-sonnet-4-6", "api_key": "sk-test",
                      "anthropic_base_url": "https://fake.example/anthropic"}))
    rows = [_json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    learned = next(r for r in rows if r["condition"] == "learned")
    assert learned["skills_promoted"] == ["spdx-header"]
    assert summary["learned"]["passed"] == 1 and summary["control"]["passed"] == 0


def test_errored_runs_are_excluded_from_statistics():
    # 第 3 对里开 Skill 那次 API 失败：不能算成"开 Skill 失败"，这一对也不参与配对检验
    recs = [{"task_id": "t", "rep": i, "condition": c, "passed": p, "error": e, "turns": 1, "cost_usd": None,
             "tool_errors": 0}
            for i, pair in enumerate([((True, ""), (False, "")), ((True, ""), (False, "")), ((False, "boom"), (True, ""))])
            for c, (p, e) in zip(("on", "off"), pair)]
    s = skill_ab.summarize(recs, "on", "off")
    assert s["on"]["n"] == 2 and s["on"]["errored"] == 1 and s["on"]["rate"] == 1.0
    assert s["off"]["n"] == 3 and s["off"]["passed"] == 1
    assert s["paired"]["pairs"] == 2 and s["paired"]["only_off"] == 0
    assert s["errored_runs"] == 1 and s["reliable"] is True
    assert skill_ab.summarize(recs[:2] + [dict(r, error="x") for r in recs[2:]], "on", "off")["reliable"] is False


def test_account_error_stops_the_experiment(tmp_path, monkeypatch):
    import anthropic
    import httpx
    from agents.agent import Agent
    suite = {"skills": SUITE["skills"], "tasks": [TASKS["pref-commit-msg"], TASKS["pref-tldr"]], "learning": []}
    calls = []

    async def broke(self, on_tool_block_complete=None, **k):
        calls.append(1)
        resp = httpx.Response(402, request=httpx.Request("POST", "https://fake.example"))
        raise anthropic.APIStatusError("Insufficient Balance", response=resp, body=None)

    monkeypatch.setattr(Agent, "_call_anthropic_stream", broke)
    summary = asyncio.run(skill_ab.run_experiment(
        "ab", reps=3, suite=suite, out_dir=tmp_path,
        agent_kwargs={"model": "claude-sonnet-4-6", "api_key": "sk-test",
                      "anthropic_base_url": "https://fake.example/anthropic"}))
    assert summary["completed_runs"] == 1 and summary["planned_runs"] == 12
    assert "Insufficient Balance" in summary["aborted"]
    assert summary["skills_on"]["n"] == 0 and summary["reliable"] is False
    assert len(calls) == 1                                    # 402 不重试
