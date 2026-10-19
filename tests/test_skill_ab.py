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
    "neg-readme-typo": ({"README.md": "# Calc\n\nA tiny calculator library.\n"}, ""),
    "neg-json-config": ({"config.json": '{\n  "timeout": 60,\n  "retries": 3\n}\n'}, ""),
    "neg-http-question": ({}, "It means the requested resource was not found on the server."),
    "neg-count-lines": ({"RESULT.txt": "5\n"}, ""),
    "neg-js-file": ({"util.js": "export function capitalize(s) {\n  return s[0].toUpperCase() + s.slice(1);\n}\n"}, ""),
    "neg-rename-var": ({"src/app/geometry.py": "def area(w, h):\n    result = w * h\n    return result\n"}, ""),
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
    "fail-module-entry": ({"out.txt": "legacy entry point: nothing generated\n"}, ""),
    "neg-readme-typo": ({}, ""),
    "neg-json-config": ({}, ""),
    "neg-http-question": ({}, "It is a client error."),
    "neg-count-lines": ({"RESULT.txt": "4\n"}, ""),
    "neg-js-file": ({}, ""),
    "neg-rename-var": ({}, ""),
}
# 副作用题：任务做对了，但误用了 Skill（文件, 回答, 执行过的命令）—— 必须被判为误用
MISUSE = {
    "neg-readme-typo": ({"README.md": "# Calc\n\nA tiny calculator library.\n", "CHANGELOG.md": "## Unreleased\n- typo\n"}, "", []),
    "neg-json-config": ({"config.json": '{"timeout": 60, "retries": 3}', "migrations/001.sql": "--"}, "", []),
    "neg-http-question": ({}, "The resource was not found.\nTL;DR: not found.", []),
    "neg-count-lines": ({"RESULT.txt": "5\n"}, "", ["python -m pytest -q"]),
    "neg-js-file": ({"util.js": "# SPDX-License-Identifier: MIT\nexport function capitalize(s) { return s; }\n"}, "", []),
    "neg-rename-var": ({"src/app/geometry.py": "def area(w, h):\n    result = w * h\n    return result\n",
                        "src/app/area_utils.py": "# new\n"}, "", []),
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
    ctx = {"files": set(task["files"]), "commands": []}
    assert skill_ab.grade(task["check"], good, GOOD[task_id][1], ctx)[0], task_id
    assert not skill_ab.grade(task["check"], naive, NAIVE[task_id][1], ctx)[0], task_id
    # 标准解法不能被误判为误用
    assert all(skill_ab.grade(m, good, GOOD[task_id][1], ctx)[0] for m in task.get("misuse") or []), task_id


@pytest.mark.parametrize("task_id", sorted(MISUSE))
def test_misuse_is_detected(tmp_path, task_id):
    task = TASKS[task_id]
    files, answer, commands = MISUSE[task_id]
    _apply(task, tmp_path, files)
    ctx = {"files": set(task["files"]), "commands": commands}
    assert skill_ab.grade(task["check"], tmp_path, answer, ctx)[0], task_id          # 任务本身做对了
    assert not all(skill_ab.grade(m, tmp_path, answer, ctx)[0] for m in task["misuse"]), task_id


def test_suite_is_consistent():
    assert len(SUITE["tasks"]) == 18 and len(SUITE["learning"]) == 9 and len(SUITE["distractors"]) == 8
    names = set(SUITE["skills"]) | set(SUITE["distractors"])
    for t in SUITE["tasks"]:
        assert set(t.get("skills") or []) <= set(SUITE["skills"]) and set(t.get("exclude_skills") or []) <= names
        assert not set(t.get("skills") or []) & set(t.get("exclude_skills") or [])
        also = set(t.get("also_applicable") or [])
        assert also <= set(SUITE["skills"]) and not also & set(t.get("skills") or []) and not also & set(t.get("exclude_skills") or [])
    kinds = [i["kind"] for i in SUITE["learning"]]
    assert kinds.count("explicit") == 4 and kinds.count("implicit") == 3 and kinds.count("oneoff") == 2
    for i in SUITE["learning"]:
        assert i["test_task"] in TASKS and i["teaching_turns"] and i["teaching_turns_alt"]


def test_skills_installed_per_condition():
    spdx = TASKS["pref-spdx"]
    on = skill_ab.skills_for(spdx, SUITE, "skills_on")
    assert "license-header" in on and "project-layout" not in on and "docker-build" in on
    assert len(on) == len(SUITE["skills"]) - 1 + len(SUITE["distractors"])
    assert skill_ab.skills_for(spdx, SUITE, "skills_isolated") == {"license-header": SUITE["skills"]["license-header"]}
    assert skill_ab.skills_for(spdx, SUITE, "skills_off") is None


def test_reset_workspace_keeps_learned_state(tmp_path):
    (tmp_path / ".moltevo" / "skills").mkdir(parents=True)
    (tmp_path / "_home").mkdir()
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "src").mkdir()
    skill_ab.reset_workspace(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".moltevo", "_home"]


@pytest.mark.parametrize("answer", [
    "HTTP 404 means the server could not find the requested resource.",
    "It means the page was not found.",
    "404 indicates the requested resource doesn't exist on the server.",
    "404 表示服务器找不到请求的资源。",
])
def test_http_question_accepts_paraphrases(tmp_path, answer):
    assert skill_ab.grade(TASKS["neg-http-question"]["check"], tmp_path, answer)[0]


def test_failure_lesson_fixtures_really_fail_without_the_trick(tmp_path):
    # 确认"失败经验"类任务的坑是真的：不照 Skill 做就会失败，照做就成功
    task = TASKS["fail-gen-env"]
    skill_ab.materialize(task, tmp_path, skills=None)
    assert "GEN_MODE" not in (tmp_path / "tools" / "gen.py").read_text()      # 源码里看不出来，要去文档里找
    env = {k: v for k, v in os.environ.items() if k != "GEN_MODE"}
    plain = subprocess.run([sys.executable, "tools/gen.py"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert plain.returncode == 3 and "registry" in plain.stdout                # 报错信息有误导性
    assert subprocess.run([sys.executable, "tools/gen.py"], cwd=tmp_path, env={**env, "GEN_MODE": "fast"}).returncode == 0
    assert sum("GEN_MODE" in (tmp_path / f).read_text() for f in task["files"] if f.startswith("docs/")) == 1

    task = TASKS["fail-tz"]
    skill_ab.materialize(task, tmp_path / "tz", skills=None)
    run = lambda tz: subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],  # noqa: E731
                                    cwd=tmp_path / "tz", capture_output=True, text=True, env={**os.environ, "TZ": tz})
    assert run("Asia/Shanghai").returncode != 0 and run("UTC").returncode == 0

    task = TASKS["fail-module-entry"]
    skill_ab.materialize(task, tmp_path / "m", skills=None)
    legacy = subprocess.run([sys.executable, "-m", "app"], cwd=tmp_path / "m", capture_output=True, text=True)
    assert legacy.returncode == 0 and "ok" in legacy.stdout                    # 看起来成功了
    assert not skill_ab.grade(task["check"], tmp_path / "m", "")[0]            # 实际没产出
    subprocess.run([sys.executable, "-m", "app.cli"], cwd=tmp_path / "m", check=True)
    assert skill_ab.grade(task["check"], tmp_path / "m", "")[0]
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
    # 漏斗：开 Skill 组在 19 个 Skill 里检索到了目标；假模型没调用 skill 工具
    funnel = summary["funnel"]["skills_on"]
    assert funnel["runs"] == 2 and funnel["retrieval_recall_at_3"] == 1.0 and funnel["invocation_rate"] == 0.0
    assert funnel["retrieval_hit_at_1"] == 1.0 and funnel["retrieval_mrr"] == 1.0
    traj = tmp_path / "trajectories" / "pref-commit-msg-skills_on-1.jsonl"
    types = [__import__("json").loads(l)["type"] for l in traj.read_text().splitlines()]
    assert "skill_retrieved" in types and "model_call" in types and types[-1] == "final_messages"
    report = (tmp_path / "report.md").read_text()
    assert "## preference" in report and "Funnel" in report
    assert skill_ab.report(tmp_path) == report


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
    approved = next(r for r in rows if r["condition"] == "learned_approved")
    gate = next(r for r in rows if r["condition"] == "learned_gate")
    assert approved["skills_promoted"] == ["spdx-header"]
    # 真实门禁：两段教学里同一个候选出现 2 次，自动晋升，不需要模拟批准
    assert gate["skills_promoted"] == ["spdx-header"] and gate["pending_candidates"] == 0
    assert summary["learned_approved"]["passed"] == 1 and summary["learned_gate"]["passed"] == 1
    assert summary["control"]["passed"] == 0
    assert summary["learning"]["promotion_rate"] == {"learned_approved": 1.0, "learned_gate": 1.0}
    assert (tmp_path / "trajectories" / "pref-spdx-learned_gate-1-teach2.jsonl").exists()


def test_one_off_instruction_counts_as_false_learning():
    recs = [{"item_id": "o", "kind": "oneoff", "task_id": "t", "condition": c, "rep": 1, "passed": False, "error": "",
             "skills_promoted": promoted}
            for c, promoted in (("learned_approved", ["wip-commits"]), ("learned_gate", []), ("control", []))]
    s = skill_ab.summarize_learning(recs)
    assert s["learning"]["false_learning_rate"] == {"learned_approved": 1.0, "learned_gate": 0.0}
    assert s["learning"]["promotion_rate"] == {"learned_approved": None, "learned_gate": None}


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


def test_invocations_are_split_into_applicable_and_wrong(tmp_path, monkeypatch):
    # 新建 Python 文件时调用 python-docstrings 是合理的（also_applicable），调用 docker-build 才是误调用
    from agents.agent import Agent
    task = TASKS["pref-spdx"]
    plan = [("skill", {"skill_name": "license-header"}), ("skill", {"skill_name": "python-docstrings"}),
            ("skill", {"skill_name": "docker-build"}),
            ("write_file", {"file_path": "strutil.py", "content": "# SPDX-License-Identifier: MIT\ndef slugify(t):\n    return t\n"})]

    async def fake_stream(self, on_tool_block_complete=None, **k):
        done = sum(1 for m in self._anthropic_messages if m["role"] == "assistant")
        if done < len(plan):
            return anth(tools=[plan[done]])
        return anth(text="done")

    monkeypatch.setattr(Agent, "_call_anthropic_stream", fake_stream)
    rec = asyncio.run(skill_ab.run_task(task, SUITE, condition="skills_on", rep=1, traj_dir=tmp_path,
                                        agent_kwargs={"model": "claude-sonnet-4-6", "api_key": "sk-test",
                                                      "anthropic_base_url": "https://fake.example/anthropic"}))
    assert rec["passed"] and rec["invoked_target"]
    assert rec["applicable_extra_invocations"] == 1 and rec["wrong_invocations"] == 1
