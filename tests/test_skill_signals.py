"""第五批 5b：执行轨迹作为学习信号、Skill 使用后的效果记录、与反馈类记忆去重。"""

import asyncio
import json

import pytest

from agents import skills as skills_mod
from agents.memory import save_memory
from fakemodel import anth, install, make_agent


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def project(workdir):
    skills_mod.reset_skill_cache()
    yield workdir
    skills_mod.reset_skill_cache()


def capture_side_query(feedback="neutral"):
    seen = []

    async def sq(system, user):
        seen.append((system, json.loads(user)))
        if "Skill Extractor" in system:
            return json.dumps({"skills": [], "feedback": feedback})
        return json.dumps({"action": "discard"})
    return sq, seen


def test_window_carries_execution_trace_with_failure_then_success(project, monkeypatch):
    a = make_agent()
    install(monkeypatch, [anth(tools=[("run_shell", {"command": "python -m pytest -q zzz_missing"})]),
                          anth(tools=[("run_shell", {"command": "echo ok"})]),
                          anth(text="fixed")])
    run(a.chat("run the tests"))
    # 假模型不走流式文字输出，这里用本轮记录下的轨迹直接建立抽取窗口
    a._set_pending_skill_extraction_window(original_user_message="run the tests", assistant_text="fixed",
                                           retrieved_reference=None)
    trace = a._pending_skill_extraction_window["execution_trace"]
    assert [t["is_error"] for t in trace] == [True, False]
    assert trace[0]["exit_code"] != 0 and "pytest" in trace[0]["args"]


def test_extraction_receives_trace_and_feedback_memories(project):
    save_memory("prefer-chinese-commits", "commit messages in Chinese", "feedback", "Use Chinese commit messages.")
    a = make_agent()
    sq, seen = capture_side_query()
    a._build_side_query = lambda **kw: sq
    window = {"messages": [{"role": "user", "content": "x"}], "latest_user": "x", "latest_assistant": "y",
              "execution_trace": [{"tool": "run_shell", "args": "make test", "is_error": True, "exit_code": 2,
                                   "infrastructure": False},
                                  {"tool": "run_shell", "args": "PYTHONPATH=src make test", "is_error": False,
                                   "exit_code": 0, "infrastructure": False}]}
    run(a._run_online_skill_evolution(window))
    payload = next(p for s, p in seen if "Skill Extractor" in s)
    assert len(payload["execution_trace"]) == 2
    assert payload["existing_feedback_memories"][0]["name"] == "prefer-chinese-commits"
    system = next(s for s, p in seen if "Skill Extractor" in s)
    assert "execution_trace" in system and "existing_feedback_memories" in system


def _stats(project):
    return json.loads((project / ".moltevo" / "skill-evolution" / "skill_usage_stats.json").read_text())


def _agent_skill(project, name="flaky-advice"):
    d = project / ".moltevo" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\nauthor: agent\n---\nbody\n")
    skills_mod.reset_skill_cache()
    return d


def _window(used, trace):
    return {"messages": [{"role": "user", "content": "x"}], "latest_user": "x", "latest_assistant": "y",
            "used_skills": used, "execution_trace": trace}


def test_outcome_recorded_and_infrastructure_errors_excluded(project):
    _agent_skill(project)
    a = make_agent()
    sq, _ = capture_side_query(feedback="corrected")
    a._build_side_query = lambda **kw: sq
    trace = [{"tool": "read_file", "args": "", "is_error": False, "exit_code": None, "infrastructure": False},
             {"tool": "mcp__x__y", "args": "", "is_error": True, "exit_code": None, "infrastructure": True},
             {"verify": "pytest", "passed": False}]
    run(a._run_online_skill_evolution(_window(["flaky-advice"], trace)))
    item = _stats(project)["flaky-advice"]
    assert item["outcomes"] == 1 and item["corrected"] == 1
    assert item["tool_error_rate_sum"] == 0.0            # MCP 不可用不算工具失败
    assert item["verify_failed"] == 1


def test_often_corrected_agent_skill_is_pruned(project):
    d = _agent_skill(project)
    a = make_agent()
    sq, _ = capture_side_query(feedback="corrected")
    a._build_side_query = lambda **kw: sq
    for i in range(5):
        w = _window(["flaky-advice"], [])
        w["latest_user"] = f"x{i}"
        run(a._run_online_skill_evolution(w))
    assert not d.exists()


def test_used_skills_attached_after_usage_judgment(project, monkeypatch):
    _agent_skill(project, "commit-style")
    a = make_agent()

    async def judge(**kw):
        return [{"name": "commit-style", "source": "project", "skill_dir": "", "retrieved": True,
                 "relevant": True, "used": True, "score": 1.0, "reason": ""}]
    monkeypatch.setattr("agents.online_skill_evolution.judge_retrieved_skill_usage", judge)
    a._last_retrieved_skill_hits = [{"name": "commit-style"}]
    a._pending_skill_extraction_window = {"latest_assistant": "reply", "used_skills": []}
    run(a._run_skill_usage_tracking("msg", "reply"))
    assert a._pending_skill_extraction_window["used_skills"] == ["commit-style"]
