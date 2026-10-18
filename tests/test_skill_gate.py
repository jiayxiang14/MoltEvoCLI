"""第五批 5a：Skill 候选区、晋升门禁、手写 Skill 只读、新的淘汰规则。"""

import asyncio
import builtins
import json
from pathlib import Path

import pytest

from agents import skill_candidates as sc
from agents import skills as skills_mod
from agents.skill_evolution import record_skill_usage_judgments
from agents.tools import check_permission
from fakemodel import make_agent


def run(coro):
    return asyncio.run(coro)


CANDIDATE = {"name": "commit-style", "description": "Write commit messages in Chinese",
             "when_to_use": "when committing", "instructions": "先写改了什么，再写为什么。",
             "evidence": "user said so", "tags": ["git"]}


@pytest.fixture(autouse=True)
def project(workdir):
    skills_mod.reset_skill_cache()
    yield workdir
    skills_mod.reset_skill_cache()


def fake_side_query(decision):
    async def sq(system, user):
        if "Skill Extractor" in system:
            return json.dumps({"skills": [CANDIDATE]})
        return json.dumps(decision)
    return sq


def evolve(a, decision, *, window="w1", session=None):
    a._build_side_query = lambda **kw: fake_side_query(decision)
    if session:
        a.session_id = session
    w = {"messages": [{"role": "user", "content": "please commit in Chinese"}],
         "latest_user": window, "latest_assistant": "ok"}
    run(a._run_online_skill_evolution(w))


def active_names():
    skills_mod.reset_skill_cache()
    return {s.name for s in skills_mod.discover_skills()}


def write_manual_skill(root: Path, name="team-rules", body="Always run make lint."):
    d = root / ".moltevo" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: team rules\n---\n{body}\n")
    skills_mod.reset_skill_cache()


def test_candidate_is_staged_not_activated(project):
    a = make_agent(permission_mode="acceptEdits")
    evolve(a, {"action": "add"})
    assert "commit-style" not in active_names()
    assert [r["name"] for r in sc.pending()] == ["commit-style"]


def test_auto_promotes_after_second_occurrence_in_accept_edits(project):
    a = make_agent(permission_mode="acceptEdits")
    evolve(a, {"action": "add"}, window="w1")
    evolve(a, {"action": "add"}, window="w1")          # 同一窗口重复处理不计数
    assert "commit-style" not in active_names()
    evolve(a, {"action": "add"}, window="w2")
    assert "commit-style" in active_names()
    skill = next(s for s in skills_mod.discover_skills() if s.name == "commit-style")
    assert sc.skill_author(skill.skill_dir) == "agent"


def test_default_mode_never_auto_promotes(project, capsys):
    a = make_agent(permission_mode="default")
    evolve(a, {"action": "add"}, window="w1")
    evolve(a, {"action": "add"}, window="w2")
    assert "commit-style" not in active_names()
    assert "/skills review" in capsys.readouterr().out


def test_merge_into_user_written_skill_needs_approval(project):
    write_manual_skill(project)
    a = make_agent(permission_mode="bypassPermissions")
    decision = {"action": "merge", "target_skill": "team-rules", "merged_instructions": "Always run make lint and make test."}
    evolve(a, decision, window="w1")
    evolve(a, decision, window="w2")
    text = (project / ".moltevo" / "skills" / "team-rules" / "SKILL.md").read_text()
    assert "make test" not in text                         # 手写 Skill 没有被自动改写
    record = sc.pending()[0]
    assert record["target_is_manual"]
    assert sc.approve(record["key"])["ok"]
    assert "make test" in (project / ".moltevo" / "skills" / "team-rules" / "SKILL.md").read_text()


def test_static_check_blocks_secrets_even_with_approval(project):
    rec = sc.stage(action="add", name="leaky", target_skill="", description="d", when_to_use="",
                   instructions="use api_key=sk-abcdefghijklmnopqrstu", tags=[], evidence="", reason="",
                   session_id="s", window_id="w", target_is_manual=False)
    assert "secret" in rec["problems"][0]
    assert not sc.approve(rec["key"])["ok"]


def test_extract_now_promotes_on_explicit_approval(project, monkeypatch):
    a = make_agent(permission_mode="default")

    async def yes(m, **kw):
        return "y"
    a.set_confirm_fn(yes)
    a._build_side_query = lambda **kw: fake_side_query({"action": "add"})
    a._pending_skill_extraction_window = {"messages": [{"role": "user", "content": "x"}],
                                          "latest_user": "x", "latest_assistant": "y"}
    run(a.extract_now())
    assert "commit-style" in active_names()


def test_repl_review_approve_and_reject(project, monkeypatch):
    for name in ("one", "two"):
        sc.stage(action="add", name=name, target_skill="", description=f"{name} desc", when_to_use="",
                 instructions=f"do {name}", tags=[], evidence="", reason="", session_id="s",
                 window_id="w", target_is_manual=False)
    answers = iter(["a", "r"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    from agents import main as main_mod
    main_mod._review_skill_candidates(make_agent())
    assert "one" in active_names() and "two" not in active_names()
    assert sc.pending() == []


def test_stale_candidates_cleaned(project):
    rec = sc.stage(action="add", name="old", target_skill="", description="d", when_to_use="",
                   instructions="i", tags=[], evidence="", reason="", session_id="s", window_id="w",
                   target_is_manual=False)
    assert sc.cleanup(now=rec["updated"] + 31 * 86400) == 1 and sc.pending() == []


def test_skill_evolve_on_user_written_skill_asks_even_in_accept_edits(project):
    write_manual_skill(project)
    d = check_permission("skill_evolve", {"skill_name": "team-rules", "lesson": "x"}, "acceptEdits")
    assert d["action"] == "confirm" and "user-written" in d["message"]


def _judge(name, skill_dir, used):
    return {"name": name, "source": "project", "skill_dir": str(skill_dir), "relevant": True, "used": used}


def test_prune_agent_skill_but_only_flag_user_written(project):
    a = make_agent(permission_mode="acceptEdits")
    evolve(a, {"action": "add"}, window="w1")
    evolve(a, {"action": "add"}, window="w2")
    agent_dir = next(s for s in skills_mod.discover_skills() if s.name == "commit-style").skill_dir
    write_manual_skill(project)
    manual_dir = project / ".moltevo" / "skills" / "team-rules"
    for _ in range(15):
        record_skill_usage_judgments([_judge("commit-style", agent_dir, False), _judge("team-rules", manual_dir, False)])
    assert not Path(agent_dir).exists()                    # agent 产生的被移到 pruned/
    assert manual_dir.exists()                             # 手写的保留
    stats = json.loads((project / ".moltevo" / "skill-evolution" / "skill_usage_stats.json").read_text())
    assert stats["team-rules"]["flagged_stale"] is True
