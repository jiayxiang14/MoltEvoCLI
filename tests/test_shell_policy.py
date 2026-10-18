"""shell 命令策略（白名单 + 解析全部命令名 + 越界写入 + 敏感路径）与文件工具的项目边界。"""

import json
from pathlib import Path

import pytest

from agents import tools
from agents.memory import get_memory_dir
from agents.tools import check_permission


def shell(cmd, mode="default"):
    return check_permission("run_shell", {"command": cmd}, mode)


@pytest.fixture(autouse=True)
def fresh_rules(workdir):
    tools.reset_permission_cache()
    yield
    tools.reset_permission_cache()


# ── 放行 ────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "ls -la",
    "cat README.md | grep foo | wc -l",
    "git status && git diff --stat",
    "git log --oneline -5; git show HEAD",
    "pytest -q tests/",
    "python -m pytest -x",
    "python script.py --flag",
    "npm test",
    "npm install",
    "FOO=1 BAR=2 make build",
    "echo hello > out.txt",
    "mkdir -p build/out && cp a.txt build/out/",
    "find . -name '*.py' | xargs grep -n TODO",
    "time pytest",
    "sed -n '1,20p' file.py",
])
def test_allowed(cmd):
    assert shell(cmd)["action"] == "allow", cmd


# ── 需要确认 ─────────────────────────────────────────────

@pytest.mark.parametrize("cmd,why", [
    ("curl https://example.com", "curl"),
    ("ls && curl https://example.com", "curl"),
    ("echo $(curl -s https://x)", "curl"),
    ("echo `wget -qO- https://x`", "wget"),
    ("(cd /tmp && nc -l 8080)", "nc"),
    ("python -c 'import os; os.system(\"x\")'", "inline code"),
    ("node -e 'require(\"fs\")'", "inline code"),
    ("bash -c 'ls'", "bash"),
    ("sh install.sh", "sh"),
    ("./run.sh", "run.sh"),
    ("find . -name '*.tmp' -exec rm {} \\;", "find"),      # 命中危险命令规则或 find -exec 校验
    ("find . -delete", "find"),
    ("pip install requests", "global install"),
    ("npm install -g typescript", "global install"),
    ("git push origin main", "git push"),
    ("git reset --hard HEAD~1", "git reset"),
    ("echo data > /etc/hosts", "outside the project"),
    ("cp secrets.txt ~/Desktop/", "outside the project"),
    ("mv build /tmp/build", "outside the project"),
    ("tee /usr/local/x < in.txt", "outside the project"),
    ("sed -i 's/a/b/' /etc/profile", "outside the project"),
    ("cat ~/.ssh/id_rsa", "sensitive"),
    ("grep -r key ~/.aws", "sensitive"),
    ("cat .env", "sensitive"),
    ("rm -rf build", "dangerous"),
    ("ls 'unterminated", "could not parse"),
])
def test_needs_confirmation(cmd, why):
    d = shell(cmd)
    assert d["action"] == "confirm", cmd
    assert why in d["message"], (cmd, d["message"])


def test_dont_ask_denies_and_bypass_allows():
    assert shell("curl https://x", "dontAsk")["action"] == "deny"
    assert shell("curl https://x", "bypassPermissions")["action"] == "allow"
    assert shell("ls", "dontAsk")["action"] == "allow"


def test_plan_mode_still_blocks_shell():
    assert shell("ls", "plan")["action"] == "deny"


def test_settings_can_extend_allowlist(workdir):
    (workdir / ".moltevo").mkdir()
    (workdir / ".moltevo" / "settings.json").write_text(json.dumps({"allowed_commands": ["curl"]}))
    tools.reset_permission_cache()
    assert shell("curl https://example.com")["action"] == "allow"


def test_allow_rule_still_takes_precedence(workdir):
    (workdir / ".moltevo").mkdir()
    (workdir / ".moltevo" / "settings.json").write_text(json.dumps({"permissions": {"allow": ["run_shell(./deploy.sh*)"]}}))
    tools.reset_permission_cache()
    assert shell("./deploy.sh staging")["action"] == "allow"


# ── 文件工具的项目边界与敏感路径 ─────────────────────────────

@pytest.mark.parametrize("mode", ["default", "acceptEdits"])
def test_write_outside_project_needs_confirmation_even_in_accept_edits(workdir, mode):
    target = Path.home() / "outside-project.txt"
    d = check_permission("write_file", {"file_path": str(target), "content": "x"}, mode)
    assert d["action"] == "confirm" and "outside the project" in d["message"]
    assert check_permission("write_file", {"file_path": str(target), "content": "x"}, "dontAsk")["action"] == "deny"


def test_write_inside_project_unchanged_in_accept_edits(workdir):
    (workdir / "a.txt").write_text("1")
    assert check_permission("edit_file", {"file_path": "a.txt", "old_string": "1", "new_string": "2"},
                            "acceptEdits")["action"] == "allow"


def test_memory_and_plan_dirs_are_allowed_write_roots(workdir):
    mem = get_memory_dir() / "note.md"
    assert check_permission("write_file", {"file_path": str(mem), "content": "x"}, "acceptEdits")["action"] == "allow"
    plan = Path.home() / ".moltevo" / "plans" / "plan-x.md"
    assert check_permission("write_file", {"file_path": str(plan), "content": "x"}, "acceptEdits")["action"] == "allow"


@pytest.mark.parametrize("tool,inp", [
    ("read_file", {"file_path": "~/.ssh/id_rsa"}),
    ("read_file", {"file_path": ".env"}),
    ("grep_search", {"pattern": "secret", "path": "~/.aws"}),
])
def test_sensitive_reads_need_confirmation(workdir, tool, inp):
    d = check_permission(tool, inp, "default")
    assert d["action"] == "confirm" and "sensitive" in d["message"]


def test_normal_reads_still_allowed(workdir):
    assert check_permission("read_file", {"file_path": "src/app.py"}, "default")["action"] == "allow"
    assert check_permission("read_file", {"file_path": ".env.example"}, "default")["action"] == "allow"
