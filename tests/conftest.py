"""
测试环境隔离：在导入 agents 之前把 HOME 指到临时目录。

agents 里不少路径常量（会话、记忆、事件日志）在导入时就按 Path.home() 求值，
所以必须在这里、而不是在单个测试里改 HOME，才能保证不碰到真实的用户目录。
"""

import os
import sys
import tempfile
from pathlib import Path

_TEST_HOME = tempfile.mkdtemp(prefix="moltevo-test-home-")
os.environ["HOME"] = _TEST_HOME
for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL"):
    os.environ.pop(var, None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """每个测试一个干净的工作目录。"""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
