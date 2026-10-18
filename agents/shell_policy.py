"""
shell 命令策略：解析出命令里的每一个命令名，白名单之外的、危险的、越界写入的、触及敏感路径的都要确认。

思路参照 anthropics/claude-quickstarts 的 autonomous-coding/security.py：
用 shlex 解析管道、&&、||、;、子 shell、$(...)、反引号里的全部命令名，逐个检查；解析失败按"需要确认"处理。

局限（需要在文档里说清楚）：白名单挡得住误操作，挡不住恶意——python、node、make 这些必须放行的命令
本身就能执行任意代码。真正的隔离要靠沙箱（容器或系统级 sandbox）。
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

# 只读命令和常见开发命令。可在 settings.json 的 "allowed_commands" 里扩充。
DEFAULT_ALLOWED_COMMANDS = {
    # 查看与文本处理
    "ls", "cat", "head", "tail", "less", "more", "wc", "grep", "egrep", "fgrep", "rg", "find", "fd", "tree",
    "pwd", "echo", "printf", "which", "type", "file", "stat", "du", "df", "sort", "uniq", "cut", "tr",
    "sed", "awk", "diff", "cmp", "comm", "basename", "dirname", "realpath", "readlink", "date", "true",
    "false", "test", "[", "jq", "yq", "column", "nl", "od", "xxd", "md5", "md5sum", "shasum", "sha256sum",
    # 文件操作（目标路径另行检查是否在项目内）
    "mkdir", "touch", "cp", "mv", "tee", "ln",
    # 版本控制（子命令另行检查）
    "git",
    # 语言与构建工具
    "python", "python3", "pytest", "pip", "pip3", "uv", "poetry", "ruff", "black", "mypy", "isort",
    "node", "npm", "npx", "yarn", "pnpm", "tsc", "eslint", "prettier", "vitest", "jest",
    "make", "cmake", "cargo", "rustc", "go", "gofmt", "javac", "java", "mvn", "gradle",
    # 进程与等待
    "sleep", "ps",
}

# 包裹其他命令的"外壳"：继续检查它包裹的那个命令。
WRAPPERS = {"env", "nohup", "time", "nice", "timeout", "xargs", "command", "builtin", "exec"}

GIT_SAFE_SUBCOMMANDS = {
    "status", "diff", "log", "show", "branch", "add", "commit", "restore", "stash", "fetch", "rev-parse",
    "ls-files", "blame", "grep", "switch", "checkout", "init", "tag", "remote", "config", "describe",
    "shortlog", "reflog", "merge-base", "cat-file", "worktree", "mv", "rm", "apply", "format-patch",
}

INLINE_CODE_FLAGS = {"python": {"-c"}, "python3": {"-c"}, "node": {"-e", "--eval", "-p", "--print"}}

# 会写文件的命令：哪些参数是写入目标
WRITE_TARGET_COMMANDS = {"cp", "mv", "ln"}          # 最后一个参数是目标
WRITE_ALL_ARGS_COMMANDS = {"touch", "mkdir", "tee"}  # 所有非选项参数都是目标

SENSITIVE_PATTERNS = [
    re.compile(r"(^|/)\.ssh(/|$)"), re.compile(r"(^|/)\.aws(/|$)"), re.compile(r"(^|/)\.gnupg(/|$)"),
    re.compile(r"(^|/)\.kube(/|$)"), re.compile(r"(^|/)\.docker/config\.json$"),
    re.compile(r"(^|/)\.netrc$"), re.compile(r"(^|/)\.npmrc$"), re.compile(r"(^|/)\.pypirc$"),
    re.compile(r"(^|/)\.env(\.[^/]*)?$"), re.compile(r"(^|/)id_(rsa|ed25519|ecdsa|dsa)(\.pub)?$"),
    re.compile(r"\.pem$"), re.compile(r"\.key$"),
]
SENSITIVE_ALLOWED_SUFFIXES = (".env.example", ".env.sample", ".env.template")

_OPERATORS = {";", "&&", "||", "|", "&", "|&", "(", ")", "{", "}", "\n"}
_REDIRECTS = {">", ">>", ">|", "&>", "&>>", "<", "<<", "<<<"}
_KEYWORDS = {"if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
             "in", "!", "function", "select"}


class ParseError(Exception):
    pass


def is_sensitive_path(raw: str) -> bool:
    path = os.path.expanduser(raw.strip("'\""))
    if path.endswith(SENSITIVE_ALLOWED_SUFFIXES):
        return False
    return any(p.search(path) for p in SENSITIVE_PATTERNS)


def is_within(raw: str, roots: list[Path]) -> bool:
    """路径是否落在任一允许写入的根目录内（相对路径按当前目录解析）。"""
    try:
        target = Path(os.path.expanduser(raw)).resolve()
    except (OSError, RuntimeError):
        return False
    if str(target) in ("/dev/null", "/dev/stdout", "/dev/stderr"):
        return True
    return any(target == r or target.is_relative_to(r) for r in roots)


def _substitutions(command: str) -> list[str]:
    """取出 $(...) 和反引号里的子命令（支持一层嵌套的括号匹配）。"""
    found = re.findall(r"`([^`]*)`", command)
    i = 0
    while True:
        start = command.find("$(", i)
        if start < 0:
            break
        depth, j = 1, start + 2
        while j < len(command) and depth:
            depth += {"(": 1, ")": -1}.get(command[j], 0)
            j += 1
        if depth:
            raise ParseError("unbalanced $(")
        found.append(command[start + 2:j - 1])
        i = j
    return found


def split_simple_commands(command: str) -> list[list[str]]:
    """把整条命令拆成若干"简单命令"（各自的 token 列表），包括命令替换里的子命令。"""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>{}")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError as e:
        raise ParseError(str(e)) from None
    commands: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok in _OPERATORS:
            if current:
                commands.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        commands.append(current)
    for inner in _substitutions(command):
        commands += split_simple_commands(inner)
    return commands


def _strip_prefix(tokens: list[str]) -> list[str]:
    """去掉开头的 shell 关键字和 VAR=value 赋值。"""
    i = 0
    while i < len(tokens) and (tokens[i] in _KEYWORDS or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i])):
        i += 1
    return tokens[i:]


def _unwrap(tokens: list[str]) -> list[list[str]]:
    """展开 env / nohup / time / xargs 等外壳：返回外壳本身和被包裹的命令。"""
    out = [tokens]
    name = os.path.basename(tokens[0])
    if name in WRAPPERS:
        rest = tokens[1:]
        if name == "timeout" and rest and not rest[0].startswith("-"):
            rest = rest[1:]                       # timeout 的时长参数
        while rest and (rest[0].startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", rest[0])):
            rest = rest[1:]
        if rest:
            out += _unwrap(rest)
    return out


def evaluate(command: str, *, allowed: set[str], write_roots: list[Path],
             dangerous: bool) -> tuple[str, str]:
    """返回 ("allow" | "confirm", 原因)。dangerous 由调用方用现有危险命令规则算好传入。"""
    if dangerous:
        return "confirm", f"dangerous command: {command}"
    try:
        simple = split_simple_commands(command)
    except ParseError as e:
        return "confirm", f"could not parse command ({e}): {command}"

    reasons: list[str] = []
    for raw in simple:
        tokens = _strip_prefix(raw)
        if not tokens:
            continue
        # 重定向目标：越界写入、读取敏感文件
        for idx, tok in enumerate(tokens):
            if tok in _REDIRECTS and idx + 1 < len(tokens):
                target = tokens[idx + 1]
                if tok.startswith(">") or tok.startswith("&>"):
                    if not is_within(target, write_roots):
                        reasons.append(f"writes outside the project: {target}")
                if is_sensitive_path(target):
                    reasons.append(f"touches a sensitive path: {target}")
        words = [t for i, t in enumerate(tokens)
                 if t not in _REDIRECTS and not (i > 0 and tokens[i - 1] in _REDIRECTS)]
        if not words:
            continue
        for part in _unwrap(words):
            name = os.path.basename(part[0])
            args = part[1:]
            if name in WRAPPERS and len(part) > 1:
                continue                          # 外壳本身放行，检查的是它包裹的命令
            if name not in allowed:
                reasons.append(f"command not on the allowlist: {name}")
                continue
            if any(is_sensitive_path(a) for a in args if not a.startswith("-")):
                reasons.append(f"touches a sensitive path ({name})")
            if name in INLINE_CODE_FLAGS and any(a in INLINE_CODE_FLAGS[name] for a in args):
                reasons.append(f"runs inline code ({name})")
            if name == "find" and any(a in ("-exec", "-execdir", "-ok", "-okdir") for a in args):
                reasons.append("find -exec runs other commands")
            if name == "find" and "-delete" in args:
                reasons.append("find -delete removes files")
            if name in ("pip", "pip3", "uv", "poetry") and any(a in ("install", "uninstall", "add", "remove") for a in args):
                reasons.append(f"global install changes the environment ({name})")
            if name in ("npm", "yarn", "pnpm") and any(a in ("-g", "--global") for a in args):
                reasons.append(f"global install changes the environment ({name})")
            if name == "git":
                sub = next((a for a in args if not a.startswith("-")), "")
                if sub and sub not in GIT_SAFE_SUBCOMMANDS:
                    reasons.append(f"git {sub}")
            if name == "sed" and any(a.startswith("-i") or a == "--in-place" for a in args):
                files = [a for a in args if not a.startswith("-")][1:]
                outside = [f for f in files if not is_within(f, write_roots)]
                if outside:
                    reasons.append(f"writes outside the project: {', '.join(outside)}")
            positional = [a for a in args if not a.startswith("-")]
            if name in WRITE_TARGET_COMMANDS and len(positional) >= 2:
                if not is_within(positional[-1], write_roots):
                    reasons.append(f"writes outside the project: {positional[-1]}")
                if name == "mv":
                    outside_src = [s for s in positional[:-1] if not is_within(s, write_roots)]
                    if outside_src:
                        reasons.append(f"moves files from outside the project: {', '.join(outside_src)}")
            if name in WRITE_ALL_ARGS_COMMANDS:
                outside = [a for a in positional if not is_within(a, write_roots)]
                if outside:
                    reasons.append(f"writes outside the project: {', '.join(outside)}")
    if reasons:
        unique = list(dict.fromkeys(reasons))
        return "confirm", f"{command}\n  ({'; '.join(unique)})"
    return "allow", ""
