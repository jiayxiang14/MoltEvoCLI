"""Terminal UI rendering — colored output, spinner, tool display."""

from __future__ import annotations

import sys
import threading
import time

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

# ─── Basic output ──────────────────────────────────────────


def _safe_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


_at_line_start = True


def _safe_stdout_write(text: object) -> None:
    global _at_line_start
    text = _safe_text(text)
    sys.stdout.write(text)
    sys.stdout.flush()
    if text:
        _at_line_start = text.endswith("\n")


def mark_line_start() -> None:
    """Tell ui.py the terminal cursor is at column 0.

    Call this after any raw input() call: the terminal itself echoes the
    typed line plus a newline, which ui.py has no way to observe on its own.
    """
    global _at_line_start
    _at_line_start = True


def _ensure_fresh_line() -> None:
    """Reset the cursor to column 0 before drawing a box.

    Streamed assistant text and the spinner write straight to stdout,
    bypassing Rich, so Rich has no idea the cursor may be mid-line. Printing
    a Panel from a mid-line cursor makes its border wrap and never close.
    """
    global _at_line_start
    if not _at_line_start:
        sys.stdout.write("\n")
        sys.stdout.flush()
        _at_line_start = True


MOLT_MARK = r"""
  ▗▖ ▗▖
  ▜▙ ▟▛
    █
  ▗███▖
  ▀▀▀▀▀
"""


def print_welcome() -> None:
    _ensure_fresh_line()
    mark = Text(MOLT_MARK, style="bold #DE7356")
    title = Text("MoltEvo CLI", style="bold #DE7356")
    subtitle = Text("Evolvable Coding Agent CLI", style="white")

    commands = Table.grid(padding=(0, 2))
    commands.add_column(style="bold cyan", no_wrap=True)
    commands.add_column(style="dim")
    commands.add_row("/plan", "read-only planning workflow")
    commands.add_row("/skills", "list reusable skills")
    commands.add_row("/skill-create", "create a reusable skill")
    commands.add_row("/skill-stats", "show skill evolution stats")
    commands.add_row("/memory", "list long-term memories")
    commands.add_row("/compact", "compact current context")
    commands.add_row("exit", "quit the session")

    left = Table.grid()
    left.add_row(Align.center(mark))
    left.add_row(Align.center(title))
    left.add_row(Align.center(subtitle))

    right = Panel(commands, title="Quick Commands", border_style="cyan", box=box.ROUNDED)

    layout = Table.grid(padding=(0, 3))
    layout.add_column(justify="center", vertical="middle")
    layout.add_column(vertical="top")
    layout.add_row(left, right)

    console.print()
    console.print(Panel(
        Align.center(layout),
        title="[bold #DE7356] * READY * [/bold #DE7356]",
        subtitle="[dim]Type your request below[/dim]",
        border_style="#DE7356",
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    console.print()


def print_user_prompt() -> None:
    global _at_line_start
    console.print("\n[bold #DE7356]Molt[/bold #DE7356][bold white]Evo[/bold white] [dim]❯[/dim] ", end="")
    _at_line_start = False


_PATH_EXTENSIONS = (
    r"py|pyi|js|jsx|ts|tsx|mjs|cjs|json|md|mdx|txt|yaml|yml|toml|cfg|ini|"
    r"sh|bash|zsh|css|scss|html|htm|svg|png|jpg|jpeg|gif|webp|csv|tsv|sql|"
    r"rs|go|java|kt|c|cc|cpp|h|hpp|rb|php|xml|env|lock|gitignore|log"
)
# Matches "dir/file.ext" style paths, or a bare "file.ext" with a recognized
# code/doc extension, optionally followed by ":line". Requiring either a
# slash or a known extension keeps this from lighting up things like "e.g."
# or version numbers.
_PATH_PATTERN = (
    r"(?<![\w/])("
    r"(?:[\w.\-]+/)+[\w.\-]+"
    r"|"
    r"[\w\-]+\.(?:" + _PATH_EXTENSIONS + r")"
    r")(?::\d+)?(?![\w])"
)


def print_assistant_text(text: str) -> None:
    global _at_line_start
    text = _safe_text(text)
    styled = Text(text)
    styled.highlight_regex(_PATH_PATTERN, style="bold #AF87FF")
    console.print(styled, end="", soft_wrap=True)
    if text:
        _at_line_start = text.endswith("\n")


def print_tool_call(name: str, inp: dict) -> None:
    _ensure_fresh_line()
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, inp)
    body = Table.grid()
    body.add_row(Text.assemble(("tool  ", "bold #DE7356"), (f"{icon} {name}", "white")))
    if summary:
        body.add_row("")
        body.add_row(Text("input", style="bold #DE7356"))
        body.add_row(Text(_safe_text(summary), style="white"))
    console.print(Panel(
        body,
        title="[bold #DE7356]Tool Call[/bold #DE7356]",
        border_style="#DE7356",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_tool_result(name: str, result: str) -> None:
    _ensure_fresh_line()
    result = _safe_text(result)
    if (name in ("edit_file", "write_file")) and not result.startswith("Error"):
        _print_file_change_result(name, result)
        return
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
    console.print(Panel(
        _safe_text(truncated),
        title=f"[dim]{name} result[/dim]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def _print_file_change_result(_name: str, result: str) -> None:
    _ensure_fresh_line()
    lines = result.split("\n")
    console.print(Panel(
        _safe_text(lines[0]),
        title="[bold green]File Change[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    ))

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
        if not line.strip():
            continue
        if line.startswith("@@"):
            console.print(f"[cyan]  {line}[/cyan]")
        elif line.startswith("- "):
            console.print(f"[red]  {line}[/red]")
        elif line.startswith("+ "):
            console.print(f"[green]  {line}[/green]")
        else:
            console.print(f"[dim]  {line}[/dim]")
    if len(content_lines) > max_display:
        console.print(f"[dim]  ... ({len(content_lines) - max_display} more lines)[/dim]")


def print_error(msg: str) -> None:
    _ensure_fresh_line()
    console.print(Panel(
        _safe_text(msg),
        title="[bold red]Error[/bold red]",
        border_style="red",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_confirmation(command: str) -> None:
    _ensure_fresh_line()
    console.print(Panel(
        _safe_text(command),
        title="[bold #FFC107]Dangerous command[/bold #FFC107]",
        border_style="#FFC107",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_divider() -> None:
    _ensure_fresh_line()
    console.rule("[dim]turn complete[/dim]", style="dim")


def print_cost(input_tokens: int, output_tokens: int) -> None:
    _ensure_fresh_line()
    cost_in = (input_tokens / 1_000_000) * 3
    cost_out = (output_tokens / 1_000_000) * 15
    total = cost_in + cost_out
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="white")
    table.add_row("input", f"{input_tokens} tokens")
    table.add_row("output", f"{output_tokens} tokens")
    table.add_row("estimate", f"${total:.4f}")
    console.print(Panel(table, title="Cost", border_style="cyan", box=box.ROUNDED))


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    _ensure_fresh_line()
    console.print(_safe_text(f"\n  [#FFC107]↻ Retry {attempt}/{max_retries}: {reason}[/#FFC107]"))


def print_info(msg: str) -> None:
    _ensure_fresh_line()
    console.print(Panel(
        _safe_text(msg),
        title="[bold cyan]Info[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_warning(msg: str) -> None:
    _ensure_fresh_line()
    console.print(Panel(
        _safe_text(msg),
        title="[bold #FFC107]Notice[/bold #FFC107]",
        border_style="#FFC107",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_goodbye() -> None:
    _ensure_fresh_line()
    console.print(Panel(
        Text("Bye. Molt saved for next time.", style="bold #DE7356"),
        border_style="#DE7356",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_interrupted() -> None:
    print_warning("Interrupted. Press Ctrl+C again to exit.")


# ─── Spinner ──────────────────────────────────────────────

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()


def start_spinner(label: str = "Thinking") -> None:
    global _spinner_thread
    if _spinner_thread is not None:
        return
    _spinner_stop.clear()

    def _run() -> None:
        frame = 0
        _safe_stdout_write(f"\n  {SPINNER_FRAMES[0]} {label}...")
        while not _spinner_stop.is_set():
            time.sleep(0.08)
            frame = (frame + 1) % len(SPINNER_FRAMES)
            _safe_stdout_write(f"\r  {SPINNER_FRAMES[frame]} {label}...")

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    global _spinner_thread
    if _spinner_thread is None:
        return
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    _safe_stdout_write("\r\033[K")
    global _at_line_start
    _at_line_start = True


# ─── Plan approval display ──────────────────────────────────


def print_plan_for_approval(plan_content: str) -> None:
    _ensure_fresh_line()
    lines = plan_content.split("\n")
    max_lines = 60
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n\n... ({len(lines) - max_lines} more lines)"
    console.print(Panel(
        _safe_text(preview),
        title="[bold cyan]Plan for Approval[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def print_plan_approval_options() -> None:
    _ensure_fresh_line()
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("choice", style="bold #FFC107", no_wrap=True)
    table.add_column("action", style="white")
    table.add_column("detail", style="dim")
    table.add_row("1", "Clear context and execute", "fresh start with auto-accept edits")
    table.add_row("2", "Execute", "keep context, auto-accept edits")
    table.add_row("3", "Manually approve edits", "keep context, confirm each edit")
    table.add_row("4", "Keep planning", "provide feedback to revise")
    console.print(Panel(table, title="[bold #FFC107]Choose an option[/bold #FFC107]", border_style="#FFC107", box=box.ROUNDED))


# ─── Sub-agent display ──────────────────────────────────────


def print_sub_agent_start(agent_type: str, description: str) -> None:
    _ensure_fresh_line()
    console.print(Panel(
        _safe_text(description),
        title=f"[bold #AF87FF]Sub-agent started: {agent_type}[/bold #AF87FF]",
        border_style="#AF87FF",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    _ensure_fresh_line()
    console.print(Panel(
        "completed",
        title=f"[bold #AF87FF]Sub-agent finished: {agent_type}[/bold #AF87FF]",
        border_style="#AF87FF",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_memory_entries(memories: list[object]) -> None:
    _ensure_fresh_line()
    table = Table(box=box.ROUNDED, header_style="bold cyan", border_style="cyan")
    table.add_column("Type", style="bold #DE7356", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Description", style="dim")
    for m in memories:
        table.add_row(
            _safe_text(getattr(m, "type", "")),
            _safe_text(getattr(m, "name", "")),
            _safe_text(getattr(m, "description", "")),
        )
    console.print(Panel(table, title="[bold cyan]Memories[/bold cyan]", border_style="cyan", box=box.ROUNDED))


def print_skill_entries(skills: list[object]) -> None:
    _ensure_fresh_line()
    table = Table(box=box.ROUNDED, header_style="bold cyan", border_style="cyan")
    table.add_column("Skill", style="bold #DE7356", no_wrap=True)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Mode", style="magenta", no_wrap=True)
    table.add_column("Description", style="white")
    for s in skills:
        name = getattr(s, "name", "")
        tag = f"/{name}" if getattr(s, "user_invocable", False) else name
        table.add_row(
            _safe_text(tag),
            _safe_text(getattr(s, "source", "")),
            _safe_text(getattr(s, "context", "")),
            _safe_text(getattr(s, "description", "")),
        )
    console.print(Panel(table, title="[bold cyan]Skills[/bold cyan]", border_style="cyan", box=box.ROUNDED))


# ─── Tool icons and summaries ───────────────────────────────

_TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "🔧",
    "list_files": "📁",
    "grep_search": "🔍",
    "run_shell": "💻",
    "skill": "⚡",
    "skill_create": "🍪",
    "skill_evolve": "🧬",
    "agent": "🤖",
}


def _get_tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔨")


def _get_tool_summary(name: str, inp: dict) -> str:
    if name == "read_file":
        return inp.get("file_path", "")
    if name == "write_file":
        return inp.get("file_path", "")
    if name == "edit_file":
        return inp.get("file_path", "")
    if name == "list_files":
        return inp.get("pattern", "")
    if name == "grep_search":
        return f'"{inp.get("pattern", "")}" in {inp.get("path", ".")}'
    if name == "run_shell":
        cmd = inp.get("command", "")
        return cmd[:60] + "..." if len(cmd) > 60 else cmd
    if name == "skill":
        return inp.get("skill_name", "")
    if name == "skill_create":
        return inp.get("name", "")
    if name == "skill_evolve":
        return inp.get("skill_name", "")
    if name == "agent":
        return f'[{inp.get("type", "general")}] {inp.get("description", "")}'
    return ""
