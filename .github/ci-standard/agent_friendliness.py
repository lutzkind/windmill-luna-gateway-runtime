#!/usr/bin/env python3
"""Canonical agent-friendliness contract for a repository checkout.

This file is the single implementation of the lightweight agent-friendliness
contract. ``lutzkind/github-actions-standard`` renders it into every managed
repository as ``.github/ci-standard/agent_friendliness.py`` and runs it as a
pull-request check; the account-wide reconciler imports the same functions for
its per-repository findings. Edit the canonical source in the standard
repository, never a rendered copy.

The contract is deliberately static and cheap: it reads the checkout and
reports precise drift findings. It never executes repository code, never talks
to the network, and never rewrites anything.

Checks, where applicable:

* ``AGENTS.md`` exists at the repository root, stays compact, and keeps the
  ``## Start here`` / ``## Commands`` / ``## Production`` sections;
* it keeps the canonical-state guidance (``git status``, ``main``/production);
* it points at the host map (``/root/REPO_MAP.md``) and at real docs;
* documented commands in ``## Commands`` still match the repository manifests
  (package scripts, test paths, script files);
* non-canonical paths are only mentioned with an explicit warning;
* repositories with Windmill sources point at ``windmill-production/OWNERS.yaml``.

A repository whose agent entry point legitimately uses another convention may
declare an exemption inside ``AGENTS.md``:

``<!-- ci-standard agent-friendliness: exempt (reason) -->``
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex
import subprocess
from pathlib import Path

MAX_AGENTS_LINES = 400
MAX_AGENTS_BYTES = 30000
REQUIRED_SECTIONS = ("Start here", "Commands", "Production")
REPO_MAP_POINTER = "/root/REPO_MAP.md"
NON_CANONICAL_PATHS = (
    "/root/agent-tmp",
    "/root/mcp-shared/chatgpt",
    "/root/backups",
    "/root/ci-selfhosted-policy",
)
WARNING_MARKERS = re.compile(
    r"(?i)(do not|don't|never|not the source|not canonical|non-canonical|not proof|"
    r"history|avoid|exclude|disposable|scratch|not deployable|retired|archived)"
)
EXEMPT_MARKER = re.compile(r"(?i)ci-standard\s+agent-friendliness\s*:\s*exempt")
DOC_POINTER = re.compile(r"(README\.md|ARCHITECTURE\.md|HANDOFF\.md|ROADMAP\.md|DECISIONS\.md|docs/)")
WINDMILL_SOURCE = re.compile(r"(?:^|/)windmill/.*(?:\.py|\.script\.lock|\.schedule\.ya?ml)$")
OWNERS_POINTER = re.compile(r"OWNERS\.yaml|windmill-production")

NPM_COMMAND = re.compile(
    r"\bnpm\s+(?:--prefix\s+(?P<pre>\S+)\s+)?(?:run\s+(?P<script>[A-Za-z0-9:_-]+)|(?P<test>test))\b"
    r"(?:\s+--prefix\s+(?P<post>\S+))?"
)
NODE_TEST = re.compile(r"\bnode\s+--test\b(?P<args>.*)$")
UNITTEST_DISCOVER = re.compile(r"\bpython3?\s+-m\s+unittest\s+discover\s+-s\s+(?P<dir>\S+)")
UNITTEST = re.compile(r"\bpython3?\s+-m\s+unittest\s+(?P<args>.+)$")
PYTEST = re.compile(r"(?:\bpython3?\s+-m\s+)?\bpytest\b(?P<args>.*)$")
SCRIPT_PATH = re.compile(
    r"(?<![\w./-])((?:scripts|bin|tests|test|src|apps|tools)/[A-Za-z0-9_./-]+\.(?:py|js|mjs|cjs|ts|tsx|sh))"
)
NODE_VALUE_FLAGS = {
    "--import",
    "--loader",
    "--require",
    "-r",
    "--test-reporter",
    "--test-reporter-destination",
    "--test-name-pattern",
    "--test-concurrency",
}
PYTEST_VALUE_FLAGS = {"-k", "-m", "-p", "--rootdir", "--maxfail", "-o", "--junitxml"}


def _placeholder(token: str) -> bool:
    return (
        not token
        or token in {"...", "\u2026"}
        or token.startswith(("<", "$", "{", "["))
        or "TODO" in token.upper()
        or "..." in token
    )


def _glob_matches(pattern: str, paths: set[str]) -> bool:
    if any(character in pattern for character in "*?["):
        return any(fnmatch.fnmatchcase(path, pattern) for path in paths)
    return pattern in paths


def _section(text: str, title: str) -> str | None:
    match = re.search(rf"(?ms)^##\s+{re.escape(title)}\b(.*?)(?=^##\s|\Z)", text)
    return match.group(1) if match else None


def _context_blocks(text: str):
    """Yield ``(current_heading, block)`` pairs for scoped context checks.

    A block is one list item, table row, fenced line, or paragraph: blank
    lines and new list/table markers start a new block, while wrapped
    continuation lines stay with the item they belong to.
    """
    heading = ""
    block: list[str] = []

    def flush():
        nonlocal block
        if block:
            yield heading, "\n".join(block)
            block = []

    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("##"):
            yield from flush()
            heading = line
            continue
        if not stripped:
            yield from flush()
            continue
        starts_item = not line[:1].isspace() and stripped.startswith(("-", "*", "|", ">"))
        if starts_item and block:
            yield from flush()
        block.append(line)
    yield from flush()


def _commands_in_section(section: str) -> list[str]:
    """Extract declared commands from table rows, fenced blocks, and list items.

    Prose paragraphs are intentionally excluded so a note that merely mentions
    a command (for example "some sample projects have an ``npm test`` that
    exits non-zero by design") is not treated as a declaration.
    """
    commands: list[str] = []
    for block in re.findall(r"(?ms)^```[^\n]*\n(.*?)^```", section):
        commands.extend(line.strip() for line in block.splitlines() if line.strip() and not line.strip().startswith("#"))
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") or stripped.startswith(("-", "*")):
            commands.extend(span.strip() for span in re.findall(r"`([^`\n]+)`", stripped) if span.strip())
    return list(dict.fromkeys(commands))


def _resolve_module(token: str, paths: set[str]) -> bool:
    parts = token.split(".")
    for length in range(len(parts), 0, -1):
        candidate = "/".join(parts[:length])
        if f"{candidate}.py" in paths or f"{candidate}/__init__.py" in paths:
            return True
    return False


def _command_findings(commands: list[str], paths: set[str], file_reader) -> list[str]:
    findings: list[str] = []

    def missing_path(command: str, token: str) -> None:
        findings.append(
            f"AGENTS.md: documented command '{command}' references missing path '{token}'"
        )

    for command in commands:
        match = NPM_COMMAND.search(command)
        if match:
            prefix = (match.group("post") or match.group("pre") or "").strip("/")
            script = match.group("script") or match.group("test")
            manifest = f"{prefix}/package.json" if prefix else "package.json"
            text = file_reader(manifest)
            if text is None:
                findings.append(
                    f"AGENTS.md: documented command '{command}' references missing {manifest}"
                )
            else:
                try:
                    scripts = json.loads(text).get("scripts", {})
                except (json.JSONDecodeError, AttributeError):
                    scripts = {}
                if not isinstance(scripts, dict) or script not in scripts:
                    findings.append(
                        f"AGENTS.md: documented command '{command}' references package.json "
                        f"script '{script}' that is not defined"
                    )
            continue

        match = NODE_TEST.search(command)
        if match:
            tokens = shlex.split(match.group("args"))
            skip = False
            for token in tokens:
                if skip:
                    skip = False
                    continue
                if token in NODE_VALUE_FLAGS:
                    skip = True
                    continue
                if token.startswith("-") or _placeholder(token):
                    continue
                if not _glob_matches(token, paths):
                    missing_path(command, token)
            continue

        match = UNITTEST_DISCOVER.search(command)
        if match:
            directory = match.group("dir").strip("/")
            if directory and directory not in paths and not any(p.startswith(directory + "/") for p in paths):
                missing_path(command, match.group("dir"))
            continue

        match = UNITTEST.search(command)
        if match and "discover" not in command:
            for token in shlex.split(match.group("args")):
                if token.startswith("-") or _placeholder(token):
                    continue
                if "/" in token or token.endswith(".py"):
                    if not _glob_matches(token.rstrip("/"), paths) and not any(
                        path.startswith(token.rstrip("/") + "/") for path in paths
                    ):
                        missing_path(command, token)
                elif "." in token and not _resolve_module(token, paths):
                    findings.append(
                        f"AGENTS.md: documented command '{command}' references unittest module "
                        f"'{token}' that does not resolve to a test file"
                    )
            continue

        match = PYTEST.search(command)
        if match:
            tokens = shlex.split(match.group("args"))
            skip = False
            for token in tokens:
                if skip:
                    skip = False
                    continue
                if token in PYTEST_VALUE_FLAGS:
                    skip = True
                    continue
                if token.startswith("-") or _placeholder(token):
                    continue
                if "/" not in token and not token.endswith(".py"):
                    continue
                if not _glob_matches(token.rstrip("/"), paths) and not any(
                    path.startswith(token.rstrip("/") + "/") for path in paths
                ):
                    missing_path(command, token)
            continue

        for token in SCRIPT_PATH.findall(command):
            if not _glob_matches(token, paths):
                missing_path(command, token)

    return list(dict.fromkeys(findings))


def agents_md_findings(text: str | None, paths: set[str], file_reader) -> list[str]:
    """Return precise agent-friendliness findings for one repository.

    ``text`` is the ``AGENTS.md`` content (``None`` when the file is absent),
    ``paths`` is the set of tracked repository-relative paths, and
    ``file_reader(path)`` returns file content or ``None``. The function is
    pure and deterministic; an empty list means the contract holds.
    """
    if text is None:
        return ["AGENTS.md missing at repository root"]
    if EXEMPT_MARKER.search(text):
        return []
    findings: list[str] = []
    line_count = text.count("\n") + 1
    if line_count > MAX_AGENTS_LINES:
        findings.append(
            f"AGENTS.md is too long for an entry point ({line_count} lines > {MAX_AGENTS_LINES})"
        )
    if len(text.encode("utf-8")) > MAX_AGENTS_BYTES:
        findings.append(
            f"AGENTS.md is too large for an entry point ({len(text.encode('utf-8'))} bytes > {MAX_AGENTS_BYTES})"
        )
    for title in REQUIRED_SECTIONS:
        if _section(text, title) is None:
            findings.append(f"AGENTS.md: missing required section '## {title}'")
    if "git status" not in text:
        findings.append("AGENTS.md: missing the local-checkout warning ('git status')")
    if not re.search(r"\borigin/main\b|\bmain\b", text):
        findings.append("AGENTS.md: does not mention 'main'; state that a checkout is not proof of origin/main")
    if REPO_MAP_POINTER not in text:
        findings.append(f"AGENTS.md: does not point at the host map ({REPO_MAP_POINTER})")
    if not DOC_POINTER.search(text):
        findings.append("AGENTS.md: does not point at canonical docs (README/ARCHITECTURE/HANDOFF/docs)")
    section = _section(text, "Commands")
    if section is not None:
        commands = _commands_in_section(section)
        if not commands:
            findings.append("AGENTS.md: '## Commands' section declares no commands or explicit no-tests statement")
        else:
            findings.extend(_command_findings(commands, paths, file_reader))
    for heading, block in _context_blocks(text):
        for path in NON_CANONICAL_PATHS:
            if path in block and not WARNING_MARKERS.search(block) and not WARNING_MARKERS.search(heading):
                findings.append(
                    f"AGENTS.md: mentions '{path}' without a warning; do not direct agents to non-canonical paths"
                )
    if any(WINDMILL_SOURCE.search(path) for path in paths) and not OWNERS_POINTER.search(text):
        findings.append(
            "AGENTS.md: repository contains Windmill sources but does not point at windmill-production/OWNERS.yaml"
        )
    return list(dict.fromkeys(findings))


def tracked_paths(root: Path) -> set[str]:
    """Return tracked repository-relative paths, falling back to a filesystem walk."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        proc = None
    if proc is not None and proc.returncode == 0 and proc.stdout.strip():
        return {line for line in proc.stdout.splitlines() if line}
    ignored = {".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__"}
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not any(part in ignored for part in path.parts)
    }


def checkout_findings(root: Path) -> list[str]:
    """Run the contract against a checkout on disk."""
    root = root.resolve()
    paths = tracked_paths(root)
    agents = root / "AGENTS.md"
    text = agents.read_text(encoding="utf-8", errors="replace") if agents.is_file() else None

    def file_reader(relative: str) -> str | None:
        candidate = root / relative
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
        return None

    return agents_md_findings(text, paths, file_reader)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="agent-friendliness contract check")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    findings = checkout_findings(args.root)
    if args.json:
        print(json.dumps({"root": str(args.root), "findings": findings}, indent=2))
    elif findings:
        print(f"agent-friendliness contract failed for {args.root}:")
        for finding in findings:
            print(f"  - {finding}")
    else:
        print(f"agent-friendliness contract ok for {args.root}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
