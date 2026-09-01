"""Repository documentation contracts exercised by the dedicated CI job."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from prefscope.cli.parser import build_parser

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[1]
_MARKDOWN_LINK = re.compile(
    r"!?\[[^\]]*\]\(([^)]+)\)|^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE
)
_FENCED_CODE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _markdown_files() -> list[Path]:
    top_level = [
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
    ]
    return [REPO_ROOT / name for name in top_level] + sorted(
        (REPO_ROOT / "docs").rglob("*.md")
    )


def _relative_link_targets(source: Path):
    text = _FENCED_CODE.sub("", source.read_text(encoding="utf-8"))
    for match in _MARKDOWN_LINK.finditer(text):
        destination = (match.group(1) or match.group(2)).strip().strip("<>")
        # A destination may have a quoted Markdown title after the URL.
        destination = destination.split(maxsplit=1)[0]
        parsed = urlsplit(destination)
        if not destination or destination.startswith("#"):
            continue
        if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
            continue
        path = unquote(parsed.path)
        if path:
            yield destination, (source.parent / path).resolve()


def test_markdown_relative_links_resolve_inside_repository():
    broken = []
    escaped = []
    for source in _markdown_files():
        for destination, target in _relative_link_targets(source):
            if target != REPO_ROOT and REPO_ROOT not in target.parents:
                escaped.append(f"{source.relative_to(REPO_ROOT)} -> {destination}")
            elif not target.exists():
                broken.append(f"{source.relative_to(REPO_ROOT)} -> {destination}")

    assert not escaped, (
        "relative links must stay inside the repository:\n" + "\n".join(escaped)
    )
    assert not broken, "broken relative Markdown links:\n" + "\n".join(broken)




def test_python_fences_are_valid_python_syntax():
    broken = []
    for source in _markdown_files():
        for index, match in enumerate(
            _PYTHON_FENCE.finditer(source.read_text(encoding="utf-8")), start=1
        ):
            try:
                compile(match.group(1), f"{source}:{index}", "exec")
            except SyntaxError as exc:
                broken.append(
                    f"{source.relative_to(REPO_ROOT)} block {index}: {exc.msg}")
    assert not broken, "invalid Python documentation blocks:\n" + "\n".join(broken)


def _github_heading_anchors(path: Path) -> set[str]:
    """Return the simple GitHub-style anchors used by this documentation."""
    anchors = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if not match:
            continue
        heading = re.sub(r"<[^>]+>", "", match.group(1))
        heading = re.sub(r"`([^`]*)`", r"\1", heading)
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        base = "".join(
            char for char in heading.strip().lower()
            if char.isalnum() or char in " _-"
        )
        base = re.sub(r"\s+", "-", base)
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def test_relative_markdown_anchors_exist():
    broken = []
    for source in _markdown_files():
        text = _FENCED_CODE.sub("", source.read_text(encoding="utf-8"))
        for match in _MARKDOWN_LINK.finditer(text):
            destination = (match.group(1) or match.group(2)).strip().strip("<>")
            destination = destination.split(maxsplit=1)[0]
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc or not parsed.fragment:
                continue
            target = (
                (source.parent / unquote(parsed.path)).resolve()
                if parsed.path else source.resolve()
            )
            if not target.is_file() or target.suffix.lower() != ".md":
                continue
            anchor = unquote(parsed.fragment).lower()
            if anchor not in _github_heading_anchors(target):
                broken.append(
                    f"{source.relative_to(REPO_ROOT)} -> {destination}")
    assert not broken, "broken relative Markdown anchors:\n" + "\n".join(broken)


def _leaf_command_paths(parser: argparse.ArgumentParser, prefix=()):
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    if not subparsers:
        if prefix:
            yield prefix
        return
    for action in subparsers:
        for name, child in action.choices.items():
            yield from _leaf_command_paths(child, (*prefix, name))


def test_every_cli_command_is_named_in_the_cli_reference():
    reference = (REPO_ROOT / "docs/reference/cli.md").read_text(encoding="utf-8")
    missing = []
    for path in _leaf_command_paths(build_parser()):
        command = " ".join(path)
        if f"`{command}`" not in reference and f"prefscope {command}" not in reference:
            missing.append(command)

    assert not missing, (
        "docs/reference/cli.md must name every parser command; missing: "
        + ", ".join(missing)
    )


def _parser_at_path(parser: argparse.ArgumentParser, path: tuple[str, ...]):
    current = parser
    for name in path:
        subparsers = next(
            action for action in current._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        current = subparsers.choices[name]
    return current


def test_cli_reference_tables_do_not_name_invalid_flags():
    reference = (REPO_ROOT / "docs/reference/cli.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^### `([^`]+)`\s*$", reference, re.MULTILINE))
    parser = build_parser()
    invalid = []
    for index, heading in enumerate(headings):
        command = heading.group(1)
        path = tuple(command.split())
        try:
            command_parser = _parser_at_path(parser, path)
        except (KeyError, StopIteration):
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(reference)
        section = reference[heading.end():end]
        valid = {
            option
            for action in command_parser._actions
            for option in action.option_strings
        }
        for line in section.splitlines():
            if not line.startswith("| `--"):
                continue
            first_cell = line.split("|", 2)[1]
            for option in re.findall(r"--[a-z0-9][a-z0-9-]*", first_cell):
                if option not in valid:
                    invalid.append(f"{command}: {option}")
    assert not invalid, "invalid CLI flags in reference tables: " + ", ".join(invalid)
