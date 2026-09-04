"""Minimal TOML-loading shim shared by repository gate tests.

Uses the stdlib `tomllib` when available (Python >= 3.11). Falls back to a
small hand-rolled parser sufficient for the restricted schemas committed in
repository test/config manifests: scalar assignments, scalar arrays (including
multiline arrays), dotted named tables, and array-of-tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python >= 3.11
except ImportError:  # pragma: no cover - exercised on Python < 3.11
    tomllib = None  # type: ignore[assignment]


def _strip_comment(raw_line: str) -> str:
    """Remove a TOML comment without treating `#` inside a string as one."""
    in_string = False
    escaped = False
    for index, char in enumerate(raw_line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "#":
            return raw_line[:index]
    return raw_line


def _array_depth(value: str) -> int:
    """Return square-bracket depth, ignoring brackets inside JSON strings."""
    depth = 0
    in_string = False
    escaped = False
    for char in value:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
    return depth


def _without_trailing_array_commas(value: str) -> str:
    """Remove TOML array trailing commas without changing string contents."""
    output: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(value):
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            output.append(char)
            continue
        if char == ",":
            cursor = index + 1
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            if cursor < len(value) and value[cursor] == "]":
                continue
        output.append(char)
    return "".join(output)


def _table_target(
    result: dict[str, Any], dotted_key: str, *, array: bool
) -> dict[str, Any]:
    """Resolve a table, anchoring dotted children to the latest AoT item."""
    parts = [part.strip() for part in dotted_key.split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid TOML table name: {dotted_key!r}")

    parent: dict[str, Any] = result
    for part in parts[:-1]:
        child = parent.setdefault(part, {})
        if isinstance(child, list):
            if not child or not isinstance(child[-1], dict):
                raise ValueError(f"table {part!r} has no current array item")
            parent = child[-1]
        elif isinstance(child, dict):
            parent = child
        else:
            raise ValueError(f"table {part!r} conflicts with a scalar")

    leaf = parts[-1]
    if array:
        items = parent.setdefault(leaf, [])
        if not isinstance(items, list):
            raise ValueError(f"array-of-tables {dotted_key!r} conflicts with a value")
        target: dict[str, Any] = {}
        items.append(target)
        return target

    existing = parent.setdefault(leaf, {})
    if not isinstance(existing, dict):
        raise ValueError(f"table {dotted_key!r} conflicts with a value")
    return existing


def parse_toml_fallback(text: str) -> dict[str, Any]:
    """Parse the restricted TOML subset without any external dependency."""
    result: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    raw_lines = text.splitlines()
    index = 0
    while index < len(raw_lines):
        line = _strip_comment(raw_lines[index]).strip()
        index += 1
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            key = line[2:-2].strip()
            current = _table_target(result, key, array=True)
            continue
        if line.startswith("[") and line.endswith("]"):
            key = line[1:-1].strip()
            current = _table_target(result, key, array=False)
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith("["):
            while _array_depth(value) > 0:
                if index >= len(raw_lines):
                    raise ValueError(f"unterminated TOML array for {key!r}")
                continuation = _strip_comment(raw_lines[index]).strip()
                index += 1
                if continuation:
                    value = f"{value} {continuation}"
            value = _without_trailing_array_commas(value)
        if value.startswith(('"', "[")):
            parsed: Any = json.loads(value)
        elif value == "true":
            parsed = True
        elif value == "false":
            parsed = False
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        if current is not None:
            current[key] = parsed
        else:
            result[key] = parsed
    return result


def load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, preferring stdlib `tomllib` when it is available."""
    path = Path(path)
    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)
    return parse_toml_fallback(path.read_text(encoding="utf-8"))
