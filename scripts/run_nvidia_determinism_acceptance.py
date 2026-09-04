#!/usr/bin/env python3
"""Produce two probe-bound NVIDIA/Vulkan determinism records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_determinism_hashes import _validate_adapter  # noqa: E402


IDENTITY_FIELDS = (
    "backend",
    "device_type",
    "name",
    "vendor",
    "device",
    "software_fallback",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class AcceptanceError(RuntimeError):
    """The NVIDIA determinism evidence is incomplete or inconsistent."""


def _read_probe(path: Path) -> dict:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot read NVIDIA adapter probe: {exc}") from exc
    if not isinstance(envelope, dict):
        raise AcceptanceError("NVIDIA adapter probe is not an object")
    if str(envelope.get("requested_backend", "")).lower() != "vulkan":
        raise AcceptanceError("NVIDIA adapter probe did not request Vulkan")
    try:
        return _validate_adapter("nvidia", envelope.get("probe"))
    except (TypeError, ValueError) as exc:
        raise AcceptanceError(f"NVIDIA adapter probe is invalid: {exc}") from exc


def _same_identity(actual: dict, expected: dict, context: str) -> None:
    for field in IDENTITY_FIELDS:
        if str(actual.get(field, "")).lower() != str(expected.get(field, "")).lower():
            raise AcceptanceError(f"{context} adapter differs at {field}")


def _parse_record(stdout: str, png_path: Path, label: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise AcceptanceError(f"{label} render produced no JSON record")
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise AcceptanceError(f"{label} render has no terminal JSON record") from exc
    if not isinstance(record, dict):
        raise AcceptanceError(f"{label} render record is not an object")
    if record.get("scene") != png_path.name.removesuffix(".repeat.png").removesuffix(
        ".png"
    ):
        raise AcceptanceError(f"{label} render record names the wrong scene")
    if not png_path.is_file():
        raise AcceptanceError(f"{label} render did not write {png_path.name}")
    actual_sha = hashlib.sha256(png_path.read_bytes()).hexdigest()
    declared_sha = str(record.get("sha256", ""))
    if SHA256_PATTERN.fullmatch(declared_sha) is None or declared_sha != actual_sha:
        raise AcceptanceError(f"{label} render SHA does not match its PNG bytes")
    try:
        record["adapter"] = _validate_adapter("nvidia", record.get("adapter"))
    except (TypeError, ValueError) as exc:
        raise AcceptanceError(f"{label} render adapter is invalid: {exc}") from exc
    return record


def _render_once(args: argparse.Namespace, artifact_dir: Path, label: str) -> dict:
    suffix = "" if label == "first" else ".repeat"
    png_path = artifact_dir / f"{args.scene}{suffix}.png"
    command = [
        sys.executable,
        "-m",
        "forge3d.determinism",
        "--scene",
        args.scene,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--out-png",
        str(png_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (artifact_dir / f"{label}.stdout.log").write_text(
        result.stdout, encoding="utf-8"
    )
    (artifact_dir / f"{label}.stderr.log").write_text(
        result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        raise AcceptanceError(f"{label} render failed with exit {result.returncode}")
    return _parse_record(result.stdout, png_path, label)


def run(args: argparse.Namespace) -> dict:
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("FORGE3D_DETERMINISTIC") != "1":
        raise AcceptanceError("FORGE3D_DETERMINISTIC=1 is required")
    if str(os.environ.get("WGPU_BACKENDS", "")).lower() != "vulkan":
        raise AcceptanceError("WGPU_BACKENDS=vulkan is required")

    expected = _read_probe(args.adapter_probe.resolve())
    first = _render_once(args, artifact_dir, "first")
    repeat = _render_once(args, artifact_dir, "repeat")
    _same_identity(first["adapter"], expected, "first render/probe")
    _same_identity(repeat["adapter"], expected, "repeat render/probe")
    _same_identity(repeat["adapter"], first["adapter"], "repeat/first render")
    if first["sha256"] != repeat["sha256"]:
        raise AcceptanceError("required NVIDIA repeat hash differs from the first render")

    (artifact_dir / f"{args.scene}.sha256").write_text(
        first["sha256"] + "\n", encoding="utf-8"
    )
    (artifact_dir / f"{args.scene}.json").write_text(
        json.dumps(first, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / f"{args.scene}.repeat.sha256").write_text(
        repeat["sha256"] + "\n", encoding="utf-8"
    )
    (artifact_dir / f"{args.scene}.repeat.json").write_text(
        json.dumps(repeat, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"first": first, "repeat": repeat, "probe": expected}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--adapter-probe", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args)
    except AcceptanceError as exc:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        (args.artifact_dir / f"{args.scene}.FAILED").write_text(
            str(exc) + "\n", encoding="utf-8"
        )
        print(f"NVIDIA determinism acceptance failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "sha256": result["first"]["sha256"],
                "adapter": result["first"]["adapter"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
