"""Validate attributable TERRA-DETERMINATA hash artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_NVIDIA_LEG = "nvidia"


def _validate_adapter(leg: str, adapter: object) -> dict:
    if not isinstance(adapter, dict):
        raise ValueError("adapter metadata is not an object")
    required = (
        "name",
        "backend",
        "device_type",
        "vendor",
        "device",
        "software_fallback",
    )
    if not all(key in adapter for key in required):
        raise ValueError("adapter metadata is incomplete")
    try:
        device = int(adapter.get("device"))
    except (TypeError, ValueError) as exc:
        raise ValueError("adapter device ID is not an integer") from exc
    if device < 0:
        raise ValueError("adapter device ID is negative")
    if adapter.get("software_fallback") is not False:
        raise ValueError("adapter does not explicitly prove software_fallback=false")
    if leg == REQUIRED_NVIDIA_LEG:
        name = str(adapter.get("name", "")).lower()
        backend = str(adapter.get("backend", "")).lower()
        device_type = str(adapter.get("device_type", "")).lower()
        if backend != "vulkan":
            raise ValueError("required NVIDIA leg did not use Vulkan")
        if device_type != "discretegpu":
            raise ValueError("required NVIDIA leg did not use a discrete GPU")
        if int(adapter.get("vendor", 0)) != 0x10DE:
            raise ValueError("required NVIDIA leg has a non-NVIDIA vendor ID")
        if "nvidia" not in name:
            raise ValueError("required NVIDIA leg has a non-NVIDIA adapter name")
    return adapter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)

    golden = (
        args.golden.read_text().split()[0].strip() if args.golden.exists() else None
    )
    produced = {}
    absent = {}
    adapters = {}
    failures = []
    gated_failure = False

    for artifact_dir in sorted(args.hashes.glob("determinism-hash-*")):
        leg = artifact_dir.name.removeprefix("determinism-hash-")
        sha_file = artifact_dir / f"{args.scene}.sha256"
        absent_file = artifact_dir / f"{args.scene}.ABSENT"
        failed_file = artifact_dir / f"{args.scene}.FAILED"
        meta_file = artifact_dir / f"{args.scene}.json"
        if sha_file.exists():
            produced[leg] = sha_file.read_text().split()[0].strip()
            try:
                adapters[leg] = _validate_adapter(
                    leg, json.loads(meta_file.read_text())["adapter"]
                )
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                failures.append(
                    f"{leg}: missing or invalid attributable adapter metadata: {exc}"
                )
        elif absent_file.exists():
            absent[leg] = absent_file.read_text().splitlines()[0]
        elif failed_file.exists():
            # Preserve loud gated failures for supplemental legs. They remain
            # visible but can never replace the required NVIDIA/Vulkan hash.
            absent[leg] = "GATED-FAILURE: " + failed_file.read_text().splitlines()[0]
            gated_failure = True

    print("produced hashes:")
    for leg, sha in sorted(produced.items()):
        adapter = adapters.get(leg)
        ident = (
            f"{adapter['name']} ({adapter['backend']}, {adapter['device_type']})"
            if adapter
            else "UNATTRIBUTED"
        )
        print(f"  {leg:8s} {sha}  adapter: {ident}")
    for leg, why in sorted(absent.items()):
        print(f"informational/absent: {leg}: {why}")
    print(f"committed golden: {golden}" if golden else "no committed golden")

    if REQUIRED_NVIDIA_LEG not in produced:
        failures.append("required NVIDIA/Vulkan leg produced no hash")
    if not produced and not gated_failure:
        failures.append("no hardware-backed leg produced a hash")
    values = set(produced.values())
    if len(values) > 1:
        failures.append(f"pairwise mismatch across legs: {produced}")
    if golden is None:
        failures.append("no committed golden found")
    elif any(sha != golden for sha in produced.values()):
        failures.append(f"mismatch against committed golden {golden}: {produced}")

    if failures:
        print("DETERMINISM FAILURE (zero-byte tolerance):", file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    print("determinism diff: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
