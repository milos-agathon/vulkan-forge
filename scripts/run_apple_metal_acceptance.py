#!/usr/bin/env python3
"""Run the checked Apple Metal matrix and emit one merged JUnit report."""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI compatibility
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from _toml_compat import load_toml as _load_toml
else:
    _load_toml = None


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "apple_metal_acceptance.toml"
RECIPE_SOURCE = ROOT / "tests" / "test_recipe_goldens.py"
PHYSICAL_TYPES = {"integratedgpu", "discretegpu"}
SOFTWARE_TOKENS = (
    "cpu",
    "llvmpipe",
    "paravirtual",
    "software",
    "swiftshader",
    "virtual",
)


@dataclass(frozen=True)
class Phase:
    name: str
    nodes: tuple[str, ...]
    marker: str | None = None


def _read_manifest() -> dict:
    if _load_toml is not None:
        return _load_toml(MANIFEST)
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)


def load_manifest() -> tuple[Phase, ...]:
    data = _read_manifest()
    if data.get("version") != 1:
        raise ValueError("unsupported Apple Metal acceptance manifest version")
    phases = tuple(
        Phase(
            str(item["name"]),
            tuple(str(node) for node in item["nodes"]),
            str(item["marker"]) if item.get("marker") else None,
        )
        for item in data.get("phases", ())
    )
    if not phases or len({phase.name for phase in phases}) != len(phases):
        raise ValueError("Apple Metal acceptance phases must be present and unique")
    if any(
        not phase.nodes
        or (phase.marker is None and any("::" not in node for node in phase.nodes))
        for phase in phases
    ):
        raise ValueError("every Apple Metal phase must contain explicit pytest nodes")
    return phases


def expected_recipe_ids() -> tuple[str, ...]:
    """Read the semantic recipe catalog without importing GPU-dependent tests."""
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "RECIPE_GOLDENS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            break
        ids = []
        for entry in node.value.elts:
            if not isinstance(entry, ast.Call) or not entry.args:
                raise ValueError("RECIPE_GOLDENS contains a non-literal entry")
            scene_id = entry.args[0]
            if not isinstance(scene_id, ast.Constant) or not isinstance(
                scene_id.value, str
            ):
                raise ValueError("RECIPE_GOLDENS scene ids must be string literals")
            ids.append(scene_id.value)
        return tuple(ids)
    raise ValueError("RECIPE_GOLDENS catalog not found")


def _name(record: dict) -> str:
    return str(
        record.get("adapter_name")
        or record.get("device_name")
        or record.get("name")
        or ""
    ).strip()


def _require_physical_apple_metal(record: dict, label: str) -> None:
    name = _name(record)
    identity = name.lower()
    if str(record.get("backend", "")).lower() != "metal":
        raise RuntimeError(f"{label}: active backend is not Metal: {record}")
    if str(record.get("device_type", "")).lower() not in PHYSICAL_TYPES:
        raise RuntimeError(f"{label}: adapter is not a physical GPU: {record}")
    if record.get("software_fallback") is not False:
        raise RuntimeError(f"{label}: software_fallback must be false: {record}")
    if "apple" not in identity or any(token in identity for token in SOFTWARE_TOKENS):
        raise RuntimeError(f"{label}: adapter is not physical Apple hardware: {record}")


def _adapter_identity(record: dict) -> tuple[str, str, str, bool | None]:
    return (
        _name(record).lower(),
        str(record.get("backend", "")).lower(),
        str(record.get("device_type", "")).lower(),
        record.get("software_fallback"),
    )


def _require_same_adapter(reference: dict, actual: dict, label: str) -> None:
    _require_physical_apple_metal(actual, label)
    if _adapter_identity(actual) != _adapter_identity(reference):
        raise RuntimeError(
            f"{label}: active Apple Metal adapter identity changed: "
            f"expected {_adapter_identity(reference)}, got {_adapter_identity(actual)}"
        )


def _initialized_engine_info() -> dict:
    from forge3d._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "engine_info"):
        raise RuntimeError("installed wheel did not expose active adapter identity")
    return dict(native.engine_info())


def _active_adapter_record(output: Path) -> dict:
    import forge3d as f3d

    probe = dict(f3d.device_probe("metal"))
    payload = {"requested_backend": "metal", "probe": probe}
    _write_json(output, payload)
    _require_physical_apple_metal(probe, "device probe")
    active = _initialized_engine_info()
    payload["active_adapter"] = active
    _write_json(output, payload)
    _require_same_adapter(probe, active, "initialized adapter")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _merge_junit(inputs: list[Path], output: Path) -> None:
    root = ET.Element("testsuites", {"name": "Apple Metal acceptance"})
    for path in inputs:
        parsed = ET.parse(path).getroot()
        if parsed.tag == "testsuite":
            root.append(parsed)
        else:
            root.extend(parsed.findall("testsuite"))
    for field in ("tests", "failures", "errors", "skipped"):
        root.set(field, str(sum(int(suite.get(field, "0")) for suite in root)))
    root.set(
        "time",
        format(
            sum((Decimal(suite.get("time", "0")) for suite in root), Decimal()),
            "f",
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def _junit_test_count(inputs: list[Path]) -> int:
    return sum(
        1
        for path in inputs
        for _ in ET.parse(path).getroot().iter("testcase")
    )


def _failure_junit(
    path: Path, message: str, *, case_name: str = "adapter-contract"
) -> None:
    suites = ET.Element(
        "testsuites", {"tests": "1", "failures": "0", "errors": "1", "skipped": "0"}
    )
    suite = ET.SubElement(
        suites,
        "testsuite",
        {
            "name": "apple-metal-infrastructure",
            "tests": "1",
            "failures": "0",
            "errors": "1",
            "skipped": "0",
            "time": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": case_name, "time": "0"})
    ET.SubElement(case, "error", {"message": message}).text = message
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suites).write(path, encoding="utf-8", xml_declaration=True)


def pytest_collection_modifyitems(items) -> None:
    """Reject active skip decorators and every xfail in the required matrix."""
    if os.environ.get("FORGE3D_APPLE_METAL_ACCEPTANCE") != "1":
        return
    forbidden = []
    for item in items:
        if item.get_closest_marker("skip") is not None:
            forbidden.append(f"{item.nodeid}: skip")
        if item.get_closest_marker("xfail") is not None:
            forbidden.append(f"{item.nodeid}: xfail")
        for marker in item.iter_markers("skipif"):
            if marker.args and bool(marker.args[0]):
                forbidden.append(f"{item.nodeid}: active skipif")
    if forbidden:
        import pytest

        raise pytest.UsageError(
            "Apple Metal acceptance forbids skip/xfail decorators:\n"
            + "\n".join(forbidden)
        )


def _pytest_main(args: list[str]) -> int:
    import pytest

    return int(pytest.main(args))


def _run_phase_in_process(phase: Phase, junit: Path, adapter: Path) -> int:
    marker_args = ["-m", phase.marker] if phase.marker else []
    code = _pytest_main(
        [
            "-p",
            "scripts.run_apple_metal_acceptance",
            *phase.nodes,
            *marker_args,
            "-v",
            "--tb=short",
            f"--junitxml={junit}",
        ]
    )
    active = _initialized_engine_info()
    _write_json(adapter, {"phase": phase.name, "active_adapter": active})
    _require_physical_apple_metal(active, f"phase {phase.name} rendered adapter")
    return code


def _validate_phase_adapter_records(
    before: dict, phases: tuple[Phase, ...], evidence_dir: Path
) -> None:
    reference = before["active_adapter"]
    _require_same_adapter(before["probe"], reference, "preflight initialized adapter")
    for phase in phases:
        path = evidence_dir / f"{phase.name}-adapter.json"
        if not path.is_file():
            raise RuntimeError(f"phase {phase.name} rendered adapter record is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("phase") != phase.name or not isinstance(
            payload.get("active_adapter"), dict
        ):
            raise RuntimeError(f"phase {phase.name} rendered adapter record is invalid")
        _require_same_adapter(
            reference,
            payload["active_adapter"],
            f"phase {phase.name} rendered adapter identity",
        )


def _phase_result_junits(
    phase: Phase, junit: Path, runner_exit: int, evidence_dir: Path
) -> list[Path]:
    outputs = [junit] if junit.is_file() else []
    if runner_exit != 0 or not outputs:
        failure = evidence_dir / f"{phase.name}-runner.xml"
        message = f"phase {phase.name} exited {runner_exit}"
        if not outputs:
            message += " without JUnit"
        _failure_junit(
            failure,
            message,
            case_name=f"{phase.name}-runner-exit",
        )
        outputs.append(failure)
    return outputs


def _run_phase(
    phase: Phase, junit: Path, adapter: Path, *, collect_only: bool
) -> int:
    if collect_only:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.run_apple_metal_acceptance",
            *phase.nodes,
        ]
        if phase.marker:
            command.extend(("-m", phase.marker))
        command.extend(("--collect-only", "-q"))
    else:
        command = [
            sys.executable,
            "-m",
            "scripts.run_apple_metal_acceptance",
            "--phase",
            phase.name,
            "--phase-junit",
            str(junit),
            "--phase-adapter",
            str(adapter),
        ]
    print(f"apple-metal phase={phase.name} command={command}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False).returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--phase")
    parser.add_argument("--phase-junit", type=Path)
    parser.add_argument("--phase-adapter", type=Path)
    args = parser.parse_args(argv)
    os.environ["FORGE3D_APPLE_METAL_ACCEPTANCE"] = "1"
    if args.phase is not None:
        if args.phase_junit is None or args.phase_adapter is None:
            parser.error("--phase-junit and --phase-adapter are required with --phase")
        matches = [phase for phase in load_manifest() if phase.name == args.phase]
        if len(matches) != 1:
            parser.error(f"unknown Apple Metal phase: {args.phase}")
        return _run_phase_in_process(matches[0], args.phase_junit, args.phase_adapter)
    if args.collect_only:
        phases = load_manifest()
        expected_cases = int(_read_manifest()["recipe_cases"])
        actual_cases = expected_recipe_ids()
        if len(actual_cases) != expected_cases:
            raise RuntimeError(
                "recipe acceptance catalog changed: "
                f"expected {expected_cases}, found {len(actual_cases)}"
            )
        codes = [
            _run_phase(phase, Path(), Path(), collect_only=True) for phase in phases
        ]
        return int(any(code != 0 for code in codes))
    if args.junit is None or args.evidence_dir is None:
        parser.error(
            "--junit and --evidence-dir are required unless --collect-only is used"
        )

    evidence_dir = args.evidence_dir.resolve()
    junit = args.junit.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR"] = str(
        evidence_dir / "recipe-goldens"
    )
    os.environ["FORGE3D_SIDERA_ARTIFACT_DIR"] = str(evidence_dir / "sidera")
    phase_junits: list[Path] = []
    result = 0
    try:
        phases = load_manifest()
        expected_cases = int(_read_manifest()["recipe_cases"])
        actual_cases = expected_recipe_ids()
        if len(actual_cases) != expected_cases:
            raise RuntimeError(
                "recipe acceptance catalog changed: "
                f"expected {expected_cases}, found {len(actual_cases)}"
            )
        before = _active_adapter_record(evidence_dir / "adapter-before.json")
        os.environ["FORGE3D_EXPECTED_ADAPTER_PROBE"] = str(
            evidence_dir / "adapter-before.json"
        )
        collected_junits: list[Path] = []
        for phase in phases:
            phase_junit = evidence_dir / f"{phase.name}.xml"
            phase_adapter = evidence_dir / f"{phase.name}-adapter.json"
            code = _run_phase(
                phase, phase_junit, phase_adapter, collect_only=False
            )
            phase_junits.extend(
                _phase_result_junits(phase, phase_junit, code, evidence_dir)
            )
            if phase_junit.is_file():
                collected_junits.append(phase_junit)
            if code != 0:
                result = 1
        _validate_phase_adapter_records(before, phases, evidence_dir)
        expected_tests = int(_read_manifest()["expected_tests"])
        actual_tests = _junit_test_count(collected_junits)
        if actual_tests != expected_tests:
            selection = evidence_dir / "selection-contract.xml"
            _failure_junit(
                selection,
                "Apple Metal manifest expected "
                f"{expected_tests} tests, collected {actual_tests}",
            )
            phase_junits.append(selection)
            result = max(result, 1)
        after = _active_adapter_record(evidence_dir / "adapter-after.json")
        _require_same_adapter(
            before["active_adapter"],
            after["active_adapter"],
            "post-render initialized adapter",
        )
    except Exception as error:
        infrastructure = evidence_dir / "adapter-contract.xml"
        _failure_junit(infrastructure, str(error))
        phase_junits.append(infrastructure)
        result = max(result, 1)
    _merge_junit(phase_junits, junit)
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
