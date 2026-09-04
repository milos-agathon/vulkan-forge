"""Consolidate and fail-closed validate TESSELLA physical-GPU evidence.

The nine core JSON files are named by gate identity. Additional JSON artifacts
are reported as supplemental, but can never replace a missing canonical gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .tessella_evidence_contract import (
        CORE_GATES,
        THRESHOLDS,
        load_gate,
        non_finite_paths,
    )
    from .tessella_evidence_provenance import (
        PROVENANCE_FILES,
        validate_provenance,
    )
else:
    from tessella_evidence_contract import (
        CORE_GATES,
        THRESHOLDS,
        load_gate,
        non_finite_paths,
    )
    from tessella_evidence_provenance import (
        PROVENANCE_FILES,
        validate_provenance,
    )


REPORT_NAME = "verification-report.json"
REPORT_SCHEMA = "forge3d.tessella_verification/1"


def verify_artifact_directory(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    results: list[dict[str, Any]] = []
    evidence_by_gate: dict[str, dict[str, Any]] = {}
    all_errors: list[str] = []
    for gate in CORE_GATES:
        filename = f"{gate}.json"
        evidence, errors = load_gate(artifact_dir / filename, gate)
        result: dict[str, Any] = {
            "file": filename,
            "gate": gate,
            "status": "fail" if errors else "pass",
            "thresholds": THRESHOLDS.get(gate, {}),
        }
        if evidence is not None and not non_finite_paths(evidence):
            result["evidence"] = evidence
            evidence_by_gate[gate] = evidence
        if errors:
            result["errors"] = errors
            all_errors.extend(errors)
        results.append(result)

    core_files = {f"{gate}.json" for gate in CORE_GATES}
    ignored = core_files | set(PROVENANCE_FILES) | {REPORT_NAME}
    supplemental = sorted(
        path.name for path in artifact_dir.glob("*.json") if path.name not in ignored
    )
    provenance, provenance_errors = validate_provenance(
        artifact_dir,
        core_files | set(PROVENANCE_FILES) | set(supplemental),
        evidence_by_gate.get("capability_degradations"),
    )
    all_errors.extend(provenance_errors)
    return {
        "schema": REPORT_SCHEMA,
        "status": "fail" if all_errors else "pass",
        "core_gate_count": len(CORE_GATES),
        "results": results,
        "provenance": provenance,
        "supplemental_json_files": supplemental,
        "validation_errors": all_errors,
    }


def write_report(artifact_dir: Path, report: dict[str, Any]) -> Path:
    path = Path(artifact_dir) / REPORT_NAME
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _print_summary(report: dict[str, Any]) -> None:
    print("TESSELLA evidence verification")
    print(f"{'gate':31} status")
    for result in report["results"]:
        print(f"{result['gate']:31} {result['status'].upper()}")
    passed = sum(result["status"] == "pass" for result in report["results"])
    print(
        f"summary: {passed}/{len(CORE_GATES)} core gates passed; "
        f"status={report['status'].upper()}"
    )
    supplemental = report["supplemental_json_files"]
    if supplemental:
        print(f"supplemental JSON (non-gating): {', '.join(supplemental)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and consolidate the nine TESSELLA evidence records."
    )
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(argv)
    if not args.artifact_dir.is_dir():
        print(
            "artifact directory does not exist or is not a directory: "
            f"{args.artifact_dir}",
            file=sys.stderr,
        )
        return 1
    report = verify_artifact_directory(args.artifact_dir)
    write_report(args.artifact_dir, report)
    _print_summary(report)
    if report["validation_errors"]:
        print("TESSELLA evidence verification failed:", file=sys.stderr)
        for error in report["validation_errors"]:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
