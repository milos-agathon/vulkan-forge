"""Validate attributable, backend-routed TERRA-DETERMINATA hash artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REQUIRED_NVIDIA_LEG = "nvidia"
COMMIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SOFTWARE_ADAPTER_TOKENS = (
    "basic render driver",
    "lavapipe",
    "llvmpipe",
    "paravirtual",
    "software",
    "swiftshader",
    "virtual",
    "virtio",
    "warp",
)


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


def backend_golden_identity(adapter: object) -> str:
    """Return the only committed-golden identity proven by ``adapter``."""
    adapter = _validate_adapter("adapter", adapter)
    if adapter.get("status", "ok") != "ok":
        raise ValueError("adapter status is not ok")
    device_type = str(adapter["device_type"]).lower()
    if device_type not in {"discretegpu", "integratedgpu"}:
        raise ValueError("adapter is not a physical GPU")
    name = str(adapter["name"]).lower()
    if not name or any(token in name for token in SOFTWARE_ADAPTER_TOKENS):
        raise ValueError("adapter name is software, virtual, or ambiguous")

    backend = str(adapter["backend"]).lower()
    if backend == "metal":
        if "apple" not in name:
            raise ValueError("the Metal golden requires an Apple adapter")
        return "metal"
    if backend == "dx12":
        if int(adapter["vendor"]) != 0x10DE or "nvidia" not in name:
            raise ValueError("the committed DX12 golden requires an NVIDIA adapter")
        return "dx12"
    if backend == "vulkan":
        _validate_adapter(REQUIRED_NVIDIA_LEG, adapter)
        return "nvidia-vulkan"
    raise ValueError(f"unsupported or ambiguous backend identity: {backend!r}")


def backend_golden_path(default_golden: Path, adapter: object) -> Path:
    """Route a render to its backend-specific hash without replacing Vulkan."""
    identity = backend_golden_identity(adapter)
    if identity == "nvidia-vulkan":
        return default_golden
    return default_golden.with_name(
        f"{default_golden.name.removesuffix('.sha256')}.{identity}.sha256"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"could not inspect candidate repository: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"could not inspect candidate repository: {detail}")
    return result.stdout


def _git_paths(repository: Path, *args: str) -> set[str]:
    return {path for path in _git(repository, *args).split("\0") if path}


def _validate_candidate_checkout(
    repository: Path,
    candidate_sha: str,
    fixture: Path,
    provenance_output: Path | None,
) -> None:
    if COMMIT_SHA_PATTERN.fullmatch(candidate_sha) is None:
        raise ValueError("candidate_sha must be a lowercase 40-hex commit SHA")

    repository = Path(repository).resolve()
    root = Path(_git(repository, "rev-parse", "--show-toplevel").strip()).resolve()
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if candidate_sha != head:
        raise ValueError(
            f"candidate_sha does not match repository HEAD: "
            f"candidate={candidate_sha}, HEAD={head}"
        )

    allowed = set()
    for output in (fixture, provenance_output):
        if output is None:
            continue
        try:
            relative = Path(output).resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "golden outputs must be inside the candidate repository"
            ) from exc
        allowed.add(relative.as_posix())

    staged = _git_paths(root, "diff", "--cached", "--name-only", "-z", "--")
    unstaged = _git_paths(root, "diff", "--name-only", "-z", "--")
    untracked = _git_paths(
        root, "ls-files", "--others", "--exclude-standard", "-z", "--"
    )
    unexpected = staged | ((unstaged | untracked) - allowed)
    if unexpected:
        raise ValueError(
            "repository has uncommitted source changes outside the generated "
            f"fixture/provenance outputs: {', '.join(sorted(unexpected))}"
        )


def golden_provenance_record(
    *,
    repository: Path,
    candidate_sha: str,
    wheel: Path,
    native: Path,
    adapter: object,
    width: int,
    height: int,
    generation_command: str,
    fixture: Path,
    provenance_output: Path | None = None,
) -> dict:
    """Build the complete provenance record required for a new golden."""
    _validate_candidate_checkout(repository, candidate_sha, fixture, provenance_output)
    if width <= 0 or height <= 0:
        raise ValueError("golden dimensions must be positive")
    if not generation_command.strip():
        raise ValueError("generation_command must be non-empty")
    identity = backend_golden_identity(adapter)
    adapter = dict(adapter)
    return {
        "schema": "forge3d.determinism-golden.v1",
        "candidate_sha": candidate_sha,
        "wheel_sha256": _sha256(wheel),
        "native_sha256": _sha256(native),
        "backend": str(adapter["backend"]).lower(),
        "golden_identity": identity,
        "adapter": adapter,
        "software_fallback": adapter["software_fallback"],
        "dimensions": {"width": width, "height": height},
        "generation_command": generation_command,
        "fixture_sha256": _sha256(fixture),
    }


def write_golden_provenance(path: Path, **fields) -> dict:
    """Write one canonical provenance record next to a generated fixture."""
    record = golden_provenance_record(provenance_output=path, **fields)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)

    produced = {}
    absent = {}
    adapters = {}
    goldens = {}
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
                continue
            try:
                golden_path = backend_golden_path(args.golden, adapters[leg])
                goldens[leg] = golden_path.read_text().split()[0].strip()
            except (OSError, ValueError) as exc:
                failures.append(f"{leg}: invalid backend golden route: {exc}")
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
    for leg, golden in sorted(goldens.items()):
        print(f"committed golden ({leg}): {golden}")

    if REQUIRED_NVIDIA_LEG not in produced:
        failures.append("required NVIDIA/Vulkan leg produced no hash")
    if not produced and not gated_failure:
        failures.append("no hardware-backed leg produced a hash")
    for leg, sha in produced.items():
        golden = goldens.get(leg)
        if golden is not None and sha != golden:
            failures.append(
                f"{leg}: mismatch against backend golden {golden}: actual={sha}"
            )

    if failures:
        print("DETERMINISM FAILURE (zero-byte tolerance):", file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    print("determinism diff: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
