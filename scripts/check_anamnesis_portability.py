"""Fail-closed native-terrain cache portability and mismatch controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import forge3d as f3d
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)


_PORTABLE_PROFILE = "terra-determinata-native-portable-v1"
_PASS_LABELS = [
    "terrain.prepare",
    "terrain.shadow",
    "terrain.forward",
    "terrain.resolve",
]


def _adapter(path: str) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    adapter = json.loads(lines[-1]).get("adapter")
    if not adapter or adapter.get("software_fallback") is not False:
        raise SystemExit("render lacks attributable physical-adapter metadata")
    return dict(adapter)


def _machine_id(path: str | None) -> str:
    if not path:
        raise SystemExit("native portability requires --machine-id-file")
    value = Path(path).read_text(encoding="utf-8").strip().lower()
    if not value or value in {"none", "unknown", "00000000-0000-0000-0000-000000000000"}:
        raise SystemExit("native portability machine identity is absent or invalid")
    return value


def _runner_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise SystemExit("native portability requires an attributable runner name")
    return value


def _native_render(
    cache: str, *, compatibility_profile: str = _PORTABLE_PROFILE
) -> tuple[bytes, dict]:
    os.environ["FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE"] = compatibility_profile
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="forge3d-anamnesis-portable-") as root:
        hdr_path = Path(root) / "environment.hdr"
        write_canonical_hdr(str(hdr_path))
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        frame = renderer.render_terrain_pbr_pom(
            f3d.MaterialSet.terrain_default(),
            f3d.IBL.from_hdr(str(hdr_path), intensity=1.0),
            f3d.TerrainRenderParams(_canonical_params_config()(512, 512)),
            heightmap,
            time_seconds=0.0,
            cache=cache,
        )
        rgba = np.ascontiguousarray(frame.to_numpy(), dtype=np.uint8).tobytes()
        return rgba, dict(renderer.last_anamnesis_cache_report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seed", "check", "mismatch"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--frame-blob")
    parser.add_argument("--golden")
    parser.add_argument("--adapter-record")
    parser.add_argument("--consumer-frame-blob")
    parser.add_argument("--consumer-golden")
    parser.add_argument("--consumer-adapter-record")
    parser.add_argument("--machine-id-file")
    parser.add_argument("--runner-name", default=os.environ.get("RUNNER_NAME", ""))
    parser.add_argument("--producer-backend", default="vulkan")
    parser.add_argument("--consumer-backend", default="dx12")
    args = parser.parse_args()
    record_path = Path(args.record)

    if args.mode == "seed":
        if not args.frame_blob or not args.golden or not args.adapter_record:
            parser.error("seed requires --frame-blob, --golden, and --adapter-record")
        png_sha256 = hashlib.sha256(Path(args.frame_blob).read_bytes()).hexdigest()
        golden_sha256 = Path(args.golden).read_text(encoding="utf-8").split()[0]
        if png_sha256 != golden_sha256:
            raise SystemExit(
                f"seed render differs from committed golden: actual={png_sha256} "
                f"golden={golden_sha256}"
            )
        adapter = _adapter(args.adapter_record)
        machine_id = _machine_id(args.machine_id_file)
        runner_name = _runner_name(args.runner_name)
        rgba, report = _native_render(args.cache)
        if (
            report["hits"]
            or report["misses"] != _PASS_LABELS
            or report["graph_command_submissions"] != 6
        ):
            raise SystemExit(f"native producer did not seed every terrain pass: {report}")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "forge3d.anamnesis.native-portability/1",
            "golden_sha256": golden_sha256,
            "rgba_sha256": hashlib.sha256(rgba).hexdigest(),
            "producer_adapter": adapter,
            "producer_machine_id": machine_id,
            "producer_runner_name": runner_name,
            "compatibility_profile": _PORTABLE_PROFILE,
            "engine_fingerprint": json.loads(f3d.anamnesis_engine_fingerprint()),
            "native_passes": _PASS_LABELS,
            "seed_report": report,
        }
        record_path.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "mode": "seed",
                    "machine_id": machine_id,
                    "runner_name": runner_name,
                    "hits": len(report["hits"]),
                    "misses": len(report["misses"]),
                    "bytes_written": report["bytes_written"],
                },
                sort_keys=True,
            )
        )
        return 0

    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != "forge3d.anamnesis.native-portability/1":
        raise SystemExit("portability record is not a native terrain graph record")
    if record.get("compatibility_profile") != _PORTABLE_PROFILE:
        raise SystemExit("portability record compatibility profile mismatch")
    if record.get("engine_fingerprint") != json.loads(
        f3d.anamnesis_engine_fingerprint()
    ):
        raise SystemExit("portability record was not produced by this exact engine head")

    if args.mode == "check":
        if not args.consumer_frame_blob or not args.consumer_golden or not args.consumer_adapter_record:
            parser.error(
                "check requires --consumer-frame-blob, --consumer-golden, and --consumer-adapter-record"
            )
        machine_id = _machine_id(args.machine_id_file)
        runner_name = _runner_name(args.runner_name)
        consumer_png_sha = hashlib.sha256(
            Path(args.consumer_frame_blob).read_bytes()
        ).hexdigest()
        consumer_golden_sha = Path(args.consumer_golden).read_text(encoding="utf-8").split()[0]
        if consumer_png_sha != consumer_golden_sha:
            raise SystemExit(
                "consumer render differs from committed golden: "
                f"actual={consumer_png_sha} golden={consumer_golden_sha}"
            )
        consumer_adapter = _adapter(args.consumer_adapter_record)
        producer_backend = str(record["producer_adapter"].get("backend", "")).lower()
        consumer_backend = str(consumer_adapter.get("backend", "")).lower()
        if (
            producer_backend != args.producer_backend.lower()
            or consumer_backend != args.consumer_backend.lower()
        ):
            raise SystemExit(
                "physical backend mismatch: "
                f"producer={producer_backend!r}, consumer={consumer_backend!r}"
            )
        rgba, report = _native_render(args.cache)
        if (
            report["hits"] != _PASS_LABELS
            or report["misses"]
            or report["graph_command_submissions"] != 0
        ):
            raise SystemExit(f"native consumer did not restore every terrain pass: {report}")
        result = {
            "mode": "check",
            "distinct_machine": machine_id != record["producer_machine_id"],
            "producer_machine_id": record["producer_machine_id"],
            "consumer_machine_id": machine_id,
            "producer_runner_name": record.get("producer_runner_name"),
            "consumer_runner_name": runner_name,
            "producer_adapter": record["producer_adapter"],
            "consumer_adapter": consumer_adapter,
            "backend_goldens_match": True,
            "hits": len(report["hits"]),
            "misses": len(report["misses"]),
            "hit_rate": report["hit_rate"],
            "bytes_read": report["bytes_read"],
        }
        print(json.dumps(result, sort_keys=True))
        return 0

    rgba, report = _native_render(
        args.cache, compatibility_profile=_PORTABLE_PROFILE + "-mismatch"
    )
    if (
        report["hits"]
        or report["misses"] != _PASS_LABELS
        or report["graph_command_submissions"] != 6
    ):
        raise SystemExit(f"native compatibility mismatch served stale terrain passes: {report}")
    print(
        json.dumps(
            {
                "mode": "mismatch",
                "hits": 0,
                "misses": len(report["misses"]),
                "hit_rate": report["hit_rate"],
                "cache_isolated": True,
                "mismatch_dimension": "compatibility_profile",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
