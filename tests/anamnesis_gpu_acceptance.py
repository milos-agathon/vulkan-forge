"""Physical 600-frame acceptance for native terrain ANAMNESIS passes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

import forge3d as f3d
from forge3d.anamnesis import render_sequence
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)


_NATIVE_PASSES = (
    "terrain.prepare",
    "terrain.shadow",
    "terrain.forward",
    "terrain.resolve",
)
_GLYPHS = {
    " ": (0, 0, 0, 0, 0),
    "N": (0x7F, 0x02, 0x04, 0x08, 0x7F),
    "O": (0x3E, 0x41, 0x41, 0x41, 0x3E),
    "d": (0x18, 0x24, 0x24, 0x12, 0x7F),
    "e": (0x38, 0x54, 0x54, 0x54, 0x18),
    "i": (0, 0x44, 0x7D, 0x40, 0),
    "l": (0, 0x41, 0x7F, 0x40, 0),
    "m": (0x7C, 0x04, 0x18, 0x04, 0x78),
    "s": (0x48, 0x54, 0x54, 0x54, 0x24),
    "t": (0x04, 0x3F, 0x44, 0x40, 0x20),
    "u": (0x3C, 0x40, 0x40, 0x20, 0x7C),
    "w": (0x3C, 0x40, 0x30, 0x40, 0x3C),
}


def _composite_label(rgba: bytes, text: str) -> bytes:
    output = np.frombuffer(rgba, dtype=np.uint8).reshape((64, 64, 4)).copy()
    for character_index, character in enumerate(text):
        for x, column in enumerate(_GLYPHS.get(character, _GLYPHS[" "])):
            pixel_x = 1 + character_index * 6 + x
            if pixel_x >= 64:
                break
            for y in range(7):
                if column & (1 << y):
                    output[1 + y, pixel_x] = (255, 255, 255, 255)
    return output.tobytes()


def _recipe(
    text: str, *, backend: str, height_sha256: str, hdr_sha256: str
) -> dict[str, Any]:
    return {
        "terrain": {
            "height_sha256": height_sha256,
            "material_set": "terrain_default/v1",
            "environment_sha256": hdr_sha256,
            "z_scale": 1.0,
        },
        "camera": {
            "path": "anamnesis-real-terrain-flythrough/v2",
            "phi_start": 20.0,
            "phi_step": 0.25,
            "theta": 45.0,
        },
        "layers": [
            {
                "kind": "label",
                "id": "summit",
                "text": text,
                "visible_frames": [250],
            }
        ],
        "anamnesis_state": {"backend": backend, "seed": 0xA11A17},
        "output": {"width": 64, "height": 64, "samples": 1, "format": "rgba8"},
    }


def _render_phase(
    *,
    renderer: Any,
    material_set: Any,
    env_maps: Any,
    heightmap: np.ndarray,
    cache: Path,
    phase: str,
) -> tuple[list[bytes], float, dict[str, Any]]:
    started = time.perf_counter()
    frames: list[bytes] = []
    hits: list[str] = []
    misses: list[str] = []
    bytes_read = 0
    bytes_written = 0
    wall_ms_saved = 0.0
    graph_command_submissions = 0
    for frame_index in range(600):
        if frame_index % 100 == 0:
            print(f"ANAMNESIS GPU {phase} frame {frame_index}/600", flush=True)
        config = _canonical_params_config()(64, 64)
        config.cam_phi_deg = 20.0 + frame_index * 0.25
        config.cam_theta_deg = 45.0
        frame = renderer.render_terrain_pbr_pom(
            material_set,
            env_maps,
            f3d.TerrainRenderParams(config),
            heightmap,
            time_seconds=frame_index / 60.0,
            cache=cache,
        )
        frames.append(np.ascontiguousarray(frame.to_numpy(), dtype=np.uint8).tobytes())
        report = dict(renderer.last_anamnesis_cache_report)
        hits.extend(report["hits"])
        misses.extend(report["misses"])
        bytes_read += int(report["bytes_read"])
        bytes_written += int(report["bytes_written"])
        wall_ms_saved += float(report["wall_ms_saved"])
        graph_command_submissions += int(report["graph_command_submissions"])
    elapsed = time.perf_counter() - started
    total = len(hits) + len(misses)
    return (
        frames,
        elapsed,
        {
            "hits": hits,
            "misses": misses,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "wall_ms_saved": wall_ms_saved,
            "graph_command_submissions": graph_command_submissions,
            "hit_rate": len(hits) / total if total else 0.0,
        },
    )


def run_acceptance(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    height_sha256 = hashlib.sha256(heightmap.tobytes()).hexdigest()
    hdr_path = root / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    hdr_sha256 = hashlib.sha256(hdr_path.read_bytes()).hexdigest()
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    backend = str(f3d.device_probe().get("backend", "unknown")).lower()

    seeded, cold_seconds, seed_report = _render_phase(
        renderer=renderer,
        material_set=material_set,
        env_maps=env_maps,
        heightmap=heightmap,
        cache=root / "warm-native",
        phase="seed",
    )
    incremental, incremental_seconds, incremental_report = _render_phase(
        renderer=renderer,
        material_set=material_set,
        env_maps=env_maps,
        heightmap=heightmap,
        cache=root / "warm-native",
        phase="incremental",
    )

    # Exercise the outer recipe scheduler too. Its terrain node consumes the
    # frames already produced by the native graph above; it never substitutes
    # for or bypasses the direct 600-frame native pass proof. The seed is the
    # cold baseline: rendering an identical second empty store would repeat the
    # same 600 native executions without strengthening the acceptance claim.
    outer_phase = ["seed"]
    outer_calls: list[tuple[str, str, int]] = []
    native_frames = {
        "seed": seeded,
        "incremental": incremental,
        "cold": seeded,
    }

    def terrain_frame(_state: Any, frame: int, _inputs: list[bytes]) -> bytes:
        outer_calls.append((outer_phase[0], "terrain.shade", frame))
        return native_frames[outer_phase[0]][frame]

    def identity(_state: Any, frame: int, inputs: list[bytes]) -> bytes:
        outer_calls.append((outer_phase[0], "identity", frame))
        return inputs[0]

    def compile_labels(
        state: list[dict[str, Any]], frame: int, _inputs: list[bytes]
    ) -> bytes:
        outer_calls.append((outer_phase[0], "label.compile", frame))
        return json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def composite_label(_state: Any, frame: int, inputs: list[bytes]) -> bytes:
        outer_calls.append((outer_phase[0], "label.composite", frame))
        return _composite_label(inputs[0], json.loads(inputs[1])[0]["text"])

    pass_executors = {
        "terrain.shade": terrain_frame,
        "accumulation": identity,
        "label.compile": compile_labels,
        "label.composite": composite_label,
        "frame.output": identity,
    }
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes() + f3d.__version__.encode("ascii")
    ).digest()
    pass_fingerprints = {
        label: hashlib.sha256(fingerprint + label.encode("utf-8")).digest()
        for label in pass_executors
    }
    pass_contexts = {
        "terrain.shade": hashlib.sha256(
            b"native-terrain-graph/v2\0"
            + heightmap.tobytes()
            + hdr_path.read_bytes()
            + f3d.__version__.encode("ascii")
        ).digest(),
        "accumulation": b"identity/no-hidden-inputs/v1",
        "label.compile": b"canonical-json-label-compiler/v1",
        "label.composite": json.dumps(
            {"glyphs": _GLYPHS, "width": 64, "height": 64},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        "frame.output": b"identity/no-hidden-inputs/v1",
    }
    original = _recipe(
        "Old summit",
        backend=backend,
        height_sha256=height_sha256,
        hdr_sha256=hdr_sha256,
    )
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"
    render_sequence(
        original,
        cache=root / "outer-warm",
        pass_executors=pass_executors,
        pass_executor_fingerprints=pass_fingerprints,
        pass_executor_contexts=pass_contexts,
        verify_reads=False,
    )
    outer_phase[0] = "incremental"
    outer_incremental = render_sequence(
        modified,
        cache=root / "outer-warm",
        pass_executors=pass_executors,
        pass_executor_fingerprints=pass_fingerprints,
        pass_executor_contexts=pass_contexts,
        verify_reads=False,
    )
    outer_phase[0] = "cold"
    outer_cold = render_sequence(
        modified,
        cache=root / "outer-cold",
        pass_executors=pass_executors,
        pass_executor_fingerprints=pass_fingerprints,
        pass_executor_contexts=pass_contexts,
        verify_reads=False,
    )
    symmetric_difference = sorted(
        set(outer_incremental.predicted_recompute)
        ^ set(outer_incremental.observed_recompute)
    )

    # Rejecting control: a one-word DEM edit must invalidate exactly every
    # dependent native terrain pass, then restore all four on repetition.
    changed_heightmap = heightmap.copy()
    changed_heightmap[0, 0] += np.float32(0.25)
    changed_config = _canonical_params_config()(64, 64)
    changed_config.cam_phi_deg = 20.0
    changed_config.cam_theta_deg = 45.0
    changed_params = f3d.TerrainRenderParams(changed_config)
    changed_frame = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        changed_params,
        changed_heightmap,
        time_seconds=0.0,
        cache=root / "warm-native",
    )
    changed_report = dict(renderer.last_anamnesis_cache_report)
    restored_changed = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        changed_params,
        changed_heightmap,
        time_seconds=0.0,
        cache=root / "warm-native",
    )
    restored_changed_report = dict(renderer.last_anamnesis_cache_report)

    expected_incremental_hits = list(_NATIVE_PASSES) * 600
    expected_seed_misses = list(_NATIVE_PASSES) * 600
    result = {
        "matching_frames": sum(
            left == right
            for left, right in zip(
                outer_incremental.frame_hashes, outer_cold.frame_hashes, strict=True
            )
        ),
        "frame_count": 600,
        "hash_lists_equal": outer_incremental.frame_hashes == outer_cold.frame_hashes,
        "predicted": len(outer_incremental.predicted_recompute),
        "observed": len(outer_incremental.observed_recompute),
        "symmetric_difference": symmetric_difference,
        "pass_labels": sorted(
            {label for _, label in outer_incremental.observed_recompute}
        ),
        "native_predicted": 0,
        "native_observed": len(incremental_report["misses"]),
        "native_incremental_hits": len(incremental_report["hits"]),
        "native_incremental_misses": len(incremental_report["misses"]),
        "native_incremental_graph_submissions": incremental_report[
            "graph_command_submissions"
        ],
        "native_invalidation_predicted": len(_NATIVE_PASSES),
        "native_invalidation_observed": len(changed_report["misses"]),
        "native_pass_labels": list(_NATIVE_PASSES),
        "cold_seconds": cold_seconds,
        "incremental_seconds": incremental_seconds,
        "speedup": cold_seconds / incremental_seconds,
        "cache_report": {
            key: value
            for key, value in incremental_report.items()
            if key not in {"hits", "misses"}
        }
        | {
            "hits": len(incremental_report["hits"]),
            "misses": len(incremental_report["misses"]),
        },
        "seed_native_misses": len(seed_report["misses"]),
        "seed_native_hits": len(seed_report["hits"]),
        "seed_graph_command_submissions": seed_report[
            "graph_command_submissions"
        ],
        "cold_native_misses": len(seed_report["misses"]),
        "cold_native_hits": len(seed_report["hits"]),
        "cold_graph_command_submissions": seed_report[
            "graph_command_submissions"
        ],
        "cold_baseline_source": "native-seed",
        "incremental_native_invocations": 600,
        "incremental_outer_terrain_executions": sum(
            1
            for phase, label, _ in outer_calls
            if phase == "incremental" and label == "terrain.shade"
        ),
        "backend": backend,
    }
    if result["matching_frames"] != 600 or not result["hash_lists_equal"]:
        raise AssertionError(f"GPU frame hash mismatch: {result}")
    if seeded != incremental:
        raise AssertionError(f"native warm restoration differs from its seed: {result}")
    if (
        result["predicted"] != 3
        or result["observed"] != 3
        or symmetric_difference
    ):
        raise AssertionError(f"outer recompute mismatch: {result}")
    if result["pass_labels"] != ["frame.output", "label.compile", "label.composite"]:
        raise AssertionError(f"unexpected outer recompute labels: {result}")
    if result["incremental_outer_terrain_executions"] != 0:
        raise AssertionError(f"outer scheduler recomputed terrain node: {result}")
    if incremental_report["hits"] != expected_incremental_hits:
        raise AssertionError(f"native pass restoration order mismatch: {result}")
    if incremental_report["misses"] or incremental_report["hit_rate"] != 1.0:
        raise AssertionError(f"incremental native terrain recomputed: {result}")
    if incremental_report["graph_command_submissions"] != 0:
        raise AssertionError(f"restored native graph submitted GPU work: {result}")
    if (
        seed_report["misses"] != expected_seed_misses
        or seed_report["hits"]
        or seed_report["graph_command_submissions"] != 3600
    ):
        raise AssertionError(f"native cold per-pass recompute set mismatch: {result}")
    if changed_report["hits"] or changed_report["misses"] != list(_NATIVE_PASSES):
        raise AssertionError(f"native DEM invalidation mismatch: {result}")
    if changed_report["graph_command_submissions"] != 6:
        raise AssertionError(f"native DEM invalidation did not execute graph work: {result}")
    if restored_changed_report["hits"] != list(_NATIVE_PASSES):
        raise AssertionError(f"changed native resources did not restore: {result}")
    if restored_changed_report["graph_command_submissions"] != 0:
        raise AssertionError(f"changed native restoration submitted GPU work: {result}")
    if changed_frame.to_numpy().tobytes() != restored_changed.to_numpy().tobytes():
        raise AssertionError(f"changed native restoration is not byte-identical: {result}")
    if result["speedup"] < 20.0:
        raise AssertionError(f"GPU speedup below 20x: {result}")

    result_path = os.environ.get("FORGE3D_ANAMNESIS_RESULT_PATH")
    if result_path:
        Path(result_path).write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forge3d-anamnesis-gpu-acceptance-") as root:
        payload = json.dumps(run_acceptance(root), sort_keys=True)
        if arguments.result is not None:
            arguments.result.write_text(payload + "\n", encoding="utf-8")
        print(payload)
