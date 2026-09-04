#!/usr/bin/env python3
"""Check whether the current adapter can run the terrain CI lanes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import forge3d as f3d
from forge3d.terrain_params import (
    AovSettings,
    PomSettings,
    SkySettings,
    make_terrain_params_config,
)


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
HARDWARE_DEVICE_TYPES = {"discretegpu", "integratedgpu", "virtualgpu"}
PHYSICAL_DEVICE_TYPES = {"discretegpu", "integratedgpu"}


def _write_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 160, 128]))


def _build_heightmap(size: int = 32) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    hill = 0.55 * np.exp(-((xx + 0.2) ** 2 * 7.5 + (yy - 0.1) ** 2 * 9.0))
    ridge = 0.25 * np.exp(-((xx - 0.35) ** 2 * 24.0 + (yy + 0.15) ** 2 * 16.0))
    slope = 0.20 * (1.0 - yy) + 0.08 * xx
    heightmap = hill + ridge + slope
    heightmap -= heightmap.min()
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#18391f"),
            (0.40, "#4e7c35"),
            (0.72, "#9a8552"),
            (1.0, "#f3f5f9"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _adapter_is_ci_safe(
    probe: dict,
    *,
    required_backend: str | None = None,
    require_nvidia: bool = False,
) -> bool:
    if probe.get("status") != "ok":
        return False
    device_type = str(probe.get("device_type", "")).lower()
    if device_type not in HARDWARE_DEVICE_TYPES:
        return False
    name = str(probe.get("name", "")).lower()
    if any(token in name for token in SOFTWARE_ADAPTER_TOKENS):
        return False
    if (
        required_backend is not None
        and str(probe.get("backend", "")).lower() != required_backend.lower()
    ):
        return False
    if require_nvidia:
        vendor = int(probe.get("vendor", 0))
        if probe.get("software_fallback") is not False:
            return False
        if device_type != "discretegpu":
            return False
        if vendor != 0x10DE:
            return False
        if "nvidia" not in name:
            return False
    return True


def _adapter_is_physical_metal(probe: dict) -> bool:
    if probe.get("status") != "ok":
        return False
    if str(probe.get("backend", "")).lower() != "metal":
        return False
    if bool(probe.get("software_fallback", False)):
        return False
    if str(probe.get("device_type", "")).lower() not in PHYSICAL_DEVICE_TYPES:
        return False
    name = str(probe.get("name", "")).lower()
    return not any(token in name for token in (*SOFTWARE_ADAPTER_TOKENS, "paravirtual"))


def _build_params(*, with_aov: bool, with_aether: bool = False) -> object:
    return f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=(96, 64),
            render_scale=1.0,
            terrain_span=2.8,
            msaa_samples=1,
            z_scale=1.35,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="colormap",
            colormap_strength=1.0,
            ibl_enabled=True,
            light_azimuth_deg=138.0,
            light_elevation_deg=24.0,
            sun_intensity=2.4,
            cam_radius=4.6,
            cam_phi_deg=138.0,
            cam_theta_deg=58.0,
            fov_y_deg=54.0,
            camera_mode="screen",
            overlays=[_build_overlay()],
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            aov=AovSettings(enabled=with_aov, albedo=True, normal=True, depth=True)
            if with_aov
            else None,
            sky=SkySettings(
                enabled=True,
                model="aether",
                turbidity=2.0,
                ozone_du=300.0,
                mie_g=0.8,
                aerial_perspective=True,
            )
            if with_aether
            else None,
        )
    )


def _smoke_render(mode: str) -> None:
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    heightmap = _build_heightmap()
    params = _build_params(
        with_aov=mode == "terrain-aov", with_aether=mode == "aether-metal"
    )

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp_hdr:
        hdr_path = Path(tmp_hdr.name)
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp_exr:
        exr_path = Path(tmp_exr.name)

    try:
        _write_test_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        if mode == "terrain-aov":
            beauty_frame, aov_frame = renderer.render_with_aov(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
            )
            aov_frame.save_exr(str(exr_path), beauty_frame)
        else:
            renderer.render_terrain_pbr_pom(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
                target=None,
                water_mask=None,
            )
    finally:
        hdr_path.unlink(missing_ok=True)
        exr_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("terrain", "terrain-aov", "sidera-metal", "aether-metal"),
        required=True,
    )
    parser.add_argument("--json", type=Path, help="write exact adapter/probe evidence")
    parser.add_argument(
        "--require-nvidia-vulkan",
        action="store_true",
        help="fail unless the selected adapter is physical NVIDIA on Vulkan",
    )
    args = parser.parse_args()

    backend = (
        "metal"
        if args.mode in {"sidera-metal", "aether-metal"}
        else os.environ.get("WGPU_BACKEND")
    )
    probe = f3d.device_probe(backend)
    print(f"terrain-ci-probe backend={backend!r} probe={probe}")

    evidence = {
        "requested_backend": backend,
        "mode": args.mode,
        "probe": probe,
    }

    def write_evidence(status: str) -> None:
        evidence["status"] = status
        if args.json is None:
            return
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.mode == "sidera-metal":
        if not _adapter_is_physical_metal(probe):
            write_evidence("absent")
            print("SIDERA reference lane ABSENT — no proven physical Metal adapter.")
            return 2
        write_evidence("passed")
        print("SIDERA reference lane has a physical Metal adapter.")
        return 0

    if args.mode == "aether-metal":
        if not _adapter_is_physical_metal(probe):
            write_evidence("absent")
            print("AETHER closure lane ABSENT — no proven physical Metal adapter.")
            return 2
        try:
            _smoke_render(args.mode)
        except Exception as exc:
            evidence["error"] = str(exc)
            write_evidence("failed")
            print(
                "AETHER closure probe: CRASH — physical Metal adapter present but "
                f"the live spectral terrain smoke failed: {exc}"
            )
            return 3
        write_evidence("passed")
        print("AETHER closure lane executed a live spectral terrain render on Metal.")
        return 0

    # Exit-code contract (CENSOR audit F-10). CI must not conflate "this
    # runner has no usable GPU" with "the renderer is broken":
    #   0 = probe positive: CI-safe hardware adapter AND the smoke render ran.
    #   2 = ABSENT: no CI-safe hardware adapter — the golden lane may honestly
    #       record an ABSENT marker and succeed.
    #   3 = CRASH: a CI-safe adapter is present but the terrain smoke render
    #       raised — a renderer defect that must FAIL the golden job.
    required_backend = "vulkan" if args.require_nvidia_vulkan else None
    if not _adapter_is_ci_safe(
        probe,
        required_backend=required_backend,
        require_nvidia=args.require_nvidia_vulkan,
    ):
        write_evidence("absent")
        print("terrain-ci-probe: ABSENT — no CI-safe hardware adapter on this runner.")
        return 2

    try:
        _smoke_render(args.mode)
    except Exception as exc:
        evidence["error"] = str(exc)
        write_evidence("failed")
        print(
            "terrain-ci-probe: CRASH — CI-safe adapter present but the terrain "
            f"smoke render failed: {exc}"
        )
        return 3

    write_evidence("passed")
    print("Terrain CI lane is supported on this adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
