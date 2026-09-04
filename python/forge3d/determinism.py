# python/forge3d/determinism.py
# TERRA-DETERMINATA: fixed-seed reference render entry point for cross-vendor
# bit-exact verification of the terrain PBR + CSM + IBL + shared-tonemap path.
# RELEVANT FILES: src/core/gpu.rs, src/shaders/includes/determinism.wgsl,
# tests/test_determinism_hash.py, .github/workflows/determinism-matrix.yml
"""Deterministic reference rendering.

``render_reference`` renders the canonical TERRA-DETERMINATA scene through the
existing offscreen ``Session``/``TerrainRenderer`` path and returns the SHA-256
of the output PNG. The measurable claim: the same scene renders to the same
bytes on every backend/vendor, proven by diffing these hashes.

Backend pinning happens in a fresh subprocess because the wgpu backend is
locked process-wide at the FIRST GPU context creation (``src/core/gpu.rs``)
and ``Session(backend=...)`` is a documented no-op wrapper. The subprocess
sets ``FORGE3D_DETERMINISTIC=1`` and ``WGPU_BACKENDS=<backend>`` in its
environment before importing forge3d, which is the only ordering that
guarantees the pin precedes GPU init. Determinism failures are loud: the
native layer panics if the pinned backend cannot be honored.

This module is pure Python over the existing native surface — no PyO3 symbols
are added, so no ``__all__``/contract/.pyi registration is required.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

# --- Canonical scene ---------------------------------------------------------
# AEQUITAS's reference scene (src/path_tracing/reference_scene.rs) is a
# sphere/plane direct-lighting scene for PT-vs-raster adjudication; it does not
# exercise terrain_pbr_pom/CSM/IBL/tonemap, so TERRA-DETERMINATA defines its
# own canonical terrain scene here and documents every constant.
#
# Scene "terra_determinata_v1":
#   - 129x129 float32 DEM: smoothstep-shaped triangle-wave ridges plus a
#     diagonal ramp, generated from IEEE-exact float32 ops only (no libm
#     transcendentals), byte-identical on every host/architecture.
#   - MaterialSet.terrain_default() materials.
#   - IBL from a procedurally written 16x8 uncompressed RGBE .hdr with a fixed
#     horizontal/vertical gradient (byte-exact by construction).
#   - Directional sun az=135 deg, el=35 deg, intensity 2.5; IBL intensity 1.0.
#   - PCF shadows (512px map, 2 cascades) so the CSM filtering path is on.
#   - Fixed orbit camera: radius 4.0, phi=45 deg, theta=45 deg, fov 55 deg.
#   - POM disabled (its raymarch is deterministic but slow; out of the minimal
#     hash surface), MSAA 1, render_scale 1.0, exposure 1.0, gamma 2.2.
CANONICAL_SCENE = "terra_determinata_v1"

_HDR_WIDTH = 16
_HDR_HEIGHT = 8


def canonical_heightmap(size: int = 129) -> "np.ndarray":
    """Closed-form deterministic DEM for the canonical scene (float32).

    Transcendentals are BANNED here: ``np.sin``/``np.cos`` route through the
    platform libm, whose float32 results are not guaranteed bit-identical
    across OS/CPU architectures — the DEM input could diverge on the
    Apple/wasm CI legs before the GPU ever runs. This field therefore uses
    only IEEE-754 exactly-specified float32 operations (multiply, add,
    subtract, divide are correctly rounded; ``floor``/``abs``/``max`` are
    exact), with every intermediate held in float32: smoothstep-shaped
    triangle waves for the ridges plus a linear ramp, matching the visual
    character of the previous sine/cosine field.
    """
    f32 = np.float32
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)

    def tri01(t: "np.ndarray") -> "np.ndarray":
        # Triangle wave in [0, 1]: |2*frac(t) - 1| (floor/abs are exact).
        u = t - np.floor(t)
        return np.abs(f32(2.0) * u - f32(1.0))

    def sstep(u: "np.ndarray") -> "np.ndarray":
        # Smoothstep polynomial u^2*(3-2u): multiplies/adds only.
        return u * u * (f32(3.0) - f32(2.0) * u)

    ux = sstep(tri01(x * f32(1.0 / 44.0)))
    uy = sstep(tri01(y * f32(1.0 / 69.0)))
    ridge = (ux * f32(36.0) - f32(18.0)) + (uy * f32(24.0) - f32(12.0))
    ramp = x * f32(0.9) + y * f32(0.4)
    dem = ridge + ramp + f32(25.0)
    assert dem.dtype == np.float32
    return dem / f32(dem.max())


def write_canonical_hdr(path: Union[str, Path]) -> None:
    """Write the fixed 16x8 uncompressed RGBE environment map, byte-exact."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {_HDR_HEIGHT} +X {_HDR_WIDTH}\n".encode())
        for yy in range(_HDR_HEIGHT):
            for xx in range(_HDR_WIDTH):
                r = (xx * 255) // (_HDR_WIDTH - 1)
                g = (yy * 255) // (_HDR_HEIGHT - 1)
                f.write(bytes([r, g, 160, 128]))


def _canonical_params_config():
    from .terrain_params import (
        ClampSettings,
        IblSettings,
        LightSettings,
        LodSettings,
        PomSettings,
        SamplingSettings,
        ShadowSettings,
        TerrainRenderParams as TerrainRenderParamsConfig,
        TriplanarSettings,
    )

    def build(width: int, height: int):
        return TerrainRenderParamsConfig(
            size_px=(width, height),
            render_scale=1.0,
            terrain_span=2.0,
            msaa_samples=1,
            z_scale=1.0,
            cam_target=[0.0, 0.0, 0.0],
            cam_radius=4.0,
            cam_phi_deg=45.0,
            cam_theta_deg=45.0,
            cam_gamma_deg=0.0,
            fov_y_deg=55.0,
            clip=(0.1, 250.0),
            light=LightSettings("Directional", 135.0, 35.0, 2.5, [1.0, 1.0, 1.0]),
            ibl=IblSettings(True, 1.0, 0.0),
            shadows=ShadowSettings(
                True, "PCF", 512, 2, 250.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
            ),
            triplanar=TriplanarSettings(6.0, 4.0, 1.0),
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            lod=LodSettings(0, 0.0, 0.0),
            sampling=SamplingSettings(
                "Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"
            ),
            clamp=ClampSettings(
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
            ),
            overlays=[],
            exposure=1.0,
            gamma=2.2,
            albedo_mode="mix",
            colormap_strength=0.5,
        )

    return build


def _render_reference_inprocess(
    scene_spec: str,
    width: int,
    height: int,
    out_png: Union[str, Path],
    certificate: Union[bool, str, Path] = False,
    cache: Union[str, Path, None] = None,
) -> str:
    """Render the canonical scene in THIS process and return the PNG SHA-256.

    Requires the deterministic env pins to be in place before forge3d touches
    the GPU; ``render_reference`` guarantees that by using a fresh subprocess.
    """
    if scene_spec != CANONICAL_SCENE:
        raise ValueError(
            f"unknown scene_spec {scene_spec!r}; only {CANONICAL_SCENE!r} is defined"
        )

    import forge3d as f3d

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    with tempfile.TemporaryDirectory() as tmpdir:
        hdr_path = os.path.join(tmpdir, "terra_determinata_env.hdr")
        write_canonical_hdr(hdr_path)
        env_maps = f3d.IBL.from_hdr(hdr_path, intensity=1.0)

    config = _canonical_params_config()(width, height)
    params = f3d.TerrainRenderParams(config)
    heightmap = canonical_heightmap()

    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=env_maps,
        params=params,
        heightmap=heightmap,
        target=None,
        certificate=certificate,
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    frame.save(str(out_png))
    return hashlib.sha256(out_png.read_bytes()).hexdigest()


def render_reference(
    scene_spec: Optional[str] = None,
    *,
    width: int = 512,
    height: int = 512,
    backend: Optional[str] = None,
    out_png: Union[str, Path],
    certificate: Union[bool, str, Path] = False,
    cache: Union[str, Path, None] = None,
) -> str:
    """Render the canonical deterministic scene and return the PNG's SHA-256.

    Args:
        scene_spec: scene name; ``None`` selects :data:`CANONICAL_SCENE`.
        width/height: output size in pixels.
        backend: single wgpu backend to pin (``"dx12"``, ``"vulkan"``,
            ``"metal"``, ``"gl"``, ``"webgpu"``). ``None`` inherits an already
            exported ``WGPU_BACKENDS``/``WGPU_BACKEND``; one of the two must
            name a backend or the native layer refuses to run (loudly).
        out_png: destination PNG path.
        certificate: ``False`` disables certificate output, a path writes the
            signed certificate there, and ``True`` writes
            ``<out_png stem>.certificate.json`` because the GPU render occurs
            in a child process.

    The render runs in a fresh Python subprocess with
    ``FORGE3D_DETERMINISTIC=1`` and the backend env var exported BEFORE any
    GPU context creation, because the backend is locked process-wide at first
    use and ``Session(backend=...)`` does not switch it.
    """
    _ = cache
    scene = scene_spec or CANONICAL_SCENE
    if backend is None and not (
        os.environ.get("WGPU_BACKENDS") or os.environ.get("WGPU_BACKEND")
    ):
        raise ValueError(
            "render_reference requires an explicit backend: pass backend=... or "
            "export WGPU_BACKENDS before calling (FORGE3D_DETERMINISTIC refuses "
            "to guess a backend)"
        )

    env = dict(os.environ)
    env["FORGE3D_DETERMINISTIC"] = "1"
    source_python = Path(__file__).resolve().parents[1]
    if source_python.name == "python":
        env["PYTHONPATH"] = os.pathsep.join(
            [str(source_python), env["PYTHONPATH"]]
            if env.get("PYTHONPATH")
            else [str(source_python)]
        )
    if backend is not None:
        env["WGPU_BACKENDS"] = backend
        env.pop("WGPU_BACKEND", None)

    cmd = [
        sys.executable,
        "-m",
        "forge3d.determinism",
        "--scene",
        scene,
        "--width",
        str(width),
        "--height",
        str(height),
        "--out-png",
        str(out_png),
    ]
    if certificate:
        certificate_path = (
            Path(certificate)
            if not isinstance(certificate, bool)
            else Path(out_png).with_suffix(".certificate.json")
        )
        cmd.extend(["--certificate", str(certificate_path)])
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"deterministic reference render failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    # The child prints a single JSON line last; tolerate warnings above it.
    last_line = proc.stdout.strip().splitlines()[-1]
    result = json.loads(last_line)
    return result["sha256"]


def _main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render the TERRA-DETERMINATA canonical scene and print its PNG SHA-256."
    )
    parser.add_argument("--scene", default=CANONICAL_SCENE)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--certificate")
    args = parser.parse_args(argv)

    render_args = (args.scene, args.width, args.height, args.out_png)
    sha = (
        _render_reference_inprocess(*render_args, certificate=args.certificate)
        if args.certificate
        else _render_reference_inprocess(*render_args)
    )
    # Attribute the hash to the actual adapter that produced it; the CI
    # artifact checker rejects unattributed or software-backed hashes.
    import forge3d as f3d

    backend_env = os.environ.get("WGPU_BACKENDS") or os.environ.get("WGPU_BACKEND")
    probe = f3d.device_probe(backend_env)
    adapter = {
        "name": probe.get("name"),
        "backend": probe.get("backend"),
        "device_type": probe.get("device_type"),
        "vendor": probe.get("vendor"),
        "device": probe.get("device"),
        "software_fallback": probe.get("software_fallback"),
    }
    print(
        json.dumps(
            {
                "scene": args.scene,
                "sha256": sha,
                "backend_env": backend_env,
                "deterministic": os.environ.get("FORGE3D_DETERMINISTIC"),
                "adapter": adapter,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
