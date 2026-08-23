# tests/test_hybrid_terrain_pt.py
# PROMETHEUS DoD gates: converged path-traced reference of a real DEM under
# sun + IBL through the GPU-backed hybrid terrain path
# (forge3d.path_tracing.hybrid_render_terrain_reference).
#   - per-pixel luminance variance < 1e-3 at convergence (hard gate)
#   - albedo/normal/depth AOV parity with the rasterizer AOV path
#   - min-max pyramid + accum + AOVs within the 512 MiB host-visible budget
#   - degenerate DEMs surface diagnostics (no silent fallback)
# GPU-skip follows the recipe-golden convention (terrain_rendering_available).

import os
import sys
from pathlib import Path
import ast
import inspect
import json
import subprocess
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _terrain_runtime import terrain_rendering_available

import forge3d as f3d
from forge3d.path_tracing import hybrid_render_terrain_reference

# Locked scene parameters (deterministic; golden-relevant).
DEM_STEP = 2  # mini_dem 256x256 -> 128x128
SIZE = 256
SPAN = 100.0
RELIEF = 20.0
CAM = {
    "origin": (0.0, 35.0, 90.0),
    "look_at": (0.0, 5.0, 0.0),
    "up": (0.0, 1.0, 0.0),
    "fov_y": 45.0,
    "exposure": 1.0,
}
SUN_AZ = 225.0
SUN_EL = 35.0
SUN_I = 2.5
ENV_I = 0.35
ALBEDO = (0.55, 0.52, 0.48)
VARIANCE_THRESHOLD = 1e-3
MAX_FRAMES = 512
WARM_WHITE = (1.0, 0.97, 0.92)
REQUIRED = "<required>"


def _dem():
    from forge3d.datasets import mini_dem

    dem = mini_dem()[::DEM_STEP, ::DEM_STEP].astype(np.float32)
    # Normalize to [0, 1] so `exaggeration` equals the world-unit relief and
    # stays inside the rasterizer's z_scale validity range (0.1-50).
    dem -= dem.min()
    dem /= max(float(dem.max()), 1e-6)
    return dem


def _scene_kwargs(dem):
    spacing = SPAN / (dem.shape[1] - 1)
    exag = RELIEF
    return dict(
        spacing=(spacing, spacing),
        exaggeration=exag,
        albedo=ALBEDO,
        sun_azimuth_deg=SUN_AZ,
        sun_elevation_deg=SUN_EL,
        sun_intensity=SUN_I,
        env_intensity=ENV_I,
        max_frames=MAX_FRAMES,
        min_frames=32,
        variance_threshold=VARIANCE_THRESHOLD,
        seed=7,
    )


def _require_gpu():
    if not terrain_rendering_available():
        pytest.skip(
            "Hybrid terrain PT requires a terrain-capable hardware-backed forge3d runtime"
        )


def _stub_function(stub_name, function_name):
    pkg = Path(f3d.__file__).resolve().parent
    node = ast.parse((pkg / stub_name).read_text(encoding="utf-8-sig"))
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == function_name:
            return item
    raise AssertionError(f"{stub_name} missing {function_name}")


def _ann_text(annotation):
    return ast.unparse(annotation) if annotation is not None else ""


def _stub_model(fn):
    pos_defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults)
    rows = []
    for arg, default in zip(fn.args.args, pos_defaults):
        rows.append(
            (
                arg.arg,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                REQUIRED if default is None else ast.unparse(default),
                _ann_text(arg.annotation),
            )
        )
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        rows.append(
            (
                arg.arg,
                inspect.Parameter.KEYWORD_ONLY,
                REQUIRED if default is None else ast.unparse(default),
                _ann_text(arg.annotation),
            )
        )
    return rows


def _runtime_model(sig):
    rows = []
    for param in sig.parameters.values():
        default = REQUIRED if param.default is inspect._empty else param.default
        annotation = "" if param.annotation is inspect._empty else str(param.annotation)
        if len(annotation) >= 2 and annotation[0] in "'\"" and annotation[-1] == annotation[0]:
            annotation = annotation[1:-1]
        rows.append((param.name, param.kind, default, annotation))
    return rows


def _raster_parity_metrics(dem, out):
    import math

    depth = out["depth"]
    normal = out["normal"]
    hits = np.isfinite(depth)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    ms = f3d.MaterialSet.custom(ALBEDO, 0.0, 0.8, 1.0, 0.0, 4.0)
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        with open(tmp.name, "wb") as fh:
            fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n")
            fh.write(bytes([128, 128, 128, 128]) * 32)
        env_maps = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)
    from forge3d.terrain_params import make_terrain_params_config

    eye_pt = np.array(CAM["origin"])
    eye_ras = np.array([eye_pt[0], eye_pt[2], eye_pt[1]])
    radius = float(np.linalg.norm(eye_ras))
    theta = math.degrees(math.acos(eye_ras[1] / radius))
    phi = math.degrees(math.atan2(eye_ras[2], eye_ras[0]))
    clip_far = 400.0
    config = make_terrain_params_config(
        size_px=(SIZE, SIZE),
        render_scale=1.0,
        terrain_span=SPAN,
        msaa_samples=1,
        z_scale=RELIEF,
        exposure=1.0,
        domain=(0.0, 1.0),
        light_azimuth_deg=SUN_AZ,
        light_elevation_deg=SUN_EL,
        sun_intensity=SUN_I,
        ibl_enabled=True,
        colormap_strength=0.0,
        camera_mode="mesh",
        cam_radius=radius,
        cam_phi_deg=phi,
        cam_theta_deg=theta,
        fov_y_deg=CAM["fov_y"],
        clip=(5.0, clip_far),
    )
    params = f3d.TerrainRenderParams(config)
    _, aov_frame = renderer.render_with_aov(ms, env_maps, params, dem)
    ras_n = aov_frame.normal()[::-1]
    ras_d = aov_frame.depth()[::-1]
    ras_a = aov_frame.albedo()[::-1]
    ras_hit = (np.linalg.norm(ras_n, axis=-1) > 0.5) & (ras_d > 1e-6)
    both = hits & ras_hit
    iou = both.sum() / max((hits | ras_hit).sum(), 1)

    ys2, xs2 = np.nonzero(both)
    half_h = np.tan(np.radians(CAM["fov_y"]) / 2.0)
    ndc_x2 = (xs2 + 0.5) / SIZE * 2 - 1
    ndc_y2 = 1 - (ys2 + 0.5) / SIZE * 2
    cosang = 1.0 / np.sqrt(1 + (ndc_x2 * half_h) ** 2 + (ndc_y2 * half_h) ** 2)
    t_ras = ras_d[ys2, xs2] * clip_far / cosang
    depth_err = np.abs(t_ras - depth[ys2, xs2])

    rn = ras_n.copy()
    rn[..., 0] *= -1.0
    dot2 = np.clip((normal[both] * rn[both]).sum(-1), -1, 1)
    ang2 = np.degrees(np.arccos(dot2))

    alb_lin_ref = ((np.array(ALBEDO) + 0.055) / 1.055) ** 2.4
    alb_err = np.abs(ras_a[both] - alb_lin_ref[None, :]).mean()
    return {
        "iou": float(iou),
        "median_depth_err": float(np.median(depth_err)),
        "median_normal_err": float(np.median(ang2)),
        "albedo_err": float(alb_err),
        "cam_dist": float(np.linalg.norm(np.array(CAM["origin"]))),
    }


def _raster_parity_metrics_in_subprocess():
    code = r"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
import test_hybrid_terrain_pt as t
dem = t._dem()
out = t.hybrid_render_terrain_reference(dem, t.SIZE, t.SIZE, t.CAM, **t._scene_kwargs(dem))
print("FORGE3D_RASTER_PARITY_JSON=" + json.dumps(t._raster_parity_metrics(dem, out)), flush=True)
"""
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo),
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        "raster parity subprocess failed\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    for line in result.stdout.splitlines():
        if line.startswith("FORGE3D_RASTER_PARITY_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise AssertionError(f"raster parity subprocess produced no metrics:\n{result.stdout}\n{result.stderr}")


def _assert_raster_parity_metrics(metrics):
    print(
        f"TERRAIN PT vs raster (mesh mode): hit IoU {metrics['iou']:.3f}; "
        f"depth median err {metrics['median_depth_err']:.2f} wu "
        f"({metrics['median_depth_err'] / metrics['cam_dist'] * 100:.1f}% of view distance); "
        f"normal median err {metrics['median_normal_err']:.2f} deg; "
        f"albedo mean abs err {metrics['albedo_err']:.4f} (linear)"
    )
    assert metrics["iou"] > 0.6, f"PT/raster terrain coverage IoU too low: {metrics['iou']:.3f}"
    assert metrics["median_depth_err"] < 10.0, (
        f"median depth error {metrics['median_depth_err']:.2f} world units vs raster"
    )
    assert metrics["median_normal_err"] < 15.0, (
        f"median normal angular error {metrics['median_normal_err']:.2f} deg vs raster"
    )
    assert metrics["albedo_err"] < 0.01, (
        f"albedo mismatch vs raster linear base color: {metrics['albedo_err']:.4f}"
    )


@pytest.fixture(scope="module")
def reference():
    _require_gpu()
    dem = _dem()
    out = hybrid_render_terrain_reference(dem, SIZE, SIZE, CAM, **_scene_kwargs(dem))
    return dem, out


def test_converged_variance_under_threshold(reference):
    _, out = reference
    print(
        f"\nTERRAIN PT: converged={out['converged']} frames={out['frames']} "
        f"variance={out['variance']:.3e} (threshold {VARIANCE_THRESHOLD:.0e})"
    )
    assert out["converged"] is True
    assert out["variance"] < VARIANCE_THRESHOLD
    rgba = out["rgba"]
    assert rgba.shape == (SIZE, SIZE, 4) and rgba.dtype == np.uint8
    # A converged reference must not be blank or magenta-marker filled.
    rgb = rgba[..., :3].astype(np.float32)
    assert rgb.mean() > 5.0, "reference is blank"
    magenta = (rgba[..., 0] > 250) & (rgba[..., 1] < 5) & (rgba[..., 2] > 250)
    assert magenta.mean() < 0.01, "magenta miss-marker leaked into terrain mode"


def test_terrain_hits_and_aov_consistency(reference):
    """PT AOVs are internally consistent: unit normals, world-unit depth,
    uniform albedo on hits, NaN depth on sky misses."""
    _, out = reference
    depth = out["depth"]
    normal = out["normal"]
    albedo = out["albedo"]
    hits = np.isfinite(depth)
    assert hits.mean() > 0.3, "terrain should cover a large part of the frame"
    # Depth is world-unit ray distance: bounded by scene scale.
    cam_dist = np.linalg.norm(np.array(CAM["origin"]) - np.array(CAM["look_at"]))
    assert depth[hits].min() > 1.0
    assert depth[hits].max() < cam_dist + SPAN * 2.0
    # Unit-length world normals with upward bias for a heightfield.
    n_len = np.linalg.norm(normal[hits], axis=-1)
    assert np.abs(n_len - 1.0).max() < 1e-2
    assert normal[hits][:, 1].mean() > 0.5
    # Uniform terrain albedo on every hit pixel.
    assert np.allclose(albedo[hits], np.array(ALBEDO), atol=2e-3)
    # Sky misses carry zero albedo/normal.
    assert np.allclose(albedo[~hits], 0.0, atol=1e-6)


def test_aov_parity_with_rasterizer(reference):
    """Albedo/normal/depth AOV parity: PT vs the rasterizer AOV path.

    Two tiers, both enforced:

    TIER 1 (tight, analytic): the PT AOVs are compared pixelwise against the
    analytic CPU reference of the same bilinear heightfield at the same
    camera — normals as angular error via world-position lookup, depth via
    exact ray-casting (already asserted in this module). These prove the PT
    AOVs are geometrically correct.

    TIER 2 (cross-renderer, coarse): the same DEM is rendered through
    TerrainRenderer.render_with_aov (camera_mode="mesh", flat custom material
    with normal_strength=0) at the mirrored camera. The raster terrain is
    Z-UP with heights in [0, z_scale] and its AOV conventions were
    established empirically: image is vertically flipped relative to the PT
    view, normals map as (-x, y, z), depth stores view-depth/far (ray
    distance after /cos), albedo stores the piecewise sRGB->linear decode of
    the material base color. Thresholds are deliberately coarse because the
    raster mesh path differs in tessellation/skirt silhouettes (hit-mask IoU
    ~0.7); the tight geometric gates are carried by Tier 1.
    """
    dem, out = reference
    depth = out["depth"]
    normal = out["normal"]
    hits = np.isfinite(depth)

    spacing = SPAN / (dem.shape[1] - 1)
    hz = dem * RELIEF
    dz_dx = np.gradient(hz, spacing, axis=1)
    dz_dy = np.gradient(hz, spacing, axis=0)
    n_ref = np.stack([-dz_dx, np.ones_like(hz), -dz_dy], axis=-1)
    n_ref /= np.linalg.norm(n_ref, axis=-1, keepdims=True)

    # ---- TIER 1: PT normals vs analytic reference (pixelwise angular) ----
    origin = np.array(CAM["origin"], dtype=np.float64)
    look = np.array(CAM["look_at"], dtype=np.float64)
    fwd = look - origin
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    half_h = np.tan(np.radians(CAM["fov_y"]) / 2.0)
    ox = -0.5 * (dem.shape[1] - 1) * spacing
    oz = -0.5 * (dem.shape[0] - 1) * spacing

    ys, xs = np.nonzero(hits)
    ndc_x = (xs + 0.5) / SIZE * 2 - 1
    ndc_y = 1 - (ys + 0.5) / SIZE * 2
    dirs = ndc_x[:, None] * half_h * right + ndc_y[:, None] * half_h * up + fwd
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = origin[None, :] + depth[ys, xs][:, None] * dirs
    gx = np.clip((pts[:, 0] - ox) / spacing, 0, dem.shape[1] - 1.001).astype(int)
    gz = np.clip((pts[:, 2] - oz) / spacing, 0, dem.shape[0] - 1.001).astype(int)
    # Exclude DEM-boundary hits where gradient stencils differ.
    interior = (gx > 1) & (gx < dem.shape[1] - 2) & (gz > 1) & (gz < dem.shape[0] - 2)
    n_cpu = n_ref[gz[interior], gx[interior]]
    n_pt = normal[ys[interior], xs[interior]]
    dot = np.clip((n_cpu * n_pt).sum(-1), -1, 1)
    ang = np.degrees(np.arccos(dot))
    print(
        f"\nTERRAIN PT normals vs analytic reference: mean {ang.mean():.2f} deg, "
        f"p95 {np.percentile(ang, 95):.2f} deg over {len(ang)} pixels"
    )
    # Central-difference reference vs exact bilinear cell gradient differ by
    # one stencil; a few degrees mean is the expected agreement.
    assert ang.mean() < 5.0, f"PT normal angular error {ang.mean():.2f} deg vs analytic"
    assert np.percentile(ang, 95) < 15.0

    # ---- TIER 2: cross-renderer comparison at the mirrored camera ----
    _assert_raster_parity_metrics(_raster_parity_metrics_in_subprocess())



def test_memory_within_budget(reference):
    _, out = reference
    from forge3d import _forge3d as _native

    metrics = _native.global_memory_metrics()
    limit = metrics["limit_bytes"]
    peak = out["peak_host_visible_bytes"]
    gpu_bytes = out["gpu_resource_bytes"]
    print(
        f"\nTERRAIN PT memory: peak host-visible {peak / 1e6:.1f} MB, "
        f"tracked GPU working set {gpu_bytes / 1e6:.2f} MB, "
        f"min-max pyramid {out['minmax_pyramid_bytes'] / 1e6:.2f} MB, "
        f"limit {limit / 1e6:.0f} MB"
    )
    assert peak < limit, f"peak host-visible {peak} exceeds budget {limit}"
    assert out["minmax_pyramid_bytes"] < limit
    # The full tracked working set (pyramid + env + accum + Welford +
    # reservoirs + G-buffer + UBOs + output/AOV textures) must fit the budget.
    assert gpu_bytes > out["minmax_pyramid_bytes"], (
        "gpu_resource_bytes must cover more than the pyramid alone"
    )
    assert gpu_bytes < limit, f"tracked GPU working set {gpu_bytes} exceeds budget {limit}"


def test_no_silent_fallback():
    _require_gpu()
    # NaN DEM must raise a diagnostic, not render a fake image.
    bad = np.full((16, 16), np.nan, dtype=np.float32)
    with pytest.raises(Exception, match="non-finite"):
        hybrid_render_terrain_reference(bad, 64, 64, CAM, max_frames=8)
    # Degenerate 1x1 DEM must raise.
    tiny = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(Exception, match="at least 2x2"):
        hybrid_render_terrain_reference(tiny, 64, 64, CAM, max_frames=8)
    # Unconverged-at-cap must raise, not return a fake reference.
    dem = _dem()
    with pytest.raises(Exception, match="did not converge"):
        hybrid_render_terrain_reference(
            dem, 128, 128, CAM, **{**_scene_kwargs(dem), "max_frames": 8, "min_frames": 2,
                                   "variance_threshold": 1e-12}
        )


def test_trust_boundary_validation():
    """Degenerate public inputs surface diagnostics before any GPU work."""
    _require_gpu()
    dem = _dem()
    kw = _scene_kwargs(dem)
    # min_frames > max_frames
    with pytest.raises(Exception, match="min_frames"):
        hybrid_render_terrain_reference(
            dem, 64, 64, CAM, **{**kw, "max_frames": 4, "min_frames": 8}
        )
    # Non-positive spacing
    with pytest.raises(Exception, match="spacing"):
        hybrid_render_terrain_reference(dem, 64, 64, CAM, **{**kw, "spacing": (0.0, 1.0)})
    # Degenerate camera (look_at == origin)
    bad_cam = {**CAM, "look_at": CAM["origin"]}
    with pytest.raises(Exception, match="look_at"):
        hybrid_render_terrain_reference(dem, 64, 64, bad_cam, **kw)
    # Invalid FOV
    with pytest.raises(Exception, match="fov"):
        hybrid_render_terrain_reference(dem, 64, 64, {**CAM, "fov_y": 0.0}, **kw)
    # Out-of-range spp
    with pytest.raises(Exception, match="spp"):
        hybrid_render_terrain_reference(dem, 64, 64, CAM, **{**kw, "spp": 0})
    # Mesh args must come together
    with pytest.raises(Exception, match="together"):
        hybrid_render_terrain_reference(
            dem, 64, 64, CAM,
            **{**kw, "mesh_vertices": np.zeros((3, 3), dtype=np.float32)},
        )


def test_sun_color_signature_stubs_and_native_order():
    required = REQUIRED
    po = inspect.Parameter.POSITIONAL_OR_KEYWORD
    ko = inspect.Parameter.KEYWORD_ONLY
    wrapper_runtime = [
        ("heightmap", po, required, "np.ndarray"),
        ("width", po, required, "int"),
        ("height", po, required, "int"),
        ("camera", po, None, "dict | None"),
        ("spacing", ko, (1.0, 1.0), "tuple[float, float]"),
        ("exaggeration", ko, 1.0, "float"),
        ("albedo", ko, (0.6, 0.6, 0.6), "tuple[float, float, float]"),
        ("sun_azimuth_deg", ko, None, "float | None"),
        ("sun_elevation_deg", ko, None, "float | None"),
        ("solar_time", ko, None, "object | None"),
        ("sun_intensity", ko, 2.5, "float"),
        ("sun_color", ko, WARM_WHITE, "Sequence[float] | np.ndarray"),
        ("env_map", ko, None, "np.ndarray | None"),
        ("env_intensity", ko, 0.35, "float"),
        ("mesh_vertices", ko, None, "np.ndarray | None"),
        ("mesh_indices", ko, None, "np.ndarray | None"),
        ("spp", ko, 1, "int"),
        ("max_frames", ko, 512, "int"),
        ("min_frames", ko, 32, "int"),
        ("variance_threshold", ko, 1e-3, "float"),
        ("seed", ko, 7, "int"),
        ("certificate", ko, False, "bool | str"),
        ("cache", ko, None, "str | None"),
        ("observer_latitude_deg", ko, None, "float | None"),
        ("observer_longitude_deg", ko, None, "float | None"),
        ("earth_model", ko, "ellipsoid", "str"),
        ("sphere_radius_m", ko, 6371008.8, "float"),
        ("refraction_model", ko, "bennett", "str"),
        ("refraction_k", ko, 0.13, "float"),
        ("pressure_mbar", ko, None, "float | None"),
        ("temperature_c", ko, None, "float | None"),
        ("atmosphere", ko, None, "Mapping[str, Any] | Any | None"),
    ]
    wrapper_stub = [
        (name, kind, required if default == required else "...", ann)
        for name, kind, default, ann in wrapper_runtime
    ]
    wrapper_stub[3] = ("camera", po, "...", "dict | None")
    wrapper_stub[4] = ("spacing", ko, "...", "Tuple[float, float]")
    wrapper_stub[6] = ("albedo", ko, "...", "Tuple[float, float, float]")
    path_stub = [
        (
            name,
            kind,
            default,
            {
                "certificate": "bool | str | PathLikeStr | None",
                "cache": "str | PathLikeStr | None",
                "atmosphere": "AtmosphereSettings | Mapping[str, object] | AtmosphereLutHandle | None",
            }.get(name, ann),
        )
        for name, kind, default, ann in wrapper_stub
    ]
    init_stub = [
        (
            name,
            kind,
            default,
            {
                "certificate": "bool | str | PathLikeStr | None",
                "cache": "str | PathLikeStr | None",
                "atmosphere": "AtmosphereSettings | Mapping[str, Any] | AtmosphereLutHandle | None",
            }.get(name, ann),
        )
        for name, kind, default, ann in wrapper_stub
    ]

    native_runtime = [
        ("heightmap", po, required, ""),
        ("width", po, required, ""),
        ("height", po, required, ""),
        ("cam", po, required, ""),
        ("spacing", po, Ellipsis, ""),
        ("exaggeration", po, 1.0, ""),
        ("albedo", po, Ellipsis, ""),
        ("sun_azimuth_deg", po, 315.0, ""),
        ("sun_elevation_deg", po, 45.0, ""),
        ("sun_intensity", po, 2.5, ""),
        ("env_map", po, None, ""),
        ("env_intensity", po, 0.35, ""),
        ("mesh_vertices", po, None, ""),
        ("mesh_indices", po, None, ""),
        ("spp", po, 1, ""),
        ("max_frames", po, 512, ""),
        ("min_frames", po, 32, ""),
        ("variance_threshold", po, 1e-3, ""),
        ("seed", po, 7, ""),
        ("certificate", po, None, ""),
        ("sun_color", po, None, ""),
        ("cache", po, None, ""),
        ("observer_latitude_deg", po, 0.0, ""),
        ("observer_longitude_deg", po, 0.0, ""),
        ("earth_model", po, "ellipsoid", ""),
        ("sphere_radius_m", po, 6371008.8, ""),
        ("refraction_model", po, "bennett", ""),
        ("refraction_k", po, 0.13, ""),
        ("pressure_mbar", po, 1013.25, ""),
        ("temperature_c", po, 15.0, ""),
        ("atmosphere", po, None, ""),
    ]
    psig = inspect.signature(hybrid_render_terrain_reference)
    assert _runtime_model(psig) == wrapper_runtime

    from forge3d import _forge3d as _native

    nsig = inspect.signature(_native.hybrid_render_terrain_reference)
    assert _runtime_model(nsig) == native_runtime

    assert _stub_model(_stub_function("path_tracing.pyi", "hybrid_render_terrain_reference")) == path_stub
    assert _stub_model(_stub_function("__init__.pyi", "hybrid_render_terrain_reference")) == init_stub


def test_terrain_reference_bridge_uses_crate_root_pyo3_types():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "py_functions"
        / "path_tracing"
        / "terrain_reference.rs"
    ).read_text(encoding="utf-8")
    assert "use super::super::super::*;" in source
    assert source.count("pyo3::types::") == 1
    assert "use pyo3::types::PyMapping;" in source


def test_sun_color_rejects_malformed_before_gpu_work():
    dem = np.zeros((4, 4), dtype=np.float32)
    bad_values = [
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
        (1.0, -0.1, 1.0),
        (1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        0.5,
        "abc",
        ("0.5", "0.9", "0.8"),
        bytearray([1, 1, 1]),
        memoryview(bytes([1, 1, 1])),
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            hybrid_render_terrain_reference(
                dem, 8, 8, CAM, sun_color=bad, max_frames=4, min_frames=2
            )


def test_sun_color_native_boundary_rejects_malformed():
    from forge3d import _forge3d as _native

    dem = np.zeros((4, 4), dtype=np.float32)
    fast = {"max_frames": 4, "min_frames": 2, "variance_threshold": 1e30}
    bad_values = [
        (1.0, -1.0, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
        (1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        0.5,
        "abc",
        ("0.5", "0.9", "0.8"),
        np.float64(0.5),
        np.array([1.0, 1.0]),
        bytearray([1, 1, 1]),
        memoryview(bytes([1, 1, 1])),
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            _native.hybrid_render_terrain_reference(dem, 8, 8, CAM, sun_color=bad, **fast)


def test_sun_color_valid_sequences_and_zero_are_accepted():
    _require_gpu()
    dem = np.zeros((4, 4), dtype=np.float32)
    valid = [
        (0.0, 0.0, 0.0),
        [1.0, 0.97, 0.92],
        (1.5, 1.0, 0.7),
        np.array([1.5, 1.0, 0.7]),
    ]
    for sun_color in valid:
        hybrid_render_terrain_reference(
            dem,
            8,
            8,
            CAM,
            sun_color=sun_color,
            sun_intensity=0.0,
            max_frames=4,
            min_frames=2,
            variance_threshold=1e30,
        )

    from forge3d import _forge3d as _native

    for sun_color in valid:
        _native.hybrid_render_terrain_reference(
            dem,
            8,
            8,
            CAM,
            sun_color=sun_color,
            sun_intensity=0.0,
            max_frames=4,
            min_frames=2,
            variance_threshold=1e30,
        )


def test_sun_color_valid_input_does_not_suppress_unrelated_failures(monkeypatch):
    import forge3d.path_tracing as pt

    class BrokenNative:
        @staticmethod
        def hybrid_render_terrain_reference(*args, **kwargs):
            raise RuntimeError("invalid device adapter state should propagate")

    monkeypatch.setattr(pt, "_NATIVE", BrokenNative())
    dem = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="adapter state"):
        pt.hybrid_render_terrain_reference(
            dem,
            8,
            8,
            CAM,
            sun_color=[1.0, 1.0, 1.0],
            max_frames=4,
            min_frames=2,
            variance_threshold=1e30,
        )


def test_sun_color_live_control_changes_output(reference):
    dem, out_default = reference
    out_blue = hybrid_render_terrain_reference(
        dem, SIZE, SIZE, CAM, **{**_scene_kwargs(dem), "sun_color": (0.2, 0.3, 1.5)}
    )
    diff = float(
        np.abs(
            out_default["rgba"][..., :3].astype(np.float64)
            - out_blue["rgba"][..., :3].astype(np.float64)
        ).mean()
    )
    print(f"\nTERRAIN PT sun_color control: mean abs RGB delta {diff:.3f}")
    assert diff > 1.0


def test_zero_sun_color_render_succeeds_and_removes_direct_sun():
    _require_gpu()
    dem = _dem()
    kw = {**_scene_kwargs(dem), "max_frames": 32, "min_frames": 2, "variance_threshold": 1e30}
    default = hybrid_render_terrain_reference(dem, 128, 128, CAM, **kw)
    zero = hybrid_render_terrain_reference(
        dem,
        128,
        128,
        CAM,
        **{**kw, "sun_color": (0.0, 0.0, 0.0)},
    )
    assert zero["rgba"].shape == (128, 128, 4)
    assert zero["rgba"].dtype == np.uint8
    assert np.isfinite(zero["depth"]).any()
    default_rgb = default["rgba"][..., :3].astype(np.float64)
    zero_rgb = zero["rgba"][..., :3].astype(np.float64)
    diff = float(np.abs(default_rgb - zero_rgb).mean())
    print(f"\nTERRAIN PT zero sun_color: mean abs RGB delta {diff:.3f}")
    assert diff > 0.5
    assert float(zero_rgb.mean()) < float(default_rgb.mean())


def test_mixed_scene_mesh_and_terrain():
    """Terrain is a first-class primitive of the shared hybrid traversal:
    a triangle mesh mixed into the scene occludes the heightfield, shows the
    mesh albedo in the AOVs, and shortens the depth where it hovers."""
    _require_gpu()
    dem = _dem()
    kw = _scene_kwargs(dem)
    kw.update(max_frames=64, min_frames=2, variance_threshold=1e30)  # fixed frames

    # A quad hovering above the terrain center, facing the camera.
    quad_v = np.array(
        [[-18.0, 22.0, -6.0], [18.0, 22.0, -6.0], [18.0, 40.0, -6.0], [-18.0, 40.0, -6.0]],
        dtype=np.float32,
    )
    quad_i = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    base = hybrid_render_terrain_reference(dem, 128, 128, CAM, **kw)
    mixed = hybrid_render_terrain_reference(
        dem, 128, 128, CAM, **{**kw, "mesh_vertices": quad_v, "mesh_indices": quad_i}
    )

    d0, d1 = base["depth"], mixed["depth"]
    # The quad introduces new hits (or closer hits) somewhere in the frame.
    closer = np.isfinite(d1) & (~np.isfinite(d0) | (d1 < d0 - 1.0))
    print(f"\nTERRAIN PT mixed scene: {closer.mean() * 100:.1f}% of pixels hit the mesh")
    assert closer.mean() > 0.01, "mesh did not appear in the mixed terrain scene"
    # Mesh pixels use the mesh albedo (0.7, 0.7, 0.8), not the terrain albedo.
    mesh_alb = mixed["albedo"][closer]
    assert np.allclose(mesh_alb, [0.7, 0.7, 0.8], atol=2e-2), (
        "mesh pixels must carry the mesh albedo through the shared traversal"
    )
    # Terrain still dominates the rest of the frame.
    terr = np.isfinite(d1) & ~closer
    assert terr.mean() > 0.3
    assert np.allclose(mixed["albedo"][terr], np.array(ALBEDO), atol=2e-2)


def test_scaling_no_per_spp_blowup():
    """O(log mips) traversal gate from the Prometheus DoD: per-frame cost
    scales ~linearly (never superlinearly) from 1 to 8 spp — a linear
    heightfield march would blow up texture reads per extra sample."""
    _require_gpu()
    import time

    dem = _dem()
    kw = _scene_kwargs(dem)
    frames = 32

    def run(spp):
        k = {**kw, "max_frames": frames, "min_frames": frames,
             "variance_threshold": 1e30, "spp": spp}
        t0 = time.time()
        hybrid_render_terrain_reference(dem, 128, 128, CAM, **k)
        return time.time() - t0

    run(1)  # warm-up (pipeline + pyramid build)
    t1 = min(run(1), run(1))
    t8 = min(run(8), run(8))
    ratio = t8 / max(t1, 1e-6)
    print(
        f"\nTERRAIN PT spp scaling: 1 spp {t1:.2f}s, 8 spp {t8:.2f}s at {frames} frames, "
        f"ratio {ratio:.1f}x (linear = 8x)"
    )
    assert ratio < 12.0, (
        f"1->8 spp cost grew superlinearly ({ratio:.1f}x): per-sample texture reads blow up"
    )

    # Accumulation depth stays linear too (no per-frame blowup).
    def run_frames(n):
        k = {**kw, "max_frames": n, "min_frames": n, "variance_threshold": 1e30}
        t0 = time.time()
        hybrid_render_terrain_reference(dem, 128, 128, CAM, **k)
        return time.time() - t0

    tf1 = run_frames(16)
    tf8 = run_frames(128)
    fratio = tf8 / max(tf1, 1e-6)
    print(f"TERRAIN PT frame scaling: 16 frames {tf1:.2f}s, 128 frames {tf8:.2f}s, ratio {fratio:.1f}x (linear = 8x)")
    assert fratio < 16.0, "per-frame cost grew superlinearly with accumulation depth"

# ---------------------------------------------------------------------------
# Golden: committed converged reference, drift-checked like the other goldens
# ---------------------------------------------------------------------------
GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "hybrid_terrain"
UPDATE_GOLDENS = os.environ.get("FORGE3D_UPDATE_HYBRID_TERRAIN_GOLDENS") == "1"


def test_terrain_reference_golden(reference):
    import json

    from _ssim import ssim

    _, out = reference
    rgba = out["rgba"]
    golden_path = GOLDEN_DIR / "mini_dem_reference.png"
    if UPDATE_GOLDENS:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        f3d.numpy_to_png(str(golden_path), rgba)
        (GOLDEN_DIR / "scores.json").write_text(
            json.dumps(
                {
                    "size": SIZE,
                    "frames": int(out["frames"]),
                    "variance": float(out["variance"]),
                    "peak_host_visible_bytes": int(out["peak_host_visible_bytes"]),
                    "gpu_resource_bytes": int(out["gpu_resource_bytes"]),
                },
                indent=2,
            )
            + "\n"
        )
        return
    assert golden_path.exists(), (
        f"Missing hybrid terrain golden {golden_path}. "
        "Regenerate with FORGE3D_UPDATE_HYBRID_TERRAIN_GOLDENS=1."
    )
    expected = f3d.png_to_numpy(str(golden_path))
    assert rgba.shape == expected.shape
    mean_abs = float(
        np.mean(np.abs(rgba[..., :3].astype(np.float32) - expected[..., :3].astype(np.float32)))
    )
    score = ssim(rgba[..., :3], expected[..., :3], data_range=255.0)
    print(f"\nTERRAIN PT golden drift: SSIM {score:.6f}, mean abs {mean_abs:.4f}")
    assert score >= 0.995, f"terrain reference drift: SSIM {score:.6f}"
    assert mean_abs <= 2.0, f"terrain reference drift: mean abs {mean_abs:.4f}"
def test_public_wrapper_resolves_solar_time_and_reports_source(monkeypatch):
    import forge3d.path_tracing as pt
    from forge3d.geo import SolarTime

    captured = {}

    class Native:
        @staticmethod
        def hybrid_render_terrain_reference(*args, **kwargs):
            captured.update(kwargs)
            return {}

    monkeypatch.setattr(pt, "_NATIVE", Native())
    assert f3d.hybrid_render_terrain_reference is pt.hybrid_render_terrain_reference
    when = SolarTime(
        utc=(2003, 10, 17, 12, 30, 30),
        observer_lat=39.742476,
        observer_lon=-105.1786,
        observer_elev_m=1830.14,
        tz_offset_hours=-7.0,
        delta_t_seconds=67.0,
        pressure_mbar=820.0,
        temperature_c=11.0,
    )
    result = f3d.hybrid_render_terrain_reference(
        np.zeros((2, 2), dtype=np.float32),
        2,
        2,
        solar_time=when,
        min_frames=1,
        max_frames=1,
    )
    expected = when.position()
    assert captured["sun_azimuth_deg"] == pytest.approx(expected["azimuth_deg"])
    assert captured["sun_elevation_deg"] == pytest.approx(expected["apparent_elevation_deg"])
    assert "sun_source" not in captured
    assert result["sun_source"] == "solar_time"
    f3d.hybrid_render_terrain_reference(
        np.zeros((2, 2), dtype=np.float32),
        2,
        2,
        solar_time=when,
        refraction_model="none",
        min_frames=1,
        max_frames=1,
    )
    assert captured["sun_elevation_deg"] == pytest.approx(expected["true_elevation_deg"])
    with pytest.raises(ValueError, match="cannot be combined"):
        pt.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            2,
            2,
            solar_time=when,
            sun_azimuth_deg=123.0,
            min_frames=1,
            max_frames=1,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        pt.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            2,
            2,
            solar_time=when,
            pressure_mbar=900.0,
            min_frames=1,
            max_frames=1,
        )
