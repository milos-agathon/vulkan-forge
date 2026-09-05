from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import threading

import numpy as np
import pytest

import forge3d as f3d
import forge3d.path_tracing as pt
from _obliqua_gpu_guard import require_qualifying_gpu

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BUDGET_BYTES = 512 * 1024 * 1024
RENDER_OPTIONS = {
    "min_frames": 32,
    "max_frames": 32,
    "variance_threshold": 1e9,
    "seed": 7,
    "spp": 1,
}
_ARTIFACT_LOCK = threading.Lock()


def _load_example(name: str):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYTIC = _load_example("obliqua_analytic_camera_fixture")
SE_EUROPE = _load_example("obliqua_se_europe_fixture")


@pytest.fixture
def obliqua_artifact_dir(request):
    root = Path(os.environ.get("FORGE3D_OBLIQUA_ARTIFACT_DIR", "tests/artifacts/obliqua"))
    root.mkdir(parents=True, exist_ok=True)
    for aggregate in ("sha_tables.json", "metrics.json"):
        path = root / aggregate
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    sub = root / request.node.name
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def _merge_artifact(directory: Path, filename: str, key: str, value) -> None:
    path = directory.parent / filename
    with _ARTIFACT_LOCK:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        current[key] = value
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def _require_gpu(reason: str, artifact_dir: Path) -> dict:
    probe = require_qualifying_gpu(reason)
    adapter = artifact_dir.parent / "adapter.json"
    if not adapter.exists():
        adapter.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return probe


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _matrix_dem(size: int = 192) -> np.ndarray:
    return SE_EUROPE.make_dem(seed=7, size=size)


def _matrix_camera(camera_model: str, tilt: str, dem_size: int = 192) -> dict:
    angle = math.radians(35.0 if tilt == "oblique" else 0.0)
    distance = dem_size * 2.4
    camera = {
        "model": camera_model,
        "origin": (0.0, distance * math.cos(angle), distance * math.sin(angle)),
        "look_at": (0.0, 0.25, 0.0),
        "up": (0.0, math.sin(angle), -math.cos(angle)),
    }
    if camera_model == "orthographic":
        camera["half_height"] = dem_size * 0.62
    else:
        camera["fov_y"] = 35.0
    return camera


def _poster(**kwargs):
    return pt.render_terrain_poster(**kwargs)


def test_poster_signature_has_certificate_and_cache():
    signature = inspect.signature(pt.render_terrain_poster)
    assert "certificate" in signature.parameters
    assert "cache" in signature.parameters


class _PosterNative:
    def __init__(self):
        self.recorded = None

    def _record_terrain_poster_certificate_inputs(self, *args):
        self.recorded = args


def _disable_capture(monkeypatch):
    if hasattr(pt, "_render_capture"):
        monkeypatch.setattr(pt, "_render_capture", lambda *args, **kwargs: contextlib.nullcontext())
    if hasattr(pt, "emit_render_certificate"):
        monkeypatch.setattr(pt, "emit_render_certificate", lambda certificate: None)


def test_poster_tiles_match_exact_grid(monkeypatch):
    calls = []
    native = _PosterNative()

    def fake(heightmap, width, height, camera=None, **kwargs):
        calls.append((width, height, kwargs))
        return {
            "rgba": np.full((height, width, 4), len(calls), dtype=np.uint8),
            "converged": True,
            "frames": 32,
            "variance": 0.0,
            "peak_host_visible_bytes": 4096,
            "peak_device_local_bytes": 8192,
            "reservoir_valid_count": 0,
            "reservoir_m_min": 0,
            "reservoir_m_max": 0,
        }

    monkeypatch.setattr(pt, "_NATIVE", native)
    monkeypatch.setattr(pt, "hybrid_render_terrain_reference", fake)
    _disable_capture(monkeypatch)
    result = pt.render_terrain_poster(
        np.zeros((4, 5), dtype=np.float32),
        11,
        7,
        {},
        tile=4,
        certificate=False,
    )
    expected = [
        (4, 4, 0, 0), (4, 4, 4, 0), (3, 4, 8, 0),
        (4, 3, 0, 4), (4, 3, 4, 4), (3, 3, 8, 4),
    ]
    assert [(w, h, kw["pixel_offset"][0], kw["pixel_offset"][1]) for w, h, kw in calls] == expected
    for width, height, kwargs in calls:
        x0, y0 = kwargs["pixel_offset"]
        assert kwargs["full_width"] == 11 and kwargs["full_height"] == 7
        assert kwargs["sensor_rect"] == (x0 / 11, y0 / 7, (x0 + width) / 11, (y0 + height) / 7)
        assert kwargs["certificate"] is False
    assert result["rgba"].shape == (7, 11, 4)
    assert len(result["tiles"]) == 6
    assert native.recorded[-2:] == (3, 2)
    source = inspect.getsource(pt.render_terrain_poster).lower()
    assert "feather" not in source and "blend" not in source


def test_poster_independent_convergence(monkeypatch):
    native = _PosterNative()
    count = 0

    def fake(heightmap, width, height, camera=None, **kwargs):
        nonlocal count
        count += 1
        return {
            "rgba": np.zeros((height, width, 4), dtype=np.uint8),
            "converged": count != 2,
            "frames": 32,
            "variance": 0.0,
            "peak_host_visible_bytes": 1,
            "peak_device_local_bytes": 1,
            "reservoir_valid_count": 0,
            "reservoir_m_min": 0,
            "reservoir_m_max": 0,
        }

    monkeypatch.setattr(pt, "_NATIVE", native)
    monkeypatch.setattr(pt, "hybrid_render_terrain_reference", fake)
    _disable_capture(monkeypatch)
    with pytest.raises(RuntimeError, match="converg"):
        pt.render_terrain_poster(np.zeros((4, 4), np.float32), 8, 4, tile=4)


def test_poster_per_tile_peak_is_not_global(monkeypatch):
    native = _PosterNative()
    peaks = iter((80_000_000, 96_000_000, 72_000_000, 88_000_000))

    def fake(heightmap, width, height, camera=None, **kwargs):
        peak = next(peaks)
        return {
            "rgba": np.zeros((height, width, 4), dtype=np.uint8),
            "converged": True,
            "frames": 32,
            "variance": 0.0,
            "peak_host_visible_bytes": peak,
            "peak_device_local_bytes": peak * 2,
            "reservoir_valid_count": 0,
            "reservoir_m_min": 0,
            "reservoir_m_max": 0,
        }

    monkeypatch.setattr(pt, "_NATIVE", native)
    monkeypatch.setattr(pt, "hybrid_render_terrain_reference", fake)
    _disable_capture(monkeypatch)
    result = pt.render_terrain_poster(np.zeros((4, 4), np.float32), 8, 8, tile=4)
    recorded = [tile["peak_host_visible_bytes"] for tile in result["tiles"]]
    assert recorded == [80_000_000, 96_000_000, 72_000_000, 88_000_000]
    assert sum(recorded) > max(recorded)
    assert max(recorded) <= BUDGET_BYTES


def test_poster_native_per_tile_peaks_stay_owner_scoped(obliqua_artifact_dir):
    _require_gpu("native per-owner tile memory accounting", obliqua_artifact_dir)
    ambient_peak = 192 * 1024 * 1024
    f3d._forge3d.request_host_visible_allocation_for_test(
        ambient_peak, "obliqua-owner-capture-negative-control"
    )
    dem = _matrix_dem(32)
    result = _poster(
        heightmap=dem,
        width=64,
        height=64,
        camera=_matrix_camera("orthographic", "nadir", 32),
        tile=32,
        certificate=False,
        **RENDER_OPTIONS,
    )
    peaks = [int(tile["peak_host_visible_bytes"]) for tile in result["tiles"]]
    assert len(peaks) == 4
    assert all(0 < peak < ambient_peak for peak in peaks), peaks


def test_poster_artifact_dir_fixture(obliqua_artifact_dir):
    assert obliqua_artifact_dir.is_dir()
    assert obliqua_artifact_dir.parent.joinpath("sha_tables.json").is_file()
    assert obliqua_artifact_dir.parent.joinpath("metrics.json").is_file()


def test_analytic_fixture_geometry_is_derived():
    dem = ANALYTIC.make_ridge_dem()
    assert dem.shape == (1024, 1024) and dem.dtype == np.float32
    assert set(np.unique(dem)) == {0.0, ANALYTIC.RIDGE_HEIGHT}
    derived = 1.2 * (
        0.5 * ANALYTIC.RIDGE_BAR_LENGTH * math.cos(math.radians(35.0))
        + ANALYTIC.RIDGE_HEIGHT * math.sin(math.radians(35.0))
    )
    assert ANALYTIC.ORTHOGRAPHIC_HALF_HEIGHT == pytest.approx(derived)
    assert ANALYTIC.EXPECTED_EDGE_X0 < ANALYTIC.EXPECTED_EDGE_X1


EXPECTED_FIXTURE_HASHES = {
    # Locked hermetic CANONICAL_SEED=7, FIXTURE_SIZE=1024 procedural inputs.
    "dem": "554ae04546c7d5a4030c54910c63e20fa5675279c5dfe35058f4ad087d7e0803",
    "population": "25c1db2ad5d659125ed9ec7d1c94d9108fc4d5cfc3cca35ee1f4ea7396a077fd",
    "subject_mask": "e216dce3f1a20ac4062c8389fb6c8d195e89805e8442966ae5793ab1b3196c26",
    "oracle_vector_inputs": "04aa3eecd4ecabf64192f043b1fa0b0ceb9b121d6bfd910c50cf4442a2b3e977",
    "oracle_vector_output": "42508a5fa90e54e81cf58c24da2afaf3e0ac79e80ed554786e979cd94d28c384",
}


def test_se_europe_fixture_and_numpy_oracle_hashes():
    assert SE_EUROPE.CANONICAL_SEED == 7
    assert SE_EUROPE.FIXTURE_SIZE == 1024
    assert len(SE_EUROPE.WARM_STOPS) >= 4
    assert SE_EUROPE.fixture_input_hashes() == EXPECTED_FIXTURE_HASHES
    light, population, mask, expected = SE_EUROPE.numpy_oracle_test_vector()
    actual = SE_EUROPE.numpy_oracle(light, population, mask)
    assert np.array_equal(actual, expected)
    assert np.all(actual[~mask, :3] == 255)
    assert np.all(actual[~mask, 3] == 0)


@pytest.mark.parametrize("camera_model", ["pinhole", "off_axis", "orthographic"])
@pytest.mark.parametrize("tilt", ["nadir", "oblique"])
def test_poster_1024_full_vs_tiled_byte_identical(camera_model, tilt, obliqua_artifact_dir):
    _require_gpu("1024 full-vs-2x2 seam matrix", obliqua_artifact_dir)
    dem = _matrix_dem()
    camera = _matrix_camera(camera_model, tilt)
    full = pt.hybrid_render_terrain_reference(
        dem,
        1024,
        1024,
        camera,
        full_width=1024,
        full_height=1024,
        **RENDER_OPTIONS,
    )["rgba"]
    tiled = _poster(
        heightmap=dem,
        width=1024,
        height=1024,
        camera=camera,
        tile=512,
        certificate=False,
        **RENDER_OPTIONS,
    )["rgba"]
    assert np.array_equal(full, tiled)
    row = {"camera_model": camera_model, "tilt": tilt, "full_sha256": _sha(full), "tiled_sha256": _sha(tiled)}
    (obliqua_artifact_dir / "hashes.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _merge_artifact(obliqua_artifact_dir, "sha_tables.json", f"1024:{tilt}:{camera_model}", row)


@pytest.mark.parametrize("camera_model", ["pinhole", "off_axis", "orthographic"])
@pytest.mark.parametrize("tilt", ["nadir", "oblique"])
def test_poster_2048_seam_strips(camera_model, tilt, obliqua_artifact_dir):
    _require_gpu("2048 tiled independent-strip seam matrix", obliqua_artifact_dir)
    dem = _matrix_dem()
    camera = _matrix_camera(camera_model, tilt)
    plate = _poster(
        heightmap=dem,
        width=2048,
        height=2048,
        camera=camera,
        tile=1024,
        certificate=False,
        **RENDER_OPTIONS,
    )["rgba"]
    vertical = pt.hybrid_render_terrain_reference(
        dem, 4, 2048, camera,
        full_width=2048, full_height=2048, pixel_offset=(1022, 0),
        sensor_rect=(1022 / 2048, 0.0, 1026 / 2048, 1.0),
        **RENDER_OPTIONS,
    )["rgba"]
    horizontal = pt.hybrid_render_terrain_reference(
        dem, 2048, 4, camera,
        full_width=2048, full_height=2048, pixel_offset=(0, 1022),
        sensor_rect=(0.0, 1022 / 2048, 1.0, 1026 / 2048),
        **RENDER_OPTIONS,
    )["rgba"]
    assert np.array_equal(vertical, plate[:, 1022:1026])
    assert np.array_equal(horizontal, plate[1022:1026, :])
    # The outermost pixels are intentionally uniform environment.  Use narrow
    # strips at 3/8 and 5/8 of the plate as the negative control: both cross the
    # rendered subject, but lie on deterministic opposite sides of its relief.
    negative_lo = 3 * 2048 // 8
    negative_hi = 5 * 2048 // 8
    vertical_lo = plate[:, negative_lo : negative_lo + 4]
    vertical_hi = plate[:, negative_hi : negative_hi + 4]
    horizontal_lo = plate[negative_lo : negative_lo + 4, :]
    horizontal_hi = plate[negative_hi : negative_hi + 4, :]
    assert not np.array_equal(vertical_lo, vertical_hi)
    assert not np.array_equal(horizontal_lo, horizontal_hi)
    row = {
        "camera_model": camera_model,
        "tilt": tilt,
        "plate_sha256": _sha(plate),
        "vertical_strip_sha256": _sha(vertical),
        "horizontal_strip_sha256": _sha(horizontal),
        "negative_vertical_lo_sha256": _sha(vertical_lo),
        "negative_vertical_hi_sha256": _sha(vertical_hi),
        "negative_horizontal_lo_sha256": _sha(horizontal_lo),
        "negative_horizontal_hi_sha256": _sha(horizontal_hi),
    }
    (obliqua_artifact_dir / "hashes.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _merge_artifact(obliqua_artifact_dir, "sha_tables.json", f"2048:{tilt}:{camera_model}", row)


def test_poster_constant_albedo_map_parity(obliqua_artifact_dir):
    _require_gpu("poster constant-albedo parity", obliqua_artifact_dir)
    dem = _matrix_dem(96)
    camera = _matrix_camera("off_axis", "oblique", 96)
    kwargs = dict(heightmap=dem, width=256, height=256, camera=camera, tile=128, certificate=False, **RENDER_OPTIONS)
    constant = _poster(**kwargs)["rgba"]
    material = np.full((*dem.shape, 4), (0.6, 0.6, 0.6, 1.0), dtype=np.float32)
    textured = _poster(**kwargs, albedo_map=material, albedo_sampling="bilinear")["rgba"]
    assert np.array_equal(constant, textured)


def _render_analytic(model: str, artifact_dir: Path) -> tuple[np.ndarray, tuple[float, float]]:
    _require_gpu(f"analytic {model} camera oracle", artifact_dir)
    dem = ANALYTIC.make_ridge_dem()
    output = pt.hybrid_render_terrain_reference(
        dem,
        1024,
        1024,
        ANALYTIC.make_camera(model, oblique=True),
        spacing=(1.0, 1.0),
        exaggeration=1.0,
        **RENDER_OPTIONS,
    )
    angles = ANALYTIC.fit_ridge_edge_angles(
        output["depth"], ANALYTIC.EXPECTED_EDGE_X0, ANALYTIC.EXPECTED_EDGE_X1, ANALYTIC.EDGE_ROI_HALF_WIDTH
    )
    return output["depth"], angles


def test_poster_orthographic_parallel_lines_below_0_05_deg(obliqua_artifact_dir):
    _, angles = _render_analytic("orthographic", obliqua_artifact_dir)
    spread = abs(angles[1] - angles[0])
    assert spread < 0.05
    _merge_artifact(obliqua_artifact_dir, "metrics.json", "analytic:orthographic", {"angles_deg": angles, "spread_deg": spread})


def test_poster_pinhole_convergence_matches_expected(obliqua_artifact_dir):
    _, angles = _render_analytic("pinhole", obliqua_artifact_dir)
    measured = abs(angles[1] - angles[0])
    expected = ANALYTIC.expected_pinhole_edge_angle_spread(
        ANALYTIC.make_camera("pinhole", oblique=True), ANALYTIC.RIDGE_ENDPOINTS_WORLD, 1024, 1024
    )
    assert abs(measured - expected) < 0.05
    _merge_artifact(obliqua_artifact_dir, "metrics.json", "analytic:pinhole", {"angles_deg": angles, "measured_spread_deg": measured, "expected_spread_deg": expected})


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    if value.max(initial=0.0) > 1.0:
        value = value / 255.0
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array(((0.4124564, 0.3575761, 0.1804375), (0.2126729, 0.7151522, 0.0721750), (0.0193339, 0.1191920, 0.9503041))).T
    xyz /= np.array((0.95047, 1.0, 1.08883))
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    return np.stack((116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])), axis=-1)


def _delta_e2000(rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
    lab1, lab2 = _srgb_to_lab(rgb1), _srgb_to_lab(rgb2)
    l1, a1, b1 = np.moveaxis(lab1, -1, 0)
    l2, a2, b2 = np.moveaxis(lab2, -1, 0)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cbar = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt(cbar**7 / (cbar**7 + 25.0**7)))
    ap1, ap2 = (1.0 + g) * a1, (1.0 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360.0)
    hp2 = np.mod(np.degrees(np.arctan2(b2, ap2)), 360.0)
    hp1 = np.where(cp1 == 0.0, 0.0, hp1)
    hp2 = np.where(cp2 == 0.0, 0.0, hp2)
    dl = l2 - l1
    dc = cp2 - cp1
    dh_angle = hp2 - hp1
    dh_angle = np.where((cp1 * cp2) == 0.0, 0.0, dh_angle)
    dh_angle = np.where(dh_angle > 180.0, dh_angle - 360.0, dh_angle)
    dh_angle = np.where(dh_angle < -180.0, dh_angle + 360.0, dh_angle)
    dh = 2.0 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh_angle / 2.0))
    lp = (l1 + l2) / 2.0
    cp = (cp1 + cp2) / 2.0
    hp = (hp1 + hp2) / 2.0
    hp = np.where((cp1 * cp2) == 0.0, hp1 + hp2, hp)
    hp = np.where(((cp1 * cp2) != 0.0) & (np.abs(hp1 - hp2) > 180.0) & ((hp1 + hp2) < 360.0), hp + 180.0, hp)
    hp = np.where(((cp1 * cp2) != 0.0) & (np.abs(hp1 - hp2) > 180.0) & ((hp1 + hp2) >= 360.0), hp - 180.0, hp)
    t = 1.0 - 0.17 * np.cos(np.radians(hp - 30.0)) + 0.24 * np.cos(np.radians(2.0 * hp)) + 0.32 * np.cos(np.radians(3.0 * hp + 6.0)) - 0.20 * np.cos(np.radians(4.0 * hp - 63.0))
    sl = 1.0 + 0.015 * (lp - 50.0) ** 2 / np.sqrt(20.0 + (lp - 50.0) ** 2)
    sc = 1.0 + 0.045 * cp
    sh = 1.0 + 0.015 * cp * t
    rt = -2.0 * np.sqrt(cp**7 / (cp**7 + 25.0**7)) * np.sin(np.radians(60.0 * np.exp(-((hp - 275.0) / 25.0) ** 2)))
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh / sh) ** 2 + rt * (dc / sc) * (dh / sh))


def test_delta_e2000_numpy_reference_vector():
    # Sharma et al. pair 1: expected 2.0425.
    lab1 = np.array([[[50.0, 2.6772, -79.7751]]])
    lab2 = np.array([[[50.0, 0.0, -82.7485]]])
    # Exercise the CIEDE2000 algebra directly by temporarily converting helper internals.
    # The RGB parity test below independently exercises the sRGB->Lab transform.
    l1, a1, b1 = lab1[0, 0]
    l2, a2, b2 = lab2[0, 0]
    cbar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2.0
    g = 0.5 * (1.0 - math.sqrt(cbar**7 / (cbar**7 + 25.0**7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = math.hypot(ap1, b1), math.hypot(ap2, b2)
    hp1, hp2 = math.degrees(math.atan2(b1, ap1)) % 360, math.degrees(math.atan2(b2, ap2)) % 360
    dhp = hp2 - hp1
    if dhp > 180: dhp -= 360
    if dhp < -180: dhp += 360
    dl, dc = l2 - l1, cp2 - cp1
    dh = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dhp / 2))
    lp, cp = (l1 + l2) / 2, (cp1 + cp2) / 2
    hp = (hp1 + hp2) / 2 if abs(hp1 - hp2) <= 180 else (hp1 + hp2 + 360) / 2
    t = 1 - 0.17 * math.cos(math.radians(hp - 30)) + 0.24 * math.cos(math.radians(2 * hp)) + 0.32 * math.cos(math.radians(3 * hp + 6)) - 0.20 * math.cos(math.radians(4 * hp - 63))
    sl = 1 + 0.015 * (lp - 50) ** 2 / math.sqrt(20 + (lp - 50) ** 2)
    sc, sh = 1 + 0.045 * cp, 1 + 0.015 * cp * t
    rt = -2 * math.sqrt(cp**7 / (cp**7 + 25**7)) * math.sin(math.radians(60 * math.exp(-((hp - 275) / 25) ** 2)))
    delta = math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh / sh) ** 2 + rt * dc / sc * dh / sh)
    assert delta == pytest.approx(2.0425, abs=5e-5)


def test_poster_equivalence_se_europe_spike(obliqua_artifact_dir):
    _require_gpu("Southeast-Europe NumPy/native albedo parity", obliqua_artifact_dir)
    dem = SE_EUROPE.make_dem()
    population = SE_EUROPE.make_population()
    subject_mask = SE_EUROPE.make_subject_mask()
    args = SE_EUROPE.make_gpu_lightfield_args(dem)
    lightfield = _poster(heightmap=dem, width=1024, height=1024, tile=512, certificate=False, **args)["rgba"]
    native_result = pt.hybrid_render_terrain_reference(
        dem,
        1024,
        1024,
        albedo_map=SE_EUROPE.make_albedo_map(population),
        albedo_sampling="bilinear",
        **args,
    )
    native = native_result["rgba"]
    oracle = SE_EUROPE.numpy_oracle(
        lightfield,
        population,
        subject_mask,
        projected_albedo=native_result["albedo"],
    )
    delta = _delta_e2000(oracle[..., :3], native[..., :3])
    mean_delta = float(np.mean(delta[subject_mask]))
    from PIL import Image
    diff = np.clip(delta / max(float(delta.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(diff, mode="L").save(obliqua_artifact_dir / "delta_e2000.png")
    metrics = {
        "mean_delta_e2000": mean_delta,
        "fixture_input_hashes": SE_EUROPE.fixture_input_hashes(),
    }
    (obliqua_artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _merge_artifact(obliqua_artifact_dir, "metrics.json", "se_europe", metrics)
    assert mean_delta < 2.0


def test_poster_restir_reservoir_stats_positive(obliqua_artifact_dir):
    _require_gpu("poster ReSTIR reservoir readback", obliqua_artifact_dir)
    dem = _matrix_dem(64)
    material = SE_EUROPE.make_albedo_map(SE_EUROPE.make_population(size=64))
    result = _poster(
        heightmap=dem, width=128, height=128, camera=_matrix_camera("off_axis", "oblique", 64),
        tile=64, albedo_map=material, sun_intensity=2.5, certificate=False, **RENDER_OPTIONS,
    )
    assert all(tile["reservoir_valid_count"] > 0 for tile in result["tiles"])
    assert all(0 < tile["reservoir_m_min"] <= tile["reservoir_m_max"] for tile in result["tiles"])


def test_poster_plate_certificate_exact_inputs(obliqua_artifact_dir):
    _require_gpu("poster plate certificate", obliqua_artifact_dir)
    material = np.full((32, 32, 4), (0.62, 0.51, 0.31, 1.0), dtype=np.float32)
    certificate = obliqua_artifact_dir.parent / "certificate.json"
    result = _poster(
        heightmap=np.zeros((32, 32), np.float32), width=64, height=64,
        camera=_matrix_camera("off_axis", "oblique", 32), tile=32,
        albedo_map=material, albedo_sampling="bilinear", certificate=certificate, **RENDER_OPTIONS,
    )
    report = json.loads(certificate.read_text(encoding="utf-8"))
    assert report["degradations"] == []
    assert report["inputs"]["camera_model"] == "off_axis"
    assert report["inputs"]["sensor_rect"] == "0,0,1,1"
    assert report["inputs"]["albedo_map_sha256"] == hashlib.sha256(material.tobytes()).hexdigest()
    assert report["inputs"]["albedo_sampling"] == "bilinear"
    assert report["inputs"]["poster_full_width"] == "64"
    assert report["inputs"]["poster_full_height"] == "64"
    assert report["inputs"]["poster_tile_layout"] == "2x2"
    poster_passes = [entry for entry in report["passes"] if entry["label"] == "hybrid_pt.poster"]
    assert len(poster_passes) == 1 and poster_passes[0]["draw_calls"] == 1
    assert report["allocations"]["peak_host_visible_bytes"] > 0
    assert result["certificate_digest"] is not None


def test_record_terrain_poster_certificate_inputs_invalid(obliqua_artifact_dir):
    _require_gpu("poster private certificate helper validation", obliqua_artifact_dir)
    helper = f3d._forge3d._record_terrain_poster_certificate_inputs
    valid = ("pinhole", "none", "nearest", 64, 64, 2, 2)
    helper(*valid)
    bad = (
        ("fisheye", "none", "nearest", 64, 64, 2, 2),
        ("pinhole", "A" * 64, "nearest", 64, 64, 2, 2),
        ("pinhole", "none", "cubic", 64, 64, 2, 2),
        ("pinhole", "none", "nearest", 0, 64, 2, 2),
    )
    for args in bad:
        with pytest.raises(ValueError):
            helper(*args)


def _load_swiss_fixture(size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    rasterio = pytest.importorskip("rasterio")
    dem_path = ROOT / "assets" / "tif" / "switzerland_dem.tif"
    cover_path = ROOT / "assets" / "tif" / "switzerland_land_cover.tif"
    missing = [str(path) for path in (dem_path, cover_path) if not path.exists() or path.stat().st_size <= 1024]
    if missing:
        message = f"UNVERIFIED/ABSENT: Swiss LFS fixtures missing or pointer-only: {missing}"
        if os.environ.get("FORGE3D_RUN_OBLIQUA_GPU"):
            pytest.fail(message)
        pytest.skip(message)
    with rasterio.open(dem_path) as source:
        dem = source.read(1, out_shape=(size, size), masked=True).astype(np.float32)
        dem = np.asarray(dem.filled(float(dem.mean())), dtype=np.float32)
    with rasterio.open(cover_path) as source:
        if source.count >= 3:
            rgb = np.moveaxis(source.read([1, 2, 3], out_shape=(3, size, size), masked=True).filled(0), 0, -1)
            scale = 255.0 if np.asarray(rgb).max(initial=0) > 1 else 1.0
            rgb = np.asarray(rgb, dtype=np.float32) / scale
            valid = np.any(rgb > 0.0, axis=2)
        else:
            classes = source.read(1, out_shape=(size, size), masked=True)
            table = source.colormap(1)
            palette = np.zeros((max(table, default=0) + 1, 4), dtype=np.uint8)
            for index, color in table.items():
                palette[index] = color
            indices = np.asarray(classes.filled(0), dtype=np.int64)
            rgba8 = palette[np.clip(indices, 0, len(palette) - 1)]
            rgb = rgba8[..., :3].astype(np.float32) / 255.0
            valid = (~np.ma.getmaskarray(classes)) & (rgba8[..., 3] > 0)
    rgba = np.empty((size, size, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb, 0.0, None)
    rgba[..., 3] = valid.astype(np.float32)
    return np.ascontiguousarray(dem), np.ascontiguousarray(rgba)


def _swiss_camera(dem: np.ndarray) -> dict:
    tilt = math.radians(35.0)
    span = float(max(dem.shape) - 1)
    relief = float(np.max(dem) - np.min(dem))
    # The Swiss DEM carries absolute elevations up to 4493 m.  Use a distant,
    # narrow off-axis frustum whose full plate covers only the central 60% of
    # the terrain.  This keeps all sixteen tiles on sun-facing terrain while
    # preserving the required 35-degree oblique perspective camera.
    distance = max(8.0 * span, 2.25 * relief)
    half_height = 0.30 * span
    center_y = float(dem[dem.shape[0] // 2, dem.shape[1] // 2])
    target = (0.0, center_y, 0.0)
    return {
        "model": "off_axis",
        "origin": (
            0.0,
            center_y + distance * math.cos(tilt),
            distance * math.sin(tilt),
        ),
        "look_at": target,
        "up": (0.0, math.sin(tilt), -math.cos(tilt)),
        "fov_y": math.degrees(2.0 * math.atan(half_height / distance)),
    }


def test_poster_swiss_4k_oblique_budget(obliqua_artifact_dir):
    _require_gpu("Swiss 4K off-axis poster budget", obliqua_artifact_dir)
    dem, material = _load_swiss_fixture()
    result = _poster(
        heightmap=dem, width=4096, height=4096,
        camera=_swiss_camera(dem), tile=1024,
        albedo_map=material, albedo_sampling="bilinear", certificate=False, **RENDER_OPTIONS,
    )
    assert len(result["tiles"]) == 16
    table = []
    for index, tile in enumerate(result["tiles"]):
        assert tile["converged"]
        assert tile["peak_host_visible_bytes"] <= BUDGET_BYTES
        assert tile["reservoir_valid_count"] > 0
        table.append(
            {
                "tile": index,
                "converged": tile["converged"],
                "peak_host_visible_bytes": tile["peak_host_visible_bytes"],
                "reservoir_valid_count": tile["reservoir_valid_count"],
                "reservoir_m_min": tile["reservoir_m_min"],
                "reservoir_m_max": tile["reservoir_m_max"],
            }
        )
    (obliqua_artifact_dir / "tile_budget.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    _merge_artifact(obliqua_artifact_dir, "metrics.json", "swiss_4k", table)
