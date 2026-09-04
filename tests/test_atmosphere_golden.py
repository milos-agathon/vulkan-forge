"""Deterministic visible AETHER sunset hue-sweep golden."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from _aether_quadrature import (
    SUNSET_DISPLAY_ORDER_DEG,
    horizon_sun_signature,
    sunset_strip,
    write_constant_hdr,
)
from test_atmosphere_reference import (
    SIZE,
    _make_metal_runtime,
    _require_physical_metal,
    _sky_params,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "atmosphere" / "aether_sunset_sweep.png"
GPU_GOLDEN = (
    ROOT / "tests" / "golden" / "atmosphere" / "aether_gpu_sunset_sweep.png"
)


def _update_goldens_enabled() -> bool:
    return os.environ.get("FORGE3D_UPDATE_AETHER_GOLDEN") == "1"


def _assert_or_update_golden(actual: np.ndarray, golden: Path = GOLDEN) -> None:
    if _update_goldens_enabled():
        golden.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(actual, mode="RGB").save(golden)

    assert golden.is_file(), (
        f"missing AETHER golden {golden}; regenerate only with "
        "FORGE3D_UPDATE_AETHER_GOLDEN=1"
    )
    expected = np.asarray(Image.open(golden).convert("RGB"), dtype=np.uint8)
    assert expected.shape == actual.shape
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert float(difference.mean()) <= 0.20
    assert int(difference.max()) <= 2


def test_sunset_hue_sweep_visibly_runs_blue_cyan_orange_red() -> None:
    blue = horizon_sun_signature(89.0)
    cyan = horizon_sun_signature(60.0)
    orange = horizon_sun_signature(10.0)
    red = horizon_sun_signature(-5.0)

    assert blue[2] > blue[1] > blue[0]
    assert cyan[2] > cyan[0] and cyan[1] > cyan[0]
    assert orange[0] > 1.25 * orange[1] and orange[1] > 1.5 * orange[2]
    assert red[0] > 1.35 * red[1] and red[0] > 2.5 * red[2]


def test_aether_sunset_sweep_matches_committed_golden() -> None:
    _assert_or_update_golden(sunset_strip())


def test_active_gpu_sky_matches_committed_sunset_golden(tmp_path: Path) -> None:
    _require_physical_metal()
    hdr_path = tmp_path / "black.hdr"
    write_constant_hdr(hdr_path)
    renderer, material, ibl, _ = _make_metal_runtime(hdr_path)
    heightmap = np.zeros((8, 8), dtype=np.float32)
    panels = []
    for elevation in SUNSET_DISPLAY_ORDER_DEG:
        frame = renderer.render_terrain_pbr_pom(
            material,
            ibl,
            _sky_params(elevation, 90.0),
            heightmap,
        )
        panels.append(
            np.asarray(frame.to_numpy(), dtype=np.uint8)[:SIZE, :SIZE, :3]
        )
    _assert_or_update_golden(np.concatenate(panels, axis=1), GPU_GOLDEN)


def test_corrupt_refresh_candidate_cannot_mutate_or_pass_committed_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = GOLDEN.read_bytes()
    monkeypatch.setenv("FORGE3D_UPDATE_AETHER_GOLDEN", "1")
    monkeypatch.delenv("FORGE3D_UPDATE_AETHER_GOLDEN")
    corrupt = np.zeros_like(sunset_strip())
    with pytest.raises(AssertionError):
        _assert_or_update_golden(corrupt)
    assert GOLDEN.read_bytes() == before
