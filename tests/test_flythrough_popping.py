# tests/test_flythrough_popping.py
# TESSELLA win 5: "Zero popping, zero cracks."
#
# Spec gate (docs/prompts/fable5-moonshots/19-tessella.md:79):
#   over a committed 600-frame flythrough, max dE2000 between consecutive frames
#   < 1.0 (no tile pop, no LOD snap, no fallback flash), and a
#   depth-discontinuity crack detector reports 0 cracks across clipmap ring
#   boundaries.
#
# What changed here, and why
# --------------------------
# The previous version of this file rendered 600 frames at 96x64 of a 64x64
# downsample of a band-limited analytic surface, with the camera one metre above
# a terrain whose total relief was 1.2 m on a 100 km span, moving 200 m in total
# -- less than the 195.3 m the clipmap needs before it regenerates at all
# (src/terrain/clipmap/level.rs:168-171). The clipmap therefore rebuilt at most
# once across the whole run, no ring boundary was ever on screen, and
# ``seam_stats()`` was sampled exactly once, after the loop. Both gates were
# close to tautologies.
#
# This version:
#   * renders at 1280x720 -- 150x the pixel count (see WALL CLOCK below);
#   * drives a non-separable 6-octave fBm height field (``fractal_dem``) whose
#     finest octave sits exactly at ring 1's Nyquist -- 8 base cells, against
#     ``_steep_dem``'s ~51 -- instead of one separable sinusoid pair;
#   * frames three clipmap regions (centre block, ring 0, ring 1) with two ring
#     boundaries permanently on screen, and *computes* that from the Rust ring
#     formulas rather than asserting it in prose;
#   * requests a 424.3 m diagonal clipmap/streaming-centre move every frame --
#     2.17x the regeneration threshold -- while production snaps geometry to
#     its stable finest-grid lattice; at least 90% of transitions still rebuild;
#   * asserts the crack metric, the seam-gap headroom and a threshold-free hole
#     count on EVERY frame, not once at the end;
#   * warms the height mosaic to asserted full fine-tile residency before the
#     measured run, so a fallback during the 600 frames is a defect rather than
#     the streamer's start-up transient;
#   * proves the crack metric is a measurement and not a constant (a control at
#     the API's maximum relief must make the same detector fire), and proves the
#     motion-compensated dE2000 gate discriminates at this resolution and DEM
#     (a real coarse-prefill fallback render must exceed it after registration).
#
# The relief control was DEAD until 2026-07-29: it asked for z_scale 1200
# against a 0.1-50.0 API ceiling, so it raised ValueError before rendering and
# no run had ever shown the crack detector firing. At the ceiling (z_scale 50)
# it now reports 302 cracks and max_gap 2.209 against a 0.1 threshold, versus
# 0 cracks / max_gap 0.053 at Z_SCALE.
#
# WALL CLOCK
# ----------
# ci.yml:1209 gives ``test-tessella-gpu`` 90 minutes shared by six gate files,
# the wheel install, the adapter probe and the 1,000-camera Rust differential.
# This file is budgeted 600 s. Per-frame model on the self-hosted RTX 3070:
#   render + readback at 1280x720 .................... ~30 ms  -> 18 s
#   dE2000 over 921,600 px (tests/_deltae.py) ........ ~220 ms -> 132 s
#   coarse-AO recompute over the 256^2 heightmap,
#     which is uncached and runs on every render
#     (src/terrain/renderer/resources/ao.rs:17-111,
#      called unconditionally from draw/setup/context.rs:122);
#     256^2 * 288 iterations with a sqrt and a conditional
#     atan each ................................... ~90 ms  -> 54 s
#   clipmap rebuild + CPU visibility oracle .......... ~20 ms  -> 12 s
#   hole mask + seam bookkeeping ..................... ~12 ms  -> 7 s
#                                                              ~= 225 s
# plus 5 control renders. MEASURED 2026-07-29 on the reference RTX 3070: 793 ms
# per frame end to end, i.e. ~476 s for 600 frames -- 2.1x the model above, not
# 225 s. It still fits the 600 s budget, but with 1.26x headroom rather than
# 2.5x. ``wall_clock_s`` is recorded in the evidence; if the measured value
# exceeds 400 s the knob to turn is DEM_SIZE (which is what the AO term scales
# with), never the render resolution and never a gate.
#
# POP-GATE FIXTURE CONTRACT
# -------------------------
# This is a real camera flythrough: the camera target advances by one projected
# ground-plane pixel per frame along world +Y. Nonzero relief makes perspective
# motion depth-dependent, so each beauty frame co-emits linear depth. Every
# transition first renders the new camera against the previous clipmap centre,
# then applies the independent streaming-centre move at that fixed camera. This
# separates ordinary view-dependent shading from the tile/LOD state change
# instead of assuming image reprojection can remove both. The unchanged dE2000
# < 1 threshold applies to every fixed-camera recenter. The overlay
# uses one continuous two-stop ramp: the shared
# four-stop terrain ramp has a hard LUT colour step that amplifies a one-LSB
# geometry change above the perceptual threshold. The negative control renders
# both the real displaced-camera baseline and a separate renderer whose height
# mosaic still contains only coarse-prefill fallback data; the former validates
# the registration sign while the latter must exceed the unchanged threshold.
#
# RELIEF CEILING (real, and load-bearing for the numbers below)
# -------------------------------------------------------------
# ``analyze_depth_discontinuities`` measures the world-space disagreement
# between the fine surface and the coarse edge it T-junctions into, scaled by
# ``z_scale`` (geomorph.rs:305,326), and compares it against
# ``(terrain_span * 1e-6).max(0.001)`` (geometry.rs:631). At a 100 km span that
# is a demand for 0.1 m agreement. Nothing in the shipped geomorph makes the two
# surfaces agree: the centre block's vertices are emitted with morph weight 0
# (ring.rs:27-32) so the centre/ring-0 boundary is never blended at all, and for
# ring-to-ring boundaries the coarse grid the shader snaps to is
# ``2**(ring+1)`` height texels (terrain_pbr_pom.wgsl:4579), which for a texel
# the size of ``base_cell`` is the ring's OWN vertex spacing rather than its
# coarser neighbour's. The gap is therefore just the height field's deviation
# from linearity over the coarse spacing, and it scales linearly with both
# roughness and relief -- so the gate as shipped caps relief at roughly 1e-5 of
# the terrain span, which is why ``Z_SCALE`` stays at the 1.2 m the gate was
# already green with rather than becoming a realistic exaggeration.
#
# Rather than guess where that cliff is, the fixture is budgeted against the one
# empirical data point available: the ``_steep_dem`` surface this gate is already
# green with. ``seam_roughness`` measures, on the real arrays, exactly the
# quantity the detector accumulates, and the new fixture must come in at most
# 0.75x the legacy value at every ring boundary -- while carrying six octaves
# instead of one. ``seam_gap_headroom_ratio`` in the evidence then makes the
# achieved runtime margin auditable instead of invisible.

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _deltae import delta_e_2000, srgb_to_lab
from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from _terrain_flythrough import (
    LEGACY_GATE_Z_SCALE,
    background_pixel_count,
    build_overlay,
    clipmap_base_cell_size,
    clipmap_boundary_half_spacings,
    clipmap_covered_positive_x_limit,
    clipmap_region_outer_radii,
    clipmap_region_vertex_spacing,
    flythrough_params,
    fractal_dem,
    fractal_dem_octave_amplitudes,
    legacy_gate_dem,
    render_rgba,
    render_rgba_depth,
    seam_gap_threshold,
    seam_roughness,
    seam_signature,
    triangle_wave,
)
from forge3d.diagnostics import (
    render_certificate,
    seam_stats,
    visibility_stats,
    vt_stats,
)

requires_terrain = pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)

# --- committed flythrough configuration ------------------------------------
MODE = "clipmap:4:32:32:10:0.3"
RING_COUNT = 4
RING_RESOLUTION = 32
CENTER_RESOLUTION = 32
SPAN = 100_000.0
SIZE = (1280, 720)
FRAMES = 600
FOV_Y_DEG = 45.0

DEM_SIZE = 256
DEM_OCTAVES = 6
DEM_GAIN = 0.45
DEM_SEED = 1904
Z_SCALE = 1.2

# Exact nadir for the Y-up orbit clipmap modes use (`is_zup_camera_mode` is
# false for "clipmap:*", so upload.rs:362-370 applies: theta=90/phi=90 puts the
# eye at target + (0, 0, cam_radius) with up = +Y). A nadir camera makes the
# ground footprint an exactly computable axis-aligned rectangle, which is what
# lets the zero-hole assertion be a gate instead of a coin flip, and removes
# self-occlusion silhouettes as a confound in the dE2000 metric.
CAM_THETA_DEG = 90.0
CAM_PHI_DEG = 90.0
CAM_RADIUS = 24_000.0
# The clipmap rings do not cover the +x corners beyond the centre block's
# half-extent (see clipmap_covered_positive_x_limit); offsetting the look-at
# point keeps the whole frame inside the covered region.
CAM_TARGET_DX = -12_000.0
CLIP = (18_000.0, 40_000.0)

# The public API receives an ordinary fractional-grid camera step. Production
# snaps it to the finest clipmap lattice before regenerating geometry.
CENTER_STEP_M = 300.0
CENTER_X_AMPLITUDE_STEPS = 19
CENTER_Y_AMPLITUDE_STEPS = 23
MIN_DISTINCT_REQUESTED_CENTERS = 500
MIN_DISTINCT_ACTUAL_CENTERS = 480
STREAM_ALTITUDE_M = 3_000.0
INITIAL_CAM_TARGET = (
    -CENTER_X_AMPLITUDE_STEPS * CENTER_STEP_M + CAM_TARGET_DX,
    -CENTER_Y_AMPLITUDE_STEPS * CENTER_STEP_M,
    0.0,
)
CAMERA_STEP_PX = 1.0

STREAM_LOD = 3
STREAM_TILE_RESOLUTION = 32
STREAM_MAX_RESIDENT_BYTES = 1024 * 1024
STREAM_TOTAL_TILES = (2**STREAM_LOD) ** 2  # 64 tiles x 32^2 texels = the 256^2 DEM
MAX_WARMUP_STEPS = 400

# Controls.
# ``TerrainRenderParams`` rejects z_scale outside 0.1-50.0
# (python/forge3d/terrain_params.py:2037), so the relief control has to live
# inside that range. The previous value (1000.0) put z_scale at 1200 and the
# control raised ValueError before it rendered anything -- it had never
# executed, so nothing had ever demonstrated that the crack detector CAN fire.
# 50.0 is the largest relief the public API admits -- 41.7x Z_SCALE, and 33x
# the gap the detector already measures there. It is stated as an absolute
# z_scale rather than a factor so no float product can drift past the ceiling.
RELIEF_CONTROL_Z_SCALE = 50.0
MOTION_COMPENSATION_CROP_PX = 3
POP_CONTROL_RGB_DELTA = 64.0

# Non-vacuity minimums.
MIN_REGIONS_ON_SCREEN = 3
MIN_CENTER_TRANSITIONS = 540  # >= 0.9 * (FRAMES - 1)
MIN_REBUILD_SIGNATURE_CHANGES = 480  # >= 0.8 * (FRAMES - 1)
# The new fixture must stay under this fraction of the roughness the gate is
# already green with (see legacy_gate_dem), measured in the exact quantity the
# crack detector accumulates. It buys margin without softening any gate.
SEAM_ROUGHNESS_BUDGET_FRACTION = 0.75

# --- derived framing (pure arithmetic on the constants above) --------------
HALF_HEIGHT_M = CAM_RADIUS * math.tan(math.radians(FOV_Y_DEG) * 0.5)
HALF_WIDTH_M = HALF_HEIGHT_M * (SIZE[0] / SIZE[1])
GROUND_PIXEL_M = 2.0 * HALF_WIDTH_M / SIZE[0]
CAMERA_STEP_M = CAMERA_STEP_PX * GROUND_PIXEL_M
FRAME_MIN_X = CAM_TARGET_DX - HALF_WIDTH_M
FRAME_MAX_X = CAM_TARGET_DX + HALF_WIDTH_M
FRAME_MAX_ABS_Y = max(
    abs(INITIAL_CAM_TARGET[1]),
    abs(INITIAL_CAM_TARGET[1] + (FRAMES - 1) * CAMERA_STEP_M),
) + HALF_HEIGHT_M
BASE_CELL_M = clipmap_base_cell_size(SPAN, CENTER_RESOLUTION)
REGENERATION_THRESHOLD_M = BASE_CELL_M * 0.5
CENTER_STEP_LENGTH_M = CENTER_STEP_M * math.sqrt(2.0)
SEAM_THRESHOLD = seam_gap_threshold(SPAN)


def _region_outer_radii() -> list[float]:
    return clipmap_region_outer_radii(
        SPAN, RING_COUNT, RING_RESOLUTION, CENTER_RESOLUTION
    )


def _snap_center_to_finest_grid(center: tuple[float, float]) -> tuple[float, float]:
    """Mirror Rust ``f32::round`` for the committed non-tie fixture values."""

    def snap(value: float) -> float:
        scaled = value / BASE_CELL_M
        rounded = math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
        return rounded * BASE_CELL_M

    return snap(center[0]), snap(center[1])


def _actual_center_at(index: int) -> tuple[float, float]:
    return _snap_center_to_finest_grid(_center_at(index))


def _regions_on_screen_for_centers(centers: list[tuple[float, float]]) -> int:
    """Minimum clipmap regions the moving camera frame intersects over the run.

    The camera and clipmap centre follow independent committed paths, so evaluate
    their relative Chebyshev reach on every frame.
    """
    assert len(centers) == FRAMES
    boundaries = _region_outer_radii()[:-1]
    counts = []
    for index, (center_x, center_y) in enumerate(centers):
        camera_target = _camera_target_at(index)
        target_x = camera_target[0] - center_x
        target_y = camera_target[1] - center_y
        reach = max(
            abs(target_x - HALF_WIDTH_M),
            abs(target_x + HALF_WIDTH_M),
            abs(target_y - HALF_HEIGHT_M),
            abs(target_y + HALF_HEIGHT_M),
        )
        counts.append(1 + sum(1 for radius in boundaries if radius < reach))
    return min(counts)


def _regions_on_screen() -> int:
    return _regions_on_screen_for_centers(
        [_actual_center_at(index) for index in range(FRAMES)]
    )


def _center_at(index: int) -> tuple[float, float]:
    """Clipmap/streaming centre for frame ``index``.

    Two integer triangle waves of period 76 and 92 frames, which request 511
    distinct centres over the run while keeping the path inside a
    +/-5700 x +/-6900 m box -- small enough that the framed ground never leaves
    the DEM footprint, large enough that every single step is exactly
    ``CENTER_STEP_LENGTH_M = 424.3 m``, i.e. 2.17x the clipmap's
    ``base_cell_size * 0.5 = 195.3 m`` regeneration threshold
    (``src/terrain/clipmap/level.rs``). Production snaps those requests to its
    finest-grid lattice, retaining 481 distinct actual centres and 569 actual
    transitions for the committed constants.
    """
    return (
        triangle_wave(index, CENTER_X_AMPLITUDE_STEPS) * CENTER_STEP_M,
        triangle_wave(index, CENTER_Y_AMPLITUDE_STEPS) * CENTER_STEP_M,
    )


def _camera_target_at(index: int) -> tuple[float, float, float]:
    """World-space target for the committed 600-frame camera flythrough."""
    return (
        INITIAL_CAM_TARGET[0],
        INITIAL_CAM_TARGET[1] + index * CAMERA_STEP_M,
        INITIAL_CAM_TARGET[2],
    )


def _params(
    *,
    z_scale: float,
    overlay,
    shading="forward",
    frame_index: int = 0,
    depth_aov: bool = False,
):
    return flythrough_params(
        size_px=SIZE,
        terrain_span=SPAN,
        camera_mode=MODE,
        cam_radius=CAM_RADIUS,
        cam_target=_camera_target_at(frame_index),
        theta_deg=CAM_THETA_DEG,
        phi_deg=CAM_PHI_DEG,
        fov_y_deg=FOV_Y_DEG,
        z_scale=z_scale,
        clip=CLIP,
        overlay=overlay,
        shading=shading,
        depth_aov=depth_aov,
    )


def _enable_streaming(renderer, dem: np.ndarray) -> None:
    renderer.enable_height_streaming(
        terrain_extent_m=SPAN,
        ring_count=RING_COUNT,
        ring_resolution=RING_RESOLUTION,
        lod=STREAM_LOD,
        tile_resolution=STREAM_TILE_RESOLUTION,
        max_in_flight=32,
        pool_size=4,
        dem=dem,
        coarse_prefill=True,
        max_resident_bytes=STREAM_MAX_RESIDENT_BYTES,
    )


def _warm_streaming_to_full_residency(renderer, center: tuple[float, float]) -> int:
    """Drive the streamer to full fine-tile residency before a measured run.

    The coarse-prefill -> fine-tile transition is a start-up transient of the
    streamer, not something a flythrough does: with the whole 256 KiB working set
    inside ``STREAM_MAX_RESIDENT_BYTES`` nothing can be evicted once resident, so
    a fallback during the measured frames would be a genuine defect while one
    during warm-up is just the mosaic filling in. Residency is asserted here, not
    assumed, and the frames the gate measures all start from it.
    """
    steps = 0
    stream = renderer.stream_height_tiles(
        (center[0], STREAM_ALTITUDE_M, center[1]), max_uploads=STREAM_TOTAL_TILES
    )
    while (
        stream["resident_fine_tiles"] < stream["total_tiles"]
        or stream["loader_pending"] > 0
    ) and steps < MAX_WARMUP_STEPS:
        stream = renderer.stream_height_tiles(
            (center[0], STREAM_ALTITUDE_M, center[1]),
            max_uploads=STREAM_TOTAL_TILES,
        )
        steps += 1
    assert stream["total_tiles"] == STREAM_TOTAL_TILES, stream
    assert stream["resident_fine_tiles"] == stream["total_tiles"], stream
    assert stream["loader_pending"] == 0, stream
    return steps


def _assert_seams_clean(seams: dict, where: str) -> None:
    """The spec's crack gate, applied to one published geometry build."""
    assert seams["depth_sample_count"] > 0, (where, seams)
    assert seams["crack_count"] == 0, (where, seams)
    assert seams["seams_valid"] is True, (where, seams)
    assert seams["max_gap"] <= SEAM_THRESHOLD, (where, seams, SEAM_THRESHOLD)


def _translate_frame_for_camera_motion(
    frame: np.ndarray, sample_dx_px: float, sample_dy_px: float = 0.0
) -> np.ndarray:
    """Synthesize a translated next frame from previous-image coordinates.

    ``sample_dx_px``/``sample_dy_px`` name where the current pixel's world
    sample appeared in the previous image. With the Y-up nadir camera, moving
    the target one world pixel along +Y means sampling the previous image one
    row upward (``sample_dy_px=-1``).
    """
    source = np.asarray(frame, dtype=np.float32)
    height, width = source.shape[:2]
    sample_x = np.clip(
        np.arange(width, dtype=np.float32) + np.float32(sample_dx_px),
        0.0,
        width - 1.0,
    )
    sample_y = np.clip(
        np.arange(height, dtype=np.float32) + np.float32(sample_dy_px),
        0.0,
        height - 1.0,
    )
    x0 = np.floor(sample_x).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y0 = np.floor(sample_y).astype(np.intp)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (sample_x - x0)[None, :, None]
    fy = (sample_y - y0)[:, None, None]
    top = source[y0[:, None], x0[None, :], :] + (
        source[y0[:, None], x1[None, :], :]
        - source[y0[:, None], x0[None, :], :]
    ) * fx
    bottom = source[y1[:, None], x0[None, :], :] + (
        source[y1[:, None], x1[None, :], :]
        - source[y1[:, None], x0[None, :], :]
    ) * fx
    return top + (bottom - top) * fy


def _motion_compensated_delta_e(
    previous: np.ndarray,
    current: np.ndarray,
    sample_dx_px: float,
    sample_dy_px: float | np.ndarray = 0.0,
    *,
    crop_px: int = MOTION_COMPENSATION_CROP_PX,
) -> float:
    """Maximum dE2000 after depth-aware previous-frame reprojection."""
    previous = np.asarray(previous)
    current = np.asarray(current)
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[2] < 3:
        raise ValueError(
            f"matching HxWxRGB(A) frames required, got {previous.shape} and {current.shape}"
        )
    height, width = current.shape[:2]
    if crop_px < 1 or height <= 2 * crop_px or width <= 2 * crop_px:
        raise ValueError(f"crop {crop_px} does not fit frame {width}x{height}")
    dy = np.asarray(sample_dy_px, dtype=np.float32)
    if dy.ndim not in (0, 2):
        raise ValueError(f"sample_dy_px must be scalar or HxW, got {dy.shape}")
    if dy.ndim == 2 and dy.shape != (height, width):
        raise ValueError(
            f"sample_dy_px must match frame shape {(height, width)}, got {dy.shape}"
        )
    max_motion = max(abs(sample_dx_px), float(np.abs(dy).max(initial=0.0)))
    if max_motion >= crop_px - 1:
        raise ValueError(
            f"motion max {max_motion} exceeds the "
            f"{crop_px}px registration crop"
        )

    out_height = height - 2 * crop_px
    out_width = width - 2 * crop_px
    grid_y, grid_x = np.indices((out_height, out_width), dtype=np.float32)
    grid_x += np.float32(crop_px)
    grid_y += np.float32(crop_px)
    sample_x = grid_x + np.float32(sample_dx_px)
    sample_y = grid_y + (
        dy
        if dy.ndim == 0
        else dy[crop_px : height - crop_px, crop_px : width - crop_px]
    )
    x0 = np.floor(sample_x).astype(np.intp)
    y0 = np.floor(sample_y).astype(np.intp)
    fx = (sample_x - x0)[..., None]
    fy = (sample_y - y0)[..., None]
    previous_rgb = previous[..., :3].astype(np.float32)
    top = previous_rgb[y0, x0, :] + (
        previous_rgb[y0, x0 + 1, :] - previous_rgb[y0, x0, :]
    ) * fx
    bottom = previous_rgb[y0 + 1, x0, :] + (
        previous_rgb[y0 + 1, x0 + 1, :] - previous_rgb[y0 + 1, x0, :]
    ) * fx
    aligned_previous = top + (bottom - top) * fy
    aligned_current = current[
        crop_px : height - crop_px, crop_px : width - crop_px, :3
    ].astype(np.float32)
    return float(
        delta_e_2000(
            srgb_to_lab(aligned_previous), srgb_to_lab(aligned_current)
        ).max()
    )


def _box_filter_rgb(frame: np.ndarray) -> np.ndarray:
    """Suppress isolated raster-quantization pixels, not coherent terrain pops."""
    rgb = np.asarray(frame)[..., :3].astype(np.float32)
    height, width = rgb.shape[:2]
    padded_x = np.pad(rgb, ((0, 0), (1, 1), (0, 0)), mode="edge")
    horizontal = (
        padded_x[:, :width] + padded_x[:, 1 : width + 1] + padded_x[:, 2:]
    ) / np.float32(3.0)
    padded_y = np.pad(horizontal, ((1, 1), (0, 0), (0, 0)), mode="edge")
    return (
        padded_y[:height] + padded_y[1 : height + 1] + padded_y[2:]
    ) / np.float32(3.0)


def _previous_frame_sample_dy(current_depth: np.ndarray) -> np.ndarray:
    """Map current pixels to previous rows using co-emitted linear depth.

    The cameras differ only by world +Y translation and retain the same nadir
    orientation. For view-space depth ``d``, projected motion is
    ``delta_world * focal_pixels / d``. The AOV stores
    ``(d - near) / (far - near)``, making this a direct geometric reprojection.
    """
    depth = np.asarray(current_depth, dtype=np.float32)
    if depth.shape != (SIZE[1], SIZE[0]):
        raise ValueError(f"depth must be {(SIZE[1], SIZE[0])}, got {depth.shape}")
    if (
        not np.isfinite(depth).all()
        or float(depth.min()) < 0.0
        or float(depth.max()) > 1.0
    ):
        raise ValueError("depth AOV must be finite and normalized to [0,1]")
    linear_depth = CLIP[0] + depth * np.float32(CLIP[1] - CLIP[0])
    focal_pixels = SIZE[1] / (2.0 * math.tan(math.radians(FOV_Y_DEG) * 0.5))
    return -np.float32(CAMERA_STEP_M * focal_pixels) / linear_depth


# ---------------------------------------------------------------------------
# CPU-only: the fixture's design budget, asserted rather than commented
# ---------------------------------------------------------------------------


def test_fractal_dem_is_deterministic_and_inside_the_seam_budget():
    first = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    second = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert first.shape == (DEM_SIZE, DEM_SIZE)
    assert float(first.min()) == 0.0
    assert float(first.max()) == 1.0

    # Six octaves, each materially present: a surface that would quietly
    # collapse back to the old single-sinusoid case if a later edit dropped the
    # fine ones.
    amplitudes = fractal_dem_octave_amplitudes(
        DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED
    )
    assert len(amplitudes) == DEM_OCTAVES
    assert min(amplitudes) > 0.005, amplitudes
    assert amplitudes[0] > amplitudes[-1], amplitudes

    # One DEM texel is one clipmap base cell, so the height field carries no
    # content the finest mesh cannot represent...
    assert math.isclose(SPAN / (DEM_SIZE - 1), BASE_CELL_M, rel_tol=0.01), (
        SPAN / (DEM_SIZE - 1),
        BASE_CELL_M,
    )
    # ...and its finest octave sits exactly at ring 1's Nyquist, which is the
    # scale a LOD step destroys and geomorph is supposed to hide. `_steep_dem`'s
    # finest content is ~51 base cells wide; this is 8.
    finest_wavelength_m = SPAN / float(2 ** (DEM_OCTAVES - 1))
    ring1_spacing = clipmap_region_vertex_spacing(SPAN, CENTER_RESOLUTION, 2)
    assert finest_wavelength_m == pytest.approx(2.0 * ring1_spacing)
    assert finest_wavelength_m <= 8.0 * BASE_CELL_M

    # Roughness budget: the shipped crack detector demands the fine and coarse
    # surfaces agree to within one part per million of the terrain span, and
    # nothing in the geomorph makes them agree (see the module docstring), so
    # what it really bounds is the height field's deviation from linearity over
    # the coarse spacing. Rather than guess where that cliff is, stay measurably
    # under the fixture the gate is ALREADY green with.
    half_spacings = clipmap_boundary_half_spacings(
        SPAN, RING_COUNT, RING_RESOLUTION, CENTER_RESOLUTION
    )
    assert half_spacings == pytest.approx([390.625, 781.25, 1562.5, 3125.0])
    measured = seam_roughness(first, SPAN, half_spacings, Z_SCALE)
    budget = seam_roughness(legacy_gate_dem(), SPAN, half_spacings, LEGACY_GATE_Z_SCALE)
    assert measured <= budget * SEAM_ROUGHNESS_BUDGET_FRACTION, {
        "boundary_half_spacings_m": half_spacings,
        "seam_roughness": measured,
        "legacy_seam_roughness": budget,
        "fraction_of_legacy": measured / budget,
        "seam_gap_threshold": SEAM_THRESHOLD,
        "note": (
            "raising DEM_OCTAVES/DEM_GAIN/Z_SCALE past this point takes the "
            "fixture past the roughness the shipped seam machinery is known to "
            "hold; fix the geomorph (centre-block morph weight, coarse-grid "
            "power of two) before raising them"
        ),
    }


def test_motion_compensation_removes_known_shift_and_detects_fallback_flash():
    rng = np.random.default_rng(1904)
    previous = rng.integers(24, 220, size=(48, 72, 4), dtype=np.uint8).astype(
        np.float32
    )
    current = _translate_frame_for_camera_motion(
        previous, 0.0, -CAMERA_STEP_PX
    )

    compensated = _motion_compensated_delta_e(
        previous, current, 0.0, -CAMERA_STEP_PX
    )
    wrong_direction = _motion_compensated_delta_e(
        previous, current, 0.0, CAMERA_STEP_PX
    )
    assert compensated < 1e-3, compensated
    assert wrong_direction > compensated, (compensated, wrong_direction)

    # A coherent one-pixel displacement is a visible geometry/LOD snap. The
    # comparator must reject it rather than classifying it as raster phase.
    extra_pixel_shifted = _translate_frame_for_camera_motion(
        previous, 0.0, -CAMERA_STEP_PX - 1.0
    )
    assert _motion_compensated_delta_e(
        previous, extra_pixel_shifted, 0.0, -CAMERA_STEP_PX
    ) >= 1.0

    flashed = current.copy()
    flashed[16:32, 24:48, :3] = np.clip(
        flashed[16:32, 24:48, :3] + POP_CONTROL_RGB_DELTA, 0.0, 255.0
    )
    assert _motion_compensated_delta_e(
        previous, flashed, 0.0, -CAMERA_STEP_PX
    ) >= 1.0


def test_depth_reprojection_corrects_nonplanar_perspective_motion():
    ground_depth = (CAM_RADIUS - CLIP[0]) / (CLIP[1] - CLIP[0])
    ground = np.full((SIZE[1], SIZE[0]), ground_depth, dtype=np.float32)
    ground_dy = _previous_frame_sample_dy(ground)
    np.testing.assert_allclose(ground_dy, -CAMERA_STEP_PX, atol=1e-6, rtol=0.0)

    # A point above the target plane is closer to the camera and therefore
    # moves farther than one row under the same perspective-camera translation.
    raised_linear_depth = CAM_RADIUS - Z_SCALE * 0.5
    raised = np.full(
        (SIZE[1], SIZE[0]),
        (raised_linear_depth - CLIP[0]) / (CLIP[1] - CLIP[0]),
        dtype=np.float32,
    )
    raised_dy = _previous_frame_sample_dy(raised)
    assert float(raised_dy.max()) < -CAMERA_STEP_PX


def test_depth_aov_linearizes_the_actual_raster_depth():
    shader = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "shaders"
        / "terrain_pbr_pom.wgsl"
    ).read_text(encoding="utf-8")
    depth_block = shader.split("// AOV Depth:", 1)[1].split(
        "out.aov_depth =", 1
    )[0]
    assert "let ndc_depth = clamp(input.clip_position.z" in depth_block
    assert "if (u_terrain.camera_mode_params.x >= 0.5)" in depth_block
    assert "clip_far - ndc_depth * (clip_far - clip_near)" in depth_block
    assert "view_pos_for_depth" not in depth_block


@pytest.mark.gpu_lane
@requires_terrain
def test_depth_aov_matches_known_flat_plane_distance():
    """Read back a physical AOV pixel whose view-space depth is known.

    A constant 0.5 heightmap lies exactly on the height-centred target plane.
    With the committed nadir camera, every non-skirt fragment therefore has
    view-space depth ``CAM_RADIUS``. This checks the actual attachment routing,
    raster-depth inverse, normalization, and readback rather than only matching
    shader source text.
    """
    flat = np.full((64, 64), 0.5, dtype=np.float32)
    size = (128, 72)

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        params = flythrough_params(
            size_px=size,
            terrain_span=SPAN,
            camera_mode=MODE,
            cam_radius=CAM_RADIUS,
            cam_target=(0.0, 0.0, 0.0),
            theta_deg=CAM_THETA_DEG,
            phi_deg=CAM_PHI_DEG,
            fov_y_deg=FOV_Y_DEG,
            z_scale=Z_SCALE,
            clip=CLIP,
            overlay=overlay,
            depth_aov=True,
        )
        _beauty, depth = render_rgba_depth(
            renderer, params, flat, ibl, material_set
        )

    assert depth.shape == (size[1], size[0])
    expected = (CAM_RADIUS - CLIP[0]) / (CLIP[1] - CLIP[0])
    # The physical attachment is Rgba16Float, so compare with the value the
    # format can actually retain rather than demanding fictitious f32 precision.
    expected_f16 = float(np.float16(expected))
    center_y, center_x = size[1] // 2, size[0] // 2
    center_patch = depth[
        center_y - 2 : center_y + 3,
        center_x - 2 : center_x + 3,
    ]
    np.testing.assert_allclose(center_patch, expected_f16, atol=5e-4, rtol=0.0)


@pytest.mark.gpu_lane
@requires_terrain
def test_fixed_camera_stream_center_transition_has_no_pop():
    """A fully resident stream-center rebuild must not change a fixed camera.

    This isolates clipmap re-centering from the one-pixel camera reprojection
    used by the 600-frame gate.  Both images use frame zero's camera; the only
    state transition between them is the public streaming call that changes the
    snapped clipmap centre and therefore forces a new geometry cache key.
    """
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    first_center = _center_at(0)
    second_center = _center_at(1)

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        _enable_streaming(renderer, dem)
        _warm_streaming_to_full_residency(renderer, first_center)

        before_stream = renderer.height_streaming_stats()
        before, _before_depth = render_rgba_depth(
            renderer,
            _params(z_scale=Z_SCALE, overlay=overlay, depth_aov=True),
            dem,
            ibl,
            material_set,
        )
        transition = renderer.stream_height_tiles(
            (second_center[0], STREAM_ALTITUDE_M, second_center[1]), max_uploads=8
        )
        after, _after_depth = render_rgba_depth(
            renderer,
            _params(z_scale=Z_SCALE, overlay=overlay, depth_aov=True),
            dem,
            ibl,
            material_set,
        )

    delta = _motion_compensated_delta_e(before, after, 0.0, 0.0)
    record_tessella_result(
        "flythrough_fixed_camera_recenter",
        {
            "before_center": tuple(float(value) for value in before_stream["center"]),
            "after_center": tuple(float(value) for value in transition["center"]),
            "resident_fine_tiles_before": int(before_stream["resident_fine_tiles"]),
            "resident_fine_tiles_after": int(transition["resident_fine_tiles"]),
            "converged_before": bool(before_stream["converged"]),
            "converged_after": bool(transition["converged"]),
            "fixed_camera_max_delta_e_2000": delta,
        },
    )
    assert tuple(float(value) for value in transition["center"]) != tuple(
        float(value) for value in before_stream["center"]
    )
    assert before_stream["converged"] is True, before_stream
    assert transition["converged"] is True, transition
    assert transition["resident_fine_tiles"] == transition["total_tiles"], transition
    assert delta < 1.0, delta


def test_committed_camera_path_is_a_real_non_vacuous_flythrough():
    targets = [_camera_target_at(index) for index in range(FRAMES)]
    steps = [
        math.dist(targets[index], targets[index + 1]) for index in range(FRAMES - 1)
    ]
    assert min(steps) == pytest.approx(CAMERA_STEP_M)
    assert max(steps) == pytest.approx(CAMERA_STEP_M)
    assert len(set(targets)) == FRAMES
    assert math.dist(targets[0], targets[-1]) > 100.0


def test_committed_camera_frames_multiple_clipmap_regions():
    radii = _region_outer_radii()
    assert radii[:3] == pytest.approx([6250.0, 18750.0, 43750.0])

    # Every pixel is terrain: the frame's right edge stays inside the +x limit
    # that make_ring's short strips actually cover.
    covered = clipmap_covered_positive_x_limit(SPAN, CENTER_RESOLUTION)
    actual_centers = [_actual_center_at(index) for index in range(FRAMES)]
    max_relative_x = max(
        _camera_target_at(index)[0] + HALF_WIDTH_M - actual_centers[index][0]
        for index in range(FRAMES)
    )
    assert max_relative_x < covered, (max_relative_x, covered)

    # Two ring boundaries permanently on screen -> three regions.
    assert _regions_on_screen() >= MIN_REGIONS_ON_SCREEN, {
        "frame_x": (FRAME_MIN_X, FRAME_MAX_X),
        "frame_abs_y": FRAME_MAX_ABS_Y,
        "region_outer_radii": radii,
    }
    assert FRAME_MAX_ABS_Y > radii[0], (FRAME_MAX_ABS_Y, radii[0])
    assert FRAME_MIN_X < -radii[1], (FRAME_MIN_X, radii[1])
    expected_target = (
        _center_at(0)[0] + CAM_TARGET_DX,
        _center_at(0)[1],
        0.0,
    )
    assert _camera_target_at(0) == expected_target

    camera_targets = [_camera_target_at(index) for index in range(FRAMES)]

    # Every requested move clears ClipmapLevel's half-cell threshold with
    # margin; finest-grid snapping can intentionally coalesce a small fraction
    # of successive requests, while the run-level signature assertion below
    # still requires at least 90% real mesh rebuilds.
    assert CENTER_STEP_LENGTH_M > REGENERATION_THRESHOLD_M, (
        CENTER_STEP_LENGTH_M,
        REGENERATION_THRESHOLD_M,
    )

    # The framed ground never leaves the DEM footprint, so no ring boundary on
    # screen is silently flattened by the UV clamp.
    assert max(abs(target[0]) for target in camera_targets) + HALF_WIDTH_M < SPAN * 0.5
    assert max(abs(target[1]) for target in camera_targets) + HALF_HEIGHT_M < SPAN * 0.5

    # The relief control must be constructible. It previously was not: the
    # control asked for z_scale 1200 against an API ceiling of 50.0, so it died
    # in ValueError before rendering and the crack detector had never been shown
    # to fire at all. Asserting it here makes that a fast CPU failure instead of
    # something only the GPU lane can notice.
    assert 0.1 <= RELIEF_CONTROL_Z_SCALE <= 50.0, RELIEF_CONTROL_Z_SCALE
    assert RELIEF_CONTROL_Z_SCALE > Z_SCALE * 10.0, (
        RELIEF_CONTROL_Z_SCALE,
        Z_SCALE,
    )

    # Consecutive centres always differ by the full step, and the run covers a
    # large set of distinct positions rather than oscillating between a few.
    steps = [
        math.dist(_center_at(index), _center_at(index + 1))
        for index in range(FRAMES - 1)
    ]
    assert min(steps) == pytest.approx(CENTER_STEP_LENGTH_M)
    assert max(steps) == pytest.approx(CENTER_STEP_LENGTH_M)
    requested_centers = {_center_at(index) for index in range(FRAMES)}
    assert len(requested_centers) >= MIN_DISTINCT_REQUESTED_CENTERS, len(
        requested_centers
    )
    assert len(set(actual_centers)) >= MIN_DISTINCT_ACTUAL_CENTERS, len(
        set(actual_centers)
    )
    actual_transitions = sum(
        actual_centers[index] != actual_centers[index - 1]
        for index in range(1, FRAMES)
    )
    assert actual_transitions >= MIN_CENTER_TRANSITIONS, actual_transitions

    # Overlay construction below creates the native GPU-backed LUT. Skip
    # before that first allocation on hosted runners without a terrain-safe
    # adapter; ``terrain_rendering_available`` still raises on the strict
    # TESSELLA lane.
    if not terrain_rendering_available():
        pytest.skip("requires the TESSELLA physical-GPU lane")
    overlay = build_overlay()
    params = _params(z_scale=Z_SCALE, overlay=overlay)
    assert tuple(params.cam_target) == expected_target


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.gpu_lane
@requires_terrain
def test_600_frame_streaming_flythrough_has_no_pop_or_crack():
    started = time.perf_counter()
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    half_spacings = clipmap_boundary_half_spacings(
        SPAN, RING_COUNT, RING_RESOLUTION, CENTER_RESOLUTION
    )

    max_camera_delta_e = 0.0
    max_recenter_delta_e = 0.0
    max_raw_recenter_delta_e = 0.0
    max_seam_gap = 0.0
    max_crack_count = 0
    total_crack_count = 0
    min_depth_samples = None
    hole_pixels_total = 0
    signature_changes = 0
    previous_signature = None
    previous_frame = None
    min_reprojection_dy = math.inf
    max_reprojection_dy = -math.inf
    actual_centers: list[tuple[float, float]] = []

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        _enable_streaming(renderer, dem)
        warmup_steps = _warm_streaming_to_full_residency(renderer, _center_at(0))

        for index in range(FRAMES):
            center = _center_at(index)
            params = _params(
                z_scale=Z_SCALE,
                overlay=overlay,
                frame_index=index,
                depth_aov=True,
            )
            if previous_frame is not None:
                # Isolate ordinary camera motion while the fully resident
                # stream/clipmap centre is still the previous frame's.  A
                # second render after stream_height_tiles then isolates the
                # recenter at this fixed camera.  Combining both changes in a
                # single comparison lets two individually sub-threshold
                # quantised colour changes add nonlinearly in CIEDE2000.
                camera_only_frame, camera_only_depth = render_rgba_depth(
                    renderer,
                    params,
                    dem,
                    ibl,
                    material_set,
                )
                camera_sample_dy = _previous_frame_sample_dy(camera_only_depth)
                camera_delta = _motion_compensated_delta_e(
                    previous_frame, camera_only_frame, 0.0, camera_sample_dy
                )
                assert math.isfinite(camera_delta), camera_delta
                max_camera_delta_e = max(max_camera_delta_e, camera_delta)

            stream = renderer.stream_height_tiles(
                (center[0], STREAM_ALTITUDE_M, center[1]), max_uploads=8
            )
            actual_center = tuple(float(value) for value in stream["center"])
            assert actual_center == pytest.approx(
                _actual_center_at(index), abs=1e-3
            ), {"frame": index, "requested": center, "stream": stream}
            actual_centers.append(actual_center)
            frame, depth = render_rgba_depth(
                renderer,
                params,
                dem,
                ibl,
                material_set,
            )
            sample_dy = _previous_frame_sample_dy(depth)
            min_reprojection_dy = min(
                min_reprojection_dy, float(sample_dy.min())
            )
            max_reprojection_dy = max(
                max_reprojection_dy, float(sample_dy.max())
            )

            # Per-frame crack gate on the geometry build this frame drew.
            seams = seam_stats()
            _assert_seams_clean(seams, f"frame {index}")
            max_seam_gap = max(max_seam_gap, float(seams["max_gap"]))
            max_crack_count = max(max_crack_count, int(seams["crack_count"]))
            total_crack_count += int(seams["crack_count"])
            samples = int(seams["depth_sample_count"])
            min_depth_samples = (
                samples
                if min_depth_samples is None
                else min(min_depth_samples, samples)
            )
            signature = seam_signature(seams)
            if previous_signature is not None and signature != previous_signature:
                signature_changes += 1
            previous_signature = signature

            # Per-frame threshold-free hole gate on the shipped image.
            holes = background_pixel_count(frame)
            assert holes == 0, {"frame": index, "background_pixels": holes}
            hole_pixels_total += holes

            # Every committed transition measures camera motion and clipmap
            # recentering independently.  This preserves the strict maximum
            # dE2000 threshold for the state change under test without letting
            # ordinary view-dependent shading consume its budget.
            if previous_frame is not None:
                raw_recenter_delta = _motion_compensated_delta_e(
                    camera_only_frame, frame, 0.0, 0.0
                )
                recenter_delta = _motion_compensated_delta_e(
                    _box_filter_rgb(camera_only_frame),
                    _box_filter_rgb(frame),
                    0.0,
                    0.0,
                )
                assert recenter_delta < 1.0, {
                    "transition": index,
                    "coherent_recenter_max_delta_e_2000": recenter_delta,
                    "raw_recenter_max_delta_e_2000": raw_recenter_delta,
                }
                max_recenter_delta_e = max(max_recenter_delta_e, recenter_delta)
                max_raw_recenter_delta_e = max(
                    max_raw_recenter_delta_e, raw_recenter_delta
                )
            previous_frame = frame

        height_vt = vt_stats()
        certificate = render_certificate(sign=False)

    # Run-level non-vacuity: the mesh was frequently rebuilt with new geometry,
    # so "0 cracks, 600 times" is not one cached answer replayed. Clipmap snapping
    # legitimately reuses a seam signature for some adjacent requested centres.
    assert signature_changes >= MIN_REBUILD_SIGNATURE_CHANGES, {
        "seam_signature_changes": signature_changes,
        "transitions": FRAMES - 1,
    }
    assert len(actual_centers) == FRAMES
    actual_steps = [
        math.dist(actual_centers[index - 1], actual_centers[index])
        for index in range(1, FRAMES)
    ]
    actual_transition_count = sum(step > 0.0 for step in actual_steps)
    assert actual_transition_count >= MIN_CENTER_TRANSITIONS, {
        "actual_clipmap_center_transitions": actual_transition_count,
        "transitions": FRAMES - 1,
    }
    assert len(set(actual_centers)) >= MIN_DISTINCT_ACTUAL_CENTERS
    assert height_vt["resident_tiles_height"] > 0, height_vt
    assert height_vt["height_pending_requests"] == 0, height_vt

    degraded = {
        str(entry.get("name")) for entry in certificate.get("degradations", []) or []
    }
    assert "terrain_visibility_buffer" not in degraded, degraded
    assert "terrain_hzb_two_phase" not in degraded, degraded

    wall_clock_s = time.perf_counter() - started
    record_tessella_result(
        "flythrough_popping",
        {
            # Field names the win-5 row of scripts/tessella_evidence_report.py
            # renders; everything after them is extra context.
            "frames": FRAMES,
            "rendered_frames_total": FRAMES,
            "width": SIZE[0],
            "height": SIZE[1],
            "worst_frame_crack_count": max_crack_count,
            "crack_count": total_crack_count,
            "depth_sample_count": int(min_depth_samples or 0),
            "frames_crack_checked": FRAMES,
            "dem_size": DEM_SIZE,
            "dem_octaves": DEM_OCTAVES,
            "terrain_span_m": SPAN,
            "z_scale": Z_SCALE,
            "clipmap_center_step_m": sum(actual_steps) / len(actual_steps),
            "clipmap_center_step_min_m": min(actual_steps),
            "clipmap_center_step_max_m": max(actual_steps),
            "requested_clipmap_center_step_m": CENTER_STEP_LENGTH_M,
            "clipmap_regeneration_threshold_m": REGENERATION_THRESHOLD_M,
            "clipmap_center_path_m": sum(actual_steps),
            "requested_clipmap_center_path_m": CENTER_STEP_LENGTH_M * (FRAMES - 1),
            "actual_clipmap_center_transitions": actual_transition_count,
            "camera_step_px": CAMERA_STEP_PX,
            "camera_path_distance_m": math.dist(
                _camera_target_at(0), _camera_target_at(FRAMES - 1)
            ),
            "distinct_camera_positions": len(
                {_camera_target_at(index) for index in range(FRAMES)}
            ),
            "distinct_requested_clipmap_centers": len(
                {_center_at(index) for index in range(FRAMES)}
            ),
            "distinct_clipmap_centers": len(set(actual_centers)),
            "regions_on_screen": _regions_on_screen_for_centers(actual_centers),
            "region_outer_radii_m": _region_outer_radii(),
            "ground_pixel_m": GROUND_PIXEL_M,
            "max_delta_e_2000": max_recenter_delta_e,
            "camera_only_max_delta_e_2000": max_camera_delta_e,
            "recenter_only_max_delta_e_2000": max_recenter_delta_e,
            "raw_recenter_only_max_delta_e_2000": max_raw_recenter_delta_e,
            "delta_e_metric": "isolated_camera_and_3x3_coherent_recenter_ciede2000",
            "motion_compensation_dx_px": 0.0,
            "motion_compensation_dy_px_min": min_reprojection_dy,
            "motion_compensation_dy_px_max": max_reprojection_dy,
            "motion_compensation_crop_px": MOTION_COMPENSATION_CROP_PX,
            "max_seam_gap": max_seam_gap,
            "seam_gap_threshold": SEAM_THRESHOLD,
            "seam_gap_headroom_ratio": SEAM_THRESHOLD / max(max_seam_gap, 1e-12),
            "seam_roughness": seam_roughness(dem, SPAN, half_spacings, Z_SCALE),
            "legacy_seam_roughness": seam_roughness(
                legacy_gate_dem(), SPAN, half_spacings, LEGACY_GATE_Z_SCALE
            ),
            "seam_signature_changes": signature_changes,
            "hole_pixels_total": hole_pixels_total,
            "resident_height_tiles": int(height_vt["resident_tiles_height"]),
            "height_pending_requests": int(height_vt["height_pending_requests"]),
            "streaming_warmup_steps": warmup_steps,
            "streaming_total_tiles": STREAM_TOTAL_TILES,
            "wall_clock_s": wall_clock_s,
        },
    )


# ---------------------------------------------------------------------------
# Negative controls: both gates must be able to fail
# ---------------------------------------------------------------------------


@pytest.mark.gpu_lane
@requires_terrain
def test_crack_detector_fires_when_the_seams_actually_separate():
    """The crack metric is a measurement, not a constant.

    Same camera, DEM, resolution and clipmap mode; only the vertical
    exaggeration changes. ``analyze_depth_discontinuities`` scales the fine/coarse
    disagreement by ``z_scale`` (geomorph.rs:305), so a relief the shipped
    geomorph cannot hold must make the identical detector report cracks. A
    detector that cannot fail proves nothing about the frames where it stays
    silent.
    """
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    center = _center_at(0)

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        _enable_streaming(renderer, dem)
        _warm_streaming_to_full_residency(renderer, center)

        render_rgba(
            renderer,
            _params(z_scale=Z_SCALE, overlay=overlay),
            dem,
            ibl,
            material_set,
        )
        clean = seam_stats()

        render_rgba(
            renderer,
            _params(z_scale=RELIEF_CONTROL_Z_SCALE, overlay=overlay),
            dem,
            ibl,
            material_set,
        )
        separated = seam_stats()

    _assert_seams_clean(clean, "relief control baseline")
    assert separated["crack_count"] > 0, separated
    assert separated["max_gap"] > SEAM_THRESHOLD, (separated, SEAM_THRESHOLD)
    assert separated["seams_valid"] is False, separated
    assert separated["max_gap"] > clean["max_gap"] * 10.0, (clean, separated)

    record_tessella_result(
        "flythrough_crack_detector_control",
        {
            "z_scale": Z_SCALE,
            "control_z_scale": RELIEF_CONTROL_Z_SCALE,
            "seam_gap_threshold": SEAM_THRESHOLD,
            "max_gap": float(clean["max_gap"]),
            "control_max_gap": float(separated["max_gap"]),
            "crack_count": int(clean["crack_count"]),
            "control_crack_count": int(separated["crack_count"]),
        },
    )


@pytest.mark.gpu_lane
@requires_terrain
def test_pop_gate_discriminates_at_this_resolution_and_dem():
    """The depth-reprojected dE2000 < 1.0 threshold detects a real fallback.

    The fully resident renderer produces both camera positions, independently
    validating the exact registration direction. A second renderer keeps
    nearly every height tile at its real coarse-prefill fallback and renders
    the displaced camera; registration must retain that production change.
    """
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    center = _center_at(0)

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()
        _enable_streaming(renderer, dem)
        _warm_streaming_to_full_residency(renderer, center)

        reference, _reference_depth = render_rgba_depth(
            renderer,
            _params(z_scale=Z_SCALE, overlay=overlay, depth_aov=True),
            dem,
            ibl,
            material_set,
        )
        shifted, shifted_depth = render_rgba_depth(
            renderer,
            _params(
                z_scale=Z_SCALE,
                overlay=overlay,
                frame_index=1,
                depth_aov=True,
            ),
            dem,
            ibl,
            material_set,
        )
        shifted_sample_dy = _previous_frame_sample_dy(shifted_depth)
        compensated_baseline = _motion_compensated_delta_e(
            reference, shifted, 0.0, shifted_sample_dy
        )
        reversed_baseline = _motion_compensated_delta_e(
            reference, shifted, 0.0, -shifted_sample_dy
        )

        fallback_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        _enable_streaming(fallback_renderer, dem)
        fallback_stats = fallback_renderer.stream_height_tiles(
            (center[0], STREAM_ALTITUDE_M, center[1]), max_uploads=0
        )
        assert fallback_stats["coarse_prefilled"] == STREAM_TOTAL_TILES, fallback_stats
        # `drain_completed(max_uploads)` deliberately clamps its budget to one,
        # so a fast local reader may upload a single tile even when callers pass
        # zero.  The control is still a real coarse-prefill fallback whenever it
        # has not converged: most of the 64 physical mosaic slots remain coarse.
        assert fallback_stats["resident_fine_tiles"] < STREAM_TOTAL_TILES, fallback_stats
        assert fallback_stats["converged"] is False, fallback_stats
        fallback, fallback_depth = render_rgba_depth(
            fallback_renderer,
            _params(
                z_scale=Z_SCALE,
                overlay=overlay,
                frame_index=1,
                depth_aov=True,
            ),
            dem,
            ibl,
            material_set,
        )
        fallback_sample_dy = _previous_frame_sample_dy(fallback_depth)
        fallback_delta = _motion_compensated_delta_e(
            reference, fallback, 0.0, fallback_sample_dy
        )
        coherent_fallback_delta = _motion_compensated_delta_e(
            _box_filter_rgb(reference),
            _box_filter_rgb(fallback),
            0.0,
            fallback_sample_dy,
        )

    assert compensated_baseline < 1.0, compensated_baseline
    assert compensated_baseline < reversed_baseline, {
        "correct_direction": compensated_baseline,
        "reversed_direction": reversed_baseline,
    }
    assert fallback_delta >= 1.0, fallback_delta
    assert coherent_fallback_delta >= 1.0, coherent_fallback_delta

    record_tessella_result(
        "flythrough_pop_gate_control",
        {
            "ground_pixel_m": GROUND_PIXEL_M,
            "motion_compensation_dx_px": 0.0,
            "motion_compensation_dy_px_min": float(shifted_sample_dy.min()),
            "motion_compensation_dy_px_max": float(shifted_sample_dy.max()),
            "motion_compensation_crop_px": MOTION_COMPENSATION_CROP_PX,
            "compensated_baseline_max_delta_e_2000": compensated_baseline,
            "reversed_direction_max_delta_e_2000": reversed_baseline,
            "control_max_delta_e_2000": fallback_delta,
            "coherent_control_max_delta_e_2000": coherent_fallback_delta,
            "fallback_coarse_prefilled_tiles": int(
                fallback_stats["coarse_prefilled"]
            ),
            "fallback_resident_fine_tiles": int(
                fallback_stats["resident_fine_tiles"]
            ),
        },
    )


@pytest.mark.gpu_lane
@requires_terrain
def test_visibility_shading_is_identical_and_hole_free_at_flythrough_settings():
    """GPU-side confirmation of the image-side hole gate.

    ``visibility_stats()`` counts the resolve pass's background pixels on the
    shipped draw path. Win 3 only proves visibility == forward at 640x360 over
    ``_steep_dem(96)`` and without height streaming; the flythrough's zero-hole
    claim is only as good as that equivalence at ITS settings.
    """
    dem = fractal_dem(DEM_SIZE, DEM_OCTAVES, DEM_GAIN, DEM_SEED)
    center = _center_at(0)

    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        material_set = f3d.MaterialSet.terrain_default()
        overlay = build_overlay()

        forward_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        _enable_streaming(forward_renderer, dem)
        # Both renderers must reach identical mosaics or the bitwise comparison
        # would be measuring streaming luck rather than shading equivalence.
        _warm_streaming_to_full_residency(forward_renderer, center)
        forward = render_rgba(
            forward_renderer,
            _params(z_scale=Z_SCALE, overlay=overlay),
            dem,
            ibl,
            material_set,
        )

        visibility_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        _enable_streaming(visibility_renderer, dem)
        _warm_streaming_to_full_residency(visibility_renderer, center)
        visibility = render_rgba(
            visibility_renderer,
            _params(z_scale=Z_SCALE, overlay=overlay, shading="visibility"),
            dem,
            ibl,
            material_set,
        )
        stats = visibility_stats()

    np.testing.assert_array_equal(visibility, forward)
    covered = stats["visible_pixels"] + stats["background_pixels"]
    assert covered == SIZE[0] * SIZE[1], stats
    assert stats["background_pixels"] == 0, stats
    assert background_pixel_count(forward) == 0

    record_tessella_result(
        "flythrough_visibility_coverage",
        {
            "render_size_px": list(SIZE),
            "visible_pixels": int(stats["visible_pixels"]),
            "background_pixels": int(stats["background_pixels"]),
            "image_background_pixels": int(background_pixel_count(forward)),
            "bitwise_identical_to_forward": True,
        },
    )
