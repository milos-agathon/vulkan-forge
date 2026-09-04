# tests/_terrain_flythrough.py
# TESSELLA win 5 ("zero popping, zero cracks"): fixtures for the 600-frame
# streaming flythrough gate in tests/test_flythrough_popping.py.
#
# These live here rather than in tests/test_terrain_clipmap_streaming.py because
# the win-5 gate needs three things the shared helper cannot give it:
#
#   * An explicit near/far pair. ``_make_params`` hard-codes
#     ``clip=(0.1, terrain_span * 1.5)``. That frustum is harmless with the
#     camera one metre from the terrain (its only current use), but with the eye
#     24 km away it leaves kilometres of Depth32Float quantisation at the ground
#     plane, which z-fights the deliberately overlapping clipmap ring strips.
#   * A height field with energy in more than one octave. ``_steep_dem`` is one
#     separable sinusoid pair plus a Gaussian.
#   * Executable mirrors of the Rust clipmap geometry so the gate can *compute*
#     which LOD regions are on screen instead of asserting it in prose.
#
# Every mirror below carries the source anchor it mirrors. If the Rust changes,
# the mirrors must change with it -- the flythrough's framing assertions are the
# thing that would go quietly vacuous otherwise.

from __future__ import annotations

import math

import numpy as np

import forge3d as f3d
from forge3d.terrain_params import AovSettings, PomSettings, make_terrain_params_config

# Background clear colour of the terrain render pass (linear 0.1/0.1/0.15).
BACKGROUND_RGB_LINEAR = np.array([0.1, 0.1, 0.15], dtype=np.float64)


# ---------------------------------------------------------------------------
# Deterministic multi-octave height field
# ---------------------------------------------------------------------------


def _value_noise_octave(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """One smoothstep-faded value-noise octave on a ``cells x cells`` lattice.

    Smoothstep (rather than linear) fades keep the octave C1, so its spectrum
    rolls off instead of carrying the broadband energy that a piecewise-linear
    lattice injects at every frequency through its slope discontinuities.
    """
    lattice = rng.random((cells + 1, cells + 1))
    coordinate = np.linspace(0.0, float(cells), size)
    low = np.minimum(np.floor(coordinate).astype(np.int64), cells - 1)
    high = low + 1
    fade = coordinate - low
    weight = fade * fade * (3.0 - 2.0 * fade)

    l00 = lattice[low][:, low]
    l01 = lattice[low][:, high]
    l10 = lattice[high][:, low]
    l11 = lattice[high][:, high]
    wx = weight[None, :]
    wy = weight[:, None]
    top = l00 * (1.0 - wx) + l01 * wx
    bottom = l10 * (1.0 - wx) + l11 * wx
    return top * (1.0 - wy) + bottom * wy


def _fbm_octave_fields(
    size: int, octaves: int, gain: float, seed: int
) -> list[np.ndarray]:
    """Per-octave contributions, largest wavelength first, before normalisation.

    Octave ``k`` uses a ``2**(k+1)``-cell lattice, i.e. it carries one full
    period every ``2`` cells, so its wavelength is ``terrain_span / 2**k``.
    """
    rng = np.random.default_rng(seed)
    fields: list[np.ndarray] = []
    amplitude = 1.0
    for octave in range(octaves):
        fields.append(amplitude * _value_noise_octave(rng, size, 2 ** (octave + 1)))
        amplitude *= gain
    return fields


def fractal_dem(
    size: int = 256, octaves: int = 4, gain: float = 0.45, seed: int = 1904
) -> np.ndarray:
    """Deterministic fBm height field normalised to ``[0, 1]``.

    Determinism contract (this array is a committed fixture in all but name --
    the gate's numbers are only comparable run to run because of it):

    * ``np.random.default_rng(seed)`` is PCG64, which is version- and
      platform-stable; no global ``np.random`` state is touched.
    * accumulation is float64 and the reduction order is fixed;
    * no sorting, no threading-dependent reductions;
    * the cast to float32 happens once, at the end.

    Unlike ``_steep_dem`` (one separable ``sin*cos`` pair plus a Gaussian, i.e.
    ~5 cycles across the grid and essentially nothing below that), this surface
    is non-separable and carries ``octaves`` distinct scales. See
    :func:`worst_case_seam_gap` for why ``octaves`` cannot simply be raised
    until the finest ring's Nyquist is reached.
    """
    fields = _fbm_octave_fields(size, octaves, gain, seed)
    total = np.zeros((size, size), dtype=np.float64)
    for field in fields:
        total += field
    low = float(total.min())
    high = float(total.max())
    return ((total - low) / max(high - low, 1e-12)).astype(np.float32)


def fractal_dem_octave_amplitudes(
    size: int = 256, octaves: int = 4, gain: float = 0.45, seed: int = 1904
) -> list[float]:
    """Half peak-to-peak of every octave, in the units :func:`fractal_dem` emits.

    Measured, not assumed: the normalisation applied by :func:`fractal_dem` is
    a function of the *summed* field, so the per-octave amplitudes are only
    knowable after the fact.
    """
    fields = _fbm_octave_fields(size, octaves, gain, seed)
    total = np.zeros((size, size), dtype=np.float64)
    for field in fields:
        total += field
    scale = 1.0 / max(float(total.max()) - float(total.min()), 1e-12)
    return [0.5 * float(field.max() - field.min()) * scale for field in fields]


# ---------------------------------------------------------------------------
# Executable mirrors of the Rust clipmap geometry
# ---------------------------------------------------------------------------


def clipmap_base_cell_size(terrain_span: float, center_resolution: int) -> float:
    """``src/terrain/clipmap/level.rs:63``."""
    return terrain_span / (center_resolution * 8.0)


def clipmap_region_outer_radii(
    terrain_span: float,
    ring_count: int,
    ring_resolution: int,
    center_resolution: int,
) -> list[float]:
    """Outer half-extent of every clipmap region, centre block first.

    Mirrors ``ClipmapLevel::generate`` (``src/terrain/clipmap/level.rs:80-140``)
    and ``ClipmapConfig::ring_extent`` (``src/terrain/clipmap/mod.rs:86-90``).
    Region 0 is the centre block; region ``i >= 1`` is clipmap ring ``i - 1``.
    """
    cell = clipmap_base_cell_size(terrain_span, center_resolution)
    radii = [cell * center_resolution * 0.5]
    for ring in range(ring_count):
        radii.append(radii[-1] + cell * float(1 << ring) * ring_resolution)
    return radii


def clipmap_region_vertex_spacing(
    terrain_span: float, center_resolution: int, region_index: int
) -> float:
    """World spacing between adjacent vertices of one clipmap region.

    The centre block steps by ``base_cell`` (``ring.rs:17,23``); ring ``r``
    steps by ``cell_size * 2.0`` where ``cell_size = base_cell * 2**r``
    (``ring.rs:68,93``), i.e. ``2**(r+1) * base_cell``. With region indices
    (centre = 0, ring ``r`` = region ``r+1``) both cases are
    ``2**region_index * base_cell``.
    """
    cell = clipmap_base_cell_size(terrain_span, center_resolution)
    return cell * float(1 << region_index)


def clipmap_covered_positive_x_limit(
    terrain_span: float, center_resolution: int
) -> float:
    """Largest ``+x`` offset from the clipmap centre that every ring covers.

    ``make_ring``'s top/bottom strips run ``col * cell_size * 2.0`` for
    ``col in 0..=resolution`` from ``center.x - outer_extent``
    (``src/terrain/clipmap/ring.rs:93,118``), i.e. they span ``2 * strip_width``
    where ``2 * outer_extent`` is needed. Their last vertex therefore lands at
    ``outer - 2 * inner``, which for every ring equals the centre block's
    half-extent, and the ``+x`` corners of each ring beyond that are left
    uncovered (``add_corner_patch`` at ``ring.rs:205-218`` is a documented
    no-op). Any frame whose right edge stays at or below this limit is fully
    covered by terrain; anything past it shows background wedges.
    """
    return clipmap_base_cell_size(terrain_span, center_resolution) * center_resolution * 0.5


def seam_gap_threshold(terrain_span: float) -> float:
    """``max_seam_gap`` used by the shipped crack detector.

    ``src/terrain/renderer/geometry.rs:631`` --
    ``max_seam_gap: (terrain_span * 1e-6).max(0.001)``. It is compared against
    ``analyze_depth_discontinuities``' gap, which is already multiplied by
    ``z_scale`` (``src/terrain/clipmap/geomorph.rs:305,326``), so the gate reads
    "the fine and coarse surfaces must agree to within one part per million of
    the terrain span, in world units".
    """
    return max(terrain_span * 1e-6, 0.001)


def clipmap_boundary_half_spacings(
    terrain_span: float,
    ring_count: int,
    ring_resolution: int,
    center_resolution: int,
) -> list[float]:
    """Half the coarse-side vertex spacing at each clipmap region boundary.

    ``analyze_depth_discontinuities`` brackets every fine boundary vertex between
    two coarse-side vertices and measures its deviation from their linear
    interpolation. The rings are strictly 2:1, so each fine vertex lands on the
    midpoint of a coarse edge and the relevant length scale is half the coarse
    spacing. Every boundary is included -- even the ones whose square reaches
    past the DEM footprint, because their sides still cross it and sample real
    terrain along the clamped edge row.
    """
    return [
        0.5
        * clipmap_region_vertex_spacing(terrain_span, center_resolution, index + 1)
        for index in range(ring_count)
    ]


def _shift_along_axis(field: np.ndarray, delta_texels: float, axis: int) -> np.ndarray:
    length = field.shape[axis]
    coordinate = np.clip(np.arange(length) + delta_texels, 0.0, length - 1.0)
    low = np.floor(coordinate).astype(np.int64)
    high = np.minimum(low + 1, length - 1)
    fade = coordinate - low
    if axis == 1:
        return field[:, low] * (1.0 - fade) + field[:, high] * fade
    return field[low, :] * (1.0 - fade[:, None]) + field[high, :] * fade[:, None]


def worst_midpoint_deviation(
    dem: np.ndarray, terrain_span: float, half_spacing_m: float
) -> float:
    """Largest ``|h(mid) - (h(mid-d) + h(mid+d)) / 2|`` over the whole DEM.

    This is exactly the quantity the shipped crack detector accumulates at a 2:1
    ring boundary, before ``z_scale``: the fine vertex sits at ``mid`` and the
    two coarse vertices bracketing it sit at ``mid -/+ d``. Measuring it on the
    real array beats modelling it, because the alternative is a spectral estimate
    that has to assume the octaves behave like pure sinusoids.

    It is an upper bound on what the detector reports at ring-to-ring boundaries,
    where the fine side is additionally smoothed by the shader's coarse-grid
    blend (``terrain_pbr_pom.wgsl:4579-4590``), and is exact at the centre/ring-0
    boundary, where the centre block's morph weight is 0.
    """
    texel_m = terrain_span / (dem.shape[0] - 1)
    delta = half_spacing_m / texel_m
    field = np.asarray(dem, dtype=np.float64)
    worst = 0.0
    for axis in (0, 1):
        low = _shift_along_axis(field, -delta, axis)
        high = _shift_along_axis(field, delta, axis)
        deviation = np.abs(field - 0.5 * (low + high))
        margin = int(math.ceil(delta)) + 1
        if deviation.shape[0] > 2 * margin + 2:
            deviation = deviation[margin:-margin, margin:-margin]
        worst = max(worst, float(deviation.max()))
    return worst


def seam_roughness(
    dem: np.ndarray,
    terrain_span: float,
    half_spacings: list[float],
    z_scale: float,
) -> float:
    """Worst T-junction gap the crack detector can see, in its own units."""
    return z_scale * max(
        worst_midpoint_deviation(dem, terrain_span, half) for half in half_spacings
    )


# The height field and vertical exaggeration this gate was green with before the
# win-5 rewrite: ``_steep_dem(256)[::4, ::4]`` at ``z_scale = 1.2``. Reproduced
# here (rather than imported) so the reference cannot drift when the shared
# helper changes, and used as the roughness budget any replacement fixture must
# stay under -- an empirical bound on what the shipped seam machinery holds,
# rather than a guess at where the cliff is.
LEGACY_GATE_Z_SCALE = 1.2


def legacy_gate_dem(size: int = 256, decimation: int = 4) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    ridges = 0.5 * np.sin(xx * np.pi * 5.0) * np.cos(yy * np.pi * 4.0)
    peak = 0.8 * np.exp(-(xx**2 + yy**2) * 3.0)
    dem = ridges + peak
    dem -= dem.min()
    dem /= max(float(dem.max()), 1e-6)
    return dem.astype(np.float32)[::decimation, ::decimation]


# ---------------------------------------------------------------------------
# Render plumbing
# ---------------------------------------------------------------------------


def flythrough_params(
    *,
    size_px: tuple[int, int],
    terrain_span: float,
    camera_mode: str,
    cam_radius: float,
    cam_target: tuple[float, float, float],
    theta_deg: float,
    phi_deg: float,
    fov_y_deg: float,
    z_scale: float,
    clip: tuple[float, float],
    overlay,
    culling: str = "frustum",
    shading: str = "forward",
    depth_aov: bool = False,
) -> "f3d.TerrainRenderParams":
    """``TerrainRenderParams`` for the win-5 gate, with an explicit frustum."""
    return f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=size_px,
            render_scale=1.0,
            terrain_span=terrain_span,
            msaa_samples=1,
            z_scale=z_scale,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="colormap",
            colormap_strength=1.0,
            ibl_enabled=True,
            light_azimuth_deg=138.0,
            light_elevation_deg=24.0,
            sun_intensity=2.4,
            cam_radius=cam_radius,
            cam_phi_deg=phi_deg,
            cam_theta_deg=theta_deg,
            cam_target=list(cam_target),
            fov_y_deg=fov_y_deg,
            camera_mode=camera_mode,
            culling=culling,
            shading=shading,
            clip=clip,
            overlays=[overlay],
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            aov=AovSettings(
                enabled=depth_aov,
                albedo=False,
                normal=False,
                depth=depth_aov,
            ),
        )
    )


def build_overlay():
    """A continuous terrain ramp, built once and reused across 600 params."""
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#18391f"), (1.0, "#f3f5f9")], domain=(0.0, 1.0)
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def render_rgba(renderer, params, heightmap, ibl, material_set) -> np.ndarray:
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
        water_mask=None,
    )
    return np.asarray(frame.to_numpy())


def render_rgba_depth(
    renderer, params, heightmap, ibl, material_set
) -> tuple[np.ndarray, np.ndarray]:
    """Render beauty and its co-emitted normalized linear-depth AOV once."""
    frame, aov_frame = renderer.render_with_aov(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        water_mask=None,
    )
    return np.asarray(frame.to_numpy()), np.asarray(aov_frame.depth())


def background_pixel_count(rgba: np.ndarray, tolerance: int = 6) -> int:
    """Pixels still showing the render pass's clear colour.

    With the frame fully inside the clipmap's covered region (see
    :func:`clipmap_covered_positive_x_limit`) every pixel must be terrain, so a
    non-zero count is a literal see-through gap: a ring-boundary crack, a missed
    skirt, or a tile that resolved to nothing.
    """
    srgb = np.where(
        BACKGROUND_RGB_LINEAR <= 0.0031308,
        BACKGROUND_RGB_LINEAR * 12.92,
        1.055 * np.power(BACKGROUND_RGB_LINEAR, 1.0 / 2.4) - 0.055,
    )
    background = np.round(srgb * 255.0).astype(np.int32)
    difference = np.abs(rgba[..., :3].astype(np.int32) - background[None, None, :])
    return int((difference <= tolerance).all(axis=-1).sum())


def seam_signature(stats: dict) -> tuple:
    """Hashable identity of one published seam analysis.

    ``SeamAnalysis`` describes the last clipmap *geometry build* only. Comparing
    consecutive signatures is how the gate proves the build actually re-ran on
    new geometry every frame rather than being served once from the geometry
    cache (``geometry.rs:616-625``).
    """
    return (
        int(stats["boundary_vertex_count"]),
        int(stats["depth_sample_count"]),
        float(stats["max_gap"]),
        float(stats["avg_gap"]),
        int(stats["crack_count"]),
    )


def triangle_wave(index: int, amplitude_steps: int) -> int:
    """Integer triangle wave in ``[-amplitude_steps, +amplitude_steps]``.

    Consecutive values always differ by exactly one step, so a path built from
    two of these advances by a fixed, non-zero distance on every frame while
    staying inside a bounded box.
    """
    span = 2 * amplitude_steps
    phase = index % (2 * span)
    ramp = phase if phase <= span else 2 * span - phase
    return ramp - amplitude_steps
