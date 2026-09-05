"""Hermetic Southeast-Europe population fixture and NumPy overlay oracle.

The palette is reconstructed from the warm ochre/coral visual language used by
population-spike maps: dark umber low values, muted orange mid values, and a
cream highlight.  It is intentionally represented by meaningful interpolation
stops rather than reverse-engineering an arbitrary fixed stop count.  No data
is downloaded; every input is procedural and hashable.
"""
from __future__ import annotations

import hashlib

import numpy as np

CANONICAL_SEED = 7
FIXTURE_SIZE = 1024
WARM_STOPS = (
    (0.00, (0.20, 0.105, 0.070)),
    (0.18, (0.38, 0.155, 0.085)),
    (0.43, (0.68, 0.285, 0.125)),
    (0.70, (0.88, 0.500, 0.235)),
    (0.90, (0.97, 0.745, 0.455)),
    (1.00, (1.00, 0.925, 0.745)),
)
CONSTANT_ALBEDO = (0.62, 0.62, 0.62)
WHISPER_FLOOR = 0.22
WHISPER_GAIN = 0.78


def _coordinates(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis)


def make_dem(seed: int = CANONICAL_SEED, size: int = FIXTURE_SIZE) -> np.ndarray:
    """Generate a smooth Balkan-like mountain field with deterministic detail."""
    if size < 16:
        raise ValueError("fixture size must be at least 16")
    x, y = _coordinates(size)
    rng = np.random.default_rng(seed)
    dem = 0.12 * (1.0 - y) + 0.05 * x
    peaks = (
        (-0.42, -0.06, 0.34, 18.0, 42.0),
        (0.06, -0.28, 0.52, 24.0, 14.0),
        (0.38, 0.18, 0.43, 32.0, 20.0),
        (-0.02, 0.46, 0.28, 16.0, 30.0),
    )
    for px, py, height, sx, sy in peaks:
        dem += height * np.exp(-((x - px) ** 2 * sx + (y - py) ** 2 * sy))
    phases = rng.uniform(0.0, 2.0 * np.pi, 4)
    dem += 0.025 * np.sin(9.0 * x + phases[0]) * np.cos(7.0 * y + phases[1])
    dem += 0.012 * np.sin(21.0 * x + phases[2]) * np.sin(17.0 * y + phases[3])
    dem -= float(dem.min())
    dem /= max(float(dem.max()), 1e-8)
    return np.ascontiguousarray(dem, dtype=np.float32)


def make_population(seed: int = CANONICAL_SEED, size: int = FIXTURE_SIZE) -> np.ndarray:
    """Generate compact urban clusters plus low-density settlement corridors."""
    x, y = _coordinates(size)
    rng = np.random.default_rng(seed + 1009)
    population = np.zeros((size, size), dtype=np.float32)
    cities = (
        (-0.38, 0.05, 1.00, 520.0),
        (-0.05, 0.12, 0.88, 680.0),
        (0.27, -0.20, 0.80, 610.0),
        (0.43, 0.26, 0.63, 760.0),
        (-0.12, -0.43, 0.56, 820.0),
    )
    for cx, cy, weight, sharpness in cities:
        population += weight * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) * sharpness)
    population += 0.075 * np.exp(-((y - 0.18 * np.sin(3.5 * x)) ** 2) * 95.0)
    population += rng.random((size, size), dtype=np.float32) * 0.006
    population -= float(population.min())
    population /= max(float(population.max()), 1e-8)
    return np.ascontiguousarray(population, dtype=np.float32)


def make_subject_mask(size: int = FIXTURE_SIZE) -> np.ndarray:
    """Return the fixed elliptical map-subject footprint."""
    x, y = _coordinates(size)
    return ((x / 0.88) ** 2 + ((y + 0.02) / 0.78) ** 2 <= 1.0)


def _palette(population: np.ndarray, stops=WARM_STOPS) -> np.ndarray:
    values = np.asarray(population, dtype=np.float32)
    positions = np.asarray([stop[0] for stop in stops], dtype=np.float32)
    colors = np.asarray([stop[1] for stop in stops], dtype=np.float32)
    flat = np.clip(values, 0.0, 1.0).ravel()
    channels = [np.interp(flat, positions, colors[:, index]) for index in range(3)]
    return np.stack(channels, axis=-1).reshape(values.shape + (3,)).astype(np.float32)


def _population_tint(population: np.ndarray, stops=WARM_STOPS) -> np.ndarray:
    whisper = WHISPER_FLOOR + WHISPER_GAIN * np.sqrt(
        np.clip(np.asarray(population, dtype=np.float32), 0.0, 1.0)
    )
    whisper[_spike_exemption_ring(population)] = 1.0
    return np.clip(_palette(population, stops) * whisper[..., None], 0.0, 1.0)


def make_albedo_map(population: np.ndarray, stops=WARM_STOPS) -> np.ndarray:
    """Build the canonical grid-aligned RGBA terrain material."""
    rgb = _population_tint(population, stops)
    rgba = np.empty(rgb.shape[:-1] + (4,), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = 1.0
    return np.ascontiguousarray(rgba)


def make_gpu_lightfield_args(dem: np.ndarray, albedo=CONSTANT_ALBEDO, **overrides) -> dict:
    """Arguments for the constant-albedo GPU light-field render."""
    height, width = np.asarray(dem).shape
    radius = 1.75 * max(width, height)
    args = {
        "camera": {
            "model": "orthographic",
            "origin": (0.0, radius * np.cos(np.deg2rad(35.0)), radius * np.sin(np.deg2rad(35.0))),
            "look_at": (0.0, 0.18, 0.0),
            "up": (0.0, np.sin(np.deg2rad(35.0)), -np.cos(np.deg2rad(35.0))),
            "half_height": 0.62 * max(width, height),
        },
        "spacing": (1.0, 1.0),
        "exaggeration": 180.0,
        "albedo": tuple(float(value) for value in albedo),
        "sun_azimuth_deg": 315.0,
        "sun_elevation_deg": 28.0,
        "sun_intensity": 2.5,
        "env_intensity": 0.15,
        "spp": 1,
        "min_frames": 32,
        "max_frames": 32,
        "variance_threshold": 1e9,
        "seed": CANONICAL_SEED,
    }
    args.update(overrides)
    return args


def _spike_exemption_ring(population: np.ndarray) -> np.ndarray:
    """Protect a thin ring around the dominant spike from whisper suppression."""
    peak_y, peak_x = np.unravel_index(int(np.argmax(population)), population.shape)
    yy, xx = np.ogrid[: population.shape[0], : population.shape[1]]
    radius = max(3.0, min(population.shape) * 0.018)
    distance2 = (xx - peak_x) ** 2 + (yy - peak_y) ** 2
    return (distance2 >= (0.70 * radius) ** 2) & (distance2 <= (1.35 * radius) ** 2)


def numpy_oracle(
    lightfield_rgba,
    population,
    subject_mask,
    stops=WARM_STOPS,
    *,
    projected_albedo=None,
) -> np.ndarray:
    """Apply warm population tint to a constant-albedo GPU light field.

    The light field supplies luminance, while the palette anchors chroma.  The
    whisper floor/gain preserves low-density relief; the dominant-spike ring is
    exempt so its halo remains readable rather than being crushed by the floor.
    """
    light = np.asarray(lightfield_rgba)
    if light.ndim != 3 or light.shape[2] != 4:
        raise ValueError("lightfield_rgba must have shape (H, W, 4)")
    pop = np.asarray(population, dtype=np.float32)
    mask = np.asarray(subject_mask, dtype=bool)
    if pop.shape != light.shape[:2] or mask.shape != pop.shape:
        raise ValueError("light field, population, and subject mask must align")
    light_rgb = light[..., :3].astype(np.float32)
    if np.issubdtype(light.dtype, np.integer):
        light_rgb /= 255.0
    # Hybrid PT stores the Reinhard-mapped linear value directly in an
    # Rgba16Float texture and the host quantizes it to UNORM bytes; there is no
    # sRGB transfer function.  Invert y=x/(1+x), divide out the constant 0.62
    # albedo, apply the canonical population tint, then run Reinhard again.
    # This preserves the GPU light field while anchoring hue to the warm map.
    mapped = np.clip(light_rgb, 0.0, 1.0 - 1e-6)
    exposed_constant = mapped / (1.0 - mapped)
    if projected_albedo is None:
        tint = _population_tint(pop, stops)
        terrain_hit = np.ones(pop.shape, dtype=bool)
    else:
        tint = np.asarray(projected_albedo, dtype=np.float32)
        if tint.shape != pop.shape + (3,):
            raise ValueError("projected_albedo must have shape (H, W, 3)")
        terrain_hit = np.any(tint > 0.0, axis=2)
    exposed_tinted = exposed_constant * (tint / float(CONSTANT_ALBEDO[0]))
    rgb = exposed_tinted / (1.0 + exposed_tinted)
    # Misses contain the environment, which is independent of terrain albedo.
    rgb[~terrain_hit] = light_rgb[~terrain_hit]
    rgb[~mask] = 1.0
    out = np.empty(light.shape, dtype=np.uint8)
    out[..., :3] = np.round(rgb * 255.0).astype(np.uint8)
    out[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return out


def numpy_oracle_test_vector() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a compact synthetic light-field vector and its oracle output."""
    y, x = np.mgrid[:9, :11]
    light = np.empty((9, 11, 4), dtype=np.uint8)
    light[..., 0] = 42 + 9 * x + 2 * y
    light[..., 1] = 38 + 7 * x + 4 * y
    light[..., 2] = 34 + 5 * x + 3 * y
    light[..., 3] = 255
    population = np.clip(((x - 2.0) ** 2 + (y - 6.0) ** 2) / 105.0, 0.0, 1.0).astype(np.float32)
    mask = ((x - 5.0) ** 2 / 30.0 + (y - 4.0) ** 2 / 18.0) <= 1.0
    return light, population, mask, numpy_oracle(light, population, mask)


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fixture_input_hashes() -> dict[str, str]:
    """Return hashes that lock only the procedural inputs, never GPU output."""
    dem = make_dem()
    population = make_population()
    subject_mask = make_subject_mask()
    light, vector_population, vector_mask, vector_output = numpy_oracle_test_vector()
    return {
        "dem": _sha(dem),
        "population": _sha(population),
        "subject_mask": _sha(subject_mask.astype(np.uint8)),
        "oracle_vector_inputs": hashlib.sha256(
            light.tobytes() + vector_population.tobytes() + vector_mask.astype(np.uint8).tobytes()
        ).hexdigest(),
        "oracle_vector_output": _sha(vector_output),
    }
