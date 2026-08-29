# Participating media

`forge3d.media.Medium` is the canonical validated participating-medium object.
It carries absorption, scattering, derived extinction, phase, density payload,
explicit version, and a deterministic full identity shared by resource caches
and temporal history.

```python
from forge3d.media import Medium, render_volumetric_reference
import numpy as np

medium = Medium.homogeneous(
    sigma_a=(0.01, 0.02, 0.03),
    sigma_s=(0.10, 0.12, 0.14),
    density=0.8,
    version=1,
)

heightmap = np.zeros((64, 64), dtype=np.float32)
camera = {
    "fov_y": 45.0,
    "terrain_camera": {
        "target": (0.0, 15.0, 0.0),
        "radius": 50.990195,
        "phi_deg": 90.0,
        "theta_deg": 78.690068,
        "mode": "mesh:yup",
    },
}
result = render_volumetric_reference(
    medium,
    heightmap,
    width=128,
    height=96,
    camera=camera,
    samples_per_pixel=16,
    homogeneous_medium_reach=250.0,
    seed=7,
    certificate=True,
)
beauty = result["beauty"]
```

The reference renderer is a GPU pixel producer and follows the repository
certificate contract. Set `certificate=True` to retain the signed execution
certificate in the diagnostics surface, or pass a path to write it. The
certificate records the negotiated adapter and capabilities, tracked
allocations, the executed terrain-trace WGSL, the media pass, and any
degradations.

Attach the same object to an offscreen configuration with the public terrain
configuration helpers:

```python
from forge3d import TerrainRenderParams
from forge3d.media import Medium
from forge3d.terrain_params import AovSettings, make_terrain_params_config

medium = Medium.homogeneous(
    sigma_a=(0.01, 0.02, 0.03),
    sigma_s=(0.10, 0.12, 0.14),
    density=0.8,
    version=1,
)
config = make_terrain_params_config(
    size_px=(128, 96),
    render_scale=1.0,
    terrain_span=64.0,
    msaa_samples=1,
    z_scale=1.0,
    exposure=1.0,
    domain=(0.0, 1.0),
    media=medium,
    aov=AovSettings(
        enabled=True,
        albedo=False,
        normal=False,
        depth=False,
        transmittance=True,
        in_scatter=True,
        cloud_shadow=True,
        optical_depth=True,
    ),
)
params = TerrainRenderParams(config)
```

Pass ``params`` to ``TerrainRenderer.render_with_aov`` together with the
material set, environment maps, and heightmap used by your existing terrain
rendering setup. Assign ``None`` to ``config.media`` before constructing new
render parameters to remove the medium. For an interactive terrain already
loaded through ``open_viewer_async``, call ``viewer.set_media(medium)`` to
attach it and ``viewer.set_media(None)`` to remove it.

The reference result contains `beauty`, `transmittance`, `in_scatter`,
`cloud_shadow`, and `optical_depth` arrays with shape `(height, width, 3)`, plus
the additive `terrain_slice` acceptance array with shape `(height, width)`.
`terrain_slice` maps the production terrain-trace hit distance into the same
64-slice logarithmic froxel coordinate used by real-time termination capture;
the mapping uses the public `clip=(near, far)` argument.
`result["diagnostics"]` uses the same schema as offscreen AOV capture and viewer
statistics: majorant proof and validity; sample and step counts; temporal
history decision and reason; separate host, froxel, density, majorant, and
staging byte counts; adapter, backend, driver, and source revision; and whether
multiple-scattering transport executed. Invalid spectra, density, phase,
majorant, camera, or transport input raises `forge3d.media.MediaError`; no
medium is silently clamped or disabled.

The returned AOV frame exposes the same `has_*`, readback, and `save_*` methods
for all four media AOVs, plus `media_diagnostics`.
Requesting one without `TerrainRenderParams.media` fails instead of returning a
placeholder texture.

`TerrainRenderParams.media = None` preserves the existing non-media renderer.
While a canonical medium is attached to the interactive viewer, its independent
legacy height-fog volumetric pass is bypassed. Analytic sky configuration stays
independent and remains valid. `viewer.get_stats()` reports
`media_diagnostics` after a successful media frame and `media_render_error`
after a failed media frame. Passing `None` to `viewer.set_media` removes the
canonical pass and restores the legacy behavior.
