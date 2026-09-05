# Map-maker recipe compatibility

## Native path-traced terrain posters

Use `forge3d.render_terrain_poster` (also available from
`forge3d.path_tracing`) for large path-traced terrain plates. The driver keeps
one global camera and pixel coordinate system while rendering exact,
non-overlapping tiles, so pinhole, off-axis, and orthographic plates do not need
feathering or seam blending.

For a terrain-grid-aligned thematic raster, pass a contiguous `(H, W, 4)` RGBA
`albedo_map` and select `albedo_sampling="nearest"` for categorical data or
`"bilinear"` for continuous data. Alpha below one selects the constant terrain
albedo before interpolation. Use `certificate=` for the final plate certificate
and `cache=` when participating in the ANAMNESIS render contract.

Reference examples:

- `examples/southeast_europe_population_pt_native.py` — hermetic procedural
  population fixture and warm continuous RGBA albedo.
- `examples/swiss_landcover_pt_oblique.py` — 4096 x 4096 off-axis land-cover
  poster assembled as 4 x 4 1024-pixel tiles.

## Legacy light-field plus NumPy overlay

The older two-stage recipe—render a constant-albedo GPU light field and tint it
with a NumPy overlay—remains a compatibility path for reproducing established
art direction. It is **legacy**, not the preferred native material workflow.
`examples/obliqua_se_europe_fixture.py` retains a pure-NumPy oracle for parity
measurement; new maps should send the canonical RGBA raster directly through
`render_terrain_poster` so lighting, ReSTIR reuse, memory accounting, and the
render certificate all describe the final pixels.
