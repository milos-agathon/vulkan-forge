# Offline 3D Map Rendering

forge3d's product direction is an offline 3D map-production workflow:

```text
MapScene + LabelPlan + ValidationReport + Bundle
```

Feature `001` establishes the diagnostics contract and support matrices used by
that workflow. Feature `003` adds deterministic `LabelPlan`, and feature `004`
adds the public `MapScene` / `SceneRecipe` API for typed validation,
GPU-terrain PNG/EXR output for renderable terrain recipes, a fatal
`MapSceneNativeUnavailable` diagnostic block whenever native rendering is
unavailable (CPU placeholder output no longer exists), named
MapScene presets, `OutputSpec` samples/denoiser/AOV controls, recipe manifests,
and deterministic review bundles.

| Area | Support level | Notes |
| --- | --- | --- |
| Structured diagnostics | `supported` | `Diagnostic` and `ValidationReport` are public Python objects. |
| Typed MapScene recipe | `underdeveloped` | `MapScene`, `SceneRecipe`, terrain, raster, vector, label, point-cloud, building-intent, camera, lighting, output, and map-furniture recipe objects are public. |
| `MapScene.validate` | `supported` | Returns deterministic structured reports with source-data, CRS, style, glyph, label-plan, memory, support-status, VT, 3D Tiles, and building diagnostics where applicable. |
| `MapScene.render` PNG/EXR path | `supported` | Fixture-backed `.npy`/GeoTIFF terrain, PNG/GeoTIFF raster, inline vector, label, and scalar building recipes render through the GPU-terrain path when the native backend is available; when it is not, render raises `MapSceneNativeUnavailable` with structured diagnostic blocks. Unsupported layer paths still block with typed diagnostics. |
| `OutputSpec` offline controls | `supported` | `samples`, `denoiser`, `aovs`, and `hdr` are public MapScene output fields. AOV and EXR writes use the native `numpy_to_exr` writer when requested. |
| Named MapScene presets | `supported` | `LightingPreset(name="rainier_showcase")` resolves camera, sun, IBL, renderer settings, and reproducibility defaults from `forge3d.presets`. |
| `recipe_manifest` | `supported` | `forge3d.recipe_manifest(scene)` returns a deterministic JSON-safe recipe manifest for bundles, CI, and review tooling. |
| `MapScene.save_bundle` review bundle | `underdeveloped` | Writes deterministic review metadata, recipe intent, validation report, label plans, and label source references. Blocking diagnostics are preserved and marked non-renderable. |
| Deterministic LabelPlan | `supported` | `LabelPlan.compile` produces stable accepted/rejected labels, diagnostics, bounds, seed, and render/export payloads for fixed inputs. |
| Unsupported-path validation | `underdeveloped` | `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and incomplete integration paths must remain diagnostic-bearing. |
| Material, VT, and large-scene polish | `underdeveloped` | P2 gaps are diagnosed before render: unsupported VT families as `vt_unsupported_family`, textured building material intent through texture/UV/fallback diagnostics, and large scenes through memory/cache/LOD/instancing summaries. |
| Web-first hosted tile delivery | `non-goal` | Offline map production remains the scope. |

Unsupported, `Pro-gated`, `placeholder/fallback`, `experimental`, or
`underdeveloped` paths must be reported before successful render completion.
The current product does not render textured PBR buildings end to end; scalar
fallback is not textured PBR support and must remain diagnostic-bearing.

`MapScene.render("map.png")` performs validation if needed and writes PNG output
for supported terrain/raster/vector/label recipes; `OutputSpec(format="exr")`
writes EXR beauty output, and `OutputSpec(aovs=[...])` writes requested AOV EXRs.
Real `.npy`/GeoTIFF terrain and PNG/GeoTIFF raster assets render through the
`gpu_terrain` path; symbolic fixture recipes without a renderable heightmap
block with a fatal `MapSceneNativeUnavailable` diagnostic instead of writing
placeholder pixels.
`MapScene.save_bundle("map.forge3d")`
records the recipe, diagnostics, support summaries, label plans, source
references, camera, lighting, output, render backend, and renderability status
through the public Python API.

## Canonical Examples

The MVP workflow is represented by three typed examples:

- `examples/mapscene_terrain_raster.py`: terrain plus raster overlay validates,
  writes GPU-terrain PNG output from generated local assets, and saves a
  review bundle.
- `examples/mapscene_vector_labels.py`: terrain plus vector overlays and labels
  compiles deterministic `LabelPlan` data, writes GPU-terrain PNG output,
  and saves a review bundle with label diagnostics.
- `examples/mapscene_buildings_labels.py`: terrain plus mixed-roof scalar
  buildings and labels renders terrain-scatter instanced building meshes with
  CSM cast/receive in the GPU terrain path, records per-building batch ids, and
  saves a review bundle.

Run the quickstart coverage with:

```powershell
python -m pytest tests/test_mapscene_quickstart.py -q
```

## Support References

Use the linked guides as the support contract for MVP review:

- `guides/label_plan_guide`
- `guides/diagnostics_reference`
- `guides/style_support_matrix`
- `guides/building_support_matrix`
- `guides/tiles3d_support_matrix`
- `guides/virtual_texturing_support_matrix`
- `guides/large_scene_support`
- `guides/competitive_positioning`
