# Gallery

The gallery is intentionally recipe-shaped. Each entry names the dataset, the
small set of APIs involved, and the core idea behind the composition.

Entries marked **Pro** require a license key to run as written.

| Preview | Entry |
| --- | --- |
| ![Mount Rainier](images/01-mount-rainier.png) | [01. Mount Rainier](01-mount-rainier.md) |
| ![Mount Fuji labels](images/02-mount-fuji-labels.png) | [02. Mount Fuji labels](02-mount-fuji-labels.md) |
| ![Swiss landcover](images/03-swiss-landcover.png) | [03. Swiss land-cover](03-swiss-landcover.md) |
| ![Luxembourg rail](images/04-luxembourg-rail-network.png) | [04. Luxembourg rail network](04-luxembourg-rail-network.md) |
| ![3D buildings](images/05-3d-buildings.png) | [05. 3D buildings](05-3d-buildings.md) **Pro** |
| ![Point cloud](images/06-point-cloud.png) | [06. Point cloud](06-point-cloud.md) |
| ![Camera flyover](images/07-camera-flyover.png) | [07. Camera flyover](07-camera-flyover.md) |
| ![Vector export](images/08-vector-export.png) | [08. Vector export](08-vector-export.md) **Pro** |
| ![Shadow comparison](images/09-shadow-comparison.png) | [09. Shadow comparison](09-shadow-comparison.md) |
| ![Map plate](images/10-map-plate.png) | [10. Map plate](10-map-plate.md) **Pro** |

## Recipe Golden Fixtures

These small deterministic fixtures back the CI recipe-golden runner. Regenerate intentionally with `FORGE3D_UPDATE_RECIPE_GOLDENS=1 pytest tests/test_recipe_goldens.py`, review the image diffs, then commit the matching `tests/golden/recipes/*.png` updates with the code change. Failure artifacts are written under `FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR` when that environment variable is set; CI uploads them from `tests/artifacts/`.

CI runs required recipe pixel goldens in the `Visual Goldens (NVIDIA Vulkan)` job. The candidate path installs `pyproj`, proves a physical NVIDIA/Vulkan adapter with `python scripts/terrain_ci_probe.py --mode terrain --require-nvidia-vulkan`, then runs terrain and recipe image comparisons with `FORGE3D_RUN_TERRAIN_GOLDENS=1`. An absent adapter, skipped test, or pixel mismatch is fatal. Candidate code never receives production signing material; signed certificate rotation remains the protected, main-only `Refresh recipe certificates` workflow. Metal coverage is an opt-in support diagnostic and is not acceptance evidence. `FORGE3D_ALLOW_SOFTWARE_GOLDENS=1` remains scoped to local recipe meta-validation and must never be treated as the render-quality gate.

To red-gate a shader-sensitive recipe change, make the smallest local shader perturbation that should affect a listed fixture, run `FORGE3D_RUN_TERRAIN_GOLDENS=1 pytest tests/test_recipe_goldens.py -k <fixture>`, and keep the generated `FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR` diff/actual/expected artifacts with the review notes. Revert the perturbation before landing. Only update a golden after the diff explains an intentional renderer change.

| Fixture | Family | Golden | Command |
| --- | --- | --- | --- |
| `mapscene_terrain_raster` | terrain_raster | `tests/golden/recipes/mapscene_terrain_raster.png` | `pytest tests/test_recipe_goldens.py -k mapscene_terrain_raster` |
| `mapscene_vector_labels` | labels_vectors | `tests/golden/recipes/mapscene_vector_labels.png` | `pytest tests/test_recipe_goldens.py -k mapscene_vector_labels` |
| `mapscene_label_halo_depth` | labels_depth_occlusion | `tests/golden/recipes/mapscene_label_halo_depth.png` | `pytest tests/test_recipe_goldens.py -k mapscene_label_halo_depth` |
| `mapscene_label_arabic_joining` | labels_complex_scripts | `tests/golden/recipes/mapscene_label_arabic_joining.png` | `pytest tests/test_recipe_goldens.py -k mapscene_label_arabic_joining` |
| `mapscene_label_occlusion_ridge` | labels_depth_occlusion | `tests/golden/recipes/mapscene_label_occlusion_ridge.png` | `pytest tests/test_recipe_goldens.py -k mapscene_label_occlusion_ridge` |
| `mapscene_vector_stroke_quality` | vector_stroke_quality | `tests/golden/recipes/mapscene_vector_stroke_quality.png` | `pytest tests/test_recipe_goldens.py -k mapscene_vector_stroke_quality` |
| `mapscene_vector_stroke_quality_4x` | vector_stroke_quality | `tests/golden/recipes/mapscene_vector_stroke_quality_4x.png` | `pytest tests/test_recipe_goldens.py -k mapscene_vector_stroke_quality_4x` |
| `mapscene_thematic_choropleth` | thematic_choropleth | `tests/golden/recipes/mapscene_thematic_choropleth.png` | `pytest tests/test_recipe_goldens.py -k mapscene_thematic_choropleth` |
| `mapscene_offline_aovs` | offline_accumulation | `tests/golden/recipes/mapscene_offline_aovs.png` | `pytest tests/test_recipe_goldens.py -k mapscene_offline_aovs` |
| `mapscene_buildings` | buildings | `tests/golden/recipes/mapscene_buildings.png` | `pytest tests/test_recipe_goldens.py -k mapscene_buildings` |
| `mapscene_screen_space_contact` | screen_space_effects | `tests/golden/recipes/mapscene_screen_space_contact.png` | `pytest tests/test_recipe_goldens.py -k mapscene_screen_space_contact` |
| `mapscene_screen_space_reflection` | screen_space_effects | `tests/golden/recipes/mapscene_screen_space_reflection.png` | `pytest tests/test_recipe_goldens.py -k mapscene_screen_space_reflection` |
| `mapscene_textured_gltf_landmark` | gltf_textured_assets | `tests/golden/recipes/mapscene_textured_gltf_landmark.png` | `pytest tests/test_recipe_goldens.py -k mapscene_textured_gltf_landmark` |
| `mapscene_furniture_graticule` | map_furniture | `tests/golden/recipes/mapscene_furniture_graticule.png` | `pytest tests/test_recipe_goldens.py -k mapscene_furniture_graticule` |
| `mapscene_alignment_utm` | alignment_crs | `tests/golden/recipes/mapscene_alignment_utm.png` | `pytest tests/test_recipe_goldens.py -k mapscene_alignment_utm` |
| `mapscene_material_maps` | terrain_materials | `tests/golden/recipes/mapscene_material_maps.png` | `pytest tests/test_recipe_goldens.py -k mapscene_material_maps` |
| `mapscene_clipmap_large_region` | clipmap_large_region | `tests/golden/recipes/mapscene_clipmap_large_region.png` | `pytest tests/test_recipe_goldens.py -k mapscene_clipmap_large_region` |
| `mapscene_auto_water` | water_masks | `tests/golden/recipes/mapscene_auto_water.png` | `pytest tests/test_recipe_goldens.py -k mapscene_auto_water` |
| `mapscene_cloud_shadows` | cloud_shadows | `tests/golden/recipes/mapscene_cloud_shadows.png` | `pytest tests/test_recipe_goldens.py -k mapscene_cloud_shadows` |
| `mapscene_tiles3d_points` | point_cloud_tiles | `tests/golden/recipes/mapscene_tiles3d_points.png` | `pytest tests/test_recipe_goldens.py -k mapscene_tiles3d_points` |
| `mapscene_copc_points` | point_cloud_tiles | `tests/golden/recipes/mapscene_copc_points.png` | `pytest tests/test_recipe_goldens.py -k mapscene_copc_points` |
| `mapscene_png16_color` | output_color | `tests/golden/recipes/mapscene_png16_color.png` | `pytest tests/test_recipe_goldens.py -k mapscene_png16_color` |

```{toctree}
:maxdepth: 1

01-mount-rainier
02-mount-fuji-labels
03-swiss-landcover
04-luxembourg-rail-network
05-3d-buildings
06-point-cloud
07-camera-flyover
08-vector-export
09-shadow-comparison
10-map-plate
```
