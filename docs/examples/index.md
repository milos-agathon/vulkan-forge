# Examples Catalog

This page lists every tracked Python example with a `__main__` entry point and
every tracked notebook in `examples/`. Catalog membership means that a runnable
entry point exists; it does not mean that optional data, network access, native
viewer support, a GPU, or video tools are available on every machine.

When a script exposes command-line flags, start with
`python examples/<path>.py --help`.

## MapScene And Public Labels

| Example | What it demonstrates | Main interfaces and runtime notes |
| --- | --- | --- |
| `examples/fuji_labels_demo.py` | Mount Fuji terrain and decluttered labels through the public typed scene path. | `MapScene`, `LabelLayer`; native MapScene render backend required |
| `examples/label_api_truth_basic.py` | Deterministic high-level viewer-label API smoke workflow. | `ViewerHandle` label methods; uses an in-process recording handle |
| `examples/mapscene_terrain_raster.py` | Canonical terrain-plus-raster MapScene. | `MapScene`, `TerrainSource`, `RasterOverlay`; native MapScene render backend required |
| `examples/mapscene_vector_labels.py` | Canonical vector-and-label MapScene. | `MapScene`, `VectorOverlay`, `LabelLayer`; native MapScene render backend required |
| `examples/mapscene_buildings_labels.py` | Typed buildings and labels. | `MapSceneBuildingLayer`, `LabelLayer`; native MapScene render backend required |
| `examples/mapscene_bundled_datasets_showcase.py` | MapScene with datasets shipped by forge3d. | `MapScene`, `forge3d.datasets`; native MapScene render backend required |
| `examples/mapscene_offline_quality.py` | Offline accumulation, AOV, and bundle output. | `MapScene.render`, `MapScene.save_bundle`; native MapScene render backend required |
| `examples/mapscene_p1_assets_bundle_showcase.py` | P1 asset adapters, bundles, and explicit unsupported-path diagnostics. | `LabelLayer`, `MapSceneBuildingLayer`, `Tiles3DLayer`; intentionally preserves diagnostic-bearing status |
| `examples/moon_south_pole.py` | Typed lunar CRS, render, diagnostics, and certificate workflow. | `MapScene`, `forge3d.gis`, `forge3d.certificate`; local lunar DEM and native render support required |

## Interactive Terrain, Overlays, And Camera

| Example | What it demonstrates | Main interfaces and runtime notes |
| --- | --- | --- |
| `examples/terrain_viewer_interactive.py` | Baseline interactive terrain viewer. | viewer IPC compatibility path; native viewer required |
| `examples/terrain_demo.py` | Terrain presets and bundle-aware CLI workflow. | `forge3d.terrain_demo`; native viewer required for rendering |
| `examples/terrain_camera_rigs_demo.py` | Orbit, rail, and target-follow terrain camera rigs. | `forge3d.camera_rigs`, `open_viewer_async`; native viewer required |
| `examples/camera_animation_demo.py` | Keyframed viewer camera paths and frame export. | `CameraAnimation`, viewer IPC compatibility path; native viewer required |
| `examples/swiss_terrain_landcover_viewer.py` | Swiss terrain with a raster land-cover overlay. | `open_viewer_async`, `ViewerHandle.load_overlay`; data/network and native viewer required |
| `examples/bosnia_terrain_landcover_viewer.py` | Bosnia terrain with raster land-cover composition. | `open_viewer_async`, raster overlay workflow; data/network and native viewer required |
| `examples/luxembourg_rail_overlay.py` | Rail vectors draped on terrain. | raw `viewer_ipc` compatibility commands; network and native viewer required |
| `examples/pointcloud_viewer_interactive.py` | Interactive LAZ/LAS point-cloud loading. | point-cloud viewer IPC compatibility path; native viewer and input data required |
| `examples/bryce_canyon_storm_timelapse.py` | Timed rain and projected cloud sheets over Bryce Canyon. | `open_viewer_async`, snapshots; native viewer and video tooling required |
| `examples/khumbu_icefall_sentinel_timelapse.py` | Sentinel-2 time series over Copernicus terrain. | `open_viewer_async`, STAC data, `ffmpeg`; network, data, and native viewer required |

## Regional Terrain And Cartography

| Example | What it demonstrates | Main interfaces and runtime notes |
| --- | --- | --- |
| `examples/colorado_rem_forge3d.py` | Snake River relative-elevation-model map. | `open_viewer_async`, terrain scatter, cartographic furniture; network/data and native viewer required |
| `examples/platte_rem_forge3d.py` | Yellowstone River relative-elevation-model plate. | `Session`, `TerrainRenderer`, `MaterialSet`; network/data and GPU-backed native build required |
| `examples/forest_cover_copernicus/italy_forest_cover_3d.py` | Copernicus forest-cover terrain map of Italy. | `open_viewer_async`, raster overlays; network/data and native viewer required |
| `examples/population_ghsl/iberia_builtup_cover_3d.py` | Iberian GHSL built-up coverage over terrain. | `open_viewer_async`, raster overlays; network/data and native viewer required |
| `examples/population_ghsl/romania_builtup_cover_3d.py` | Romanian GHSL built-up coverage over terrain. | `open_viewer_async`, raster overlays; network/data and native viewer required |
| `examples/population_spike_worldpop/poland_population_spikes.py` | WorldPop density as a 3D spike map. | raw viewer IPC compatibility path; local data and native viewer required |
| `examples/population_spike_worldpop/poland_population_spikes_height_shade.py` | WorldPop density with height-shade styling. | shared Poland viewer pipeline; local data and native viewer required |
| `examples/population_spike_worldpop/poland_population_contour_3d.py` | Stepped population contours over 3D terrain. | shared Poland viewer pipeline; local data and native viewer required |
| `examples/population_spike_worldpop/germany_population_spikes_height_shade.py` | Germany WorldPop height-shade variant. | shared WorldPop viewer pipeline; local data, optional palette dependency, and native viewer required |
| `examples/population_spike_worldpop/france_population_spikes_height_shade.py` | France WorldPop height-shade variant. | shared WorldPop viewer pipeline; local data and native viewer required |
| `examples/rotterdam_solar_potential_shadow_study.py` | Rotterdam roof-solar suitability and selected-time shadows. | 3D BAG/PVGIS/OSM acquisition with deterministic preview; network required for uncached data |
| `examples/osm_city_demo.py` | OSM building extrusion with a deterministic preview renderer. | `forge3d.io.import_osm_buildings_from_geojson`; network required for uncached data |
| `examples/osm_city_daycycle.py` | Animated sunlight and shadows over the OSM city scene. | builds on `examples/osm_city_demo.py`; network/cache and `ffmpeg` required |
| `examples/helsinki_transit_daycycle.py` | Helsinki transit and road-flow day cycle. | builds on the OSM city scripts; network/cache and `ffmpeg` required |
| `examples/turkiye_river_basins_3d.py` | Deterministic poster helpers for a Turkiye river-basins composition. | helper-backed CLI entry point; current entry point reports the target poster dimensions |
| `examples/uk_ireland_lighthouse_map.py` | Night terrain poster with OSM lighthouse placement. | `Session`, `TerrainRenderer`; network/data and GPU-backed native build required |

## Smoke, Atmosphere, And Video Composition

| Example | What it demonstrates | Main interfaces and runtime notes |
| --- | --- | --- |
| `examples/california_fire_smoke_effect.py` | Deterministic geospatial smoke transport overlays. | NumPy/Pillow composition; no live incident-data claim |
| `examples/california_wildfire_smoke_video.py` | California terrain, wildfire, wind, and smoke-exposure video. | data acquisition and Python composition; network/cache and `ffmpeg` required |
| `examples/california_cigar_smoke_demo.py` | August Complex hybrid volumetric-smoke demonstration. | `forge3d.smoke` when available plus cached assets and video tooling |
| `examples/humanity_globe_video.py` | Offline population-density globe video. | `numpy_to_png`, GPW-v4 input, `ffmpeg` |

## Notebooks

| Notebook | What it demonstrates |
| --- | --- |
| `examples/notebooks/quickstart.ipynb` | First terrain-viewer workflow in notebook form. |
| `examples/notebooks/terrain_explorer.ipynb` | Notebook-centric terrain exploration. |
| `examples/notebooks/map_plate.ipynb` | Map-plate composition and cartographic output. |

## Support Files

These tracked files have no standalone Python entry point:

- `examples/_import_shim.py`: repository-import helper used by scripts
- `examples/_terrain_feature_demo.py`: shared file-backed rendering helpers for the TV terrain demos
- `examples/terrain_tv4_material_variation_demo.py`: callable real-DEM procedural material-variation demo
- `examples/terrain_tv6_heterogeneous_volumetrics_demo.py`: callable real-DEM heterogeneous-volume viewer demo
- `examples/terrain_tv10_subsurface_demo.py`: callable real-mountain-DEM terrain subsurface-response demo
- `examples/terrain_tv21_blending_demo.py`: callable real-DEM terrain mesh-blending demo
- `examples/terrain_tv24_reflection_probe_demo.py`: callable real-DEM local-reflection-probe and water demo
- `examples/sample_style.json`: sample style input
- `examples/presets/baseline_no_vector_overlays.json`: baseline viewer preset
- `examples/presets/rainier_showcase.json`: Rainier showcase preset

## Where To Go Next

- Use the [3D Map Project Ideas](3d-map-project-ideas.md) page for candidate
  future examples; it is not a list of current runnable files.
- Use the [Feature Map](../guides/feature_map.md) to choose the right module family.
- Use the [Tutorials](../tutorials/index.md) for guided onboarding.
- Use the [API Reference](../api/api_reference.rst) for exact symbols.
