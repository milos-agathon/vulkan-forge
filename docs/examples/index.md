# Examples Catalog

This page covers every runnable example and notebook in the repo. Use it as the
index for `examples/`.

General rule: start with `python examples/<name>.py --help` when the script
exposes CLI flags.

## Foundational Sanity Checks

| Example | What it demonstrates | Main APIs |
| --- | --- | --- |
| `triangle_png.py` | Smallest end-to-end render check. Creates a PNG with the fallback `Renderer`. | `Renderer.render_triangle_png()` |
| `png_numpy_roundtrip.py` | Round-trip image IO from NumPy to PNG and back. | `numpy_to_png()`, `png_to_numpy()` |
| `terrain_single_tile.py` | Minimal terrain-data processing example that writes a grayscale image from the bundled DEM. | `mini_dem()`, `numpy_to_png()` |
| `terrain_demo.py` | Thin CLI wrapper around the terrain demo module, including preset and bundle-aware workflows. | `forge3d.terrain_demo`, `bundle` |

## Interactive Terrain And Cartography

| Example | What it demonstrates | Main APIs |
| --- | --- | --- |
| `terrain_viewer_interactive.py` | Baseline interactive terrain viewer with camera, sun, AA, and PBR-oriented controls. | `open_viewer_async()`, `ViewerHandle`, terrain IPC |
| `swiss_terrain_landcover_viewer.py` | Terrain plus raster land-cover drape, tuned for polished snapshot output. | `ViewerHandle.load_overlay()` |
| `bosnia_terrain_landcover_viewer.py` | Raster-overlay terrain workflow with final image compositing outside the viewer. | `open_viewer_async()`, `load_overlay()`, PIL composition |
| `belgium_bivariate_climate_map.py` | Climate rasters projected onto terrain and turned into a polished final map. | viewer path, raster overlays, xarray/rioxarray-side prep |
| `poland_population_spikes_height_shade.py` | Terrain-plus-overlay composition for a height-shaded thematic map. | viewer path, overlay controls |
| `pnoa_river_showcase.py` | Pure-Python companion composition example for terrain storytelling and post-processing. | downstream composition around terrain assets |
| `pnoa_river_showcase_video.py` | High-resolution cinematic terrain sequence with overlay styling and final video/frame assembly. | viewer path, `terrain_scatter.viewer_orbit_radius()` |
| `terrain_atmosphere_path_demo.py` | Lower-level terrain-native rendering path without the interactive viewer. | `Session`, `TerrainRenderer`, `TerrainRenderParams`, `MaterialSet`, `IBL` |
| `southeast_europe_population_pt_native.py` | Tiled path-traced terrain poster with population-derived per-pixel albedo and an orthographic camera. | `render_terrain_poster()`, `albedo_map`, orthographic camera |
| `swiss_landcover_pt_oblique.py` | 4K, 4x4 tiled path-traced terrain poster with per-pixel Swiss land-cover albedo and an off-axis camera. | `render_terrain_poster()`, `albedo_map`, off-axis camera |
| `uk_ireland_lighthouse_map.py` | British Isles terrain poster rendered through `TerrainRenderer`, with OSM lighthouses driving night-time glow and spotlight placement. | `Session`, `TerrainRenderer`, Overpass, Terrarium DEM tiles |

## Overlays, Labels, Styles, And Picking

| Example | What it demonstrates | Main APIs |
| --- | --- | --- |
| `luxembourg_rail_overlay.py` | Vector rail overlay draped onto terrain. | `viewer_ipc`, `add_vector_overlay`, terrain IPC |
| `fuji_labels_demo.py` | Label placement, typography, priorities, zoom ranges, and decluttering. | `viewer_ipc.add_label`, label settings |
| `style_viewer_interactive.py` | Mapbox-style import and style-driven vector overlay workflow. | `forge3d.style` |
| `picking_demo.py` | Premium picking path including lasso selection, highlight styles, and rich pick events. | `viewer_ipc` picking helpers |
| `picking_test_interactive.py` | Interactive/manual verification harness for picking behavior. | `viewer_ipc` picking helpers |

## Point Clouds, Buildings, And Large Assets

| Example | What it demonstrates | Main APIs |
| --- | --- | --- |
| `pointcloud_viewer_interactive.py` | LAZ/LAS point-cloud viewing with size and color controls. | `ViewerHandle.load_point_cloud()`, point-cloud IPC |
| `cog_streaming_demo.py` | Direct COG access for tile/window reads, stats, and benchmarking. | `forge3d.cog.open_cog()` |
| `buildings_viewer_interactive.py` | Building import flows from GeoJSON, CityJSON, and 3D Tiles-backed sources. | `forge3d.buildings` |
| `osm_city_demo.py` | OSM-driven city scene around a center point, using forge3d extrusion helpers and a deterministic preview renderer. | `forge3d.io.import_osm_buildings_from_geojson()` |
| `rotterdam_solar_potential_shadow_study.py` | Rotterdam 3D BAG LoD2.2 roof solar suitability with LoD1.2 fallback coverage, selected-time shadows, and optional day-cycle export. | 3D BAG WFS, PVGIS, OSM context, deterministic preview renderer |

## Animation And Camera Automation

| Example | What it demonstrates | Main APIs |
| --- | --- | --- |
| `camera_animation_demo.py` | Keyframe-based camera paths, interpolation preview, and frame export. | `forge3d.animation.CameraAnimation`, viewer IPC |
| `terrain_camera_rigs_demo.py` | Higher-level terrain camera rigs such as orbit, rail, and follow shots. | `forge3d.camera_rigs`, viewer path |
| `khumbu_icefall_sentinel_timelapse.py` | Multi-year Sentinel-2 time-series terrain animation over the Khumbu Icefall with a 5 m DEM render grid and MP4 output. | Microsoft Planetary Computer STAC, Copernicus DEM GLO-30, `open_viewer_async()`, `ffmpeg` |
| `humanity_globe_video.py` | Offline rotating-globe recreation of the Humanity Globe GPW-v4 population-density video with stepped turbo colors and MP4 output. | GPW-v4 population density, `numpy_to_png()`, `ffmpeg` |

## Notebooks

| Notebook | What it demonstrates |
| --- | --- |
| `examples/notebooks/quickstart.ipynb` | First terrain viewer workflow in notebook form. |
| `examples/notebooks/terrain_explorer.ipynb` | Notebook-centric terrain exploration. |
| `examples/notebooks/map_plate.ipynb` | Map-plate composition and cartographic output. |

## Support Files In `examples/`

These files are part of the examples directory but are support artifacts rather
than standalone demos:

- `sample_style.json`: sample input for style-driven overlay workflows
- `presets/*.json`: reusable example presets
- `_import_shim.py` and `_png.py`: internal helpers used by example scripts

## Where To Go Next

- Use the [3D Map Project Ideas](3d-map-project-ideas.md) page for candidate
  future examples that do not duplicate the current catalog.
- Use the [Feature Map](../guides/feature_map.md) to choose the right module family.
- Use the [Tutorials](../tutorials/index.md) for guided onboarding.
- Use the [API Reference](../api/api_reference.rst) when you already know the workflow and need the exact symbol.
