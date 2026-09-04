# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and follows SemVer (pre-1.0 may include breaking changes).

## [Unreleased]

## [1.35.0] - 2026-08-13
### Added
- HELIOS: added public NREL SPA solar positioning via `forge3d.geo.solar_position` while preserving the existing NOAA-compatible `sun_position` APIs, geodetic-curvature/refraction viewshed and shadow APIs, and shared PROMETHEUS min-max terrain traversal for viewshed and shadow queries.
- HELIOS acceptance includes independent Whitebox and golden-image evidence, deterministic matching DX12/Vulkan physical-GPU results, and zero measured renderer memory growth.

### Changed
- Bumped the package and PyPI version to `1.35.0`.

## [1.34.0] - 2026-07-25
### Added
- DUPLA: double-float (DD) arithmetic in lockstep Rust and WGSL, with Knuth/Dekker transforms and Joldes-Muller-Popescu error bounds. Adds the `forge3d.precision` module plus the `dd_selftest`, `dd_harness`, and `dd_jitter_demo` entry points, backend exactness canaries, tracked GPU proof buffers, and certificate `precision`/`jitter` evidence. Measured on an RTX 3070 across DX12 and Vulkan, every operation stays inside its cited bound (add 2.39/3 u2, mul 5.63/7, div 5.92/15, sqrt 3.34/15) over 100M generated plus 1M adversarial vectors per operation with zero mirror mismatches.
- EUCLIDEA: exact geometric predicates and a boolean overlay engine (`src/geometry/exact`, `src/geometry/overlay`), with a verification oracle, snap rounding, ring/face reconstruction, and validity checks, reached through `forge3d.gis`.
- ANAMNESIS: content-addressed render caching. Adds the `forge3d.anamnesis` module and the `anamnesis_leaf_key`, `anamnesis_pass_key`, `anamnesis_engine_fingerprint`, `anamnesis_store_verify`, `anamnesis_store_gc`, `anamnesis_store_put_leaf`, `anamnesis_store_get`, and `anamnesis_restore_rgba8` entry points. Render entry points take a `cache=` keyword, and cross-backend portability is proven physically: a store seeded on NVIDIA Vulkan restores through DX12 in a separate job and clean workspace.
- COMPENDIUM: the F3DZ DEM codec. Adds the `forge3d.codec` module with `compress_dem`, `decompress_dem`, and `verify_dem`, a fail-closed decoder, CRC and error-bound verification, certificate `codec` evidence, and a cross-platform determinism lane proving every corpus page is byte-identical between CPU and GPU decode.
- LIMES: analytic vector coverage. Adds `vector_render_analytic_py` and `vector_coverage_primitives_py`, and a `quality="analytic"` option on `VectorScene.render_oit`/`render_snapshot` that computes exact round-stroke coverage on the GPU instead of tessellating. Agreement with a committed 64x64 supersampled reference is within 1e-3 mean and 0.5/255 maximum absolute error across the torture sheet.

### Fixed
- `determinism-matrix.yml`: the `f3dz-stream` job checked out without pinning `ref` to the pull-request head, unlike every other checkout in that workflow. It is now exact-head pinned, and the provenance gate covers every workflow rather than `ci.yml` alone.
- The LIMES numerical, ablation, mosaic, and determinism gates required only `f3d.has_gpu()`, which is also true for the CPU rasterizers hosted runners expose. On hosted Windows WARP one torture case took 17 minutes and failed the reference gate, against 0.13 s and a pass on real adapters; ten such cases overran the 35-minute job budget and cancelled the lane. They now require a hardware adapter.
- Docs workflow: pull-request runs shared one global `pages` concurrency group with `cancel-in-progress`, so any two overlapping runs cancelled each other and the loser reported a red `Build Docs` check. Pull requests now use a per-ref group; `pages` stays serialized for the runs that actually deploy.

### Changed
- Bumped the package and PyPI version to `1.34.0`.
- LIMES ingest and resolve CPU cost reduced: `push_arc` reuses a scratch buffer instead of allocating one vector per arc (~400k allocations on the 100k-segment throughput scene, 63.3 -> 51.5 ms), and `linear_to_straight_rgba8` writes through a pre-sized buffer instead of 8.3M individual pushes (18.7 -> 14.6 ms). Output is byte-identical.
- The LIMES throughput gate now measures frame time, as `31-limes.md` requirement 6 states, rather than summed `gpu_ms`. The previous metric compared GPU time against a default path that spends 0.089 ms of a ~54 ms frame on the GPU, the rest being CPU tessellation. The gate carries a documented deviation: the measured ratio is 4.2-4.6x rather than the required 2x, so it is set as a regression bound at 5.5x, and the shortfall is recorded rather than hidden.

## [1.33.0] - 2026-07-14
### Added
- Windowed remote `read_cog` now streams over HTTP range requests for chunky-planar **striped and tiled** COGs, fetching only the strips or tiles that overlap the requested window instead of downloading the whole object (`src/gis/cog_range.rs` wraps `RangeReader` in a `Read+Seek` cursor feeding a window-selective strip/tile decoder). Measured on a 1024² 64-tile fixture: a 400×400 window transfers ~22.7% of the object (1.19× the intersecting tile payload); a 40×40 in-tile window ~4.0%.
- Added honest, distinct diagnostics for the remote COG path: `range_read` for a streamed window, `range_fallback` (with `unsupported_layout`/`range_not_supported`/`invalid_range_response` reasons) for a full-object fallback, and `cache_hit`/`cache_miss` for the fetch cache.

### Fixed
- `RangeReader` validates every HTTP 206 partial response — `Content-Range` start/end/total and body length — before the bytes may enter the byte cache or a decoder, and accepts case-varied `bytes` range-unit names per RFC 9110 §14.1.
- Out-of-bounds windows raise `InvalidArgument` after dimensions are established via metadata range requests, without ever triggering a full-object download — including on layouts the windowed decoder cannot assemble.
- A server that ignores `Range` or returns an invalid 206 falls back to a correct full fetch-to-cache decode rather than corrupting the read; fatal decode errors (e.g. a corrupt TIFF header) propagate without any fallback download.

### Changed
- Bumped the package and PyPI version to `1.33.0`.
- Split the raster reader (`raster_info.rs` into `raster_read`/`raster_tags`/`raster_values`/`raster_window`) and the COG range reader (`range_reader.rs` into `range_cache`/`range_stats`/`content_range`) to honor the repository's module-size guideline.

## [1.32.0] - 2026-07-14
### Added
- Added signed render certificates with live GPU-pass timing and verified shader-source provenance across the production render surface.
- Added CI gates for resource allocation ownership, negotiated capabilities, dead render structures, certificate coverage, clean-checkout fixtures, and determinism artifact attribution.
- MENSURA: added a pure-Rust CRS/projection engine (Transverse Mercator, Lambert Conformal Conic 2SP, Albers Equal-Area, Mercator, Polar Stereographic, and geocentric) reached through a single dispatch table, an expanded ITRF Helmert datum-transform surface, Karney-geodesic distance/area, and EGM96 geoid-height lookups.
- MENSURA: added a first-class `height_system` threaded through `RasterInfo` and DEM preparation (with chart-datum carry-through), a canonical antimeridian polygon splitter, and world-coordinate f32-narrowing anchoring guards.
- MENSURA: added geodesy/CRS regression coverage — a projection oracle, dateline-splitter and height-system test matrices, per-method conformance residual evidence, and a compile-fail world-coordinate f32 source gate.

### Fixed
- Refused software and hypervisor-virtualized adapters in deterministic mode so their hashes cannot masquerade as physical-hardware evidence; real hardware hashes remain pairwise and golden checked with zero-byte tolerance.
- Made the canonical determinism inputs and IBL precompute source deterministic, including explicit adapter metadata for every accepted hash.
- Corrected clean-checkout golden, LFS, font, packaging, and cross-platform CI setup defects.

### Changed
- CENSOR render paths now report explicit degradations rather than silently claiming unavailable GPU timing or fallback behavior as successful execution.
- MENSURA: `forge3d.gis.reproject_raster` and vector reprojection now raise structured errors (`InvalidCrs` at parse, `TransformFailed` / `BackendUnavailable` at transform) with per-pixel failure accounting, instead of silently returning an unchanged or all-nodata raster with a success status.
- No migration is required; existing Python APIs remain compatible.

## [1.31.0] - 2026-07-09
### Added
- Completed the Blender-outmatch P2/P3 map rendering quality pass: COG/cache streaming, indexed clipmap terrain rendering, native 3D Tiles/COPC point rendering, terrain screen-space effects, VT timing/metadata, GPU water/cloud map-scene controls, memory/timing diagnostics, textured glTF landmarks, thematic GPU styling, and Arabic label shaping coverage.
- Added the P2/P3 implementation audit and validation evidence to `docs/3d-map-rendering-quality-blender-outmatch-plan.md`, with all P2 tasks full and evidence-based P3 deferrals recorded.
- Added the P2 recipe golden baselines for clipmaps, 3D Tiles/COPC points, screen-space effects, water/clouds, textured glTF, thematic choropleths, and Arabic label joining.

### Changed
- Bumped the package and PyPI version to `1.31.0`.
- Removed the recipe-golden ignore rules that kept intended P2/P3 baselines out of version control.

## [1.30.0] - 2026-07-04
### Added
- Completed the Blender-outmatch P0/P1 map rendering quality block: MapScene GPU terrain rendering, labels/vectors, offline output, presets, alignment diagnostics, SDF text atlas support, material maps, buildings, cartographic furniture, Hosek-Wilkie sky, 16-bit/color pipeline work, and recipe-level golden gates.
- Added the P0/P1 implementation audit to `docs/3d-map-rendering-quality-blender-outmatch-plan.md`, marking all BOP-P0 and BOP-P1 tasks full with residual polish tracked separately.

### Changed
- Bumped the package and PyPI version to `1.30.0`.

## [1.29.0] - 2026-07-02
### Added
- G-002c: Added Rust-first GIS vector metadata/read, CRS reprojection, geometry measurement/topology, rasterization, geometry masks, raster masks, and thematic raster APIs.
- Added `forge3d.gis.normalize_raster(...)` and `forge3d.gis.classify_raster(...)` for nodata-aware normalization and explicit-bin classification with class tables.
- Added the G-002c C6 thematic raster implementation plan and regression coverage for the thematic raster surface.

### Changed
- Updated package metadata to describe forge3d's Rust-first GIS vector, rasterization, mask, thematic, CRS, affine, windowing, and reprojection support.
- Bumped the package and PyPI version to `1.29.0`.

## [1.28.0] - 2026-06-30
### Added
- G-002a1: Added public Rust-backed `forge3d.gis.read_raster(path, bands=None, window=None, masked=False)` with 1-based band selection, in-bounds pixel windows, serialized `RasterInfo` metadata, per-band nodata summaries, and true-valid masks when requested.
- G-002b: Added Rust-first local TIFF GIS CRS/affine/bounds/windowing/resampling/alignment/reprojection APIs with stable diagnostics and a support matrix documenting built-in EPSG:4326/EPSG:3857 limits.

### Changed
- Sealed the G-002a1 local raster read/write foundation before G-002c planning: `read_raster_info`, `read_raster`, `write_raster`, `read_raster_window`, and `read_raster_mask` now share Rust-backed TIFF read internals.
- Updated package metadata to describe forge3d's local GIS raster read/write, CRS, affine, windowing, and reprojection support.
- Bumped the package version to `1.28.0`.

## [1.27.0] - 2026-06-29
### Added
- G-001: Added the public `forge3d.gis` module surface with `RasterInfo`, `read_raster_info(path)`, and `write_raster(...)`.
- G-002a: Added Rust TIFF-backed local raster metadata reading, PyO3 bindings, Python wrappers, type stubs, and API contract coverage.
- G-002a1: Added GeoTIFF writing support covering CRS authority metadata, affine transforms, nodata, creation options, overwrite validation, warning codes, and round-trip raster tests.

### Changed
- Updated package metadata to describe forge3d's Rust-backed local GIS raster read/write support.
- Bumped the package version to `1.27.0`.

## [1.26.0] - 2026-05-18
### Added
- Implemented Spec 006 P2 material, virtual-texture, and large-scene polish: deterministic VT family validation, textured-building material diagnostics, advanced static label planning, and large-scene resource summaries.
- Added P2 support diagnostics for unsupported VT families, missing texture paths, missing UVs, unsupported texture formats, unavailable cache/LOD stats, unsupported instancing paths, estimated GPU memory, experimental label paths, and explicit scalar/material fallbacks.
- Added support docs and audit coverage for VT normal/mask status, building texture prerequisites and fallback behavior, advanced labels, large-scene offline scope, and non-MVP-blocking P2 deferrals.

### Changed
- Updated package metadata to describe forge3d's P2 material/VT diagnostics, advanced labels, large-scene summaries, and reproducible offline map bundles.
- Bumped the package version to `1.26.0`.

## [1.25.0] - 2026-05-18
### Added
- Implemented Spec 005 P1 map asset workflows: data-driven `LabelLayer` ingestion from features, typography coverage and fallback declarations, public `MapSceneBuildingLayer` and `Tiles3DLayer` scene adapters, and deterministic map-scene bundle round-trip state.
- Added structured P1 diagnostics for missing label fields, Unicode coverage gaps, unsupported tile formats/features, missing external assets, unavailable terrain sampling, Pro-gated paths, and placeholder/fallback paths.
- Added `examples/mapscene_p1_assets_bundle_showcase.py`, which uses repo datasets plus a local synthetic tileset to exercise labels, typography, buildings, 3D Tiles review metadata, validation diagnostics, and bundle reload.

### Changed
- Updated package metadata to describe forge3d's P1 map asset layers, 3D Tiles review support, diagnostics, and reproducible bundle workflow.
- Bumped the package version to `1.25.0`.

## [1.24.0] - 2026-05-18
### Added
- Implemented specs 001-004 for offline 3D map workflows: structured diagnostics/support matrices, label API truthfulness, deterministic `LabelPlan`, and the typed `MapScene` MVP.
- Added public `Diagnostic`, `ValidationReport`, support-status, and render-failure policy contracts with deterministic serialization and bundle-ready reporting.
- Added deterministic label planning with priority classes, keepouts, accepted/rejected label summaries, and validation diagnostics.
- Added `MapScene`, `SceneRecipe`, typed terrain/raster/vector/label/building/point-cloud recipe objects, pre-render validation, PNG rendering, and deterministic review-bundle save support.
- Added canonical MapScene examples plus `examples/mapscene_bundled_datasets_showcase.py`, which uses the bundled `mini_dem` and `sample_boundaries` datasets to exercise specs 001-004 together.

### Changed
- Updated package metadata to describe forge3d as a Rust/WebGPU 3D map engine with diagnostics, labels, and reproducible scene bundles.
- Bumped the package version to `1.24.0`.

## [1.23.0] - 2026-03-30
### Added
- `pip install forge3d` now installs the `interactive_viewer` command, so standard viewer usage no longer requires a separate `cargo build --release --bin interactive_viewer` step.
- Completed Epic TV22 scatter wind animation across the terrain renderer and interactive viewer workflows.
- Added regression coverage for `render_with_aov(..., time_seconds=...)` and an opt-in live viewer wind path covering viewer time accumulation and camera updates.

### Fixed
- Viewer IPC now rejects invalid terrain scatter wind payloads instead of silently replacing them with defaults.
- Hardened the terrain scatter Python contract so invalid `wind` objects fail fast before serialization.
- Updated viewer and example documentation to use the installed `interactive_viewer` command by default, while keeping direct Cargo execution as a source-checkout fallback.

### Changed
- Bumped the package version to `1.23.0`.

## [1.22.0] - 2026-03-30

### Added
- Implemented Epic TV17 terrain camera rigs with reusable orbit, rail, and target-follow rig builders in `forge3d.camera_rigs`.
- Added target-aware `CameraKeyframe` editing APIs, a dedicated terrain camera rig demo, and regression coverage for determinism, clearance enforcement, and target-bearing playback/export.

### Changed
- Bumped the package version to `1.22.0`.

## [1.21.0] - 2026-03-29

### Added
- Implemented Epic TV16 terrain scene variants and review layers across scene bundles, the Python viewer handle surface, and the viewer IPC/runtime state path.
- Added named `ReviewLayer` and `SceneVariant` bundle schema support, atomic list/apply/query APIs, and regression coverage for variant persistence, round-trips, and manual layer overrides.
- Added TV16 bundle documentation updates covering the canonical `scene/state.json` payload and the Python scene-bundle workflow.

### Changed
- Bumped the package version to `1.21.0`.

## [1.19.0] - 2026-03-23

### Added
- Implemented Epic TV13: Terrain Population LOD Pipeline with QEM mesh simplification, auto-generated scatter LOD chains, and HLOD clustering wired through the renderer, viewer, and IPC paths.
- Added `forge3d.geometry.simplify_mesh()`, `forge3d.geometry.generate_lod_chain()`, `forge3d.terrain_scatter.auto_lod_levels()`, and `forge3d.terrain_scatter.HLODPolicy`.
- Added TV13 documentation, an end-to-end Mt. Fuji demo, and targeted Rust/Python regression coverage for the new workflow.

### Fixed
- Hardened TV13 mesh simplification bounds validation so malformed meshes are rejected across all target ratios instead of propagating corrupt data.
- Corrected HLOD cluster activation bounds to account for mesh extents at per-instance scale, preventing premature activation for large transformed instances.
- Fixed default auto-LOD ratio generation for `lod_count > 5`.
- Fixed the TV13 demo import fallback so stale installed `forge3d` wheels are replaced by the repo copy when needed.

### Changed
- Bumped the package version to `1.19.0`.

## [1.18.0] - 2026-03-22

### Added
- Implemented Epic TV6: Heterogeneous Terrain Volumetrics with bounded density volumes, viewer IPC wiring, and Python preset constructors for valley fog, plume, and localized haze.
- Added TV6 documentation, a terrain volumetrics demo, and regression coverage for density-volume config export, viewer IPC reporting, and real-DEM render-budget validation.

### Changed
- Bumped the package version to `1.18.0`.

## [1.17.0] - 2026-03-22

### Added
- Implemented Epic TV10: Terrain Subsurface Materials with per-layer subsurface controls for snow, rock, and wetness terrain layers.
- Added terrain-side subsurface shader wiring, native uniform/config plumbing, a dedicated TV10 workflow doc page, a real-DEM example, and dedicated runtime plus golden-image coverage.

### Changed
- Published the TV10 workflow in the main docs navigation and API reference.
- Extended the terrain golden CI lane to run the new TV10 goldens and hardened the TV10 real-DEM example test to skip on unsupported hosted adapters.

## [1.20.0] - 2026-03-24

### Added
- Implemented Epic TV12 terrain offline render quality with deterministic offline accumulation, adaptive sampling, HDR resolve/export, aligned terrain AOV support, and optional OIDN denoising through the public Python API.
- Added the TV12 public docs page, example workflow, offline controller, HDR frame binding, native offline shaders/passes, and regression coverage for architecture, controller behavior, and runtime image quality.
- Added the adjacent terrain renderer updates required by the shipped TV12 path, including depth/readback plumbing, terrain data revision handling, reflection-probe terrain integration, and terrain-scatter blend controls that share the same renderer surface.

### Changed
- Tightened the offline contract so `render_offline()` requires explicit `OfflineQualitySettings(enabled=True)`, adaptive jitter budgeting covers the full planned sample budget, and `HdrFrame.save()` requires an explicit `.exr` suffix while releasing the GIL during export.
- Bumped the package version to `1.20.0`.

## [1.16.0] - 2026-03-22

### Added
- Implemented Epic TV5: Local Probe Lighting for Terrain Scenes — SH L2 irradiance probes for terrain global illumination.
- Added probe baker, GPU types, and `SHL2` spherical harmonics infrastructure in `src/terrain/probes/`.
- Added `terrain_probes.wgsl` shader with probe sampling, blending, and debug visualization modes.
- Added local probe lighting overview documentation page for TV5.

### Changed
- Bumped the package version to `1.16.0`.

## [1.15.1] - 2026-03-21

### Fixed
- Hardened the TV4 GPU test gating so generic hosted-runner Python jobs skip the new terrain-heavy tests unless the current adapter is verified as terrain-render-capable. This keeps the dedicated probed terrain lanes authoritative while restoring green CI for the release branch.

## [1.15.0] - 2026-03-21

### Added
- Added the public TV4 terrain workflow documentation page and linked it from the main docs spine and API reference.
- Added a shared terrain-noise shader unit for terrain material variation and existing terrain detail-noise callsites.
- Added bounded TV4 terrain material variation controls through `MaterialNoiseSettings` and `MaterialLayerSettings.variation`.
- Added a real-DEM TV4 example plus regression coverage for shared noise wiring, zero-regression defaults, visible snow/rock/wetness variation output, and the TV4 render-time budget.
- Added a CI example lane for the TV4 terrain material variation demo.

### Changed
- Bumped the package version to `1.15.0`.

## [1.14.0] - 2026-03-20

### Added
- Added terrain workflow documentation pages for the first three terrain-visualization epics:
  - TV1 terrain atmosphere path parity
  - TV2 terrain AOV and multi-channel EXR export
  - TV3 terrain scatter and population
- Added the `forge3d.terrain_scatter` Python workflow to the published docs spine and API reference.
- Added terrain scatter upload, viewer IPC wiring, and renderer stats/memory-report surfaces for terrain-native population workflows.
- Added terrain AOV multi-channel EXR export coverage and a dedicated TV2 example path in CI.

### Changed
- Bumped the package version to `1.14.0`.
- Promoted the TV1-TV3 terrain workflows into the main documentation navigation so they ship as first-class public docs instead of notes/examples only.

## [1.13.1] - 2026-03-19

### Fixed
- Stopped `Scene.enable_soft_light_radius()` from panicking during pipeline creation by removing eager construction of unsupported multi-light and soft-shadow pipelines.
- Fixed the point/spot light shader to use depth-array shadow resource types that match the Rust bind-group layout, eliminating the `invalid function call` panic in `Scene.enable_point_spot_lights()`.
- Stopped eager construction of the unused point/spot shadow pipeline, avoiding a second validation failure once the shader module is created successfully.

### Added
- Added `tests/test_light_feature_enablement.py` to lock the Windows notebook repro sequence: create `Scene`, upload `mini_dem`, enable soft light radius, enable point/spot lights, add/clear lights, and render successfully.

## [1.13.0] - 2026-03-17

### Changed
- Removed the legacy `render_raster`, `render_polygons`, and `render_raytrace_mesh` Python API in favor of the viewer/IPC rendering path.
- Raised the Python support floor to 3.10+ and aligned the Rust abi3 target with that floor (`abi3-py310`).
- Refreshed package metadata, docs version sourcing, and public API smoke coverage for the developer-platform packaging work.

### Added
- Added `tests/test_install_smoke.py` to gate import/public-surface/version smoke checks.
- Added Linux `aarch64` wheel builds to CI and a new tag-driven `publish.yml` workflow for PyPI releases.
- Added `docs/product/pro-boundary-notes.md` as the Phase 1 decision log for the future open/Pro split.
- Added an offline Phase 3 license module, Pro-gating tests, and a shared pytest fixture for Pro-only workflows.

### Documentation
- Rewrote the README around the developer-platform launch story and open/Pro split.
- Added Pro callouts to gated tutorials and gallery entries.
- Added `CONTRIBUTING.md`, `SECURITY.md`, and `docs/product/launch-blog.md` for launch readiness.

## [1.12.2] - 2026-02-20

### Added
- **API consolidation execution coverage**
  - Added Scene/runtime behavior tests for SSGI/SSR and bloom execution wiring.
  - Added integration coverage for point-cloud GPU path and COPC/LAZ fixture gates.
  - Added API docs pages for reflections and cloud shadows and refreshed audit snapshots.

### Fixed
- **Scene SSAO uniform wiring**
  - Updated Scene-side SSAO uniform layout to match WGSL `SsaoSettings` (including sample count and projection parameters), preventing silent zero-sample occlusion.
  - Corrected runtime `proj_scale` upload to use the documented formula `0.5 * height * P[1][1]` from the active projection matrix so SSAO/GTAO screen-space radius tracks camera FOV.
  - Aligned SSAO `ao_min` default to `0.35` (matching shader defaults/docs), restoring the intended ambient occlusion floor in rendered output.
- **Bloom CPU fallback threshold behavior**
  - Replaced threshold remap/clamp path with shader-consistent soft-threshold extraction to avoid collapse at higher threshold values.
- **Test safety**
  - Updated GPU scene probe helpers to catch `Exception` instead of `BaseException`, preserving `KeyboardInterrupt`/`SystemExit` semantics.

## [1.12.1] - Vector Export (SVG/PDF)

### Added
- **Priority 5 — Vector Export for Print-Grade Overlays**
  - New `src/export` module with projection, SVG, and label text generation.
  - `project_3d_to_2d()`: 3D to 2D projection with view-projection matrix.
  - `project_2d_to_screen()`: 2D bounds to screen coordinate mapping.
  - `Bounds2D`: Axis-aligned bounding box for coordinate mapping.
  - `vectors_to_svg()`: Generate SVG from polygon and polyline definitions.
  - `labels_to_svg_text()`: Generate SVG text elements with halo support.
  - `SvgExportConfig`: Configuration for SVG export (precision, background, line styles).
  - New `python/forge3d/export.py` module with pure-Python API.
  - `VectorScene`: Container class for collecting polygons, polylines, and labels.
  - `generate_svg()`: Generate SVG string from VectorScene.
  - `export_svg()`: Export VectorScene to SVG file.
  - `export_pdf()`: Export to PDF via optional cairosvg dependency.
  - `validate_svg()`: XML structure validation for SVG output.
  - Support for polygons with holes (evenodd fill-rule).
  - Label halo rendering via stroke for better readability.
  - New example: `examples/vector_export_demo.py` with 4 demo types (simple, contours, features, full).
  - 47 new tests for SVG generation and coordinate projection.

## [1.12.0] - 3D Buildings Pipeline

### Added
- **Priority 4 — 3D Buildings Pipeline**
  - **Roof Type Inference (M4.1)**: `RoofType` enum with 10 roof shapes (flat, gabled, hipped, pyramidal, dome, mansard, shed, gambrel, onion, skillion).
  - `infer_roof_type()` function to infer roof type from OSM tags (`building:roof:shape`, `roof:shape`, building type).
  - Height multiplier per roof type for realistic building extrusion.
  - **Material Presets (M4.2)**: `BuildingMaterial` struct with PBR properties (albedo, roughness, metallic, IOR, emissive).
  - 18 material presets: brick, concrete, glass, steel, aluminum, wood, plaster, stone, sandstone, granite, marble, and roof materials.
  - `material_from_tags()` to infer materials from OSM tags (`building:material`, `building:facade:material`).
  - `material_from_name()` for preset lookup.
  - CSS color parsing (`#RGB`, `#RRGGBB`, named colors) for building colors.
  - **CityJSON Parser (M4.3)**: Full CityJSON 1.1 format parser (`parse_cityjson()`).
  - Support for Building and BuildingPart city objects.
  - LOD selection (prefers highest available LOD).
  - Transform (scale/translate) application to vertices.
  - CRS extraction from metadata.
  - Automatic normal generation for parsed geometry.
  - **Terrain Integration (M4.4)**: `BuildingRenderData` struct for GPU-ready building batches.
  - `Tiles3dRenderer::prepare_buildings()` and `get_visible_buildings()` methods.
  - Distance-based building culling.
  - **Python API (M4.5)**: New `forge3d.buildings` module.
  - `Building`, `BuildingLayer`, `BuildingMaterial` dataclasses.
  - `add_buildings()` - Load buildings from GeoJSON with extrusion.
  - `add_buildings_cityjson()` - Load buildings from CityJSON.
  - `add_buildings_3dtiles()` - Load building metadata from 3D Tiles.
  - Python bindings: `infer_roof_type_py()`, `material_from_tags_py()`, `material_from_name_py()`, `parse_cityjson_py()`.
  - 44 new tests covering extrusion, materials, CityJSON parsing, and roof inference.

## [1.11.1] - CRS Reprojection Support

### Added
- **Priority 3 — On-the-fly CRS Reprojection (PROJ)**
  - Feature-gated PROJ library integration (`proj` Cargo feature).
  - New `src/geo` module with `reproject_coords`, `reproject_point`, `validate_crs`, `crs_equal` functions.
  - Python bindings: `proj_available()`, `reproject_coords()` exposed to Python.
  - Python wrapper module `python/forge3d/crs.py` with pyproj fallback.
  - `transform_coords()`: Transform coordinate arrays between CRS (e.g., WGS84 to UTM).
  - `reproject_geom()`: Reproject Shapely geometries.
  - `crs_to_epsg()`: Parse EPSG codes from CRS strings.
  - `get_crs_from_rasterio()`, `get_crs_from_geopandas()`: Extract CRS from geospatial files.
  - `render_polygons()` now accepts `target_crs` parameter for automatic reprojection.
  - `TerrainRenderParams` now includes `terrain_crs` field for terrain coordinate system.
  - Support for both modern pyproj (>= 2.0) and legacy pyproj APIs.
  - Comprehensive test suite: `test_crs_reproject.py`, `test_crs_auto.py`.

## [1.11.0] - Scene Bundle & Style Spec

### Added
- **Priority 1 — Scene Bundle (.forge3d) Support**
  - Implemented reproducible scene archive format containing terrain, overlays, camera state, and render settings.
  - New `src/bundle` module with `BundleManifest` schema and checksum validation.
  - Python API: `save_bundle()`, `load_bundle()`, `is_bundle()` for easy artifact management.
  - Interactive Viewer integration: Save and load full scene state via IPC commands.
  - CLI integration: `--save-bundle` and `--load-bundle` flags for batch processing and restoration.

- **Priority 2 — Mapbox Style Spec Import**
  - Full Mapbox GL Style Spec (v8) JSON parser for ecosystem compatibility.
  - New `src/style` module with types, parser, converters, expressions, and sprite loading.
  - Supported layer types: `fill`, `line`, `symbol`, `background`, `circle`.
  - Paint properties: `fill-color`, `fill-opacity`, `line-color`, `line-width`, `text-color`, `text-halo-*`.
  - Filter expressions: `==`, `!=`, `all`, `any`, `has`, `!`, `in`.
  - CSS color parsing: hex (`#RGB`, `#RRGGBB`), `rgb()`, `rgba()`, `hsl()`, named colors.
  - **Data-driven expression evaluation**: `interpolate`, `step`, `match`, `case`, `coalesce`.
  - **Math/logic operators**: `+`, `-`, `*`, `/`, `%`, `^`, `sqrt`, comparison, boolean logic.
  - **Sprite atlas loading**: Load sprite.json metadata and atlas image dimensions.
  - **Glyph/font support**: PBF glyph range management for Mapbox fonts.
  - Python API: `load_style()`, `apply_style()`, `parse_color()`, `evaluate_color_expr()`, `evaluate_number_expr()`.
  - Visual diff tests for data-driven styling verification.
  - Interactive viewer example: `examples/style_viewer_interactive.py`.

## [1.10.1] - OIT Transparency & Shadow Quality

### Added
- **Order-Independent Transparency (OIT)**
  - Implemented Weighted Blended OIT (WBOIT) for correct rendering of overlapping transparent surfaces (e.g., vector overlays).
  - Added `enable_oit()` and `disable_oit()` methods to `Scene` and `Viewer`.
  - Dual-source blending pipeline integration for `VectorOverlayStack`.
- **Shadow Quality Improvements**
  - Added compute-shader based separable Gaussian blur pass for VSM/EVSM/MSM moment maps.
  - Improved shadow softness and reduced artifacts in variance shadow maps.

### Fixed
- **Terrain Rendering**
  - Fixed "edge cliff" artifacts by correcting UV clamping in `terrain.wgsl`.
  - Added deterministic analytic height fallback for sentinel tiles to prevent flat shading anomalies.

## [1.10.0] - 3D Tiles & Point Cloud Platform

### Added
- **Phase 5 / P5 — 3D Geospatial Platform (3D Tiles & Point Clouds)**
  - **3D Tiles Support** (OGC Standard)
    - Native `tileset.json` parser with hierarchical traversal support.
    - B3DM (Batched 3D Model) payload decoder with glTF 2.0 integration.
    - PNTS (Point Cloud) payload decoder with feature table parsing.
    - Screen-Space Error (SSE) based LOD selection (`compute_sse`, `should_refine`).
    - Hierarchical culling and traversal iterator (`TilesetTraverser`).
  - **Point Cloud System** (Massive Dataset Support)
    - COPC (Cloud Optimized Point Cloud) reader with LAZ 1.4 decompression.
    - EPT (Entwine Point Tile) schema support for octree-based streaming.
    - Out-of-core octree traversal with frustum culling and LOD management.
    - Quantitative attribute extraction (intensity, classification, color).
  - **Python API Bindings**
    - `Tiles3dRenderer` and `PointCloudRenderer` for high-performance drawing.
    - `Tileset` and `CopcDataset` classes for asset management.
    - Comprehensive test suite coverage: `test_3dtiles_parse.py`, `test_copc_parse.py`.

### Documentation
- Added Phase 5 implementation plan details to `docs/roadmap/roadmap.md`.
- Updated API reference with new geospatial modules.

## [1.9.9] - Creator Workflow & Map Plate Compositor

### Added
- **Phase 4 / P4.1–P4.3 — Map Plate Compositor (Creator Workflow)**
  - P4.1: `MapPlate` class with configurable region layout for professional cartographic output.
  - Map plate region system supporting main map area, legend, scale bar, title, and attribution zones.
  - Flexible layout engine with percentage-based or absolute positioning for each plate region.
  - P4.2: Auto-generated legends from colormap with customizable styling and positioning.
  - Scale bar generation with automatic unit selection (meters, kilometers) based on map extent.
  - North arrow integration with configurable style and placement within plate layout.
  - P4.3: Export pipeline supporting PNG and PDF output formats with embedded layout metadata.
  - High-resolution export capability (up to 16K) maintaining plate region proportions.
  - Batch export workflows for generating multiple map variants from single configuration.
  - Title and attribution text rendering with configurable fonts, sizes, and alignment.
  - Interactive viewer integration via `map_plate` command for immediate visual feedback.
  - Template system for reusable map plate configurations across different datasets.
  - Professional-grade output suitable for publication, presentation, and print media.
  - Python API: `render_map_plate(terrain, layout_config, output_path)` with full customization.
  - Comprehensive test suite: `test_map_plate_layout.py`, `test_legend_generation.py`, `test_plate_export.py`.

### Documentation
- Added map plate compositor guide with layout examples and best practices.
- Updated creator workflow documentation with template library and styling reference.

## [1.9.8] - Cloud-Native Data & COG Streaming

### Added
- **Phase 3 / M6 — COG Streaming (Cloud-Optimized GeoTIFF)**
  - P3.1: HTTP range request adapter enabling efficient remote tile access without full file download.
  - P3.2: IFD (Image File Directory) parsing with automatic overview selection for optimal LOD streaming.
  - P3.3: LRU (Least Recently Used) cache with configurable memory budget preventing unbounded growth.
  - Native COG decoder with `reqwest`-based HTTP client for range reads (bytes=start-end headers).
  - Rasterio fallback path for comprehensive format support when native decoder unavailable.
  - Speculative prefetch strategies reducing latency on sequential tile access patterns.
  - Aggressive caching layer with hit/miss statistics exposed for monitoring and tuning.
  - Deterministic tile decode ensuring reproducible results across cache states.
  - Memory budget enforcement preventing OOM on large-scale streaming operations.
  - Overview pyramid integration selecting appropriate resolution based on viewport and LOD requirements.
  - Python CLI demo: `cog_streaming_demo.py` demonstrating remote DEM streaming without local copy.
  - Comprehensive test suite: `test_cog_range_read.py`, `test_cog_overviews.py`, `test_cog_cache_eviction.py`.

### Performance
- Streamed datasets render without pre-tiling or full file download, enabling continent-scale terrain from cloud storage.
- Range request optimization reduces bandwidth consumption by fetching only visible tiles.
- LRU cache provides sub-millisecond tile retrieval on cache hits.

### Documentation
- Added COG streaming architecture documentation with cache behavior and prefetch strategies.
- Updated data ingestion pipeline guide with cloud-native workflow examples.

## [1.9.7] - Terrain Scale & Clipmap Structure

### Added
- **Phase 2 / P2.1 / M5 — Clipmap Structure (Terrain Scalability)**
  - Nested-ring clipmap mesh system replacing single-grid terrain draw for true large-scale rendering.
  - Four-ring clipmap configuration with configurable resolution per ring (default 64x64 vertices).
  - Automatic LOD selection based on distance from camera with seamless transitions between rings.
  - Vertex morphing weights for smooth LOD blending preventing popping artifacts at ring boundaries.
  - Skirt geometry support for crack-free terrain edges and horizon closure.
  - Triangle budget optimization achieving 99.9% reduction versus full-resolution DEM (10,240 triangles vs 13M+).
  - Meets P2.1 exit criteria: ≥40% triangle reduction at distance (actual: 92.2% internal reduction).
  - Integration with existing height mosaic and page table for streaming LOD tile requests.
  - UV coordinate mapping preserving texture sampling across all clipmap rings.
  - Center block with high-detail geometry surrounded by progressively coarser rings.
  - Configurable clipmap extent, ring count, and morph range for quality/performance tuning.
  - Python CLI demo: `clipmap_demo.py` with detailed mesh statistics and LOD verification.
  - Comprehensive test coverage: `test_clipmap_structure.py` validating geometry, LOD reduction, and seam correctness.

### Performance
- Terrain triangle count reduced from 13.5M to 10K triangles (99.9% reduction) while maintaining visual quality.
- Stable per-frame triangle budget regardless of terrain size, enabling continental-scale rendering.

### Documentation
- Added clipmap configuration reference with ring layout diagrams.
- Updated terrain rendering pipeline documentation with LOD selection and morphing details.

## [1.9.6] - TAA Foundation & Motion Vectors

### Added
- **Phase 1 — TAA Foundation (Temporal Anti-Aliasing)**
  - Motion vectors / velocity buffer system enabling temporal reprojection for camera and object motion.
  - GBuffer velocity channel with shader output capturing per-pixel motion information.
  - Halton 2,3 sequence jitter for sub-pixel sampling and temporal convergence.
  - Projection matrix jitter integration for TAA sample distribution.
  - TAA resolve pass with history color buffer reprojection using depth and camera matrices.
  - Neighborhood clamping for history rejection preventing ghosting artifacts.
  - Reactive mask support for overlays and water to handle transparency correctly.
  - YCoCg color space conversion for improved temporal clamping and reduced color bleeding.
  - Demonstrable shimmer reduction in thin-feature scenes (power lines, fences, foliage edges).
  - CLI integration via `--taa` flag with preset support for easy configuration.
  - Variance reduction metrics vs no-TAA baseline showing measurable quality improvement.
  - Python tests: `test_motion_vectors.py`, `test_jitter_sequence.py`, `test_taa_convergence.py`, `test_taa_toggle.py`.

### Fixed
- Camera motion ghosting eliminated through proper motion vector generation.
- Temporal aliasing on thin geometry reduced via sub-pixel jittering and accumulation.

### Documentation
- Added TAA usage examples with quality/performance trade-offs.
- Updated rendering pipeline documentation with motion vector and temporal resolve integration.

## [1.9.5] - Shadow System Productization

### Added
- **P0.2 / M3 — Shadow Filtering Productization (VSM/EVSM/MSM)**
  - Production-ready Variance Shadow Maps (VSM), Exponential Variance Shadow Maps (EVSM), and Moment Shadow Maps (MSM) techniques fully integrated and validated.
  - Aligned Python configuration validation with Rust implementation: VSM, EVSM, and MSM now properly accepted in `ShadowSettings`.
  - Implemented separable Gaussian blur pass for moment-based shadow maps with configurable kernel size.
  - Added light bleeding reduction controls: EVSM positive/negative exponents, moment bias parameters, and memory budget enforcement.
  - CLI shadow technique selection via `--shadow-technique` flag supporting all seven methods: Hard, PCF, PCSS, VSM, EVSM, MSM, CSM.
  - Forced-edge regression test suite with per-technique validation and penumbra/leak metrics.
  - Shadow technique A/B comparison renders demonstrating quality differences across all filtering methods.
  - Memory budget tracking ensures moment-based shadow maps respect ≤512 MiB host-visible memory constraint.
  - Integration tests for each shadow technique with non-trivial numeric diff validation.

### Fixed
- Python config validation no longer incorrectly rejects VSM, EVSM, and MSM shadow techniques.
- Moment blur path properly integrated into shadow pipeline with depth-aware edge preservation.

### Documentation
- Added shadow technique comparison documentation with quality/performance characteristics.
- Updated rendering options guide with complete shadow filtering reference.

## [1.9.4] - Sun Ephemeris & Time-of-Day Controls

### Added
- **P0.3 / M2 — Sun Ephemeris + Time-of-Day Controls**
  - Deterministic sun position calculation from geographic coordinates and UTC datetime.
  - Python API: `sun_position(lat, lon, datetime)` returns accurate azimuth and elevation angles.
  - Viewer/preset configuration keys: `--sun-lat`, `--sun-lon`, `--sun-datetime` for precise solar positioning.
  - Automatic sun angle computation for realistic lighting at any location and time.
  - Validation against NOAA solar calculator reference data for accuracy.
  - Measurable shadow direction changes based on temporal and geographic parameters.
  - Interactive viewer integration with ephemeris-driven lighting controls.

### Documentation
- Added sun ephemeris usage examples with coordinate and datetime inputs.
- Updated viewer documentation with time-of-day lighting configuration.

## [1.9.3] - Transparency & OIT Productization

### Added
- **P0.1 — Order-Independent Transparency (OIT) Productization**
  - Unified transparency strategy selection for the main rendering pipeline.
  - Three OIT modes: Standard alpha blending, Weighted Blended OIT (WBOIT), and Dual-Source blending with automatic fallback.
  - Dual-source OIT renderer with hardware detection and WBOIT fallback (`src/core/dual_source_oit.rs`).
  - Integration of vector overlays and multi-layer transparent draw calls through unified OIT compositor.
  - Python API: `enable_oit()` method for programmatic transparency mode control.
  - Transparency modes exposed via `DualSourceOITMode` enum (Disabled, DualSource, WBOITFallback, Automatic).
  - Backend-specific hardware detection for dual-source blending support across Vulkan, Metal, and DX12.
  - Improved rendering quality for overlapping transparent surfaces (water, volumetrics, vector overlays).

### Documentation
- Added transparency/OIT usage examples and API reference.
- Updated rendering pipeline documentation with OIT integration details.

## [1.9.2] - Camera Animation System

### Added
- **Feature C: Camera Path + Keyframe Animation**
  - Keyframe-based camera animation system with cubic Hermite interpolation.
  - Interactive preview in the viewer and offline rendering to PNG sequences.
  - Direct MP4 video export via ffmpeg integration.
  - Dynamic sun lighting that follows the camera for dramatic effect.
  - `CameraAnimation` Python API for programmatic path creation.
  - Three pre-built animation types: `orbit`, `flyover`, `sunrise`.
  - Example: `examples/camera_animation_demo.py`.

## [1.9.1] - Snapshot Refactoring & Cleanup

### Changed
- Refactored snapshot rendering to separate it from window rendering.
- Removed HUD overlay to simplify viewer.
- Removed unused large data file `glad_cropland_2019.tif`.
- Updated `fuji_labels.png` asset.

## [1.9.0] - Terrain Labels & Fuji Demo

### Added
- **Terrain Labels System** — Support for screen-space text labels anchored to world coordinates.
  - Rendered via a font atlas (default: `assets/fonts/default_atlas.png`).
  - Screen-space rendering with depth occlusion support.
  - IPC commands: `add_label`, `clear_labels`, `set_labels_enabled`, `load_label_atlas`.
- **Mount Fuji Labels Demo** (`examples/fuji_labels_demo.py`)
  - Demonstrates loading OSM place names from a GeoPackage and overlaying them on a 30m DEM of Mount Fuji.
  - Features high-quality PBR rendering presets, tonemapping, and depth-of-field integration.
  - Usage: `python examples/fuji_labels_demo.py --preset high_quality`



## [1.8.0] - Draped Terrain Overlays

### Added
- **Draped Terrain Overlays (Option A)** — Full implementation of terrain overlay system with lit/shadowed texture overlays
  - Overlay textures sampled in terrain UV space and blended into albedo before lighting
  - Overlays receive full PBR lighting: sun diffuse, shadows, ambient occlusion
  - Multiple overlay layers with deterministic stacking order (z_order)
  - Three blend modes: Normal (alpha blend), Multiply (darken), Overlay (Photoshop-style contrast)
  - Per-layer opacity, visibility, and extent controls
  - Global overlay opacity multiplier for quick adjustments
  - Overlay system disabled by default (existing behavior preserved)
  - GPU resources: Per-layer textures + composite texture (Rgba8UnormSrgb)
  - Memory budget: ~80 MB for 4 layers at 2k resolution (well within 512 MiB limit)
  - **Rust API**: `add_overlay_raster()`, `add_overlay_image()`, `remove_overlay()`, `set_overlay_visible()`, `set_overlay_opacity()`, `list_overlays()`, `overlay_count()`, `set_global_overlay_opacity()`, `set_overlays_enabled()`
  - **IPC Commands**: `load_overlay`, `remove_overlay`, `set_overlay_visible`, `set_overlay_opacity`, `set_global_overlay_opacity`, `set_overlays_enabled`, `list_overlays`
  - **Python API**: `OverlaySettings`, `OverlayLayerConfig`, `OverlayBlendMode` in `terrain_params.py`

- **Swiss Terrain Land Cover Viewer Example** (`examples/swiss_terrain_landcover_viewer.py`)
  - Interactive 3D viewer for Switzerland DEM with land cover classification overlay
  - Land cover resampled to match DEM resolution via rasterio/GDAL
  - EPSG:3035 (LAEA Europe) projection support
  - Four high-quality snapshot presets (hq1-hq4) with varying effects
  - Automatic legend generation with transparent background
  - Demonstrates draped overlay integration with full PBR lighting

### Documentation
- Added `docs/terrain_overlays.rst` — User guide for overlay system
- Updated `docs/plan_overlays_option_a_draped_textures.md` — Full implementation plan

### Fixed
- Removed unused import warning for `ViewerTerrainPbrConfig` in terrain module
- Suppressed dead code warning for `BloomCompositeUniforms` (reserved for future bloom composite pass)

## [1.7.0] - Interactive Viewer Post-Processing Pipeline

### Added
- **P5 — Volumetric Fog + Light Shafts**
  - Height-based fog with exponential density falloff and configurable absorption/scattering
  - God rays (volumetric shadows) via shadow map sampling during ray march
  - Henyey-Greenstein phase function for realistic single-scattering
  - Half-resolution rendering option with bilateral depth-aware upsampling
  - CLI: `--volumetrics --vol-mode height --vol-density 0.02 --vol-light-shafts --vol-shaft-intensity 2.0`

- **P4 — Motion Blur (Temporal Accumulation)**
  - Camera motion blur via multi-frame temporal accumulation across shutter interval
  - Configurable sample count (8-32), shutter angle, and camera delta parameters
  - Accumulation buffer (Rgba32Float) with final resolve pass
  - CLI: `--motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-phi-delta 5`

- **P3 — Depth of Field (Separable Blur)**
  - Circle of Confusion (CoC) calculation with configurable f-stop, focus distance, focal length
  - Two-pass separable Gaussian blur weighted by CoC
  - Tilt-shift support via tilted focal plane (pitch/yaw parameters)
  - Quality presets (low=4, medium=8, high=16 samples)
  - CLI: `--dof --dof-f-stop 2.8 --dof-focus-distance 300 --dof-quality high`

- **P2 — Barrel Distortion + Chromatic Aberration**
  - Brown-Conrady barrel/pincushion distortion model
  - Radial RGB split for chromatic aberration effect
  - Edge clamping to prevent sampling outside texture bounds
  - CLI: `--lens-effects --lens-distortion 0.15 --lens-ca 0.03 --lens-vignette 0.3`

- **P1 — Full-Screen Post-Process Infrastructure**
  - Reusable post-process pass with ping-pong textures for UV-remapping effects
  - Full-screen triangle vertex shader with screen-space uniforms
  - Integration point in ViewerTerrainScene::render() after main pass
  - Bit-exact passthrough when all effects disabled

### Documentation
- Extended `docs/pbm_pom_viewer.md` with comprehensive P1-P5 post-processing examples
- Added combined scene examples demonstrating full post-processing stack

## [1.6.0] - P4 Motion Blur

### Added
- Temporal accumulation motion blur for camera movement
- Multi-frame render loop with camera interpolation across shutter interval
- Configurable shutter angle and sample count

## [1.5.0] - P3 Depth of Field

### Added
- Separable blur DoF with Circle of Confusion calculation
- Tilt-shift focal plane support for miniature effects
- Quality presets for blur kernel radius

## [1.4.0] - P2 Lens Effects

### Added
- Barrel distortion and chromatic aberration post-process effects
- Vignette integration with lens effects pass

## [1.3.0] - P1 Post-Process Infrastructure

### Added
- Full-screen post-process pass infrastructure
- Ping-pong texture management for multi-pass effects
- Screen-space uniform buffer for lens parameters

## [1.2.0] - QA, Memory Budget Enforcement, and IBL Refresh

- **P4 - IBL pipeline refresh**
  - Compute-driven equirectangular -> cubemap conversion plus irradiance/specular prefiltering and BRDF LUT generation (WGSL: `ibl_equirect.wgsl`, `ibl_prefilter.wgsl`, `ibl_brdf.wgsl`)
  - On-disk `.iblcache` reuse keyed by HDR + resolution with new CLI flags `--ibl-res` and `--ibl-cache`
  - Runtime split-sum shader path now sources bind group @group(2) with shared sampler and BRDF uniform controls

- **P9 — QA: Golden Images, Unit Tests, CI Matrix**
  - Implemented comprehensive QA infrastructure for cross-platform correctness and visual regression testing
  - **Shader parameter packing tests** (`tests/test_shader_params_p5_p8.rs`):
    - GPU alignment validation for all P5-P8 types (SSAOSettings, SSGISettings, SSRSettings, SkySettings, VolumetricSettings)
    - WGSL uniform buffer layout verification (16-byte alignment, size validation)
    - Bytemuck Pod/Zeroable trait tests for safe GPU transmission
    - Parameter range validation (roughness [0.04, 1.0], turbidity [1.0, 10.0], etc.)
    - 12 comprehensive test cases covering layout, alignment, defaults, and edge cases
  - **Golden image regression testing** (`tests/golden_images.rs`):
    - 12 reference images at 1280×920 resolution covering BRDF × shadow × GI combinations
    - SSIM (Structural Similarity Index) comparison with epsilon ≥ 0.98 threshold
    - Configurations: Lambert, Phong, GGX, Disney, Oren-Nayar, Toon, Ashikhmin, Ward, Blinn-Phong
    - Shadow techniques: Hard, PCF, PCSS, VSM, EVSM, MSM, CSM
    - GI modes: None, IBL, SSAO, GTAO, SSGI, SSR
    - Diff image generation and artifact storage on failure
    - Golden image generator script (`scripts/generate_golden_images.py`)
  - **CI/CD matrix** (`.github/workflows/ci.yml`):
    - Cross-platform testing: Windows (win_amd64), Linux (linux_x86_64), macOS (macos_universal2)
    - Rust test suite with cargo check, cargo test, cargo clippy
    - Python wheel builds with maturin for all platforms (abi3 support)
    - Python tests across 3 versions (3.9, 3.11, 3.12)
    - Example sanity tests: Render 640×360 frames within 90s timeout
    - Golden image validation with artifact upload on failure
    - Shader parameter packing validation on all platforms
    - Documentation build (Rust docs + Sphinx)
    - Artifact preservation (wheels, docs, golden image diffs)
  - **Troubleshooting documentation** (`docs/troubleshooting_visuals.rst`):
    - Comprehensive visual debugging guide organized by subsystem
    - Checklists for: Lighting & Shadows, BRDF & Materials, Global Illumination, Atmospherics & Sky, Screen-Space Effects
    - Performance & memory debugging (OOM, slow rendering)
    - Platform-specific issues (Windows/MSVC, Linux/GNU, macOS/Clang)
    - Shader compilation errors (WGSL validation, SPIR-V translation)
    - Golden image test failures and CI debugging
    - Quick reference with code examples for common fixes
  - **CI job structure**:
    - `test-rust`: Cargo tests on all platforms
    - `build-wheels`: Maturin wheel builds for win/linux/macos
    - `test-python`: Pytest across Python versions
    - `test-golden-images`: Visual regression with SSIM validation
    - `test-examples`: Sanity checks for all example scripts
    - `test-shader-params`: GPU alignment validation
    - `build-docs`: Rust + Sphinx documentation builds
    - `ci-success`: Final gating job for PR merges
  - Test coverage: 12 golden images, 12 shader param tests, 10+ example sanity tests
  - Acceptance: CI green on all platforms, golden image diffs stored as artifacts on failure
- **P8 — Performance & Memory Budget Enforcement (≤512 MiB host-visible)**
  - Implemented GPU memory budget tracking and auto-downscaling to enforce 512 MiB host-visible memory constraint
  - **Memory budget infrastructure** (`src/render/memory_budget.rs`):
    - `GpuMemoryBudget`: Thread-safe atomic tracking of GPU allocations by category
    - Categories: VertexBuffers, IndexBuffers, UniformBuffers, Textures (RGBA8, RGBA16F, R32F, Depth), ShadowMaps, IBL Cubemaps, BRDF LUT, Froxel Grid, Screen-space Effects
    - Budget enforcement: Warn at 90% utilization, track peak allocation, provide detailed breakdown
    - Auto-downscaling functions for resource quality when budget exceeded
  - **Auto-downscaling strategies**:
    - `auto_downscale_shadow_map()`: Progressive resolution halving (4096→2048→1024→512 minimum)
    - `auto_downscale_ibl_cubemap()`: Cubemap resolution downscale with mip chain accounting (512→256→128→32 minimum)
    - `auto_downscale_froxel_grid()`: Froxel dimension reduction prioritizing Z-axis (depth) first
    - Log warnings when downscaling occurs with original→final resolution
  - **Rendering statistics API** (`Scene.get_stats()`):
    - Returns Python dict with GPU memory usage (current, peak, budget, utilization %)
    - Lists enabled rendering passes (terrain, ssao, reflections, dof, clouds, IBL, etc.)
    - Frame time tracking (placeholder: 0.0 ms, ready for future timing integration)
    - Memory estimations based on texture dimensions and buffer sizes
  - **RenderStats struct**: Structured statistics with memory breakdown by category
  - **Thread-safe atomic counters**: All memory tracking uses `Arc<AtomicUsize>` for concurrent safety
  - **Validation via unit tests**: Budget overflow, auto-downscaling, utilization calculation tests
  - CLI-ready: `--gpu-budget-mib 512` parameter support (default: 512 MiB)
  - Console logging: Budget decisions logged at INFO level, warnings at WARN level
  - Acceptance criteria: Renderer respects ≤512 MiB, stats accessible via `scene.get_stats()`, auto-downscaling prevents OOM
- **P7 — Python UX Polish: High-level Presets & Validation + Examples**
  - Implemented high-level rendering presets for common scenarios while preserving low-level control
  - **forge3d.presets module** with production-ready configurations:
    - `studio_pbr()`: Indoor studio lighting (directional + IBL + PCF shadows + Disney Principled BRDF)
    - `outdoor_sun()`: Outdoor scenes (Hosek-Wilkie sky + sun + CSM shadows + Cook-Torrance GGX)
    - `toon_viz()`: Stylized NPR rendering (Toon BRDF + hard shadows + no GI)
    - `minimal()`: Fast previews (Lambert + single light + no shadows)
    - `high_quality()`: Final renders (PCSS soft shadows + GTAO + SSR + IBL)
  - **Preset features**:
    - Configuration dictionaries ready for `Renderer(**config)`
    - Sensible defaults with parameter overrides support
    - Factory pattern with keyword arguments
    - Programmatic access via `get_preset(name, **kwargs)`
  - **Example gallery scripts** for visual comparison:
    - `examples/lighting_gallery.py`: Grid render comparing 12 BRDF models (Lambert, Phong, Oren-Nayar, GGX, Disney, Ashikhmin, Ward, Toon, Minnaert)
    - `examples/shadow_gallery.py`: Side-by-side comparison of 7 shadow techniques (Hard, PCF, PCSS, VSM, EVSM, MSM, CSM) with quality/performance table
    - `examples/ibl_gallery.py`: HDR environment rotation sweep, roughness sweep (0-1), metallic vs. dielectric comparison
  - **CLI integration ready**: Examples demonstrate command-line usage patterns
  - All galleries support configurable output resolution, grid layout, and tile sizes
  - Comprehensive docstrings with usage examples and parameter documentation
  - Acceptance: Users can reproduce figures with single command: `python examples/terrain_demo.py --preset outdoor_sun --brdf cooktorrance-ggx --shadows csm`
- **P6 — Atmospherics & Sky (Hosek-Wilkie/Preetham, Volumetric Fog/God-rays)**
  - Implemented physical sky models and volumetric fog with single-scattering for realistic atmospheric rendering
  - Physical sky models: Hosek-Wilkie (2012) and Preetham (1999) analytic atmospheric scattering
    - Hosek-Wilkie: More accurate sky model based on measured atmospheric data with turbidity-based coefficients
    - Preetham: Classic Perez function-based sky with turbidity parameter
    - Turbidity range [1.0-10.0]: 2.0=clear sky, 6.0=hazy, 10.0=very hazy
    - Ground albedo [0-1] for realistic ground bounce lighting influence
    - Sun disk rendering with limb darkening and corona/glow
    - Sunrise/sunset color shifts with proper solar elevation angle calculation
    - Exposure control and simple tonemapping (Reinhard)
  - Volumetric fog with Henyey-Greenstein phase function for single-scattering
    - Exponential height fog: Denser near ground with configurable falloff
    - Henyey-Greenstein phase: Configurable asymmetry parameter g [-1 to 1] for forward/backward scattering
    - View-ray marching with jittered sampling (16-128 steps, configurable)
    - Beer-Lambert extinction law for physically-based transmittance
    - Temporal reprojection with configurable alpha [0-0.9] for stable, noise-free results
  - God-rays (volumetric shadows) from shadow maps
    - Shadow map sampling during ray march for volumetric occlusion
    - PCF shadow filtering for smooth god-ray beams
    - Directional in-scattering from sun with phase function
    - Ambient sky contribution for omnidirectional scatter
    - Beams visible when sun occluded by terrain/geometry
  - Alternative froxelized volumetric approach (16×8×64 grid) for performance
    - Precomputed froxel grid with scattering and extinction
    - Fast lookup during final render pass
    - Memory budget: ~64 KiB for froxel grid (well under 512 MiB)
  - WGSL shaders: `sky.wgsl` (compute + fragment variants), `volumetric.wgsl` (view-ray + froxel approaches)
  - Rust types: `SkySettings` (48 bytes), `VolumetricSettings` (80 bytes), `AtmosphericsSettings` (all GPU-aligned)
  - Enums: `SkyModel` (Off, Preetham, HosekWilkie), `VolumetricPhase` (Isotropic, HenyeyGreenstein)
  - Python bindings: `SkySettings`, `VolumetricSettings` classes with factory methods and validation
  - Factory methods: `SkySettings.hosek_wilkie()`, `SkySettings.preetham()`, `VolumetricSettings.with_god_rays()`, `VolumetricSettings.uniform_fog()`
  - Sun direction synced from directional light tagged as "sun"
  - CLI-ready: `--sky hosek-wilkie --turbidity 2.5 --ground-albedo 0.2`, `--volumetric 'density=0.015,phase=hg,g=0.7,max_steps=48'`
  - Acceptance criteria: Sunrise/sunset color shift visible, god-ray beams when sun occluded, temporally stable fog
- **P5 — Screen-space Effects (SSAO/GTAO, SSGI, SSR)**
  - Implemented comprehensive screen-space rendering effects as optional toggleable passes
  - GBuffer pass: Depth, view-space normals, material properties (albedo, roughness, metallic) for screen-space techniques
  - SSAO/GTAO: Two ambient occlusion techniques with bilateral blur and optional temporal filtering
    - SSAO: Hemisphere sampling with spiral pattern (16-64 samples/pixel)
    - GTAO: Horizon-based ground-truth AO for higher accuracy
    - Bilateral blur: Edge-preserving denoising using depth and normal weights
    - Configurable radius, intensity, bias, and temporal accumulation (0-0.95 alpha)
  - SSGI: Screen-space global illumination with half-res ray marching and IBL fallback
    - Ray marching in depth buffer for indirect diffuse lighting (16-32 steps/ray)
    - Cosine-weighted hemisphere sampling with low-discrepancy noise
    - IBL fallback for ray misses with configurable contribution (0-1)
    - Half-resolution mode with bilateral upsampling for performance
    - Temporal accumulation for noise reduction (0-0.9 alpha)
  - SSR: Screen-space reflections with hierarchical ray marching and environment fallback
    - Adaptive hierarchical ray marching with binary search refinement (32-64 steps)
    - Thickness-based hit detection with configurable tolerance
    - Roughness-based fade and screen edge fade for artifact reduction
    - Environment map fallback for off-screen/missed rays with mip-based roughness
    - Fresnel-based reflection intensity with metallic/dielectric support
    - Temporal accumulation for stable reflections (0-0.85 alpha)
  - WGSL shaders: `gbuffer.wgsl`, `ssao_gtao.wgsl`, `ssgi.wgsl`, `ssr.wgsl`
  - Depth reconstruction utilities: Fast view-space position reconstruction from linear depth
  - Rust types: `SSAOSettings`, `SSGISettings`, `SSRSettings`, `ScreenSpaceSettings` (all 32-byte GPU-aligned)
  - Python bindings: `SSAOSettings`, `SSGISettings`, `SSRSettings` classes with validation
  - Factory methods: `SSAOSettings.ssao()`, `SSAOSettings.gtao()` for quick configuration
  - CLI-ready: Designed for `--gi ssao`, `--gi ssgi`, `--gi ssr` command-line options
  - Acceptance criteria: AO visibly darkens creases, SSGI adds diffuse bounce on walls, SSR reflects sky & bright objects
- **P4 — IBL Pipeline (Diffuse Irradiance + Specular Prefilter + BRDF LUT)**
  - Implemented complete Image-Based Lighting pipeline for high-quality environment lighting
  - Compute shaders for offline/first-frame precomputation: irradiance convolution, GGX specular prefilter, BRDF 2D LUT
  - Irradiance convolution: Lambertian diffuse sampling with hemisphere integration (512-2048 samples/pixel)
  - Specular prefilter: GGX importance sampling with Hammersley low-discrepancy sequence (1024 samples/pixel)
  - BRDF LUT: Split-sum approximation integration (NdotV × roughness) with geometry term
  - Split-sum evaluation shader: `eval_ibl()` combining prefiltered specular, diffuse irradiance, and BRDF LUT
  - Energy-conserving IBL with proper Fresnel and metallic/dielectric blending
  - Disk cache support (`.iblcache`) keyed by HDR path, resolution, and GGX settings
  - `IblResourceCache` manages BRDF LUT (512×512 RG16F), irradiance cube (32×32 RGBA16F), specular cube (128×128 RGBA16F, 6 mips)
  - Memory budget: ~3 MiB total (well under 64 MiB P0 budget)
  - WGSL modules: `ibl/irradiance_convolution.wgsl`, `ibl/specular_prefilter.wgsl`, `ibl/brdf_lut.wgsl`, `ibl/eval_ibl.wgsl`
  - Hammersley sequence and radical inverse for low-discrepancy sampling
  - Tonemapping helpers: Reinhard and ACES filmic
- **P3 — Shadow System (Hard, PCF, PCSS, VSM/EVSM/MSM, CSM)**
  - Implemented comprehensive pluggable shadow system with 7 techniques
  - Shadow techniques: Hard (single sample), PCF (Poisson disk), PCSS (soft shadows with blocker search), VSM (variance), EVSM (exponential variance), MSM (4-moment), CSM (cascaded)
  - Complete WGSL implementation: `shadows_p3.wgsl` with all techniques and unified dispatcher
  - Extended `ShadowTechnique` enum with P3 variants and helper methods (`name()`, `from_name()`, `requires_moments()`, `channels()`)
  - PCSS: Blocker search + penumbra estimation + adaptive PCF filtering
  - VSM/EVSM/MSM: Moment-based shadow maps with light leak reduction and Chebyshev's inequality
  - CSM: Cascade selection with smooth transitions and per-cascade stabilization support
  - Python bindings: `ShadowSettings` supports all 7 techniques via string names
  - GPU-aligned `ShadowParamsGPU` struct (64 bytes) with all technique parameters
  - Poisson disk sampling (16 and 32-sample sets) for high-quality PCF/PCSS
- **P2 — BRDF Library + Material Routing**
  - Implemented comprehensive BRDF library with 10 switchable shading models
  - BRDF models: Lambert, Phong, Oren-Nayar, Cook-Torrance (GGX & Beckmann), Disney Principled, Ashikhmin-Shirley, Ward, Toon, Minnaert
  - WGSL shader modules: `brdf/common.wgsl` (geometry terms, NDFs, Fresnel), individual model shaders, and `brdf/dispatch.wgsl` (model switcher)
  - Extended `MaterialShading` struct with new parameters: sheen, clearcoat, subsurface, anisotropy (total 32 bytes, GPU-aligned)
  - Python bindings: `MaterialShading` class with factory methods (`lambert()`, `phong()`, `disney()`, `anisotropic()`)
  - Validation example: `examples/brdf_comparison.py` demonstrating all BRDF models
  - Cross-platform shader compilation verified (win_amd64, linux_x86_64, macos_universal2 compatible)
- Workstream I1 — Interactive Viewer
  - I1: Interactive windowed viewer with winit 0.29 integration providing real-time exploration
  - Orbit camera mode: rotate around target, zoom, and pan with mouse controls
  - FPS camera mode: WASD movement with mouse look, Q/E vertical movement, Shift speed boost
  - Tab to toggle between camera modes, 60+ FPS on simple scenes
  - DPI-aware rendering with live FPS counter in window title
  - Rust binary: `cargo run --bin interactive_viewer`
  - Documentation: `docs/interactive_viewer.rst`
- Renderer configuration plumbing with typed enums and validation exposed via `Renderer(config=..., **kwargs)` and `Renderer.get_config()`, including CLI overrides for lighting, BRDF, shadows, GI, and atmosphere in `examples/terrain_demo.py`.
- Light buffer SSBO with R2 sampling seeds, WGSL sampling stubs, and Python `Renderer.set_lights()` for multi-light uploads.
- WGSL BRDF library and dispatch covering Lambert, Phong, Blinn-Phong, Oren-Nayar, Cook-Torrance (GGX/Beckmann), Disney Principled, Ashikhmin-Shirley, Ward, Toon, and Minnaert, exposed via `RendererConfig.brdf_override`.
- Python memory helpers now expose real resident and staging telemetry via forge3d.mem and top-level shortcuts.
- Regression coverage for GPU adapter probes exercises native callbacks and fallback shims.

### Changed
- Wired staging rings and virtual texture residency updates into the global memory tracker so Python budgets reflect GPU usage.
- Aligned CMake, Sphinx, and ignore rules with the release metadata to keep generated docs out of the repo.
- Terrain PBR shader clamps POM steps, applies ACES tonemapping with explicit gamma, and honors global colormap strength while the renderer now outputs `Rgba8Unorm` to avoid double sRGB conversion.
- Terrain asset pipelines now auto-select material and IBL quality tiers to respect the 512 MiB budget, emitting memory usage logs and downscaling textures when devices report iGPU-class limits.
- Terrain renderer now queries adapter sample-count support, downgrades unsupported MSAA requests, and logs the effective sample count to avoid WebGPU validation errors.

### Fixed
- WI-3 GGX debug outputs match ground truth; D-only emits the raw GGX NDF and G-only applies the Schlick-Smith masking term while leaving the specular path unchanged (`src/shaders/brdf_tile.wgsl`).
- Terrain renderer honors TerrainRenderParams toggles for MSAA and parallax occlusion flags so user settings directly drive the GPU pipeline.

### Documentation
- Added ``docs/rendering_options.rst`` summarizing lighting, BRDF, shadow, GI, and volumetric options alongside the new CLI flags.
- Point auxiliary CLAUDE files at the canonical root guide to avoid drift.

## [1.1.0] - Interactive Viewer Documentation & Codebase Polish

### Added
- **Interactive Viewer Documentation** (`docs/viewer/interactive_viewer.md`)
  - Comprehensive user guide with quick start examples for terrain viewing and PBR mode.
  - Terminal command reference: `set` for multi-parameter adjustments (camera, lighting, terrain, water).
  - PBR mode commands (`pbr on|off`, `pbr exposure=`, `pbr shadows=`).
  - High-resolution snapshot support up to 16K (16384×16384, 270 megapixels).
  - IPC protocol documentation with JSON/TCP examples for programmatic control.
  - Platform support matrix (macOS/Metal, Linux/Vulkan, Windows/DX12).

### Changed
- Applied rustfmt code formatting across 50+ source files for consistent style.
- Added 45-file viewer refactoring plan targeting ≤300 LOC per module.
- Enhanced AGENTS.md with 40 AI evidence and stop-condition rules for quality assurance.

## [1.0.0] - PBR Terrain Viewer Phase 6: Documentation

### Added
- **Phase 6 — Documentation**
  - Complete user guide (`docs/pbm_pom_viewer.md`) covering CLI options, interactive commands, API reference, and troubleshooting.
  - Files modified table documenting all backend (Rust) and frontend (Python) changes.
  - Known limitations section identifying future work: CSM shadows, IBL cubemaps, POM displacement.

## [0.99.0] - PBR Terrain Viewer Phase 5: End-to-End Testing

### Added
- **Phase 5 — End-to-End Testing**
  - Integration test suite (`tests/test_terrain_viewer_pbr.py`) with 5 test cases validating legacy/PBR modes.
  - Test coverage: legacy mode renders, PBR mode enables, PBR produces different output, exposure affects output, PBR can be disabled.
  - Pytest-compatible test harness with viewer binary detection.

## [0.98.0] - PBR Terrain Viewer Phase 4: PBR Shader & Pipeline

### Added
- **Phase 4 — PBR Shader + Pipeline Integration**
  - PBR WGSL shader (`src/viewer/terrain/shader_pbr.rs`) with Blinn-Phong specular, soft self-shadows, and sky-gradient ambient.
  - Height-based material zones (vegetation → rock → snow), slope-based rock exposure, roughness variation.
  - ACES tonemapping, gamma correction, and configurable exposure post-processing.
  - Dual render path in `src/viewer/terrain/render.rs` for legacy/PBR switching.

## [0.97.0] - PBR Terrain Viewer Phase 3: Python CLI & IPC

### Added
- **Phase 3 — Python CLI Args + IPC Commands**
  - CLI options in `examples/terrain_viewer_interactive.py`: `--pbr`, `--exposure`, `--normal-strength`, `--ibl-intensity`, `--shadows`.
  - Interactive console commands: `pbr on|off`, `pbr exposure=<float>`, `pbr normal=<float>`, `pbr status`.
  - IPC command routing from Python to Rust viewer process.

## [0.96.0] - PBR Terrain Viewer Phase 2: Config Struct

### Added
- **Phase 2 — ViewerTerrainPbrConfig Struct**
  - `ViewerTerrainPbrConfig` Rust struct with fields: enabled, exposure, normal_strength, ibl_intensity, shadow_technique, shadow_map_res, hdr_path, msaa.
  - Default values: enabled=false (legacy mode), exposure=1.0, normal_strength=1.0, ibl_intensity=1.0.
  - PBR renderer module (`src/viewer/terrain/pbr_renderer.rs`) with config management.

## [0.95.0] - PBR Terrain Viewer Phase 1: IPC Protocol

### Added
- **Phase 1 — IPC Protocol + ViewerCmd Extension**
  - `ViewerCmd::SetTerrainPbr` enum variant in `src/viewer/viewer_enums.rs`.
  - IPC JSON command `set_terrain_pbr` with optional fields: enabled, exposure, normal_strength, ibl_intensity, shadow_technique.
  - Protocol mapping in `src/viewer/ipc/protocol.rs` and command handler in `src/viewer/cmd/handler.rs`.

## [0.94.0] - P6 Micro-Detail

### Added
- **P6 — Micro-Detail**
  - Triplanar detail normal sampling (2 m repeat) blended via RNM with distance-based fade to avoid LOD popping.
  - Procedural albedo brightness noise (±10%) in stable world-space coordinates for close-range variation.
  - Validation artifacts: `phase_p6.png`, `phase_p6_diff.png`, `p6_run.log`, `p6_result.json` with shimmer checks and fade distances.

## [0.93.0] - P5 Ambient Occlusion Enhancement

### Added
- **P5 — Ambient Occlusion Enhancement**
  - Debug mode 28 outputs the raw SSAO buffer for verification.
  - Coarse horizon AO precomputed from the heightmap, bound as an optional multiplier with default weight 0 (no-op by default).
  - Validation artifacts: `phase_p5.png`, `p5_run.log`, `p5_result.json` confirming SSAO presence and AO fallback path.

## [0.92.0] - P4 Water Planar Reflections

### Added
- **P4 — Water Planar Reflections**
  - Planar reflection render pass with mirrored camera, clip plane, and half-resolution target.
  - Shader sampling with wave-based distortion plus Fresnel mixing; shore attenuation reduces wave intensity near land.
  - Validation artifacts: `phase_p4.png`, `phase_p4_diff.png`, `p4_run.log`, `p4_result.json` logging clip plane, resolution, and wave params.

## [0.91.0] - P3 Normal Anti-Aliasing Fix

### Added
- **P3 — Normal Anti-Aliasing Fix**
  - Normal-variance mipchain generated at heightmap upload; Toksvig-style roughness adjustment applied to specular only.
  - Roughness floors lowered to 0.25 for land and 0.02 for water with clamping for stability; water branch unchanged.
  - Validation artifacts: `phase_p3.png`, `p3_run.log`, `p3_result.json` with energy histograms and roughness floor confirmation; debug modes 23–25 preserved.

## [0.90.0] - P2 Atmospheric Depth

### Added
- **P2 — Atmospheric Depth**
  - Height-based fog pipeline with uniform struct and bind group entries; applied after PBR and before tonemap.
  - CLI parameters `--fog-density`, `--fog-height-falloff`, `--fog-inscatter` (all default 0) keep baseline identical when fog is off.
  - Validation artifacts: `phase_p2.png`, `phase_p2_diff.png`, `p2_run.log`, `p2_result.json` including fog-on notes and SSIM vs P1 with fog off.

## [0.89.0] - P1 Cascaded Shadows

### Added
- **P1 — Cascaded Shadows**
  - Single source of truth for `TERRAIN_USE_SHADOWS` with real CSM resources bound in `TerrainRenderer` (group 3); defaults to one cascade if params missing.
  - Cascade splits config `[50, 200, 800, 3000]` behind a toggle plus optional PCSS light-size parameter retaining prior hard-shadow behavior by default.
  - Compile-time debug overlay for cascade boundaries and validation artifacts: `phase_p1.png`, `phase_p1_diff.png`, `p1_run.log`, `p1_result.json` with SSIM and shadow config logs.

## [0.88.0] - P7 Python UX Polish

### Added
- **P7 — Python UX Polish: High-level Presets & Validation + Examples**
  - **Presets API** (`python/forge3d/presets.py`):
    - `studio_pbr()`: Indoor studio lighting with directional key + IBL, PCF shadows, Disney Principled BRDF, HDRI sky
    - `outdoor_sun()`: Outdoor scenes with Hosek-Wilkie sky, sun directional light, CSM shadows, Cook-Torrance GGX BRDF
    - `toon_viz()`: Stylized NPR rendering with Toon BRDF, hard shadows, no GI
    - `get(name)` and `available()` helpers for case-insensitive preset resolution
    - All presets return mappings compatible with `RendererConfig.from_mapping()` for clean merging
  - **Renderer.apply_preset(name, **overrides)** method:
    - Merges preset into current `RendererConfig` instance
    - Applies user overrides after preset application
    - Raises `ValueError` for invalid preset names with helpful error messages
  - **CLI integration** in `examples/terrain_demo.py`:
    - `--preset <name>` flag to apply preset as base configuration
    - CLI overrides (--brdf, --shadows, --hdr, etc.) take precedence over preset defaults
    - Acceptance one-liner: `python examples/terrain_demo.py --preset outdoor_sun --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr`
  - **Example galleries**:
    - `examples/lighting_gallery.py`: Grid render comparing BRDF models (Lambert, Phong, Oren-Nayar, GGX, Disney, Ashikhmin, Ward, Toon, Minnaert) with configurable output
    - `examples/shadow_gallery.py`: Side-by-side comparison of shadow techniques (Hard, PCF, PCSS, VSM, EVSM, MSM, CSM) with quality/performance notes
    - `examples/ibl_gallery.py`: HDR environment rotation sweep, roughness sweep (0-1), metallic vs. dielectric comparisons
  - **Unit tests** (`tests/test_presets.py`, `tests/test_renderer_apply_preset.py`):
    - Validates all presets return valid mappings accepted by `RendererConfig.from_mapping()`
    - Verifies expected fields for each preset (sky model, shadow technique, BRDF)
    - Tests preset application and override precedence

### Documentation
- Added Sphinx how-to pages for presets (`docs/user/presets_overview.rst`, per-preset pages)
- Example gallery documentation with CLI usage and reproducibility tips
- Import-safe design ensures presets work in CPU-only environments without native dependencies

## [0.87.0] - P6 Performance & Polish

### Added
- Half-resolution volumetric fog path with bilateral, depth-aware upsampling to full resolution.
- New viewer commands for fog performance/quality tuning:
  - `:fog-half on|off` to enable/disable half-resolution fog rendering.
  - `:fog-edges on|off` to enable/disable depth-aware bilateral upsample.
  - `:fog-upsigma <float>` to tune bilateral depth sigma.
- Step-count heuristic in half-res mode to maintain quality at lower cost.

### Changed
- Integrated the half-res compute + upsample pass into the viewer render graph, preserving the single-terminal workflow and automatic snapshot path.
- Minor cleanup and uniform consistency checks around camera/fog parameters.

### Documentation
- Updated `docs/rendering_options.rst` with the new fog controls, usage, and tuning notes.

## [0.86.0] - P5 Screen-space Effects

### Added
- SSAO/GTAO, SSGI, and SSR passes integrated and toggleable from the viewer and Python API.
- Bilateral blur, temporal accumulation controls, and GI debug viz.
- Golden image generator updated and made resilient to slow-exiting viewers; automatic snapshot plumbing stabilized.

## [0.85.0] - P4 Image-Based Lighting

### Added
- End-to-end IBL pipeline: HDR import, irradiance convolution, specular prefilter, BRDF LUT.
- Cache-aware resource init with conservative memory budget selection and auto-downgrade heuristics.
- Viewer Lit viz path with `--lit-sun`/`--lit-ibl` parameters.

## [0.84.0] - P3 Shadows

### Added
- Pluggable shadow techniques: Hard, PCF, PCSS, VSM, EVSM, MSM, and CSM.
- Shadow manager, atlas allocation, and quality controls; PCSS blocker search and PCF filtering.

## [0.83.0] - P2 BRDF Library

### Added
- BRDF library and dispatcher covering Lambert, Phong, Blinn-Phong, Oren–Nayar, Cook–Torrance (GGX/Beckmann), Disney Principled, Ashikhmin–Shirley, Ward, Toon, Minnaert.
- Material uniform extensions and global override plumbed via `RendererConfig.brdf_override`.

## [0.82.0] - P1 Light System

### Added
- Typed light buffer (std430) with multiple light types and per-frame sampling seeds.
- Python `Renderer.set_lights(...)` and viewer light controls.

## [0.81.0] - P0 Config & CLI Plumbing

### Added
- Typed `RendererConfig` with enums for lights, BRDF, shadows, GI, sky/volumetrics; serde + validation.
- CLI/Viewer flag plumbing for selecting viz/gi modes; Python wrappers and docs.

## [0.80.0]

### Added
- Workstream H — Vector & Graph Layers completion and Python surface polish
  - H1/H2: Finalized point impostors with LOD and shape modes; global setters `set_point_shape_mode()` and `set_point_lod_threshold()` exposed in `python/forge3d/__init__.py`.
  - H3/H7: Anti-aliased polyline renderer stabilized with `LineCap`/`LineJoin` controls.
  - H4: Weighted blended OIT paths integrated for points/lines; compose to `Rgba8UnormSrgb`.
  - H5: Full-scene picking via `R32Uint` path; Python helper `vector_render_pick_map_py()` returns an `np.ndarray(H,W)` of IDs.
  - H12/H13: Graph renderer exposed through `python/forge3d/vector.py` including node/edge convenience APIs.
  - High-level `VectorScene` wrapper enabling batched authoring and one-call render or pick-map.

### Changed
- Python package initializer now ensures the native extension is initialized exactly once across both `forge3d` and `python.forge3d` import paths to avoid PyO3 double-init errors.
- Added `numpy` import and shipped `composite_rgba_over(...)` utility for RGBA compositing with configurable premultiplication.

### Fixed
- Robust buffer `map_async` error propagation in native OIT/pick readbacks; explicit `PyRuntimeError` mapping rather than implicit conversions.
- Closed an unbalanced brace in `src/lib.rs` OIT combined path.

### Tests
- Version assertions updated to this release.
- Full test suite remains green on reference environment: 1003 passed, 229 skipped, 2 xfailed.

### Documentation
- API surface for vector and picking helpers documented via docstrings; CHANGELOG entry for Workstream H completion.

## [0.79.0]

### Added
- Workstream G — Datashader Interop
  - G1: Datashader Adapter v1 completed with premultiplied alpha support (`premultiply=True`), optional transform passthrough for alignment, zero-copy RGBA views where possible, and API exports in `adapters/__init__.py`.
  - G2: Bench & Docs completed — `docs/user/datashader_interop.rst` updated with benchmarking playbook, usage, and plotting guidance.

### Changed
- Planar Reflections (B5) performance tuning to meet perf thresholds without changing public APIs:
  - Medium quality reflection resolution reduced from 1024 → 512.
  - Reflection render target format switched from `Rgba16Float` → `Rgba8Unorm`.
  - WGSL blur kernel sampling loops optimized to compact half-kernel extents.
- Vector module stabilized: consolidated bind group layout for points to always include atlas slots; added OIT/pick pipelines groundwork for vector primitives.

### Tests
- Version assertions updated to `0.79.0`.
- Datashader performance tests referenced via `tests/perf/test_datashader_zoom.py`.
- Full test suite remains green on reference environment: 1003 passed, 224 skipped, 2 xfailed.

### Documentation
- Expanded Datashader interop guide in `docs/user/datashader_interop.rst` with usage and benchmarking notes.

## [0.78.0]

### Added
- Workstream F — Geometry & IO (recorded from `roadmap2.csv`)
  - F1: Polygon Extrusion — GPU prism mesh generation; normals/UVs validated.
  - F2: OSM Building Footprints — import footprints and apply heights.
  - F3: Thick Polylines 3D — constant pixel/world width, joins/caps, no z-fighting.
  - F4: Import OBJ + MTL — OBJ parser and materials.
  - F5: Export to OBJ — round-trip tri/UV/normals; Blender opens.
  - F6: Export STL (3D Print) — watertight meshes; binary STL.
  - F7: MultipolygonZ → OBJ — extrude MultiPolygonZ with materials.
  - F8: Consistent Normals & Weld — weld + smooth normals; deterministic import hygiene.
  - F9: Primitive Mesh Library — planes/boxes/spheres/cylinders/cones/torus; text3D.
  - F10: UV Unwrap Helpers — planar/spherical UVs; texel-density mindful.
  - F11: Subdivision Surface — Loop subdivision with crease/boundary preservation; UV interpolation; normals recomputed.
  - F12: Displacement Modifiers — heightmap/procedural displacement; tangent/normals workflow.
  - F13: PLY Import — attributes import (pos/nrm/uv/color) with bbox/stats parity.
  - F14: Mesh Transform Ops — center/scale/flip/swap and bounds reporting.
  - F15: Mesh Info & Validate — diagnostics for duplicates/degenerates/non‑manifold edges and stats.
  - F16: Mesh Instancing (Raster) — CPU and GPU paths; indirect draws; examples.
  - F17: Curve & Tube Primitives — ribbons/tubes with joins (miter/bevel/round), caps, and UVs.
  - F18: glTF 2.0 import (+Draco) — loader and PBR mapping.

### Changed
- Scene SSAO/blur/composite pipeline alignment and bind group corrections; fixed validation errors in demos.
- Terrain tile bind group size fix (page table dummy buffer sized to 32 bytes) to satisfy WGSL struct layout.

### Tests
- All Workstream F tests enabled and passing: subdivision (including creases), UV/tangent generation, transforms, validate, instancing, and curves joins.

## [0.60.0]

### Added
- Workstream E1m – Loader diagnostics counters
  - Added per-loader counters (requests, enqueued, dropped_by_policy, canceled, send_fail, completed) for height and overlay async loaders
  - Python diagnostics: `debug_async_loader_counters()`, `debug_async_overlay_loader_counters()`

## [0.59.0]

### Added
- Workstream E1k – Coalescing policy selection and request prioritization
  - Coalescing policy configurable: `coalesce_policy='coarse'|'fine'` for height and overlay loaders
  - Near-to-far priority ordering of visible tiles before issuing requests/uploads
  - Request queue switched to bounded `sync_channel` with non-blocking `try_send()`

## [0.58.0]

### Added
- Workstream E1j – Pluggable real data ingestion
  - File-backed readers: `FileHeightReader(template, scale, offset)` and `FileOverlayReader(template)`
  - Python: `enable_async_loader(..., template=..., scale=..., offset=...)`, `enable_async_overlay_loader(..., template=...)`

## [0.57.0]

### Added
- Workstream E1i – Cancellation and LOD-aware coalescing
  - Cancel requests for tiles leaving visibility; drop results if canceled
  - Coalescing rules: Prefer coarse (cancel descendants) or prefer fine (cancel ancestors)

## [0.56.0]

### Added
- Workstream E1h – Overlay async IO parity
  - `AsyncOverlayLoader` (pool, dedup/backpressure, cancellation)
  - `stream_tiles_to_overlay_mosaic_at_lod(...)` with neighbor prefetch and per-frame upload budget

## [0.55.0]

### Added
- Workstream E1g – Neighbor prefetch
  - 4-neighborhood prefetch under in-flight budget for smoother streaming

## [0.54.0]

### Added
- Workstream E1f – Thread pool and Python API
  - Dispatcher + N worker threads; tunable `pool_size`
  - Python: `enable_async_loader(tile_resolution, max_in_flight, pool_size)` and `debug_async_loader_stats()`

## [0.53.0]

### Added
- Workstream E1e – Deduplication and backpressure
  - Pending set eliminates duplicate in-flight requests
  - In-flight cap limits outstanding requests; per-frame upload budget enforced

## [0.52.0]

### Added
- Workstream E1c – Async tile IO (height)
  - `AsyncTileLoader` and integration in `stream_tiles_to_height_mosaic_at_lod(...)`

## [0.51.0]

### Added
- Workstream E1b – Shader-side UVs via TileSlot + MosaicParams
  - Removed CPU `uv_remap`; shaders compute UVs from page table entries

## [0.50.0]

### Added
- Workstream D10 – Title/Compass/Scale/Text overlays
  - Title bar, compass rose, scale bar, and text overlays with enable/disable APIs

## [0.49.0]

### Added
- Workstream D8 – Hillshade/Shadow overlay
  - Azimuth/altitude-driven shadow overlay with blend modes

## [0.48.0]

### Added
- Workstream D6/D7 – Contours
  - Contour generation and contour overlay rendering

## [0.47.0]

### Added
- Workstream D5 – Altitude overlay
  - Altitude/height-based overlay composited over terrain

## [0.46.0]

### Added
- Workstream D4 – Drape raster overlay
  - Offset/scale controls; alpha blending

## [0.45.0]

### Added
- Workstream D3 – Compass overlay
  - Compass rose overlay with positioning and styling controls

## [0.44.0]

### Added
- Workstream D2 – Text and scale overlays
  - Text overlay API; scale bar overlay support

## [0.43.0]

### Added
- Workstream D1 – Overlays infrastructure
  - Overlay composition scaffolding and renderer wiring

## [0.42.0]

### Added
- Workstream C3 – Shoreline foam overlay
  - Foam overlay controls: width, intensity, noise scale; enable/disable API

## [0.41.0]

### Added
- Workstream C2 – Water material and depth-aware coloration
  - Water surface material with depth colors and alpha

## [0.40.0]

### Added
- Workstream C1 – Water detection & masking
  - `detect_water_from_dem(method='auto'|'flat', smooth_iters)` and external mask support with validation

## [0.39.0] - 2025-09-19

### Added
- Workstream A25 – End-to-end GPU LBVH with fully GPU radix sort
  - Enabled a complete GPU build-and-refit path for triangle BVHs
  - Morton codes on GPU; sorting on GPU (bitonic for ≤256, multi-pass radix with atomics for larger arrays)
  - Topology link (`init_leaves`, `link_nodes`) and iterative refit kernels
  - Example: `examples/accel_lbvh_refit.rs` (prints world AABB before/after)
  - Validation: GPU-gated tests for refit and BVH node invariants

## [0.38.0]

### Added
- Workstream A24 – Wavefront parity and performance stabilization
  - Tightened parity between MEGAKERNEL and WAVEFRONT engines; performance harness refinements

## [0.37.0]

### Added
- Workstream A23 – Queue compaction and scheduling tuning
  - Improved scatter/compact stages and queue depth heuristics for wavefront PT

## [0.36.0]

### Added
- Workstream A22 – Triangle mesh traversal parity
  - BVH traversal tweaks and CPU-GPU parity checks for mesh rendering

## [0.35.0]

### Added
- Workstream A21 – Scene cache controls
  - `enable_scene_cache`, `reset_scene_cache`, `cache_stats` exposed for offline re-renders

## [0.34.0]

### Added
- Workstream A20 – Engine selection API unification
  - `TracerEngine` selection and CLI forwarding in example wrappers

## [0.33.0]

### Added
- Workstream A19 – Scene cache infrastructure
  - Reuse scene-dependent precomputations for deterministic parity

## [0.32.0]

### Added
- Workstream A18 – Progressive tiling refinements
  - Hooks for large offline renders with progress callbacks

## [0.31.0]

### Added
- Workstream A17 – RNG & determinism hardening
  - Seed plumbing and test harness consistency across engines

## [0.30.0]

### Added
- Workstream A16 – Ray and hit buffers optimization
  - Memory layout and bandwidth improvements for path tracing queues

## [0.29.0]

### Added
- Workstream A15 – Progressive tiling (initial)
  - Tiled rendering support for high-resolution outputs

## [0.28.0]

### Added
- Workstream A14 – Material/shader interop
  - Consistent CPU/WGSL paths for selected BRDF features

## [0.27.0]

### Added
- Workstream A13 – Lighting utilities
  - Shared helpers for MIS-friendly direct lighting integration

## [0.26.0]

### Added
- Workstream A12 – Wavefront Path Tracer
  - Raygen/intersect/shade/scatter/compact kernels and scheduler

## [0.25.0]

### Added
- Workstream A11 – Participating media helpers (CPU)
  - HG phase, sampling, and height-fog utilities for prototyping

## [0.24.0]

### Added
- Workstream A10 – Tonemap integration for PT outputs
  - HDR → tonemap flow unification for parity tests

## [0.23.0]

### Added
- Workstream A9 – PBR texture inputs (test scaffold)
  - Deterministic effect of textures in CPU fallback

## [0.22.0]

### Added
- Workstream A8 – ReSTIR DI controls and integration
  - Temporal/spatial reuse toggles and example wrapper flags

## [0.21.0]

### Added
- Workstream A7 – GPU LBVH foundations
  - Morton codes and BVH link scaffolding for triangle meshes

## [0.20.0]

### Added
- Workstream A6 – Dielectric water (offline shading)
  - Schlick Fresnel and Beer–Lambert helpers

## [0.19.0]

### Added
- Workstream A5 – Camera & sampling paths
  - Camera RNG sampling alignment between CPU and GPU paths

## [0.18.0]

### Added
- Workstream A4 – RNG plumbing
  - Deterministic XorShift wiring for repeatable renders

## [0.17.0]

### Added
- Workstream A3 – Triangle mesh pipeline integration points
  - Mesh upload/BVH hooks for future GPU traversal

## [0.16.0]

### Added
- Workstream A2 – Path tracer API surface (Python)
  - Minimal tracer interface, camera constructors, and flags

## [0.15.0]

### Added
- Workstream A1 – Path tracer MVP
  - Single-sphere baseline and deterministic CPU fallback

## [0.14.0] - 2025-09-12

### Added
- Workstream V – Datashader Interop
  - V1: Datashader adapter providing zero-copy RGBA views (`rgba_view_from_agg`) and alignment validation (`validate_alignment`), plus overlay packaging (`to_overlay_texture`) and convenience (`shade_to_overlay`).
  - V2: Performance and fidelity harness: deterministic dataset, zoom tests (Z0/Z4/Z8/Z12), SSIM checks vs goldens, and frame-time/memory metrics.
  - Demo: `examples/datashader_overlay_demo.py` writes `examples/output/datashader_overlay_demo.png` and prints `OK`.
  - CI: `.github/workflows/datashader-perf.yml` runs perf tests, enforces thresholds, uploads artifacts.
- Docs: `docs/user/datashader_interop.rst` added to ToC; covers zero-copy, alignment, and memory guidance.
- Optional dependency handling: graceful skips when Datashader is missing; adapter availability probe.

### Changed
- Public API exposes `forge3d.adapters` with Datashader utilities guarded by availability.
- `validate_alignment` usable without Datashader for basic extent checks.

### Tests
- Datashader unit/perf tests skip cleanly when optional deps absent.

## [0.13.0] - 2025-09-12

### Added
- Workstream U – Basemaps & Tiles
  - U1: XYZ/WMTS tile client with on-disk cache, offline mode, and conditional GET (ETag/Last-Modified). Mosaic composition to RGBA via Pillow.
  - U2: Attribution overlay utility with readable text/logo, DPI-aware placement, and TL/TR/BL/BR presets.
  - U3: Cartopy GeoAxes interop example (Agg backend), overlaying forge3d tile mosaic with extent alignment.
- Provider policy support: polite `User-Agent` and `Referer` on requests, configurable via env (`FORGE3D_TILE_USER_AGENT`, `FORGE3D_TILE_REFERER`). OSM attribution defaults to “© OpenStreetMap contributors”.
- Optional extras: `tiles` (Pillow) and `cartopy` (Cartopy + Matplotlib) added to `pyproject.toml`.
- Docs: `docs/tiles/xyz_wmts.md` and `docs/integration/cartopy.md` added; Sphinx ToC updated.

### Tests
- Added `tests/test_tiles_client.py` and `tests/test_tiles_overlay.py` with mocked HTTP; 7 tests pass locally.

### Artifacts
- Demo outputs saved to `reports/u1_tiles.png` and `reports/u3_cartopy.png`.

## [0.12.0] - 2025-09-12

### Added
- Workstream S — Raster IO & Streaming
  - S1: RasterIO windowed reads and block iterator
    - `windowed_read(dataset, window, out_shape, resampling)` parity with requested window/out_shape
    - `block_iterator(dataset, blocksize)` covers extent without gaps/overlaps
    - Demo: `examples/raster_window_demo.py`; Docs: `docs/ingest/rasterio_tiles.md`
  - S2: Nodata/mask → alpha propagation
    - `extract_masks(dataset)` and RGBA alpha synthesis; color channels preserved
    - Demo: `examples/mask_to_alpha_demo.py`
  - S3: CRS normalization via WarpedVRT + pyproj
    - `WarpedVRTWrapper` and `reproject_window` with resampling handling
    - `get_crs_info(crs)` for CRS inspection; Docs: `docs/ingest/reprojection.md`
    - Demo: `examples/reproject_window_demo.py`
  - S6: Overview selection
    - `select_overview_level` and `windowed_read_with_overview` for byte-reduction at coarse zooms
    - Demo: `examples/overview_selection_demo.py`; Docs: `docs/ingest/overviews.md`
  - S4: xarray/rioxarray DataArray ingestion
    - `ingest_dataarray(da)` preserves dims `(y,x[,band])`, dtype conversions, CRS/transform passthrough
    - Demo: `examples/xarray_ingest_demo.py`; Docs: `docs/ingest/xarray.md`
  - S5: Dask-chunked ingestion
    - `ingest_dask_array`, streaming `materialize_dask_array_streaming`, planning & memory guardrails
    - Demo: `examples/dask_ingest_demo.py`; Docs: `docs/ingest/dask.md`

### Changed
- Optional dependency handling hardened; imports are lazy and degrade gracefully without rasterio/xarray/dask/pyproj
- Demos write artifacts via a tiny PNG fallback when native module is unavailable
- Bumped versions to 0.12.0 (Python package and Cargo crate)

### Tests
- All Workstream S tests pass; full test suite: 560 passed, 94 skipped, 4 xfailed

## [0.11.0] - 2025-09-12

### Added
- Workstream R — Matplotlib & Array Interop
  - R1: Matplotlib colormap interop and linear Normalize support
    - Accepts Matplotlib colormap names and `Colormap` objects; produces RGBA LUTs suitable for forge3d
    - Linear Normalize parity with Matplotlib (per-channel max abs diff ≤ 1e-7 on randomized arrays)
    - Ramp image RGBA parity vs Matplotlib reference (SSIM ≥ 0.999; fallback PSNR ≥ 45 dB)
    - Optional dependency handling with clear ImportError hint when Matplotlib is absent
  - R3: Normalization presets interop (LogNorm, PowerNorm, BoundaryNorm)
    - Parity vs Matplotlib within 1e-7 on representative inputs
    - Unit tests cover common branches and edge cases
  - R4: Display helpers `imshow_rgba(ax, rgba, extent=None, dpi=None)`
    - Correct orientation/aspect; honors extent and DPI
    - Zero-copy path for C-contiguous `uint8` inputs (no pre-copy by helper)
  - Demos and docs
    - Examples: `examples/mpl_cmap_demo.py`, `examples/mpl_norms_demo.py`, `examples/mpl_imshow_demo.py`
    - Docs: `docs/integration/matplotlib.md`

### Changed
- Bumped version to 0.11.0 across Python, Cargo, and packaging metadata
- Simplified Matplotlib backend setup to avoid deprecated internals, improving compatibility

### Tests
- Added `tests/test_mpl_cmap.py`, `tests/test_mpl_norms.py`, and `tests/test_mpl_display.py`
- All tests pass locally; GPU-dependent tests auto-skip without an adapter

## [0.10.0] - 2025-09-11

### Added
- Workstream Q — initial deliverables:
  - Q1 PostFX chain: Python API to enable/disable/list effects and presets; HDR path with tonemap integration.
  - Q5 Bloom: compute passes scaffolded and wired — bright-pass + separable blur (H/V) + composite into HDR prior to tonemap.
  - Q2 LOD: impostors scaffolding (`src/terrain/impostors.rs`), atlas shader stub, sweep demo and perf test.
  - Q3 GPU metrics: surfaces present; Python `gpu_metrics.get_available_metrics()` exposing `vector_indirect_culling` and other labels.
  - Demos: `examples/postfx_chain_demo.py`, `examples/bloom_demo.py`, `examples/lod_impostors_demo.py` (artifacts in `reports/`).
  - Docs: PostFX page added and Sphinx build configured.

### Changed
- Renderer integrates an HDR → PostFX → tonemap path behind a per-renderer toggle; Python helpers added for toggling in user code.

### Notes
- Bloom composite currently writes blurred result into HDR target; additive composite strength control can be expanded in a follow-up.
- Bench/example cargo targets may require additional assets/features; Python builds/tests and docs build remain green.

## [0.9.0] - 2025-09-11

### Changed
- Bumped crate and Python package version to 0.9.0 (Cargo.toml, pyproject.toml, Python `__version__`).
- Aligned packaging and metadata; ensured maturin uses `release-lto` profile as documented.

### Documentation
- Updated README to reflect 0.9.0 in Versioning and added a short “What’s new in v0.9.0”.
- Updated CHANGELOG with this release entry.

### Notes
- No functional code changes in this release; housekeeping for the 0.9.0 cut.

## [0.8.0] - 2025-09-09

### Added
- Test compatibility shims exposed at top-level API (pure-Python) to keep legacy tests green:
  - `c10_parent_z90_child_unitx_world`, `c6_parallel_record_metrics`, `c7_run_compute_prepass`, `c9_push_pop_roundtrip`.
- WGSL shader headers documenting bind groups, bindings, and formats:
  - `src/shaders/pbr.wgsl`, `src/shaders/shadows.wgsl`.
- Documentation updates:
  - `docs/build.md` (CMake wrapper), `examples/README.md` (running examples/outputs).

### Changed
- Python PBR: default parity math with optional perceptual gain gated by env `F3D_PBR_PERCEPTUAL` (enabled by default for specular luma tests; set to `0` to disable).
- PBR material constructor clamps inputs (base_color, metallic, roughness≥0.04, normal_scale, occlusion_strength, emissive) for safer defaults.
- Texture setters accept RGB/RGBA; base-color RGB upgraded to RGBA (alpha=255); metallic-roughness accepts RGB/RGBA (G=roughness, B=metallic).
- Async readback is now opt-in under Cargo feature `async_readback`; `tokio` is optional.

### Fixed
- Readback path error propagation in `src/lib.rs` (no `.expect`; proper `Result` mapping for `map_async`).

### Infrastructure
- Version bump to 0.8.0 across Cargo, Python package, and Sphinx docs.

## [0.7.0] - 2025-09-06

Workstream N – Advanced Rendering Systems (from roadmap.csv):

- N1: PBR material pipeline — Cook–Torrance (GGX) metallic/roughness with material uniforms and Python APIs; SSIM≥0.95; energy conservation validated.
- N2: Shadow mapping (CSM) — 4-cascade, PCF 3×3, bias uniforms; stable across cascades; <10 ms overhead @1080p.
- N3: HDR pipeline & tonemapping — RGBA16F targets; ACES/Reinhard with exposure/gamma; cross-backend consistent.
- N4: Render bundles — Pre-encoded command sequences with cache/invalidation and metrics; 2–5× CPU encode speedups for static scenes.
- N5: Environment mapping/IBL — Cubemap loader, prefiltered env maps, BRDF LUT, irradiance probes; proper roughness→mip mapping.
- N6: TBN generation — Per-vertex tangent/bitangent (MikkTSpace-like) and vertex attrs; pipeline layout accepts new attributes.
- N7: Normal mapping — Tangent-space sampling/decoding to [-1,1], TBN transform, strength blend; docs/examples.
- N8: HDR off-screen target + tone-map enforcement — RGBA16F off-screen with post-pass tonemapper; asserts correct formats; no double-gamma.

### Added
- **Comprehensive Audit Remediation**: Systematic improvements addressing code quality, memory management, and API stability
  - R7: Optional CMake wrapper for cross-platform builds (`CMakeLists.txt`, `cmake/`)
  - R9: Async/double-buffered readback system with buffer pooling and resource management (`src/core/async_readback.rs`)
  - R10: Complete Sphinx API reference documentation with GPU memory management guide (`docs/`)
  - R13: 10 advanced examples showcasing current capabilities:
    - Advanced terrain + shadows + PBR integration (`examples/advanced_terrain_shadows_pbr.py`)
    - Contour overlay visualization with topographic mapping (`examples/contour_overlay_demo.py`)
    - HDR tone mapping comparison with multiple operators (`examples/hdr_tonemap_comparison.py`)
    - Vector OIT layering with transparency demonstration (`examples/vector_oit_layering.py`)
    - Normal mapping on terrain with surface detail (`examples/normal_mapping_terrain.py`)
    - IBL environment lighting with spherical harmonics (`examples/ibl_env_lighting.py`)
    - Multi-threaded command recording with parallel workloads (`examples/multithreaded_command_recording.py`)
    - Async compute prepass for depth optimization (`examples/async_compute_prepass.py`)
    - Large texture upload policies with memory management (`examples/large_texture_upload_policies.py`)
    - Device capability probe with comprehensive GPU analysis (`examples/device_capability_probe.py`)
  - R15: Comprehensive CI/CD workflows for automated testing and releases:
    - Multi-platform CI with Rust fmt/clippy and Python pytest (`.github/workflows/ci.yml`)
    - Automated wheel building and PyPI publishing (`.github/workflows/release.yml`)
    - Performance benchmarking and nightly builds (`.github/workflows/benchmarks.yml`)
    - Dependency monitoring and code quality metrics (`.github/workflows/maintenance.yml`)

### Changed
- **R1**: Unified shadows.get_preset_config() with comprehensive memory validation and legacy compatibility
- **R2**: Implemented Drop trait for ResourceHandle to ensure automatic GPU memory cleanup
- **R3**: Replaced .expect() with RenderError categorization across all FFI boundaries for better error handling
- **R4**: Documented all WGSL bind group layouts with comprehensive pipeline documentation
- **R5**: Aligned CPU PBR implementation with WGSL shaders and clearly documented remaining differences
- **R6**: Improved packaging flow by excluding compiled artifacts and enhancing MANIFEST.in
- **R8**: Expanded texture size accounting to support all GPU formats including compressed and depth formats
- **R11**: Clarified shadows preset memory policy with 256 MiB atlas constraint enforcement
- **R12**: Hardened Python input validation across all APIs with comprehensive dtype/shape/contiguity checks
- **R14**: Finalized public API exports by removing internal functions and establishing materials module policy

### Fixed
- Memory constraint validation now prevents shadow atlas configurations exceeding 256 MiB
- Python input validation provides precise error messages with expected vs. actual parameter descriptions
- Resource cleanup is now automatic through Drop trait implementation, preventing memory leaks
- Error handling across FFI boundaries is now categorized and user-friendly rather than causing panics

### Documentation
- Added comprehensive API policy documentation (`python/forge3d/api_policy.md`)
- Enhanced module docstrings with clear import patterns and stability indicators
- Materials module policy established: `forge3d.pbr` is primary, `forge3d.materials` is compatibility shim
- All advanced examples include detailed documentation and performance metrics

### Infrastructure
- Complete GitHub Actions CI/CD pipeline with multi-platform support
- Automated wheel building for win_amd64, linux_x86_64, and macos_universal2
- Documentation building with Sphinx and automated deployment
- Performance benchmarking with nightly builds and memory stress testing
- Dependency monitoring and security audits

<!-- Future work goes here -->

## [0.6.0] - 2025-09-03

### Added
- **Workstream I – WebGPU Fundamentals**: Advanced GPU memory management and performance optimization
  - I6: Split-buffers performance benchmarking with bind group churn comparison (`examples/perf/split_vs_single_bg.rs`)
  - I7: Big buffer pattern implementation with 64-byte aligned ring allocator (`src/core/big_buffer.rs`)
  - I8: Double-buffering for per-frame data with ping-pong buffer support (`src/core/double_buffer.rs`)
  - I9: Upload policy benchmark harness comparing multiple upload strategies (`bench/upload_policies/policies.rs`)
  - Feature flags (`wsI_bigbuf`, `wsI_double_buf`) for optional adoption
  - Comprehensive performance validation and memory tracking integration
- **Workstream L – Advanced Rendering**: Texture processing and descriptor indexing enhancements
  - L3: Descriptor indexing capability detection and terrain pipeline texture array support
  - HDR (Radiance) image loading and processing utilities (`python/forge3d/hdr.py`)
  - Texture processing and mipmap generation utilities (`python/forge3d/texture.py`)
  - Advanced sampler modes and texture filtering (`src/core/sampler_modes.rs`)
  - GPU-based mipmap generation with gamma-aware downsampling (`src/core/mipmap.rs`)
  - Terrain palette switching with both descriptor indexing and fallback support

### Changed
- Enhanced terrain rendering pipeline to support dynamic palette switching without pipeline rebuilds
- Improved device capability reporting to include descriptor indexing and texture array limits

### Fixed
- Terrain palette switching now produces visually distinct colors when changing palettes
- Memory budget compliance maintained across all new big buffer and double-buffer implementations

## [0.5.0] - 2025-08-31

### Added
- **Workstream H – Vector & Graph Layers**: Complete vector graphics rendering pipeline with GPU acceleration
  - Full vector graphics API with polygons, polylines, points, and graphs (`src/vector/api.rs`)
  - Anti-aliased line rendering with caps and joins support (H8, H9)
  - Instanced point rendering with texture atlas and debug modes (H11, H20, H21, H22)
  - Order Independent Transparency (OIT) for proper alpha blending (H16)
  - GPU culling and indirect drawing for large-scale rendering performance (H17, H19)
  - Polygon fill pipeline with hole support and proper sRGB output (H5, H6)
  - Graph rendering system with separate node/edge pipelines (H12, H13)
  - Comprehensive batching and visibility culling with AABB computation (H4, H10)

## [0.4.0] - 2025-08-30

### Added
- **Zero-Copy NumPy Interoperability**: Implemented zero-copy pathways between NumPy arrays and Rust GPU memory system
  - Added test-only hooks for pointer validation: `render_triangle_rgba_with_ptr()`, `debug_last_height_src_ptr()`
  - Float32 C-contiguous heightmap arrays processed without copying via direct memory access
  - RGBA output buffers returned as NumPy arrays sharing memory with Rust allocations
  - Comprehensive test suite in `tests/test_numpy_interop.py` with 13 validation tests
  - Zero-copy profiler tool `python/tools/profile_copies.py` with "zero-copy OK" validation
  - Added validation helpers in `python/forge3d/_validate.py` for compatibility checking
  - Documentation: `docs/interop_zero_copy.rst` with usage patterns and troubleshooting
- **Memory Budget Tracking**: Implemented 512 MiB host-visible memory budget enforcement
  - Created memory tracker module `src/core/memory_tracker.rs` with atomic resource counters  
  - Budget checking prevents out-of-memory errors with descriptive failure messages
  - Real-time memory metrics via `get_memory_metrics()` API with utilization ratios
  - Thread-safe tracking of buffer/texture allocations and deallocations
  - Fixed readback buffer accounting in render methods with proper budget validation
  - Memory budget test suite in `tests/test_memory_budget.py` with 15 validation tests
  - Documentation: `docs/memory_budget.rst` with usage patterns and best practices

## [0.3.0] - 2025-08-29

### Fixed
- **Terrain UBO size mismatch**: Fixed WGPU validation error "Buffer is bound with size X where shader expects Y"
  - Reduced terrain uniform buffer from 656 bytes to 176 bytes (std140-compatible layout)
  - Removed complex lighting data (point/spot lights, normal matrix) to simplify uniform structure  
  - Updated WGSL shader to match simplified 176-byte layout with 5 fields: view(64B) + proj(64B) + sun_exposure(16B) + spacing_h_exag_pad(16B) + _pad_tail(16B)
  - Added compile-time size assertions and runtime validation to prevent future drift
  - Updated uniform debug interface to return exactly 44 floats (176 bytes / 4)
  - Created comprehensive documentation in `docs/uniforms.rst` explaining the new layout

### Added
- **Model transforms & math helpers**: Complete T/R/S transformation system with math utilities
  - Added comprehensive transform functions: `translate()`, `rotate_x/y/z()`, `scale()`, `scale_uniform()`
  - Implemented `compose_trs()` for T*R*S matrix composition with quaternion-based rotations
  - Added matrix utilities: `multiply_matrices()`, `invert_matrix()`, `look_at_transform()`
  - Created `src/transforms.rs` module with NumPy interop and proper column/row-major conversion
  - 12 comprehensive tests in `tests/test_d4_transforms.py` including acceptance criterion validation
- **Orthographic projection**: Pixel-aligned 2D camera mode for UI and precise rendering
  - Added `camera_orthographic()` function with left/right/bottom/top/near/far parameters
  - Implemented manual orthographic matrix construction with GL↔WGPU clip-space conversion
  - Full support for both GL [-1,1] and WGPU [0,1] depth ranges via `clip_space` parameter
  - 7 validation tests in `tests/test_d5_ortho_camera.py` with pixel-alignment verification
- **Camera uniforms with viewWorldPosition**: Enhanced uniform system for specular lighting
  - Extended `TerrainUniforms` with `view_world_position` field for camera world position
  - Added `camera_world_position_from_view()` utility for automatic extraction from view matrices
  - Updated WGSL shader to access camera position for distance-based lighting effects
  - 8 comprehensive tests in `tests/test_d6_camera_uniforms.py` with matrix validation
- **Normal matrix computation**: Proper normal transformation for non-uniform scaling
  - Added `compute_normal_matrix()` function computing inverse-transpose for correct normal transformation
  - Integrated normal matrix into terrain uniform buffer (64-byte mat4x4 field)
  - Updated WGSL terrain shader to transform normals using normal matrix for accurate lighting
  - 12 mathematical tests in `tests/test_d7_normal_matrix.py` validating transform properties

## [0.2.0] - 2025-08-28

### Added
- **Engine layout & error type**: Added centralized `RenderError` enum with PyErr conversion; created modular layout shims (`src/context.rs`, `src/core/framegraph.rs`, `src/core/gpu_types.rs`) for deliverable compliance
- **Off-screen target preservation**: Added regression tests to ensure 512×512 PNG round-trip remains deterministic; existing row-padding and readback functionality preserved  
- **Device diagnostics integration**: Added `Renderer.report_device()` method returning structured device capabilities including backend, limits, and MSAA support; MSAA automatically gated based on device capabilities
- **Explicit tonemap functions**: Added `reinhard()` and `gamma_correct()` functions to `terrain.wgsl` with explicit gamma 2.2 correction; created comprehensive color management documentation

## [0.1.0] - 2025-08-19

### Added

* Expanded README to cover all implemented ROADMAP items (T2.x, T3.x, T4.2, T5.1, T5.2).
* New examples: grid generation and terrain normalization/height-range.
* Documented timing harness API & CLI with usage guidance.

### Changed

* Version bumped to `0.1.0` (Cargo & Python). `__version__` now reports `0.1.0`.

### Fixed

* Clarified PathLike support and C-contiguity requirements.

## 0.0.9 — T4.1 Scene integration
- Added `scene` module with `Scene` Py API (camera, height upload, render to PNG).
- Reused T3 terrain pipeline and kept bind groups cached.
- **T4.2 PNG & NumPy round-trip ✅**
  - Added `png_to_numpy`, `numpy_to_png`
  - Added `Scene.render_rgba()`
  - Added tests for round-trip and parity with `render_png`
  - T4.2: Fixed `numpy_to_png` to properly accept uint8 arrays of shape **(H,W,3)** (RGB) in addition to (H,W,4) and (H,W).
  - Tests: Added RGB/Gray PNG↔NumPy round-trip coverage.
- Docs: README usage snippet; ROADMAP updated.

## [0.0.8] — 2025-08-16
### Added
- **Workstream T3 — Terrain Shaders & Pipeline.**
- `TerrainPipeline` in `src/terrain/pipeline.rs` with:
  - Bind group layouts: (0) Globals UBO, (1) height **R32Float** + **NonFiltering** sampler, (2) LUT texture + Filtering sampler.
  - Vertex layout: `position.xy` and `uv` as two `Float32x2` attributes.
  - sRGB color target (recommended): `Rgba8UnormSrgb`.
- Python-facing spike `TerrainSpike` for offscreen rendering and PNG output.
- `ColormapLUT` supporting runtime format selection; defaults to sRGB, can force UNORM via `VF_FORCE_LUT_UNORM`.

### Changed
- Cached pipeline and bind groups now used in the render pass (no runtime re-creation).
- Documentation updates:
  - Exact single-line docstring for `build_grid_xyuv` clarifying `[x, z, u, v]` layout.
  - Local comment explaining **NonFiltering (nearest)** requirement for `R32Float` height textures.

### Fixed
- Verified uniform block layout (176 bytes, std140-compatible) and WGPU clip-space projection via tests.

## [0.0.7] — 2025-08-15
### Added
- Completed **Workstream T2 — Uniforms, Camera, and Lighting**.
- New Rust module `src/camera.rs` with:
  - `camera_look_at()`, `camera_perspective(clip_space={'wgpu','gl'})`, `camera_view_proj()` exposed to Python (PyO3).
  - Precise parameter validation and exact error messages.
  - NumPy-friendly outputs: C-contiguous, `float32`, shape `(4,4)`.
- Terrain uniforms:
  - `TerrainUniforms` struct (std140-compatible, **176 bytes**, 16-byte aligned).
  - `Globals` container and `TerrainSpike::debug_uniforms_f32()` to inspect 44-float UBO layout.
- Terrain camera integration:
  - `TerrainSpike::set_camera_look_at(...)` computes aspect from framebuffer and updates UBO.
  - Default projection switched to **WGPU clip space** via `camera::perspective_wgpu()`.

### Changed
- `build_view_matrices()` now uses WGPU depth range [0,1] (was GL [-1,1]).
- GL→WGPU depth conversion refactored to `gl_to_wgpu()` helper.

### Tests
- Rust: unit test guarantees `TerrainUniforms` size and alignment.
- Rust: unit test verifies default projection is WGPU clip space.
- Python: `tests/test_camera.py` (~20 tests) covering camera math, validation, and TerrainSpike integration.

### Docs
- README updated to document WGPU clip-space default and camera API examples.

## [0.0.6] - 2025-08-15
### Workstream T1 — CPU Mesh & GPU Resources
**Status:** Complete

### Added
- **CPU grid mesh generator**
  - New `terrain::mesh::{make_grid, GridMesh, GridVertex, Indices}` with CCW winding and centered origin.
  - Python API `_forge3d.grid_generate(nx, nz, spacing, origin)` returning NumPy arrays:
    - `XY: (N,2) float32`, `UV: (N,2) float32`, `indices: (M,) uint32`.
  - Validation of shapes/dtypes; zero-copy where possible.
- **Height texture upload (R32Float)**
  - `Renderer.upload_height_r32f()` with proper 256-byte `bytes_per_row` padding.
  - Debug helpers: `debug_read_height_patch()` and `read_full_height_texture()`.
- **Colormap LUT system**
  - Central registry `src/colormap/mod.rs` with embedded 256×1 PNG assets: `viridis`, `magma`, `terrain`.
  - Unconditional Python/Rust discovery: `colormap_supported()`.
  - Runtime texture format selection:
    - Prefer `Rgba8UnormSrgb`; fallback to `Rgba8Unorm` with CPU sRGB→linear conversion.
    - Env toggle `VF_FORCE_LUT_UNORM=1` to exercise fallback path.
  - `TerrainSpike` integration (feature-gated): bind group layout `(0=UBO, 1=texture, 2=sampler)`, linear-filtered LUT sampling.
  - `TerrainSpike.debug_lut_format()` for inspection.
- **Docs & API polish**
  - Expanded README sections (T11, T1.2, T1.3).
  - Rich docstrings in `python/forge3d/__init__.py`.

### Changed
- Removed stale `grid` module; Python keeps a `generate_grid` alias to the new `grid_generate` for compatibility.
- `TerrainSpike` now seeds lighting from the computed light vector.
- WGSL shader cleaned up and aligned with binding layout.

### Fixed
- WGSL parsing errors (commas vs semicolons) and linear-space lighting correctness.

### Tests
- `tests/test_grid_generate.py`: shapes, dtypes, UV corners, CCW winding, u16/u32 index switch, large grids.
- `tests/test_colormap.py`: registry/discovery, format fallback (including `VF_FORCE_LUT_UNORM`), shader sanity.

### Compatibility
- No breaking Python API changes; `TerrainSpike` remains feature-gated.
- Rust callers should migrate imports from `crate::grid` → `crate::terrain::mesh`.
  - Python alias preserves old name: `generate_grid = grid_generate`.


## \[0.0.5] - 2025-08-08

### Added

* **T33 – Colormap LUT & assets:** Embedded 256×1 PNG LUTs (`viridis`, `magma`, `terrain`) and a central registry `colormap.rs` exposing `SUPPORTED` and `resolve_bytes()`. LUTs are sampled in the fragment shader for height-mapped color.
* **A2 – Terrain spike renderer:** `TerrainSpike(width, height, grid=128, colormap='viridis')` headless renderer with off-screen target and `render_png(path)` for test coverage.
* **T2.2 – Sun direction & tonemap:** Uniforms now carry `sun_dir` and `exposure`; shader computes diffuse `N·L` and applies Reinhard tonemap for perceptually non-flat output. Python helpers `set_sun(elevation_deg, azimuth_deg)` and `set_exposure(exposure)` (gated by `terrain_spike` feature).
* **T1.1 – Grid index/vertex generator:** CPU grid mesh (positions + normals) for the spike terrain; Python wrapper `grid_generate(nx, nz, spacing=(dx,dy), origin='center')` returning NumPy arrays.

### Changed

* **Uniform layout:** `TerrainUniforms` repacked to std140-compatible **176 bytes**:
  `view(64) + proj(64) + sun_exposure(16) + spacing_h_exag_pad(16) + _pad_tail(16)`.
  Matches WGSL reflection and avoids validation errors.
* **Shader pipeline:** `terrain.wgsl` updated to consume the new uniform layout, sample the LUT, apply diffuse lighting, and tonemap; bindings: `@group(0) @binding(0)=UBO`, `1=LUT texture_2d`, `2=Sampler`.
* **Colormap selection:** Strict, case-sensitive names validated against `SUPPORTED`; shared error text across Rust/Python to keep tests deterministic.

### Fixed

* **wgpu validation panic** “buffer size 164, shader expects 176”: corrected by the new UBO layout.
* **WGSL parse error** (“expected ',', found ';'”): struct fields now comma-separated; shader module creation no longer fails.
* **wgpu 0.19 API mismatch**: `ImageDataLayout.{bytes_per_row,rows_per_image}` now `Option<u32>`—converted `NonZeroU32` via `.into()` at all call sites.
* **Colormap ‘magma’ rejected**: registry and asset mapping added; constructor accepts `"magma"`.
* **Uniform PNG output (\~710B)**: shader now maps height→LUT and lights scene; PNG sizes comfortably exceed the test threshold.

### Technical Notes

* Off-screen color target `Rgba8UnormSrgb` with 256-byte row alignment for copies; readback performs CPU unpadding.
* Validation layers remain enabled in Debug; any device/shader error is a test failure.
* Paths kept zero-copy for NumPy interop; no unnecessary heap churn during readbacks.

## [0.0.4] - 2025-08-05
### Added
- **T0.1 – Public API & validation:** `Renderer.add_terrain(heightmap, spacing, exaggeration, colormap)` with robust NumPy array validation; accepts `float32`/`float64` with shape `(H, W)` and C‑contiguous requirement; clear `PyRuntimeError` for invalid inputs.
- **T0.2 – DEM statistics & normalization:** Automatic `h_min`/`h_max` computation from heightmap with optional percentile clamping; `Renderer.set_height_range(min, max)` override method for custom height ranges.
- **T1.1 – Grid index/vertex generator:** CPU mesh generation in `terrain/mesh.rs` with `make_grid(W, H, dx, dy)` producing indexed triangle grids; vertex attributes include world‑space `position.xy` and `uv` coordinates for height sampling; automatic `u16`/`u32` index format selection based on vertex count.
- **T1.2 – Height texture upload:** GPU height texture creation with `R32Float` format and `TEXTURE_BINDING | COPY_DST` usage; 256‑byte row alignment handling for cross‑platform compatibility; linear clamp sampler configuration.
- **T1.3 – Colormap LUT texture:** Built‑in terrain colormaps (`viridis`, `magma`, `terrain`) as 256×1 `RGBA8UnormSrgb` textures; height‑to‑color mapping with `h_min`/`h_max` normalization uniforms; CPU reference implementation for unit testing.

### Changed
- Enhanced `Renderer` constructor to support terrain rendering pipeline initialization.
- Terrain metadata storage including `dx`, `dy`, `h_min`, `h_max`, `exaggeration`, and colormap selection.

### Fixed
- Proper error handling for unsupported heightmap dtypes and shapes.
- Memory‑efficient reuse of vertex/index buffers for terrain mesh generation.

### Technical Notes
- Grid generation optimized for 1024×1024 heightmaps with sub‑40ms performance target.
- Cross‑platform texture upload with proper row padding handled automatically.
- Colormap validation ensures known scalar inputs map to expected palette colors.

## [0.0.3] - 2025-08-01
### Added
- **A1.9 – Device diagnostics & failure modes:** Rust PyO3 APIs `enumerate_adapters()` and `device_probe(backend)`; Python CLI `python/tools/device_diagnostics.py` that writes JSON and classifies outcomes.
- **A1.10 – Performance sanity:** `python/tools/perf_sanity.py` measuring init/steady timings with JSON/CSV output; optional budget/baseline enforcement via `VF_ENFORCE_PERF=1`.
- **A1.8 – CI matrix & artifacts:** New workflow `.github/workflows/ci.yml` running pytest, determinism harness, and (Win/macOS) cross-backend runner with uploaded artifacts.
- **Docs (A1.11):** Quickstart, Tools, Testing, CI, and Troubleshooting sections updated.

### Changed
- README guidance for Python 3.13 builds using `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.

### Fixed
- Robust import paths for the compiled module in tools; improved error messages.


## [0.0.2] - 2025-07-31
### Added
- **A1.4 – Off-screen target & readback**: persistent RGBA8 UNORM SRGB color target (`RENDER_ATTACHMENT | COPY_SRC`) and persistent readback buffer with 256-byte row alignment + CPU unpadding.
- **A1.5 – Python API surface**: public package `forge3d` with:
  - `Renderer(width, height)`
  - `render_triangle_rgba(width, height) -> (H,W,4) uint8`
  - `render_triangle_png(path, width, height) -> None`
  - `__version__` metadata
- Legacy alias **`vshade`** re-exports the public API for compatibility.

### Changed
- PyO3 text signatures and docstrings for `__init__`, `render_triangle_rgba`, `render_triangle_png`, and `info`.
- Robust import in `forge3d/__init__.py` (supports top-level or package-internal `_forge3d`).

### Fixed
- Import symbol mismatch by standardizing the module to `#[pymodule] fn _forge3d(...)`.
- Deterministic pipeline preserved (blend=None, CLEAR_COLOR, CCW + back-cull, fixed viewport/scissor).

### Notes
- For Python 3.13 + PyO3 0.21, build with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
