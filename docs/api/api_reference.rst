API Reference
=============

This reference covers the current public Python surface exposed by the package.
It is grouped by workflow so the module layout is easier to navigate.

Top-Level Package
-----------------

.. automodule:: forge3d
   :members:
   :no-index:

Viewer, Notebook, And IPC
-------------------------

.. automodule:: forge3d.viewer
   :members:
   :no-index:

Label API Truth
~~~~~~~~~~~~~~~

The public label workflow is exposed through ``forge3d.viewer.ViewerHandle``.
For feature ``002-label-api-truth``, use ``ViewerHandle.add_label``,
``ViewerHandle.add_labels``, ``ViewerHandle.add_line_label``,
``ViewerHandle.add_curved_label``, ``ViewerHandle.add_callout``,
``ViewerHandle.add_vector_overlay``, ``ViewerHandle.load_label_atlas``,
``ViewerHandle.set_labels_enabled``, ``ViewerHandle.clear_labels``,
``ViewerHandle.remove_label``, ``ViewerHandle.set_label_typography``, and
``ViewerHandle.set_declutter_algorithm``.
Run ``python examples/label_api_truth_basic.py --json`` to exercise the public
workflow and inspect deterministic ids, diagnostics, layout metrics, and
placement policy output.

Successful point, batch, line, callout, and vector-overlay create operations
return stable ids where later inspection, removal, export, or review needs
them. Typography controls update native label-manager state and expose
serializable layout metrics. Decluttering controls update native label-manager
state and expose a deterministic placement policy. Curved labels and
terrain-elevated line labels are currently ``experimental`` public paths that
return typed ``experimental_feature`` diagnostics rather than unqualified
success. Empty text, invalid line geometry, and unsupported declutter
algorithms return ``placeholder_fallback`` diagnostics; known glyph coverage
gaps return ``missing_glyphs`` diagnostics.

Deterministic LabelPlan
~~~~~~~~~~~~~~~~~~~~~~~

``forge3d.label_plan`` exposes the offline deterministic label compiler. Use
``LabelPlan.compile`` with labels, camera/output context, terrain samples,
keepouts, priority classes, typography, glyph coverage, and a seed to produce
accepted labels, rejected labels, diagnostics, bounds, and render/export
payload data. Unsupported payload backends return typed
``placeholder_fallback`` diagnostics rather than empty success.

.. automodule:: forge3d.label_plan
   :members:
   :no-index:

Typed MapScene API
~~~~~~~~~~~~~~~~~~

``forge3d.map_scene`` exposes the typed offline map-production contract. Use
``MapScene`` with ``SceneRecipe`` or keyword recipe components to validate
scene intent, compile deterministic frozen render plans with
``MapScene.compile_plan()``, write GPU-terrain PNG output or EXR output for
renderable terrain recipes, raise ``MapSceneNativeUnavailable`` with
structured diagnostic blocks when native rendering is unavailable (there is
no CPU placeholder output), and save
deterministic review bundles. ``OutputSpec`` exposes ``samples``,
``denoiser``, ``aovs``, and ``hdr`` fields for offline-quality MapScene output,
and ``LightingPreset(name="rainier_showcase")`` resolves the self-contained
camera/sun/IBL/reproducibility preset from ``forge3d.presets``.
Unsupported, ``Pro-gated``,
``placeholder/fallback``, ``experimental``, missing-source, and blocking
diagnostic paths are reported before successful render completion.

``MapScene.render`` performs validation before rendering, writes deterministic
PNG or EXR output for supported terrain/raster/vector/label scene recipes,
records the last backend on ``MapScene.last_render_backend`` and sample/AOV
details on ``MapScene.last_render_metadata``, and still blocks unsupported layer
paths through typed diagnostics instead of representing them as successful
renders. ``forge3d.recipe_manifest(scene)`` returns a deterministic JSON-safe
manifest for CI/review tooling. ``MapScene.save_bundle`` writes review metadata
and diagnostics; blocked scenes are recorded as non-renderable instead of being
represented as successful renders.

Feature ``005-map-assets-bundles-p1`` extends this product path without
changing the legacy building module export. Use ``forge3d.map_scene.LabelLayer``
constructors ``LabelLayer.from_features``, ``LabelLayer.from_geodataframe``,
``LabelLayer.from_style_layer``, and ``LabelLayer.compile_labels`` for
data-driven label input. Typography inputs use ``FontAtlas.default_latin``,
``FontAtlas.from_font``, ``FontFallbackRange``, and ``TypographySettings`` for
bundled Basic Latin coverage, typed font asset diagnostics, deterministic
fallback declarations, kerning/tracking/line-height metrics, multiline labels,
and callout layout metadata. Use ``forge3d.map_scene.BuildingLayer`` methods
``BuildingLayer.from_geojson``, ``BuildingLayer.from_cityjson``, and
``BuildingLayer.from_mesh`` for product-scene building intent; the top-level
``MapSceneBuildingLayer`` alias points to this product class, while the legacy ``forge3d.BuildingLayer`` name still points to ``forge3d.buildings`` for
backward compatibility. Use ``Tiles3DLayer.from_tileset_json`` and
``Tiles3DLayer.from_b3dm`` for typed 3D Tiles scene intent.

These P1 asset APIs are still ``underdeveloped`` until their story-specific
tests complete. Unsupported, ``Pro-gated``, ``placeholder/fallback``, and
``experimental`` paths must remain diagnostic-bearing. Feature-local diagnostic
codes include ``missing_label_field``, ``unicode_coverage_gap``,
``unsupported_tile_format``, ``unsupported_tile_feature``,
``missing_external_asset``, and ``unavailable_terrain_sampler``.
``MapScene.load_bundle`` reconstructs the deterministic recipe payload where
available and preserves validation diagnostics; richer renderable bundle replay
remains owned by the P1 bundle tasks.

.. automodule:: forge3d.map_scene
   :members:
   :no-index:

.. automodule:: forge3d.viewer_ipc
   :members:
   :no-index:

.. automodule:: forge3d.widgets
   :members:
   :no-index:

.. automodule:: forge3d.interactive
   :members:
   :no-index:

Terrain Configuration And Automation
------------------------------------

.. automodule:: forge3d.terrain_params
   :members:
   :no-index:

.. automodule:: forge3d.presets
   :members:
   :no-index:

.. automodule:: forge3d.animation
   :members:
   :no-index:

.. automodule:: forge3d.camera_rigs
   :members:
   :no-index:

.. automodule:: forge3d.terrain_scatter
   :members:
   :no-index:

Data Access And Scene Inputs
----------------------------

.. automodule:: forge3d.datasets
   :members:
   :no-index:

.. automodule:: forge3d.crs
   :members:
   :no-index:

.. automodule:: forge3d.cog
   :members:
   :no-index:

.. automodule:: forge3d.pointcloud
   :members:
   :no-index:

.. automodule:: forge3d.tiles3d
   :members:
   :no-index:

Scene Assets And Packaging
--------------------------

.. automodule:: forge3d.buildings
   :members:
   :no-index:

.. automodule:: forge3d.bundle
   :members:
   :no-index:

.. automodule:: forge3d.style
   :members:
   :no-index:

.. automodule:: forge3d.style_expressions
   :members:
   :no-index:

Cartography And Export
----------------------

.. automodule:: forge3d.map_plate
   :members:
   :no-index:

.. automodule:: forge3d.legend
   :members:
   :no-index:

.. automodule:: forge3d.scale_bar
   :members:
   :no-index:

.. automodule:: forge3d.north_arrow
   :members:
   :no-index:

.. automodule:: forge3d.export
   :members:
   :no-index:

Native Rendering And Quality
----------------------------

.. automodule:: forge3d.media
   :members:
   :no-index:

.. automodule:: forge3d.offline
   :members:
   :no-index:

.. automodule:: forge3d.denoise_oidn
   :members:
   :no-index:

.. automodule:: forge3d.path_tracing
   :members:
   :no-index:

.. automodule:: forge3d.sdf
   :members:
   :no-index:

.. automodule:: forge3d.lighting
   :members:
   :no-index:

Geometry, Mesh, Vector, And IO
------------------------------

.. automodule:: forge3d.geometry
   :members:
   :no-index:

.. automodule:: forge3d.io
   :members:
   :no-index:

.. automodule:: forge3d.mesh
   :members:
   :no-index:

.. automodule:: forge3d.vector
   :members:
   :no-index:

Materials, Textures, And Utilities
----------------------------------

.. automodule:: forge3d.materials
   :members:
   :no-index:

.. automodule:: forge3d.textures
   :members:
   :no-index:

.. automodule:: forge3d.colors
   :members:
   :no-index:

.. automodule:: forge3d.mem
   :members:
   :no-index:

.. automodule:: forge3d.config
   :members:
   :no-index:

CLI-Oriented Helpers
--------------------

.. automodule:: forge3d.terrain_demo
   :members:
   :no-index:

.. automodule:: forge3d.terrain_pbr_pom
   :members:
   :no-index:
