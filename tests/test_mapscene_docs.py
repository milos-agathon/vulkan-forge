from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


REQUIRED_MAPSCENE_DIAGNOSTIC_CODES = (
    "missing_crs",
    "missing_source_identity",
    "missing_renderable_data",
    "missing_external_asset",
    "unsupported_asset_format",
    "unsupported_output_format",
    "unsupported_layer_type",
    "unsupported_feature",
)


SUPPORT_MATRIX_DOCS = (
    "docs/guides/style_support_matrix.md",
    "docs/guides/building_support_matrix.md",
    "docs/guides/tiles3d_support_matrix.md",
    "docs/guides/virtual_texturing_support_matrix.md",
    "docs/guides/competitive_positioning.md",
)

ALLOWED_SUPPORT_LEVELS = {
    "supported",
    "underdeveloped",
    "missing",
    "Pro-gated",
    "placeholder/fallback",
    "experimental",
    "unsupported",
    "non-goal",
}


def test_mapscene_public_api_is_documented_with_truthful_support_levels():
    offline = _text("docs/guides/offline_3d_map_rendering.md")
    api = _text("docs/api/api_reference.rst")

    assert "forge3d.map_scene" in api
    assert "MapScene.render" in offline
    assert "MapScene.save_bundle" in offline
    assert "| `MapScene.render` PNG/EXR path | `supported` |" in offline
    assert "GPU-terrain" in api
    assert "allow_placeholder" not in api, "the placeholder escape hatch was removed by SUTURA"
    assert "MapSceneNativeUnavailable" in api
    assert "last_render_backend" in api
    assert "GPU-terrain PNG/EXR output" in offline
    assert "recipe_manifest" in api
    assert "Full MapScene rendering | `missing`" not in offline
    assert "Deterministic LabelPlan | `missing`" not in offline
    assert "`unsupported`" in offline
    assert "`Pro-gated`" in offline
    assert "`placeholder/fallback`" in offline


def test_feature_004_artifacts_do_not_keep_stale_tbd_or_later_owned_wording():
    targets = ["docs/guides/offline_3d_map_rendering.md"]
    forbidden = ("TBD", "later-owned", "Owned by feature `004`", "Owned by feature `003`")

    for target in targets:
        text = _text(target)
        for marker in forbidden:
            assert marker not in text, f"{marker!r} remains in {target}"


def test_mapscene_validation_diagnostic_codes_are_documented():
    diagnostics = _text("docs/guides/diagnostics_reference.md")

    for code in REQUIRED_MAPSCENE_DIAGNOSTIC_CODES:
        assert f"`{code}`" in diagnostics, f"{code!r} missing from diagnostics reference"


def test_offline_mapscene_guide_links_canonical_examples_and_support_guides():
    offline = _text("docs/guides/offline_3d_map_rendering.md")

    required_markers = (
        "examples/mapscene_terrain_raster.py",
        "examples/mapscene_vector_labels.py",
        "examples/mapscene_buildings_labels.py",
        "guides/label_plan_guide",
        "guides/diagnostics_reference",
        "guides/style_support_matrix",
        "guides/building_support_matrix",
        "guides/tiles3d_support_matrix",
        "guides/virtual_texturing_support_matrix",
        "guides/competitive_positioning",
        "python -m pytest tests/test_mapscene_quickstart.py -q",
    )

    for marker in required_markers:
        assert marker in offline, f"{marker!r} missing from offline MapScene guide"


def test_support_matrices_record_current_mapscene_diagnostics_and_ownership():
    style = _text("docs/guides/style_support_matrix.md")
    buildings = _text("docs/guides/building_support_matrix.md")
    tiles3d = _text("docs/guides/tiles3d_support_matrix.md")
    vt = _text("docs/guides/virtual_texturing_support_matrix.md")

    assert "LabelPlan" in style
    assert "MapScene.validate" in style
    assert "later features" not in style

    for marker in ("MapSceneBuildingLayer", "missing_external_asset", "unsupported_asset_format"):
        assert marker in buildings
    assert "later work" not in buildings

    assert "MapScene" in tiles3d
    assert "python_public_3dtiles_incomplete" in tiles3d
    assert "Owned by later MapScene" not in tiles3d

    assert "MapScene.validate" in vt
    assert "albedo-only" in vt
    assert "vt_unsupported_family" in vt


def test_support_matrix_rows_use_prd_support_taxonomy():
    for target in SUPPORT_MATRIX_DOCS:
        text = _text(target)
        in_support_table = False
        for line in text.splitlines():
            if not line.startswith("| "):
                in_support_table = False
                continue
            if line.startswith("| ---"):
                continue
            if "Support level" in line:
                in_support_table = True
                continue
            if not in_support_table:
                continue
            cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            level_index = 2 if target.endswith("competitive_positioning.md") else 1
            if len(cells) > level_index:
                assert cells[level_index] in ALLOWED_SUPPORT_LEVELS, (
                    f"{target} has non-PRD support level {cells[level_index]!r}"
                )


def test_competitive_positioning_lists_required_non_goal_boundaries():
    competitive = _text("docs/guides/competitive_positioning.md")

    required_boundaries = (
        "Streamed browser map delivery",
        "Complete Mapbox style parity",
        "Cesium-grade global runtime parity",
        "General DCC or game-editor workflows",
        "Textured PBR buildings",
        "VT normal/mask runtime",
    )
    for boundary in required_boundaries:
        assert boundary in competitive
    assert "production-ready 3D Tiles" not in competitive
