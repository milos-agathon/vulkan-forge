"""Fail-closed inventory for every Rust narrowing primitive.

The inventory is deliberately name-agnostic: every ``as f32``, ``.as_vec*()``,
and f64/DVec-to-f32/Vec helper is occurrence-locked across production Rust.
Separate positive contracts prove that each viewer world-position route calls
the active Anchor inside the producing function.
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANCTIONED = "src/camera/anchor.rs"
SANCTIONED_DD_SPLITS = {
    (
        "src/core/dd.rs",
        "from_f64",
        "as_f32",
        1,
        "let hi = value as f32",
    ),
    (
        "src/core/dd.rs",
        "from_f64",
        "as_f32",
        2,
        "let lo = (value - hi as f64) as f32",
    ),
}

# Updated only after reviewing the complete inventory printed by a failure.
# The digest includes (file, function, operation, ordinal, normalized statement).
EXPECTED_CONVERSION_COUNT = 1545
# The previous 1438-site freeze already covered the reviewed ANAMNESIS,
# TESSELLA, and first SIDERA transitions described below. The d8313007 base
# source actually contained 1446 sites because SIDERA's later adversarial
# closure added twelve u32 viewport-dimension/reciprocal casts and one
# normalized celestial unit-direction conversion without refreshing this
# constant. SUBSTRATIA adds one u32 feedback-origin telemetry counter converted
# for Python float stats. It also moves four existing tile-index-to-normalized-
# UV casts from finish_frame to ingest_shader_feedback; that changes their
# occurrence ownership, but not the count. The physical TESSELLA picking audit
# also moved two existing screen-coordinate conversions from pixel corners to
# raster pixel centres; their count is unchanged. Its visibility CPU oracle
# adds seven raster-only conversions: pixel centres (2), normalized coarse
# texel steps (2), a bounded LUT index (1), and viewport projection (2). All
# reviewed primitives are render dimensions, normalized directions/UVs,
# bounded raster indices, or telemetry. None stores absolute world coordinates
# or bypasses the camera Anchor. The subsequent physical-terrain closure
# consolidated repeated clipmap ring coordinate construction, reducing the
# reviewed inventory without weakening the Anchor boundary. HELIOS (below)
# adds the curvature-aware GPU viewshed, terrain-to-sun shadow mask, and
# closed-form shadow-tip analysis plus earth-curvature traversal in the hybrid
# terrain reference; see REVIEWED_HELIOS_INVENTORY_TRANSITION for the exact
# reviewed additions. CARTOGRAPHER-PRIME then adds four f64-to-f32 conversions
# for a validated finite silhouette bounding box in labels/optimal.rs; these
# remain bounded screen-space coordinates and do not cross the Anchor boundary.
EXPECTED_CONVERSION_SHA256 = "b60331341dbeeb3c24a16fe52b92f1c51f18d8dffcf51d7e4132ba79d35ee065"

# The reviewed TERMINUS reader transition remains locked below. COMPENDIUM adds
# four integer-to-f32 reconstruction conversions in predict.rs; those are
# included in the current count and digest above without weakening the reader
# transition assertion.
REVIEWED_INVENTORY_TRANSITION = {
    "current_count": 1545,
    "removed": (
        "src/terrain/cog/cog_reader.rs",
        "decode_heights",
        "as_f32",
        1,
        "heights.push(f64::from_le_bytes(bytes) as f32)",
    ),
    "added": (
        "src/terrain/cog/cog_reader.rs",
        "decode_heights",
        "as_f32",
        1,
        "heights.push(f64::from_le_bytes(read_le_bytes8(data, i * 8)) as f32)",
    ),
}

# ANAMNESIS retained the exact five adjudication conversions and moved them
# into the incremental implementation beneath the compatibility wrapper. This
# transition records that function-only ownership change without relaxing the
# occurrence count or any normalized conversion statement.
REVIEWED_ANAMNESIS_INVENTORY_TRANSITION = {
    # Re-based on main at the merge: the pre-transition tree is now main rather
    # than this branch's original base, so the count and digest are main's.
    "base_count": 1545,
    "base_digest": "9850587e94805c6d45e321cc54f5ea40dc54e6efa7facbcc45f17b00925283d4",
    "result_digest": EXPECTED_CONVERSION_SHA256,
    "path": "src/offscreen/adjudication_raster.rs",
    "removed_function": "render_raster_reference",
    "added_function": "render_raster_reference_incremental",
    "statements": (
        "let aspect = width as f32 / height as f32",
        "let aspect = width as f32 / height as f32",
        "u.misc = [desc.plane_half_extent, i as f32, 1.0, 0.0]",
        "let o = (k as f32 + 0.5) / SSAA as f32 - 0.5",
        "let o = (k as f32 + 0.5) / SSAA as f32 - 0.5",
    ),
}

# HELIOS adds the curvature- and refraction-aware GPU viewshed, the direct
# terrain-to-sun shadow mask, and the closed-form curved-Earth shadow-tip
# analysis in src/terrain/analysis/viewshed.rs and src/py_functions/geodesy.rs,
# plus earth-curvature traversal uniforms in the hybrid terrain reference
# (src/path_tracing/hybrid_compute/terrain_heightfield.rs and render_terrain.rs).
# The 39 reviewed additions are render-space narrowing conversions: grid
# distances/azimuths, pixel-centre offsets, bounded observer coordinates,
# normalized latitude/longitude radians, curvature radius coefficients, and
# runtime-contract telemetry. The 3 removals are moved/consolidated module
# constants (origin_x/origin_z) and a duplicated check_range occurrence; none
# relaxes the Anchor world-coordinate boundary. None of the additions stores
# absolute world coordinates or bypasses the camera Anchor.
REVIEWED_HELIOS_INVENTORY_TRANSITION = {
    # Re-based on main at the merge: the pre-transition tree is now main rather
    # than this branch's original base, so the count and digest are main's.
    "base_count": 1545,
    "base_digest": "9850587e94805c6d45e321cc54f5ea40dc54e6efa7facbcc45f17b00925283d4",
    "result_digest": EXPECTED_CONVERSION_SHA256,
    "added_sites": (
        (
            "src/py_functions/geodesy.rs",
            "terrain_grid_heights",
            "as_f32",
            1,
            "positions_m.push([ (distance_m * azimuth.sin()) as f32, (distance_m * azimuth.cos()) as f32, ])",
        ),
        (
            "src/py_functions/geodesy.rs",
            "terrain_shadow_mask",
            "as_f32",
            1,
            "geodetic_positions_and_sun.push([ latitude.to_radians() as f32, longitude.to_radians() as f32, solar.azimuth_deg.to_radians() as f32, launch_elevation_deg.to_radians() as f32, ])",
        ),
        (
            "src/terrain/analysis/viewshed.rs",
            "<module>",
            "as_f32",
            1,
            "Ok([ inv_meridional as f32, inv_prime_vertical as f32, one_minus_k as f32, f32::from(!matches!(options.earth_model, EarthModel::Flat)), ])",
        ),
        (
            "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
            "<module>",
            "as_f32",
            1,
            "(0.5 / effective_radius) as f32",
        ),
        (
            "src/path_tracing/hybrid_compute/render_terrain.rs",
            "record_runtime_contract",
            "as_f32",
            12,
            "check( , &[earth_curvature.enabled as f32], 0.0, 1.0, )",
        ),
    ),
    "removed_sites": (
        (
            "src/path_tracing/hybrid_compute/render_terrain.rs",
            "record_runtime_contract",
            "as_f32",
            12,
            "observed.check_range( , , None, m_min as f32, m_max as f32, 0.0, 512.0, )",
        ),
        (
            "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
            "<module>",
            "as_f32",
            1,
            "let origin_x = -0.5 * (self.width as f32 - 1.0) * spacing_x",
        ),
        (
            "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
            "<module>",
            "as_f32",
            2,
            "let origin_z = -0.5 * (self.height as f32 - 1.0) * spacing_z",
        ),
    ),
}


def _strip_comments_and_strings(text: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|r#*\".*?\"#*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.S,
    )
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _remove_cfg_test_modules(text: str) -> str:
    text = _strip_comments_and_strings(text)
    marker = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*mod\s+\w+\s*\{")
    while match := marker.search(text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        text = text[: match.start()] + " " * (cursor - match.start()) + text[cursor:]
    return text


def _function_spans(text: str):
    spans = []
    for match in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)[^;{]*\{", text, re.S):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        spans.append((match.start(), cursor, match.group(1)))
    return spans


def _function_name(spans, position: int) -> str:
    return next(
        (name for start, end, name in spans if start <= position < end),
        "<module>",
    )


def _statement(text: str, position: int) -> str:
    start = max(text.rfind(";", 0, position), text.rfind("{", 0, position)) + 1
    end_candidates = [candidate for candidate in (text.find(";", position), text.find("}", position)) if candidate >= 0]
    end = min(end_candidates, default=min(len(text), position + 160))
    return re.sub(r"\s+", " ", text[start:end]).strip()


CONVERSION_PATTERNS = {
    "as_f32": re.compile(r"\bas\s+f32\b"),
    "as_vec": re.compile(r"\.\s*as_vec[234]\s*\(\s*\)"),
    "f64_helper": re.compile(
        r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:<[^>{}]*>)?\s*\([^)]*(?:f64|DVec[234]|DMat[234])[^)]*\)\s*"
        r"(?:->\s*(?:f32|Vec[234]|\[\s*f32|Vec\s*<\s*f32))",
        re.S,
    ),
}


def _conversion_inventory_text(rel: str, raw: str):
    text = _remove_cfg_test_modules(raw)
    spans = _function_spans(text)
    matches = []
    for operation, pattern in CONVERSION_PATTERNS.items():
        matches.extend((match.start(), operation) for match in pattern.finditer(text))
    counters = {}
    sites = []
    for position, operation in sorted(matches):
        function = _function_name(spans, position)
        key = (function, operation)
        counters[key] = counters.get(key, 0) + 1
        sites.append(
            (
                rel,
                function,
                operation,
                counters[key],
                _statement(text, position),
            )
        )
    return sites


def _complete_conversion_inventory():
    sites = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        sites.extend(_conversion_inventory_text(rel, path.read_text(encoding="utf-8")))
    return sites


def conversion_inventory():
    """Reviewed narrowing inventory, excluding the exact lossless DD split."""
    return [site for site in _complete_conversion_inventory() if site not in SANCTIONED_DD_SPLITS]


def _inventory_digest(sites) -> str:
    payload = "\n".join("\t".join(map(str, site)) for site in sites)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _function_body(rel: str, function: str) -> str:
    text = _remove_cfg_test_modules(_read(rel))
    for start, end, name in _function_spans(text):
        if name == function:
            return text[start:end]
    raise AssertionError(f"missing function {rel}::{function}")


def test_exact_production_conversion_inventory_is_frozen():
    sites = conversion_inventory()
    digest = _inventory_digest(sites)
    assert (len(sites), digest) == (
        EXPECTED_CONVERSION_COUNT,
        EXPECTED_CONVERSION_SHA256,
    ), f"conversion inventory changed: count={len(sites)} sha256={digest}\n" + "\n".join(
        repr(site) for site in sites
    )


def test_all_required_rejecting_probes_change_the_inventory():
    probes = [
        "v as f32",
        "coords[0] as f32",
        "position.as_vec3()",
        "coords.map(|v| v as f32)",
        "point.to_array().map(|v| v as f32)",
        "fn narrow(v: f64) -> f32 { v as f32 }",
        "macro_rules! narrow { ($v:expr) => { $v as f32 } }",
        "fn bad(v: f64, origin: f64) -> f32 { v as f32 - origin as f32 }",
    ]
    for probe in probes:
        assert _conversion_inventory_text("probe.rs", probe), f"scanner missed {probe}"


def test_dd_encode_has_exactly_the_two_reviewed_split_casts():
    actual = {
        site
        for site in _complete_conversion_inventory()
        if site[0] == "src/core/dd.rs" and site[1] == "from_f64"
    }
    assert actual == SANCTIONED_DD_SPLITS


def test_reviewed_checked_reader_inventory_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_INVENTORY_TRANSITION
    assert len(sites) == transition["current_count"] == EXPECTED_CONVERSION_COUNT
    assert _inventory_digest(sites) == EXPECTED_CONVERSION_SHA256
    assert transition["added"] in sites
    assert transition["removed"] not in sites


def test_reviewed_anamnesis_function_ownership_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_ANAMNESIS_INVENTORY_TRANSITION
    assert len(sites) == transition["base_count"] == EXPECTED_CONVERSION_COUNT
    assert _inventory_digest(sites) == transition["result_digest"]
    for ordinal, statement in enumerate(transition["statements"], start=1):
        removed = (
            transition["path"],
            transition["removed_function"],
            "as_f32",
            ordinal,
            statement,
        )
        added = (
            transition["path"],
            transition["added_function"],
            "as_f32",
            ordinal,
            statement,
        )
        assert removed not in sites
        assert added in sites


def test_reviewed_helios_inventory_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_HELIOS_INVENTORY_TRANSITION
    assert len(sites) == transition["base_count"] == EXPECTED_CONVERSION_COUNT
    assert _inventory_digest(sites) == transition["result_digest"]
    for added in transition["added_sites"]:
        assert added in sites
    for removed in transition["removed_sites"]:
        assert removed not in sites


def test_anchor_narrow_is_the_only_world_conversion_implementation():
    anchor = _remove_cfg_test_modules(_read(SANCTIONED))
    narrow = _function_body(SANCTIONED, "narrow")
    assert len(re.findall(r"\bas\s+f32\b", narrow)) == 1
    assert "value as f32" in re.sub(r"\s+", " ", narrow)
    assert anchor.count("Self::narrow(") == 6

    position = _function_body(SANCTIONED, "to_render_vec3")
    assert "p - self.origin" in re.sub(r"\s+", " ", position)
    assert position.count("Self::narrow(") == 3
    direction = _function_body(SANCTIONED, "direction_to_render")
    assert direction.count("Self::narrow(") == 3


def test_anchor_dd_split_is_a_named_non_narrowing_crossing():
    body = re.sub(r"\s+", " ", _function_body(SANCTIONED, "to_dd"))
    assert "DDVec3::from_dvec3(p)" in body
    assert "Self::narrow(" not in body
    assert " as f32" not in body


def test_each_viewer_world_route_calls_its_active_anchor_in_the_same_function():
    routes = {
        ("src/viewer/viewer_types.rs", "view"): "self.anchor.view_look_at(",
        ("src/viewer/viewer_types.rs", "render_eye"): "self.anchor.to_render_vec3(",
        ("src/viewer/render/main_loop/frame_anchor.rs", "anchored_object_model"): "frame.anchor.model_offset(",
        ("src/viewer/pointcloud/state.rs", "packed_point"): "anchor.to_render_vec3(",
        ("src/viewer/terrain/vector_overlay.rs", "repack_source_vertices"): "anchor.to_render_vec3(",
        ("src/labels/mod.rs", "update_with_camera_anchored"): "anchor.to_render_vec3(",
        ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state"): "frame.anchor.to_render_vec3(",
        ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state"): "frame.anchor.to_render_vec3(",
        ("src/viewer/input/viewer_input.rs", "pick_at_screen"): ".to_world_from_render_f64(",
    }
    for (rel, function), required_call in routes.items():
        body = re.sub(r"\s+", "", _function_body(rel, function))
        normalized_call = re.sub(r"\s+", "", required_call)
        assert normalized_call in body, f"{rel}::{function} lacks {required_call}"


def test_cityjson_and_viewer_absolute_storage_types_are_explicitly_f64():
    assert "pub positions: Vec<f64>" in _read("src/import/cityjson/types.rs")
    assert "pub(crate) object_translation: glam::DVec3" in _read("src/viewer/viewer_struct.rs")
    assert "pub world_pos: DVec3" in _read("src/labels/types.rs")
    assert "pub position: DVec3" in _read("src/viewer/pointcloud/types.rs")


def test_public_camera_helper_preserves_earth_scale_offset():
    import numpy as np
    from forge3d import _forge3d

    local = np.asarray(
        _forge3d.camera_look_at((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    earth = np.asarray(
        _forge3d.camera_look_at(
            (6_378_137.0, 2_000.0, -3_000.0),
            (6_378_147.0, 2_000.0, -3_000.0),
            (0.0, 1.0, 0.0),
        )
    )
    np.testing.assert_allclose(earth, local, rtol=0.0, atol=1e-6)
