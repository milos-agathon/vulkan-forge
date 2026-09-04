"""Occurrence-level inventory of every interactive-viewer matrix operation."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "src/viewer", ROOT / "src/shadows")

EXPECTED = {
    ("src/shadows/csm_renderer.rs", "compute_cascade", "matrix_compose", 1),
    ("src/shadows/csm_renderer.rs", "compute_cascade", "projection_ctor", 1),
    ("src/shadows/csm_renderer.rs", "update_cascades", "matrix_compose", 1),
    ("src/shadows/csm_renderer.rs", "update_cascades", "matrix_inverse", 1),
    ("src/viewer/camera_controller.rs", "view_matrix", "anchor_view", 1),
    ("src/viewer/camera_controller.rs", "view_matrix", "anchor_view", 2),
    ("src/viewer/camera_controller.rs", "view_matrix", "view_delegate", 1),
    ("src/viewer/camera_controller.rs", "view_matrix", "view_delegate", 2),
    ("src/viewer/input/viewer_input.rs", "pick_at_screen", "frame_projection", 1),
    ("src/viewer/input/viewer_input.rs", "pick_at_screen", "frame_view", 1),
    ("src/viewer/input/viewer_input.rs", "pick_at_screen", "matrix_compose", 1),
    ("src/viewer/input/viewer_input.rs", "pick_at_screen", "matrix_inverse", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "refresh_after_rebase", "frame_projection", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "refresh_after_rebase", "frame_view", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "refresh_after_rebase", "matrix_compose", 1),
    ("src/viewer/render/main_loop/frame_anchor.rs", "refresh_after_rebase", "previous_vp_write", 1),
    ("src/viewer/render/main_loop/frame_setup.rs", "prepare_render_frame", "frame_projection", 1),
    ("src/viewer/render/main_loop/frame_setup.rs", "prepare_render_frame", "frame_view", 1),
    ("src/viewer/render/main_loop/frame_setup.rs", "prepare_render_frame", "matrix_inverse", 1),
    ("src/viewer/render/main_loop/frame_setup.rs", "prepare_render_frame", "matrix_inverse", 2),
    ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog", "frame_projection", 1),
    ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog", "frame_view", 1),
    ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog", "matrix_compose", 1),
    ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog", "matrix_inverse", 1),
    ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog", "matrix_inverse", 2),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "frame_projection", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "frame_view", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "matrix_compose", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "matrix_compose", 2),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "matrix_inverse", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "matrix_inverse", 2),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "previous_vp_read", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass", "previous_vp_write", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_object_overlay", "frame_projection", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_object_overlay", "frame_view", 1),
    ("src/viewer/render/main_loop/geometry/pass.rs", "render_object_overlay", "matrix_compose", 1),
    ("src/viewer/render/main_loop/secondary.rs", "render_secondary_paths", "frame_view_projection", 1),
    ("src/viewer/render/main_loop/secondary.rs", "render_secondary_paths", "frame_view_projection", 2),
    ("src/viewer/render/main_loop/snapshot_sky.rs", "encode_snapshot_sky", "frame_projection", 1),
    ("src/viewer/render/main_loop/snapshot_sky.rs", "encode_snapshot_sky", "frame_view", 1),
    ("src/viewer/render/main_loop/snapshot_sky.rs", "encode_snapshot_sky", "matrix_inverse", 1),
    ("src/viewer/render/main_loop/snapshot_sky.rs", "encode_snapshot_sky", "matrix_inverse", 2),
    ("src/viewer/state/labels.rs", "update_labels_for_frame", "frame_view_projection", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "frame_projection", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "frame_view", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "matrix_compose", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "matrix_compose", 2),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "matrix_inverse", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "matrix_inverse", 2),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "previous_vp_read", 1),
    ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once", "previous_vp_write", 1),
    ("src/viewer/terrain/render/offscreen/effects.rs", "apply_snapshot_effects", "matrix_inverse", 1),
    ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state", "frame_projection", 1),
    ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state", "frame_view", 1),
    ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state", "matrix_compose", 1),
    ("src/viewer/terrain/render/screen/effects.rs", "apply_screen_effects", "matrix_inverse", 1),
    ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state", "frame_projection", 1),
    ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state", "frame_view", 1),
    ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state", "matrix_compose", 1),
    ("src/viewer/viewer_types.rs", "extend_with_mesh", "matrix_inverse", 1),
    ("src/viewer/viewer_types.rs", "projection", "projection_ctor", 1),
    ("src/viewer/viewer_types.rs", "view", "anchor_view", 1),
    ("src/viewer/viewer_types.rs", "view_projection", "matrix_compose", 1),
}

NO_MATRIX_FILES = {
    "src/viewer/terrain/dof/pass/execute.rs",
    "src/viewer/terrain/overlay/stack/composite.rs",
    "src/viewer/terrain/volume_density.rs",
}


def _strip(text: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|r#*\".*?\"#*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.S,
    )
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _remove_cfg_test_modules(text: str) -> str:
    text = _strip(text)
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


def _matrix_variables(function_text: str) -> set[str]:
    variables = set(
        re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?:&\s*)?(?:D?Mat[234])\b", function_text)
    )
    hints = re.compile(r"(?:Mat[234]::|\.view\s*\(|\.projection\s*\(|\.view_projection\s*\(|\.view_matrix\s*\(|\.inverse\s*\(|\.transpose\s*\()")
    assignments = list(
        re.finditer(
            r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=;]+)?=\s*([^;]+);",
            function_text,
            re.S,
        )
    )
    changed = True
    while changed:
        changed = False
        for match in assignments:
            name, rhs = match.group(1), match.group(2)
            operands = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs))
            if hints.search(rhs) or operands.intersection(variables):
                if name not in variables:
                    variables.add(name)
                    changed = True
    return variables


def _inventory_function(rel: str, function: str, body: str, base: int):
    matches = []
    fixed = {
        "look_at_ctor": re.compile(r"\b(?:D?Mat4)::look_at_[a-z_]+\s*\("),
        "projection_ctor": re.compile(r"\b(?:D?Mat4)::(?:perspective|orthographic)_[a-z_]+\s*\("),
        "anchor_view": re.compile(r"\b(?:self\s*\.\s*)?anchor\s*\.\s*view_look_at\s*\("),
        "frame_view": re.compile(r"\bframe\s*\.\s*view\s*\("),
        "frame_projection": re.compile(r"\bframe\s*\.\s*projection\s*\("),
        "frame_view_projection": re.compile(r"\bframe\s*\.\s*view_projection\s*\("),
        "view_delegate": re.compile(r"\b(?:self\s*\.\s*)?(?:orbit|fps|camera)\s*\.\s*view_matrix\s*\("),
    }
    for operation, pattern in fixed.items():
        matches.extend((base + match.start(), operation) for match in pattern.finditer(body))

    variables = _matrix_variables(body)
    hint = re.compile(r"(?:view|proj|projection|matrix|model|transform|^p$|^v$|^vp$)", re.I)
    multiply = re.compile(
        r"(?P<left>[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^;()]*\))?)"
        r"\s*\*\s*"
        r"(?P<right>[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^;()]*\))?)",
        re.S,
    )
    for match in multiply.finditer(body):
        left_text = match.group("left")
        right_text = match.group("right")
        left = re.split(r"\s*\.\s*", left_text, maxsplit=1)[0]
        right = re.split(r"\s*\.\s*", right_text, maxsplit=1)[0]
        if (left in variables or hint.search(left_text)) and (
            right in variables or hint.search(right_text)
        ):
            matches.append((base + match.start(), "matrix_compose"))

    inverse = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*inverse\s*\(")
    for match in inverse.finditer(body):
        name = match.group(1)
        if name in variables or hint.search(name):
            matches.append((base + match.start(), "matrix_inverse"))
    for match in re.finditer(r"\(([^;{}]+)\)\s*\.\s*inverse\s*\(", body, re.S):
        operands = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", match.group(1)))
        if operands.intersection(variables) or hint.search(match.group(1)):
            matches.append((base + match.start(), "matrix_inverse"))

    for match in re.finditer(r"\bprev_view_proj(?:_matrix)?\b", body):
        tail = body[match.end() : match.end() + 8]
        if re.match(r"\s*:", tail):
            continue
        operation = "previous_vp_write" if re.match(r"\s*=\s*[^=]", tail) else "previous_vp_read"
        matches.append((base + match.start(), operation))

    return [(position, function, operation) for position, operation in sorted(matches)]


def _inventory_text(rel: str, raw: str):
    text = _remove_cfg_test_modules(raw)
    matches = []
    for start, end, function in _function_spans(text):
        matches.extend(_inventory_function(rel, function, text[start:end], start))
    counters = {}
    sites = []
    for _position, function, operation in sorted(matches):
        key = (function, operation)
        counters[key] = counters.get(key, 0) + 1
        sites.append((rel, function, operation, counters[key]))
    return sites


def inventory_sites():
    sites = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.rs")):
            rel = path.relative_to(ROOT).as_posix()
            sites.extend(_inventory_text(rel, path.read_text(encoding="utf-8")))
    return sites


def _function_body(rel: str, function: str) -> str:
    text = _remove_cfg_test_modules((ROOT / rel).read_text(encoding="utf-8"))
    return next(text[start:end] for start, end, name in _function_spans(text) if name == function)


def test_matrix_operation_inventory_is_exact_by_file_function_operation_ordinal():
    actual = set(inventory_sites())
    assert actual == EXPECTED, (
        f"missing={sorted(EXPECTED - actual)}\n"
        f"unexpected={sorted(actual - EXPECTED)}\n"
        f"actual={sorted(actual)}"
    )


def test_alias_multiline_inverse_previous_vp_and_duplicate_probes_are_detected():
    probe = """
    fn render(p: Mat4, v: Mat4, view_proj: Mat4) {
        let a = p *\n v;
        let b = p * v;
        let inv = view_proj.inverse();
        let old = self.prev_view_proj;
        self.prev_view_proj = a;
    }
    """
    sites = _inventory_text("probe.rs", probe)
    expected = {
        ("probe.rs", "render", "matrix_compose", 1),
        ("probe.rs", "render", "matrix_compose", 2),
        ("probe.rs", "render", "matrix_inverse", 1),
        ("probe.rs", "render", "previous_vp_read", 1),
        ("probe.rs", "render", "previous_vp_write", 1),
    }
    assert expected.issubset(set(sites)), sites


def test_direct_look_at_is_owned_only_by_anchor_module():
    paths = [ROOT / "src/camera/mod.rs"]
    for root in SCAN_ROOTS:
        paths.extend(sorted(root.rglob("*.rs")))
    pattern = re.compile(r"\b(?:D?Mat4)::look_at_[a-z_]+\s*\(")
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel == "src/camera/anchor.rs":
            continue
        assert not pattern.search(_remove_cfg_test_modules(path.read_text(encoding="utf-8"))), rel


def test_each_matrix_producer_uses_the_frozen_frame_or_explicit_delegate():
    required = {
        ("src/viewer/camera_controller.rs", "view_matrix"): "anchor: &Anchor",
        ("src/viewer/viewer_types.rs", "view"): "self.anchor.view_look_at",
        ("src/viewer/viewer_types.rs", "projection"): "self.fov_deg",
        ("src/viewer/viewer_types.rs", "view_projection"): "self.view()",
        ("src/viewer/viewer_types.rs", "extend_with_mesh"): "transform: Mat4",
        ("src/viewer/render/main_loop/frame_anchor.rs", "refresh_after_rebase"): "frame: FrameCamera",
        ("src/viewer/render/main_loop/geometry/pass.rs", "render_geometry_pass"): "self.current_frame_camera()",
        ("src/viewer/render/main_loop/geometry/pass.rs", "render_object_overlay"): "self.current_frame_camera()",
        ("src/viewer/render/main_loop/geometry/fog.rs", "render_geometry_fog"): "self.current_frame_camera()",
        ("src/viewer/render/main_loop/frame_setup.rs", "prepare_render_frame"): "self.current_frame_camera()",
        ("src/viewer/render/main_loop/secondary.rs", "render_secondary_paths"): "self.current_frame_camera()",
        ("src/viewer/render/main_loop/snapshot_sky.rs", "encode_snapshot_sky"): "frame: FrameCamera",
        ("src/viewer/input/viewer_input.rs", "pick_at_screen"): "self.current_frame_camera()",
        ("src/viewer/state/labels.rs", "update_labels_for_frame"): "frame: crate::viewer::viewer_types::FrameCamera",
        ("src/viewer/state/viewer_helpers/gi/geometry.rs", "render_geometry_to_gbuffer_once"): "self.current_frame_camera()",
        ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state"): "frame: crate::viewer::viewer_types::FrameCamera",
        ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state"): "frame: crate::viewer::viewer_types::FrameCamera",
        ("src/viewer/terrain/render/screen/effects.rs", "apply_screen_effects"): "state.view_proj",
        ("src/viewer/terrain/render/offscreen/effects.rs", "apply_snapshot_effects"): "state.view_proj",
        ("src/shadows/csm_renderer.rs", "update_cascades"): "camera_view: Mat4",
        ("src/shadows/csm_renderer.rs", "compute_cascade"): "inv_view_proj: Mat4",
    }
    inventoried = {(rel, function) for rel, function, _operation, _ordinal in EXPECTED}
    assert inventoried == set(required), f"unclassified producers: {sorted(inventoried - set(required))}"
    for (rel, function), token in required.items():
        body = re.sub(r"\s+", "", _function_body(rel, function))
        assert re.sub(r"\s+", "", token) in body, f"{rel}::{function} lacks {token}"


def test_declared_no_matrix_consumers_remain_matrix_free():
    for rel in NO_MATRIX_FILES:
        assert not _inventory_text(rel, (ROOT / rel).read_text(encoding="utf-8")), rel


def test_generic_object_fog_shadow_uses_viewer_128_byte_model_abi_once():
    types = (ROOT / "src/viewer/viewer_types.rs").read_text(encoding="utf-8")
    init = (ROOT / "src/viewer/init/viewer_new.rs").read_text(encoding="utf-8")
    fog = (ROOT / "src/viewer/render/main_loop/geometry/fog.rs").read_text(
        encoding="utf-8"
    )
    shared = (ROOT / "src/shaders/terrain_shadow_depth.wgsl").read_text(
        encoding="utf-8"
    )

    assert "pub(crate) struct ViewerShadowUniforms" in types
    assert "size_of::<ViewerShadowUniforms>(), 128" in types
    assert "offset_of!(ViewerShadowUniforms, object_model)" in types
    assert init.count("uShadow.object_model * vec4<f32>(inp.pos, 1.0)") == 1
    assert "self.anchored_object_model(frame).to_cols_array_2d()" in fog
    assert "let Some(vb) = self.geom_vb.as_ref() else" in fog
    assert "Size: 112 bytes" in shared


def test_terrain_uniform_wgsl_layouts_and_minimum_bindings_are_exact():
    rust = (ROOT / "src/viewer/terrain/render.rs").read_text(encoding="utf-8")
    simple = (ROOT / "src/viewer/terrain/shader.rs").read_text(encoding="utf-8")
    pbr = (ROOT / "src/viewer/terrain/shader_pbr/terrain_pbr.wgsl").read_text(
        encoding="utf-8"
    )
    shadow = (
        ROOT / "src/viewer/terrain/shader_pbr/terrain_shadow_depth.wgsl"
    ).read_text(encoding="utf-8")
    core = (ROOT / "src/viewer/terrain/scene/core.rs").read_text(encoding="utf-8")
    pbr_layout = (ROOT / "src/viewer/terrain/scene/pbr_compute.rs").read_text(
        encoding="utf-8"
    )
    shadow_layout = (ROOT / "src/viewer/terrain/scene/pipeline_init.rs").read_text(
        encoding="utf-8"
    )
    viewer_shadow_layout = (ROOT / "src/viewer/init/viewer_new.rs").read_text(
        encoding="utf-8"
    )

    for name, size, origin, span in (
        ("TerrainUniforms", 160, 144, 152),
        ("ShadowPassUniforms", 128, 112, 120),
        ("TerrainPbrUniforms", 256, 240, 248),
    ):
        assert f"size_of::<{name}>()];" in rust
        assert f"offset_of!({name}, render_origin_xz)];" in rust
        assert f"offset_of!({name}, render_span_xz)];" in rust
        assert f"const _: [(); {size}]" in rust
        assert f"const _: [(); {origin}]" in rust
        assert f"const _: [(); {span}]" in rust

    def fields(text: str, struct_name: str) -> list[tuple[str, str]]:
        body = re.search(rf"struct\s+{struct_name}\s*\{{(.*?)\}}", text, re.S)
        assert body is not None, struct_name
        return re.findall(r"^\s*([A-Za-z_]\w*)\s*:\s*([^,]+),", body.group(1), re.M)

    assert fields(simple, "Uniforms")[-2:] == [
        ("render_origin_xz", "vec2<f32>"),
        ("render_span_xz", "vec2<f32>"),
    ]
    assert fields(pbr, "Uniforms")[-2:] == [
        ("render_origin_xz", "vec2<f32>"),
        ("render_span_xz", "vec2<f32>"),
    ]
    assert fields(shadow, "ShadowPassUniforms")[-2:] == [
        ("render_origin_xz", "vec2<f32>"),
        ("render_span_xz", "vec2<f32>"),
    ]
    assert "min_binding_size: wgpu::BufferSize::new(160)" in core
    assert "min_binding_size: wgpu::BufferSize::new(256)" in pbr_layout
    assert "min_binding_size: wgpu::BufferSize::new(128)" in shadow_layout
    assert "min_binding_size: wgpu::BufferSize::new(128)" in viewer_shadow_layout

    mutated = simple.replace(
        "render_origin_xz: vec2<f32>",
        "render_origin_xz: vec4<f32>",
        1,
    )
    assert fields(mutated, "Uniforms")[-2:] != fields(simple, "Uniforms")[-2:]
