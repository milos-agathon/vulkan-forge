use super::*;
use crate::verify::contract::parse_contract;

fn contract(inputs: &str, output: &str) -> EntryContract {
    parse_contract(&format!(
        "[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [{inputs}]\noutputs = [\"return:{output}\"]\n"
    ))
    .unwrap()
    .entries
    .remove(0)
}

#[test]
fn validated_ir_rejects_a_contract_that_allows_zero_denominator() {
    let source = "fn main(denom: f32) -> f32 { return 1.0 / denom; }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:denom:-1:1\"", "-3.4e38:3.4e38"),
    )
    .unwrap();
    assert_eq!(proof.alarms[0].kind, "possible_nan_or_inf");
    assert_eq!(proof.alarms[0].line, 1);
}

#[test]
fn strict_positive_branch_excludes_zero_denominator() {
    let source =
        "fn main(denom: f32) -> f32 { if denom > 0.0 { return 1.0 / denom; } return 0.0; }";
    let proof = prove_wgsl(source, "main", &contract("\"value:denom:0:1\"", "0:3.4e38")).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn positive_fraction_does_not_bypass_subnormal_ftz() {
    let source =
        "fn main(value: f32, delta: f32) -> f32 { return value / (value + delta * delta); }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:value:1e-45:1e-45\", \"value:delta:0:0\"", "0:1"),
    )
    .unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn positive_fraction_preserves_native_division_ulp_margin_near_one() {
    let source =
        "fn main(value: f32, delta: f32) -> f32 { return value / (value + delta * delta); }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:value:1:1\", \"value:delta:0:0\"", "0:1"),
    )
    .unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "output_range"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn explicitly_allowed_nan_output_still_rejects_infinity() {
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:dummy:0:0\"]\noutputs = [\"return:0:1:nan\"]\n").unwrap();
    let nan = prove_wgsl(
        "fn main(dummy: f32) -> f32 { return bitcast<f32>(0x7fc00000u); }",
        "main",
        &parsed.entries[0],
    )
    .unwrap();
    assert!(nan.alarms.is_empty(), "{:?}", nan.alarms);
    let inf = prove_wgsl(
        "fn main(dummy: f32) -> f32 { return bitcast<f32>(0x7f800000u); }",
        "main",
        &parsed.entries[0],
    )
    .unwrap();
    assert!(inf.alarms.iter().any(|alarm| alarm.kind == "output_range"));
}

#[test]
fn non_convergent_loop_emits_widening_alarm() {
    let source = "fn main(value: f32) -> f32 { var x = value; loop { x = x + 1.0; } return x; }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:value:0:0\"", "-3.4e38:3.4e38"),
    )
    .unwrap();
    assert!(proof
        .alarms
        .iter()
        .any(|alarm| alarm.kind == "loop_widening"));
}

#[test]
fn vector_arithmetic_is_proved_from_component_ranges() {
    let source = "fn main(value: vec3<f32>) -> vec3<f32> { return value + vec3<f32>(1.0); }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:value:0:1\"", "0.999999:2.000001"),
    )
    .unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn calls_locals_and_branches_are_interpreted() {
    let source = r#"
fn safe_rcp(x: f32) -> f32 {
var magnitude = x;
if magnitude < 0.0 { magnitude = -magnitude; }
return 1.0 / max(magnitude, 0.25);
}
fn main(x: f32) -> f32 { return safe_rcp(x); }
"#;
    let proof = prove_wgsl(source, "main", &contract("\"value:x:-1:1\"", "0:4.00001")).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn deterministic_division_accepts_a_positive_max_guard() {
    let source = format!(
        "{}\nfn main(lo: f32, hi: f32) -> f32 {{\n    let range = max(hi - lo, 0.000001);\n    return det_div(1.0, range);\n}}",
        include_str!("../../shaders/includes/determinism.wgsl")
    );
    let proof = prove_wgsl(
        &source,
        "main",
        &contract(
            "\"value:lo:-1000:1000\", \"value:hi:-1000:1000\"",
            "-1:1000001",
        ),
    )
    .unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn entry_structs_and_uniform_members_are_seeded_by_name() {
    let source = r#"
struct Uniforms { denominator: f32 }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct Input { @location(0) value: f32 }
@fragment fn main(input: Input) -> @location(0) f32 {
return input.value / max(uniforms.denominator, 0.25);
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:input.value:0:1\", \"uniform:uniforms.denominator:0:1\"]\noutputs = [\"location0:0:4.00001\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn vector_components_are_seeded_by_name() {
    let source = r#"
struct Uniforms { values: vec4<f32> }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
fn main() -> f32 { return 1.0 / uniforms.values.y; }
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"uniform:uniforms.values.x:0:0\", \"uniform:uniforms.values.y:2:2\", \"uniform:uniforms.values.z:0:0\", \"uniform:uniforms.values.w:0:0\"]\noutputs = [\"return:0.499999:0.500001\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn false_equality_edge_excludes_zero_for_division() {
    let source =
        "fn main(denom: f32) -> f32 { if denom == 0.0 { return 0.0; } return 1.0 / denom; }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:denom:-1:1\"", "-1e38:1e38"),
    )
    .unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn bounded_loop_reaches_a_fixed_point() {
    let source = r#"fn main(x: f32) -> f32 {
        var value = x;
        for (var i = 0u; i < 3u; i = i + 1u) { value = value * 0.5; }
        return value;
    }"#;
    let proof = prove_wgsl(source, "main", &contract("\"value:x:0:1\"", "0:1.000001")).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn dynamic_buffer_access_uses_the_declared_length() {
    let source = "@group(0) @binding(0) var<storage, read> values: array<f32>; fn main(index: u32) -> f32 { return values[index]; }";
    let parse = |maximum: u32| {
        parse_contract(&format!("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:index:0:{maximum}\", \"buffer:values:0:1:4\"]\noutputs = [\"return:0:1\"]\n")).unwrap().entries.remove(0)
    };
    let safe = prove_wgsl(source, "main", &parse(3)).unwrap();
    assert!(safe.alarms.is_empty(), "{:?}", safe.alarms);
    let proof = prove_wgsl(source, "main", &parse(4)).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_oob"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn texture_load_uses_declared_dimensions_and_sample_range() {
    let source = "@group(0) @binding(0) var image: texture_2d<f32>; fn main(coord: vec2<i32>) -> vec4<f32> { return textureLoad(image, coord, 0); }";
    let parse = |maximum: i32| {
        parse_contract(&format!("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:coord:0:{maximum}\", \"texture:image:0:1:2:min4:min4\"]\noutputs = [\"return:0:1\"]\n")).unwrap().entries.remove(0)
    };
    let safe = prove_wgsl(source, "main", &parse(3)).unwrap();
    assert!(safe.alarms.is_empty(), "{:?}", safe.alarms);
    let proof = prove_wgsl(source, "main", &parse(4)).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_oob"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn array_texture_load_uses_runtime_layer_count_guard() {
    let guarded = r#"
@group(0) @binding(0) var image: texture_2d_array<f32>;
fn main(layer: u32) -> vec4<f32> {
    if (layer >= textureNumLayers(image)) {
        return vec4<f32>(0.0);
    }
    return textureLoad(image, vec2<i32>(0), i32(layer), 0);
}
"#;
    let contract = parse_contract(
        "[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:layer:0:3\", \"texture:image:0:1:3:image.width:image.height:image.layers\"]\noutputs = [\"return:0:1\"]\n",
    )
    .unwrap()
    .entries
    .remove(0);
    let proof = prove_wgsl(guarded, "main", &contract).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);

    let unguarded = guarded.replace(
        "    if (layer >= textureNumLayers(image)) {\n        return vec4<f32>(0.0);\n    }\n",
        "",
    );
    let proof = prove_wgsl(&unguarded, "main", &contract).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_oob"),
        "{:?}",
        proof.alarms
    );

    for axis in ["x", "y"] {
        let wrong_dimension_guard = guarded.replace(
            "textureNumLayers(image)",
            &format!("textureDimensions(image).{axis}"),
        );
        let proof = prove_wgsl(&wrong_dimension_guard, "main", &contract).unwrap();
        assert!(
            proof
                .alarms
                .iter()
                .any(|alarm| alarm.kind == "possible_oob"),
            "{axis}-dimension guard incorrectly proved the array layer: {:?}",
            proof.alarms
        );
    }

    let clamped = r#"
@group(0) @binding(0) var image: texture_2d_array<f32>;
fn main(layer: u32) -> vec4<f32> {
    let upper = max(i32(textureNumLayers(image)) - 1, 0);
    let safe_layer = clamp(i32(layer), 0, upper);
    return textureLoad(image, vec2<i32>(0), safe_layer, 0);
}
"#;
    let proof = prove_wgsl(clamped, "main", &contract).unwrap();
    assert!(
        proof.alarms.is_empty(),
        "actual layer clamp was not proved: {:?}",
        proof.alarms
    );
    for axis in ["x", "y"] {
        let wrong_dimension_clamp = clamped.replace(
            "textureNumLayers(image)",
            &format!("textureDimensions(image).{axis}"),
        );
        let proof = prove_wgsl(&wrong_dimension_clamp, "main", &contract).unwrap();
        assert!(
            proof
                .alarms
                .iter()
                .any(|alarm| alarm.kind == "possible_oob"),
            "{axis}-dimension clamp incorrectly proved the array layer: {:?}",
            proof.alarms
        );
    }
}

#[test]
fn rounded_unit_coordinate_is_bounded_by_texture_dimensions() {
    let source = r#"
@group(0) @binding(0) var image: texture_2d<f32>;
fn main(t: f32) -> vec4<f32> {
    let dims = textureDimensions(image);
    let max_x = max(i32(dims.x) - 1, 0);
    let x = i32(round(clamp(t, 0.0, 1.0) * f32(max_x)));
    return textureLoad(image, vec2<i32>(x, 0), 0);
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:t:-10:10\", \"texture:image:0:1:2:image.width:image.height\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn independently_clamped_neighbor_axes_compose_to_an_in_bounds_coordinate() {
    let source = r#"
@group(0) @binding(0) var image: texture_2d<f32>;
fn main(t: vec2<f32>) -> vec4<f32> {
    let dims = textureDimensions(image);
    let max_x = max(i32(dims.x) - 1, 0);
    let max_y = max(i32(dims.y) - 1, 0);
    let x0 = i32(floor(clamp(t.x, 0.0, 1.0) * f32(max_x)));
    let y0 = i32(floor(clamp(t.y, 0.0, 1.0) * f32(max_y)));
    let x1 = clamp(x0 + 1, 0, max_x);
    let y1 = clamp(y0 + 1, 0, max_y);
    return textureLoad(image, vec2<i32>(x1, y1), 0);
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:t:-10:10\", \"texture:image:0:1:2:image.width:image.height\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn portable_height_bilinear_load_is_bounded_and_range_preserving() {
    let source = r#"
@group(0) @binding(0) var image: texture_2d<f32>;
fn main(uv: vec2<f32>, lod: f32) -> f32 {
    let last_level = i32(textureNumLevels(image)) - 1;
    let level = clamp(i32(floor(lod + 0.5)), 0, last_level);
    let dims = textureDimensions(image, level);
    let max_x = max(i32(dims.x) - 1, 0);
    let max_y = max(i32(dims.y) - 1, 0);
    let texel_x = clamp(uv.x, 0.0, 1.0) * f32(max_x);
    let texel_y = clamp(uv.y, 0.0, 1.0) * f32(max_y);
    let x0 = i32(floor(texel_x));
    let y0 = i32(floor(texel_y));
    let x1 = clamp(x0 + 1, 0, max_x);
    let y1 = clamp(y0 + 1, 0, max_y);
    let blend = clamp(
        vec2<f32>(texel_x - f32(x0), texel_y - f32(y0)),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let h00 = textureLoad(image, vec2<i32>(x0, y0), level).r;
    let h10 = textureLoad(image, vec2<i32>(x1, y0), level).r;
    let h01 = textureLoad(image, vec2<i32>(x0, y1), level).r;
    let h11 = textureLoad(image, vec2<i32>(x1, y1), level).r;
    let top = mix(h00, h10, blend.x);
    let bottom = mix(h01, h11, blend.x);
    return mix(top, bottom, blend.y);
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:uv:-10:10\", \"value:lod:-100:100\", \"texture:image:0:1:2:image.width:image.height\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn entry_point_output_ranges_are_checked() {
    let source = "@fragment fn main(@location(0) value: f32) -> @location(0) f32 { return value; }";
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:value:0:2\"]\noutputs = [\"location0:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "output_range"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn private_global_initializer_seeds_entry_output() {
    let source = "var<private> value: u32 = 3u; @fragment fn main(@location(0) dummy: u32) -> @location(0) u32 { return value + dummy; }";
    let parsed = parse_contract(
        "[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:dummy:0:0\"]\noutputs = [\"location0:3:3\"]\n",
    )
    .unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);

    let mutant = source.replace("= 3u", "= 4u");
    let proof = prove_wgsl(&mutant, "main", &parsed.entries[0]).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "output_range"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn vector_math_broadcasts_scalar_bounds() {
    let source = "fn main(value: vec3<f32>, factor: f32) -> vec3<f32> { return clamp(mix(value, vec3<f32>(1.0), factor), vec3<f32>(0.0), vec3<f32>(1.0)); }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:value:-1:1\", \"value:factor:0:1\"", "0:1"),
    )
    .unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn switch_and_array_length_are_interpreted() {
    let source = r#"
@group(0) @binding(0) var<storage, read> values: array<f32>;
fn main(selector: u32) -> f32 {
    var index = 0u;
    switch selector {
        case 0u: { index = 1u; }
        default: { index = arrayLength(&values) - 1u; }
    }
    return values[index];
}

"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:selector:0:1\", \"buffer:values:0:1:4\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn runtime_array_length_guard_proves_dynamic_access() {
    let source = r#"
@group(0) @binding(0) var<storage, read> values: array<f32>;
fn main(index: u32) -> f32 {
    if index >= arrayLength(&values) { return 0.0; }
    return values[index];
}

"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:index:0:4294967000\", \"buffer:values:0:1:dynamic\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn pointer_argument_helpers_are_interpreted_and_copied_back() {
    let source = r#"
fn next(state: ptr<function, u32>) -> f32 {
    var value = *state;
    value = value ^ (value << 5u);
    *state = value;
    return f32(value) / 4294967296.0;
}
fn main(seed: u32) -> f32 {
    var state = seed;
    return next(&state);
}"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:seed:0:4294967295\"]\noutputs = [\"return:0:1.001\"]\n")
        .unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn image_store_checks_coordinates_and_written_range() {
    let source = r#"
@group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    textureStore(image, gid.xy, vec4<f32>(0.5));
}
"#;
    let parse = |maximum: u32| {
        parse_contract(&format!("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:gid:0:{maximum}\", \"texture:image:0:1:2:min4:min4\"]\noutputs = [\"image:0:1\"]\n")).unwrap().entries.remove(0)
    };
    let safe = prove_wgsl(source, "main", &parse(3)).unwrap();
    assert!(safe.alarms.is_empty(), "{:?}", safe.alarms);
    let unsafe_proof = prove_wgsl(source, "main", &parse(4)).unwrap();
    assert!(
        unsafe_proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_oob"),
        "{:?}",
        unsafe_proof.alarms
    );
}

#[test]
fn texture_level_queries_are_bounded_integers() {
    let source = "@group(0) @binding(0) var image: texture_2d<f32>; fn main() -> u32 { return textureNumLevels(image); }";
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"texture:image:0:1:2:min4:min4\"]\noutputs = [\"return:1:32\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn integer_shift_ranges_keep_bit_trick_seeds_finite() {
    let source = "fn main(x: f32) -> f32 { let xc = max(x, 1.17549435e-38); return bitcast<f32>(0x5f3759dfu - (bitcast<u32>(xc) >> 1u)); }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:x:0.000001:65536\"", "-3.4e38:3.4e38"),
    )
    .unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn exceptional_math_results_are_rejected() {
    let source = "fn main(x: f32) -> f32 { return sqrt(x); }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:x:-1:1\"", "-3.4e38:3.4e38"),
    )
    .unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn sqrt_of_nonnegative_interval_is_finite() {
    let source = "fn main(x: f32) -> f32 { return sqrt(x); }";
    let proof = prove_wgsl(source, "main", &contract("\"value:x:0:4\"", "0:2.00001")).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn normalize_uses_any_definitely_nonzero_component() {
    let source =
        "fn main(x: f32, y: f32) -> vec3<f32> { return normalize(vec3<f32>(x, y, -1.0)); }";
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:x:-16:16\", \"value:y:-16:16\"", "-1.001:1.001"),
    )
    .unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn count_within_invariant_bounds_dynamic_prefix_indices() {
    let source = r#"
@group(0) @binding(0) var<storage, read> values: array<f32>;
fn main(count: u32) -> f32 {
    var last = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        last = values[i];
    }
    return last;
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:count:0:4294967295\", \"buffer:values:0:1:dynamic\"]\noutputs = [\"return:0:1\"]\ninvariants = [\"count_within:count:values\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn uninitialized_function_locals_start_at_wgsl_zero() {
    let source = r#"
struct Pair { a: f32, b: u32 }
fn main(x: f32) -> Pair {
    var value: Pair;
    return value;
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:x:0:0\"]\noutputs = [\"a:0:0\", \"b:0:0\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn distance_and_nonzero_branch_prove_normalize() {
    let source = r#"
fn main(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    var h = a - b;
    if all(h == vec3<f32>(0.0)) { h = vec3<f32>(1.0, 0.0, 0.0); }
    return normalize(h);
}"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:a:-1:1\", \"value:b:-1:1\"]\noutputs = [\"return:-1.001:1.001\"]\ninvariants = [\"distance_ge:a:b:0.000001\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn length_guard_proves_normalize() {
    let source = r#"
fn main(a: vec3<f32>) -> vec3<f32> {
    let n = length(a);
    if n < 0.000001 { return vec3<f32>(0.0); }
    return normalize(a);
}"#;
    let proof = prove_wgsl(
        source,
        "main",
        &contract("\"value:a:-1:1\"", "-1.001:1.001"),
    )
    .unwrap();
    assert!(proof.alarms.is_empty(), "{:?}", proof.alarms);
}

#[test]
fn det_body_mutation_invalidates_summary() {
    let source = r#"
fn det_barrier(x: f32) -> f32 { return x; }
fn det_fma(a: f32, b: f32, c: f32) -> f32 { return a * b + c; }
fn det_mix(a: f32, b: f32, t: f32) -> f32 { return mix(a, b, t); }
fn det_rcp(x: f32) -> f32 { return 1.0 / max(abs(x), 0.000001); }
fn det_div(a: f32, b: f32) -> f32 { return a / max(abs(b), 0.000001); }
fn det_sqrt(x: f32) -> f32 { return 1.0 / (x - x); }
fn det_normalize3(v: vec3<f32>) -> vec3<f32> { return normalize(v); }
fn det_pow(a: f32, b: f32) -> f32 { return pow(a, b); }
fn det_sin(x: f32) -> f32 { return sin(x); }
fn det_atan2(y: f32, x: f32) -> f32 { return atan2(y, x); }
fn det_acos(x: f32) -> f32 { return acos(x); }
"#;
    let proof = prove_wgsl(
        source,
        "det_sqrt",
        &contract("\"value:x:0:1\"", "-3.4e38:3.4e38"),
    )
    .unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn terrain_renderer_source_hash_is_pinned() {
    let actual = super::engine::stable_hash(crate::shader_sources::terrain().as_bytes());
    assert_eq!(
        actual,
        super::engine::PINNED_TERRAIN_SOURCE_HASH,
        "{actual:#018x}"
    );
}

#[test]
fn determinism_source_hash_is_pinned() {
    let actual = super::engine::stable_hash(
        include_str!("../../shaders/includes/determinism.wgsl").as_bytes(),
    );
    assert_eq!(
        actual,
        super::engine::PINNED_DETERMINISM_SOURCE_HASH,
        "{actual:#018x}"
    );
}

#[test]
fn hybrid_renderer_source_hash_is_pinned() {
    let actual = super::engine::stable_hash(crate::shader_sources::hybrid_kernel().as_bytes());
    assert_eq!(
        actual,
        super::engine::PINNED_HYBRID_KERNEL_SOURCE_HASH,
        "{actual:#018x}"
    );
}

#[test]
fn source_hash_ignores_platform_line_endings() {
    assert_eq!(
        super::engine::stable_hash(b"first\nsecond\n"),
        super::engine::stable_hash(b"first\r\nsecond\r\n")
    );
}

#[test]
fn xorshift_body_mutation_invalidates_callers() {
    let source = r#"
fn xorshift32(state: ptr<function, u32>) -> f32 {
    let denom = f32(*state - *state);
    return 1.0 / denom;
}
fn main(seed: u32) -> f32 {
    var state = seed;
    return xorshift32(&state);
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:seed:0:15\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_nan_or_inf"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn negative_signed_buffer_index_is_oob() {
    let source = r#"
@group(0) @binding(0) var<storage, read> values: array<f32>;
fn main(index: i32) -> f32 {
    if index < i32(arrayLength(&values)) {
        return values[index];
    }
    return 0.0;
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:index:-1:3\", \"buffer:values:0:1:dynamic\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_oob"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn negative_signed_texture_coordinate_is_oob() {
    let source = r#"
@group(0) @binding(0) var image: texture_2d<f32>;
fn main(coord: vec2<i32>) -> vec4<f32> {
    let size = textureDimensions(image);
    if coord.x < i32(size.x) && coord.y < i32(size.y) {
        return textureLoad(image, coord, 0);
    }
    return vec4<f32>(0.0);
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:coord:-1:3\", \"texture:image:0:1:2:min4:min4\"]\noutputs = [\"return:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(
        proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_oob"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn cast_coordinates_preserve_guard_relations() {
    let source = r#"
@group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
fn main(gid: vec3<u32>, width: u32, height: u32) {
    if gid.x >= width || gid.y >= height { return; }
    textureStore(image, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.5));
}
"#;
    let parsed = parse_contract("[module]\npath = \"tests/data/shader_proofs/fixture.wgsl\"\nowner = \"test\"\nexpiry = \"2027-01-17\"\n\n[[entry]]\nname = \"main\"\nproof_status = \"proven\"\ninputs = [\"value:gid:0:16384\", \"value:width:1:16384\", \"value:height:1:16384\", \"texture:image:0:1:2:width:height\"]\noutputs = [\"image:0:1\"]\n").unwrap();
    let proof = prove_wgsl(source, "main", &parsed.entries[0]).unwrap();
    assert!(
        !proof
            .alarms
            .iter()
            .any(|alarm| alarm.kind == "possible_oob"),
        "{:?}",
        proof.alarms
    );
}

#[test]
fn current_determinism_lemmas_are_proved_through_ir() {
    let source = include_str!("../../shaders/includes/determinism.wgsl");
    let parsed =
        parse_contract(include_str!("../../../shaders/contracts/determinism.toml")).unwrap();
    let mut failures = Vec::new();
    for contract in &parsed.entries {
        let proof = prove_wgsl(source, &contract.name, contract).unwrap();
        if !proof.alarms.is_empty() {
            failures.push((contract.name.as_str(), proof.alarms));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn inspect_deterministic_kernel_ir_shape() {
    let module =
        naga::front::wgsl::parse_str(include_str!("../../shaders/includes/determinism.wgsl"))
            .unwrap();
    let mut summaries = Vec::new();
    for (_, function) in module.functions.iter().filter(|(_, function)| {
        matches!(
            function.name.as_deref(),
            Some("det_rcp" | "det_inverse_sqrt")
        )
    }) {
        let binary = function
            .expressions
            .iter()
            .filter(|(_, expression)| matches!(expression, naga::Expression::Binary { .. }))
            .count();
        let bitcast = function
            .expressions
            .iter()
            .filter(|(_, expression)| {
                matches!(expression, naga::Expression::As { convert: None, .. })
            })
            .count();
        let math = function
            .expressions
            .iter()
            .filter(|(_, expression)| matches!(expression, naga::Expression::Math { .. }))
            .count();
        let calls = function
            .body
            .iter()
            .filter(|statement| matches!(statement, naga::Statement::Call { .. }))
            .count();
        summaries.push(format!(
            "{} binary={binary} bitcast={bitcast} math={math} calls={calls}",
            function.name.as_deref().unwrap()
        ));
    }
    panic!("{}", summaries.join("\n"));
}
