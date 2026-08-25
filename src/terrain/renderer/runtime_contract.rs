use super::OverlayUniforms;
use crate::core::shader_contract_runtime::RuntimeContractObservation;

pub(super) fn build_observation(
    scene: &str,
    terrain: &[f32],
    shading: &[f32],
    overlay: &OverlayUniforms,
    heightmap: &[f32],
    width: u32,
    height: u32,
) -> Result<RuntimeContractObservation, String> {
    if terrain.len() != 44 || shading.len() != 44 {
        return Err("terrain contract uniforms have an unexpected ABI length".to_string());
    }
    if width == 0 || height == 0 || heightmap.len() != width as usize * height as usize {
        return Err(
            "terrain contract height texture dimensions do not match its samples".to_string(),
        );
    }

    let mut observation = RuntimeContractObservation::new(
        scene,
        "terrain_pbr_pom.pipeline",
        "terrain_pbr_pom",
        "src/shaders/terrain_pbr_pom.wgsl",
        "fs_main",
        "shaders/contracts/terrain_pbr_pom.toml",
    );
    check_slice(
        &mut observation,
        "u_terrain.view",
        0,
        &terrain[..16],
        -65_536.0,
        65_536.0,
    );
    check_slice(
        &mut observation,
        "u_terrain.proj",
        0,
        &terrain[16..32],
        -65_536.0,
        65_536.0,
    );
    check_slice(
        &mut observation,
        "u_terrain.sun_exposure",
        0,
        &terrain[32..36],
        -1.0,
        65_536.0,
    );
    // BOP-P2-02's declared large-region floor is a 100 km square.
    for (name, value, allowed_min, allowed_max) in [
        ("u_terrain.spacing_h_exag.x", terrain[36], 1e-6, 100_000.0),
        ("u_terrain.spacing_h_exag.y", terrain[37], 1e-6, 100_000.0),
        ("u_terrain.spacing_h_exag.z", terrain[38], 1e-6, 65_536.0),
        ("u_terrain.spacing_h_exag.w", terrain[39], 0.0, 65_536.0),
    ] {
        observation.check_range(
            "uniform",
            name,
            Some(0),
            value,
            value,
            allowed_min,
            allowed_max,
        );
    }
    check_slice(
        &mut observation,
        "u_terrain.camera_mode_params",
        0,
        &terrain[40..44],
        0.0,
        10_000_000.0,
    );

    for (name, range, allowed) in [
        ("u_shading.triplanar_params", 0..4, (0.0, 65_536.0)),
        ("u_shading.pom_steps", 4..8, (0.0, 128.0)),
        ("u_shading.layer_heights", 8..12, (0.0, 1.0)),
        ("u_shading.layer_roughness", 12..16, (0.0, 1.0)),
        ("u_shading.layer_metallic", 16..20, (0.0, 1.0)),
        ("u_shading.layer_control", 20..24, (-32.0, 32.0)),
        ("u_shading.light_params", 24..28, (0.0, 65_536.0)),
        ("u_shading.clamp0", 28..32, (-65_536.0, 65_536.0)),
        ("u_shading.clamp1", 32..36, (0.0, 65_536.0)),
        ("u_shading.clamp2", 36..40, (0.0, 65_536.0)),
        ("u_shading.height_curve", 40..44, (0.0, 65_536.0)),
    ] {
        check_slice(
            &mut observation,
            name,
            5,
            &shading[range],
            allowed.0,
            allowed.1,
        );
    }
    observation.check_range(
        "uniform_relation",
        "u_shading.clamp0.height_range",
        Some(5),
        shading[29] - shading[28],
        shading[29] - shading[28],
        1e-6,
        131_072.0,
    );

    for (index, values) in [
        overlay.params0,
        overlay.params1,
        overlay.params2,
        overlay.params3,
        overlay.params4,
        overlay.params5,
    ]
    .iter()
    .enumerate()
    {
        check_slice(
            &mut observation,
            &format!("u_overlay.params{index}"),
            8,
            values,
            -10_000_000.0,
            10_000_000.0,
        );
    }
    check_slice(
        &mut observation,
        "height_tex.samples",
        1,
        heightmap,
        -65_536.0,
        65_536.0,
    );
    observation.check_length(
        "texture_dimension",
        "height_tex.width",
        Some(1),
        width.into(),
        1,
    );
    observation.check_length(
        "texture_dimension",
        "height_tex.height",
        Some(1),
        height.into(),
        1,
    );
    Ok(observation)
}

pub(super) fn record_observation(
    scene: &str,
    terrain: &[f32],
    shading: &[f32],
    overlay: &OverlayUniforms,
    heightmap: &[f32],
    width: u32,
    height: u32,
) -> Result<(), String> {
    if !crate::core::shader_contract_runtime::capture_active() {
        return Ok(());
    }
    crate::core::shader_contract_runtime::record_observation(build_observation(
        scene, terrain, shading, overlay, heightmap, width, height,
    )?)
}

fn check_slice(
    observation: &mut RuntimeContractObservation,
    name: &str,
    binding: u32,
    values: &[f32],
    allowed_min: f32,
    allowed_max: f32,
) {
    let (observed_min, observed_max) = values
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        });
    observation.check_range(
        "uniform",
        name,
        Some(binding),
        observed_min,
        observed_max,
        allowed_min,
        allowed_max,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_runtime_observation_checks_uploaded_uniforms_and_height_texture() {
        let mut terrain = vec![0.0; 44];
        terrain[32..36].copy_from_slice(&[0.0, 1.0, 0.0, 1.0]);
        terrain[36..40].copy_from_slice(&[100_000.0, 100_000.0, 1.0, 1.0]);
        terrain[40..44].copy_from_slice(&[0.0, 64.0, 0.1, 1_000.0]);
        let mut shading = vec![0.0; 44];
        shading[0..4].copy_from_slice(&[1.0, 4.0, 1.0, 0.0]);
        shading[8..12].copy_from_slice(&[0.0, 0.33, 0.66, 1.0]);
        shading[12..16].fill(0.5);
        shading[20..24].copy_from_slice(&[4.0, 0.125, 0.0, 0.0]);
        shading[24..28].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        shading[28..32].copy_from_slice(&[0.0, 1.0, 0.0, 1.0]);
        shading[32..36].copy_from_slice(&[0.0, 1.0, 0.0, 1.0]);
        shading[36..40].copy_from_slice(&[0.0, 1.0, 0.0, 1.0]);
        shading[40..44].copy_from_slice(&[0.0, 1.0, 1.0, 0.5]);

        let observation = build_observation(
            "terrain.render_internal",
            &terrain,
            &shading,
            &OverlayUniforms {
                params0: [0.0; 4],
                params1: [0.0; 4],
                params2: [0.0; 4],
                params3: [0.0; 4],
                params4: [0.0; 4],
                params5: [0.0; 4],
            },
            &[0.0, 0.5, 1.0, 0.25],
            2,
            2,
        )
        .unwrap();
        let names = observation
            .checked_bindings
            .iter()
            .map(|binding| binding.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(names.contains("u_terrain.view"));
        assert!(names.contains("u_terrain.spacing_h_exag.x"));
        assert!(names.contains("u_shading.clamp0.height_range"));
        assert!(names.contains("height_tex.samples"));
        assert_eq!(observation.status, "passed");
    }
}
