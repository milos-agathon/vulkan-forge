use super::*;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ShadowPassUniforms {
    pub(super) light_view_proj: [[f32; 4]; 4],
    pub(super) terrain_params: [f32; 4],
    pub(super) grid_params: [f32; 4],
    pub(super) height_curve: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct OverlayUniforms {
    pub(super) params0: [f32; 4],
    pub(super) params1: [f32; 4],
    pub(super) params2: [f32; 4],
    pub(super) params3: [f32; 4],
    pub(super) params4: [f32; 4],
    pub(super) params5: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct FogUniforms {
    pub(super) params0: [f32; 4],
    pub(super) fog_inscatter: [f32; 4],
    pub(super) sky_params0: [f32; 4],
    pub(super) sky_params1: [f32; 4],
    pub(super) aether_sun_direction: [f32; 4],
    pub(super) aether_planet_lut: [f32; 4],
}

impl FogUniforms {
    pub(super) fn disabled() -> Self {
        Self {
            params0: [0.0, 0.0, 0.0, 0.0],
            fog_inscatter: [1.0, 1.0, 1.0, 0.0],
            sky_params0: [0.0, 0.0, 0.0, 0.0],
            sky_params1: [0.0, 0.0, 0.0, 0.0],
            aether_sun_direction: [0.0, 0.0, 1.0, 0.0],
            aether_planet_lut: [6_360_000.0, 6_460_000.0, 2.0, 2.0],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct MaterialLayerUniforms {
    pub(super) snow_params0: [f32; 4],
    pub(super) snow_params1: [f32; 4],
    pub(super) snow_color: [f32; 4],
    pub(super) snow_sss_tint: [f32; 4],
    pub(super) rock_params: [f32; 4],
    pub(super) rock_color: [f32; 4],
    pub(super) rock_sss_tint: [f32; 4],
    pub(super) wetness_params: [f32; 4],
    pub(super) wetness_sss_tint: [f32; 4],
    pub(super) variation_params0: [f32; 4],
    pub(super) snow_variation: [f32; 4],
    pub(super) rock_variation: [f32; 4],
    pub(super) wetness_variation: [f32; 4],
    pub(super) map_flags: [f32; 4],
}

impl MaterialLayerUniforms {
    pub(super) fn disabled() -> Self {
        Self {
            snow_params0: [2000.0, 500.0, 0.785, 0.262],
            snow_params1: [0.3, 0.4, 0.0, 0.0],
            snow_color: [0.95, 0.95, 0.98, 0.0],
            snow_sss_tint: [1.0, 1.0, 1.0, 0.0],
            rock_params: [0.785, 0.175, 0.8, 0.0],
            rock_color: [0.35, 0.32, 0.28, 0.0],
            rock_sss_tint: [1.0, 1.0, 1.0, 0.0],
            wetness_params: [0.3, 0.5, 0.0, 0.0],
            wetness_sss_tint: [1.0, 1.0, 1.0, 0.0],
            variation_params0: [3.5, 18.0, 4.0, 0.0],
            snow_variation: [0.0, 0.0, 0.0, 0.0],
            rock_variation: [0.0, 0.0, 0.0, 0.0],
            wetness_variation: [0.0, 0.0, 0.0, 0.0],
            map_flags: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct HeightAoUniforms {
    pub(super) params0: [f32; 4],
    pub(super) params1: [f32; 4],
    pub(super) params2: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SunVisUniforms {
    pub(super) params0: [f32; 4],
    pub(super) params1: [f32; 4],
    pub(super) params2: [f32; 4],
    pub(super) params3: [f32; 4],
}
