use super::decode_atmosphere::*;
use super::decode_core::*;
use super::decode_effects::*;
use super::decode_lighting::*;
use super::decode_materials::*;
use super::decode_postfx::*;
use super::decode_probes::*;
use super::decode_vt::*;
use super::parse::*;
use super::*;

impl TerrainRenderParams {
    pub(crate) fn from_python_params(py: Python<'_>, params: Bound<'_, PyAny>) -> PyResult<Self> {
        let core = parse_core_params(&params)?;

        let light = params.getattr("light")?;
        let ibl = params.getattr("ibl")?;
        let shadows = params.getattr("shadows")?;
        let triplanar = params.getattr("triplanar")?;
        let pom = params.getattr("pom")?;
        let lod = params.getattr("lod")?;
        let sampling = params.getattr("sampling")?;
        let clamp = params.getattr("clamp")?;

        let decoded = DecodedTerrainSettings {
            light: parse_light_settings(&light)?,
            triplanar: parse_triplanar_settings(&triplanar)?,
            pom: parse_pom_settings(&pom)?,
            lod: parse_lod_settings(&lod)?,
            clamp: parse_clamp_settings(&clamp)?,
            sampling: parse_sampling_settings(&sampling)?,
            shadow: parse_shadow_settings(&shadows)?,
            fog: parse_fog_settings(&params),
            reflection: parse_reflection_settings(&params),
            detail: parse_detail_settings(&params),
            height_ao: parse_height_ao_settings(&params),
            sun_visibility: parse_sun_visibility_settings(&params),
            bloom: parse_bloom_settings(&params),
            materials: parse_material_layer_settings(&params),
            vector_overlay: parse_vector_overlay_settings(&params),
            tonemap: parse_tonemap_settings(&params),
            aov: parse_aov_settings(&params),
            screen_space: parse_screen_space_settings(&params),
            dof: parse_dof_settings(&params),
            motion_blur: parse_motion_blur_settings(&params),
            lens_effects: parse_lens_effects_settings(&params),
            denoise: parse_denoise_settings(&params),
            volumetrics: parse_volumetrics_settings(&params),
            sky: parse_sky_settings(&params)?,
            probes: parse_probe_settings(&params),
            reflection_probes: parse_reflection_probe_settings(&params),
            vt: parse_vt_settings(&params),
        };

        let overlays = extract_overlays(params.getattr("overlays")?.as_gil_ref())?;

        Ok(Self {
            size_px: core.size_px,
            render_scale: core.render_scale,
            terrain_span: core.terrain_span,
            msaa_samples: core.msaa_samples,
            z_scale: core.z_scale,
            cam_target: core.cam_target,
            cam_radius: core.cam_radius,
            cam_phi_deg: core.cam_phi_deg,
            cam_theta_deg: core.cam_theta_deg,
            cam_gamma_deg: core.cam_gamma_deg,
            fov_y_deg: core.fov_y_deg,
            clip: core.clip,
            exposure: core.exposure,
            gamma: core.gamma,
            albedo_mode: core.albedo_mode,
            colormap_strength: core.colormap_strength,
            hue_variation_strength: core.hue_variation_strength,
            ao_weight: core.ao_weight,
            height_curve_mode: core.height_curve_mode,
            height_curve_strength: core.height_curve_strength,
            height_curve_power: core.height_curve_power,
            lambert_contrast: core.lambert_contrast,
            colormap_srgb: core.colormap_srgb,
            output_srgb_eotf: core.output_srgb_eotf,
            camera_mode: core.camera_mode,
            culling: core.culling,
            shading: core.shading,
            vt_store_path: core.vt_store_path,
            prefetch_horizon_ms: core.prefetch_horizon_ms,
            vt_upload_budget_bytes: core.vt_upload_budget_bytes,
            debug_mode: core.debug_mode,
            aa_samples: core.aa_samples,
            aa_seed: core.aa_seed,
            terrain_data_revision: core.terrain_data_revision,
            height_curve_lut: core.height_curve_lut,
            overlays,
            light: light.unbind(),
            ibl: ibl.unbind(),
            shadows: shadows.unbind(),
            triplanar: triplanar.unbind(),
            pom: pom.unbind(),
            lod: lod.unbind(),
            sampling: sampling.unbind(),
            clamp: clamp.unbind(),
            python_object: params.into_py(py),
            decoded,
        })
    }

    pub(crate) fn decoded(&self) -> &DecodedTerrainSettings {
        &self.decoded
    }
}
