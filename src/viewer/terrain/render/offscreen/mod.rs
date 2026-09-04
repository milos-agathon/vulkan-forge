mod effects;
mod scene;
mod setup;

use super::*;

pub(super) struct SnapshotRenderState {
    pub(super) use_pbr: bool,
    pub(super) view_mat: glam::Mat4,
    pub(super) proj: glam::Mat4,
    pub(super) view_proj: glam::Mat4,
    pub(super) sun_dir: glam::Vec3,
    pub(super) eye: glam::Vec3,
    pub(super) render_origin_span: [f32; 4],
    pub(super) h_range: f32,
    pub(super) shader_z_scale: f32,
    pub(super) vo_view_proj: [[f32; 4]; 4],
    pub(super) vo_sun_dir: [f32; 3],
    pub(super) vo_lighting: [f32; 4],
}

impl ViewerTerrainScene {
    pub fn render_to_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        selected_feature_id: u32,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) -> Option<crate::core::resource_tracker::TrackedTexture> {
        self.snapshot_depth_texture = None;
        eprintln!("[DEBUG render_to_texture ENTRY] {}x{}", width, height);
        if self.terrain.is_none() {
            eprintln!("[DEBUG render_to_texture] No terrain, returning None");
            return None;
        }

        self.prepare_snapshot_resources(width, height);
        let (color_tex, color_view) = match self.create_snapshot_color_target(
            "terrain_viewer.snapshot_color",
            target_format,
            width,
            height,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[terrain] failed to allocate snapshot color target: {e}");
                return None;
            }
        };
        let (depth_tex, depth_view) = match self.create_snapshot_depth_target(width, height) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[terrain] failed to allocate snapshot depth target: {e}");
                return None;
            }
        };
        let state = self.build_snapshot_render_state(encoder, target_format, width, height, frame);
        let has_vector_overlays = self.prepare_snapshot_overlays();

        self.render_snapshot_scene_pass(
            encoder,
            &color_view,
            &depth_view,
            selected_feature_id,
            &state,
            has_vector_overlays,
        );
        self.render_snapshot_oit_pass(
            encoder,
            &color_view,
            &depth_view,
            width,
            height,
            selected_feature_id,
            &state,
            has_vector_overlays,
        );

        let output = self.apply_snapshot_effects(
            encoder,
            target_format,
            width,
            height,
            &depth_view,
            color_tex,
            color_view,
            &state,
        );
        self.snapshot_depth_texture = Some(depth_tex);
        Some(output)
    }
}
