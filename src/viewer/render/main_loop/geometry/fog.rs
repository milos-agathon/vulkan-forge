use glam::Mat4;

use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::viewer::viewer_enums::FogMode;
use crate::viewer::{
    CameraFrustum, FogCameraUniforms, Viewer, ViewerShadowUniforms, VolumetricUniformsStd140,
};

use super::mat4_to_array;

impl Viewer {
    pub(super) fn render_geometry_fog(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gi: &mut ScreenSpaceEffectsManager,
    ) {
        if !self.fog_enabled {
            return;
        }

        let frame = self.current_frame_camera();
        let proj = frame.projection(self.config.width, self.config.height);
        let view_mat = frame.view();

        if let Some(ref mut csm) = self.csm {
            let frustum = CameraFrustum::from_matrices(&view_mat, &proj);
            csm.update_cascades(&self.queue, &frustum);
        }

        if self.fog_use_shadows {
            self.render_fog_shadow_cascades(encoder, frame);
        }

        let inv_view = view_mat.inverse();
        let inv_proj = proj.inverse();
        let eye = frame.render_eye();
        let fog_cam = FogCameraUniforms {
            view: mat4_to_array(view_mat),
            proj: mat4_to_array(proj),
            inv_view: mat4_to_array(inv_view),
            inv_proj: mat4_to_array(inv_proj),
            view_proj: mat4_to_array(proj * view_mat),
            eye_position: [eye.x, eye.y, eye.z],
            near: frame.near,
            far: frame.far,
            _pad_far: [0.0; 3],
            _pad: [0.0; 3],
            _pad_end: 0.0,
        };
        self.queue
            .write_buffer(&self.fog_camera, 0, bytemuck::bytes_of(&fog_cam));

        let sun_dir_ws = glam::Vec3::from_array(self.lit_sun_direction_ws).normalize();
        let steps = if self.fog_half_res_enabled {
            (self.fog_steps.max(1) / 2).max(16)
        } else {
            self.fog_steps.max(1)
        };
        let fog_params_packed = VolumetricUniformsStd140 {
            density: self.fog_density.max(0.0),
            height_falloff: 0.1,
            phase_g: self.fog_g.clamp(-0.999, 0.999),
            max_steps: steps,
            start_distance: 0.1,
            max_distance: frame.far,
            _pad_a0: 0.0,
            _pad_a1: 0.0,
            scattering_color: [1.0, 1.0, 1.0],
            absorption: 1.0,
            sun_direction: [sun_dir_ws.x, sun_dir_ws.y, sun_dir_ws.z],
            sun_intensity: if sun_dir_ws.y > 0.0 {
                (self.lit_sun_intensity * self.lit_directional_scale).max(0.0)
            } else {
                0.0
            },
            ambient_color: [0.2, 0.25, 0.3],
            temporal_alpha: self
                .fog_history_state
                .blend_alpha(self.fog_temporal_alpha.clamp(0.0, 0.9)),
            use_shadows: if self.fog_use_shadows { 1 } else { 0 },
            jitter_strength: 0.8,
            frame_index: self.fog_frame_index,
            _pad0: 0,
        };
        self.queue
            .write_buffer(&self.fog_params, 0, bytemuck::bytes_of(&fog_params_packed));

        let mut fog_shadow_mat = Mat4::IDENTITY;
        if self.fog_use_shadows {
            if let Some(ref csm) = self.csm {
                let cascades = csm.cascades();
                if let Some(c0) = cascades.get(0) {
                    fog_shadow_mat = Mat4::from_cols_array_2d(&c0.light_projection);
                }
            }
        }
        self.queue.write_buffer(
            &self.fog_shadow_matrix,
            0,
            bytemuck::bytes_of(&mat4_to_array(fog_shadow_mat)),
        );

        self.ensure_fog_bind_groups(gi);
        let bg0 = self.fog_bg0_cache.borrow();
        let bg1 = self.fog_bg1_cache.borrow();
        let bg2 = self.fog_bg2_cache.borrow();

        if matches!(self.fog_mode, FogMode::Raymarch) {
            self.dispatch_raymarch_fog(
                encoder,
                gi,
                bg0.as_ref().unwrap(),
                bg1.as_ref().unwrap(),
                bg2.as_ref().unwrap(),
            );
        } else {
            self.dispatch_froxel_fog(
                encoder,
                bg0.as_ref().unwrap(),
                bg1.as_ref().unwrap(),
                bg2.as_ref().unwrap(),
            );
        }

        self.fog_history_state.mark_populated();
        self.fog_frame_index = self.fog_frame_index.wrapping_add(1);
    }

    fn render_fog_shadow_cascades(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) {
        let Some(vb) = self.geom_vb.as_ref() else {
            // Terrain shadows use their own viewer-specific pipeline. A
            // terrain-only scene has no generic geometry buffer to draw here.
            return;
        };
        if let (Some(ref csm), Some(ref csm_pipe), Some(ref csm_cam_buf)) = (
            self.csm.as_ref(),
            self.csm_depth_pipeline.as_ref(),
            self.csm_depth_camera.as_ref(),
        ) {
            let cascade_count = csm.cascade_count() as usize;
            for cascade_idx in 0..cascade_count {
                if let (Some(depth_view), Some(light_vp)) = (
                    csm.cascade_depth_view(cascade_idx),
                    csm.cascade_projection(cascade_idx),
                ) {
                    let shadow_uniforms = ViewerShadowUniforms {
                        light_view_proj: light_vp.to_cols_array_2d(),
                        object_model: self.anchored_object_model(frame).to_cols_array_2d(),
                    };
                    self.queue
                        .write_buffer(csm_cam_buf, 0, bytemuck::bytes_of(&shadow_uniforms));
                    let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.csm.depth"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    shadow_pass.set_pipeline(csm_pipe);
                    shadow_pass.set_bind_group(0, self.csm_depth_bind_group.as_ref().unwrap(), &[]);
                    shadow_pass.set_vertex_buffer(0, vb.slice(..));
                    if let Some(ib) = self.geom_ib.as_ref() {
                        shadow_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        shadow_pass.draw_indexed(0..self.geom_index_count, 0, 0..1);
                    } else {
                        shadow_pass.draw(0..self.geom_index_count, 0..1);
                    }
                }
            }
        }
    }
}
