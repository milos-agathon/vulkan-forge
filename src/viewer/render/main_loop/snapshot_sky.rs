use glam::Mat4;

use crate::viewer::viewer_types::FrameCamera;
use crate::viewer::Viewer;

impl Viewer {
    pub(super) fn encode_snapshot_sky(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        frame: FrameCamera,
    ) -> bool {
        if let Err(error) = self.ensure_snapshot_sky_resources(width, height) {
            crate::core::degradation::record_degradation(
                "allocation_fallback",
                "viewer.sky.snapshot",
                "dedicated snapshot sky unavailable; snapshot omits celestial sky",
            );
            eprintln!("[viewer] failed to allocate snapshot sky resources: {error}");
            return false;
        }

        let cache = self
            .sky_snapshot_cache
            .as_ref()
            .expect("snapshot sky cache initialized");
        let projection = frame.projection(width, height);
        let view = frame.view();
        let matrices = [
            matrix_columns(view),
            matrix_columns(projection),
            matrix_columns(view.inverse()),
            matrix_columns(projection.inverse()),
        ];
        self.queue
            .write_buffer(&cache.camera, 0, bytemuck::cast_slice(&matrices));
        let eye = frame.render_eye();
        let eye_offset = (std::mem::size_of::<[[f32; 4]; 4]>() * 4) as u64;
        self.queue.write_buffer(
            &cache.camera,
            eye_offset,
            bytemuck::cast_slice(&[eye.x, eye.y, eye.z, 0.0]),
        );
        self.queue.write_buffer(
            &cache.camera,
            eye_offset + std::mem::size_of::<[f32; 4]>() as u64,
            bytemuck::cast_slice(&[
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ]),
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viewer.sky.snapshot.compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &cache.compute_bind_group, &[]);
            pass.set_bind_group(1, &cache.camera_bind_group, &[]);
            pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }
        if self.night_instance_count > 0 {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.night.snapshot.overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cache.output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.night_pipeline);
            pass.set_bind_group(0, &self.night_bind_group, &[]);
            pass.set_bind_group(1, &cache.camera_bind_group, &[]);
            pass.draw(0..6, 0..self.night_instance_count);
        }
        true
    }

    pub(super) fn present_snapshot_sky(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: Option<&wgpu::TextureView>,
    ) {
        let cache = self
            .sky_snapshot_cache
            .as_ref()
            .expect("snapshot sky encoded before presentation");
        let depth_attachment = depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewer.sky.snapshot.present"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(if depth.is_some() {
            &self.sky_present_depth_pipeline
        } else {
            &self.sky_present_flat_pipeline
        });
        pass.set_bind_group(0, &cache.present_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn matrix_columns(matrix: Mat4) -> [[f32; 4]; 4] {
    let values = matrix.to_cols_array();
    [
        [values[0], values[1], values[2], values[3]],
        [values[4], values[5], values[6], values[7]],
        [values[8], values[9], values[10], values[11]],
        [values[12], values[13], values[14], values[15]],
    ]
}
