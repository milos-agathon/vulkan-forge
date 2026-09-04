use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_buffer_init, tracked_create_texture};
use crate::viewer::viewer_struct::SkySnapshotResources;
use crate::viewer::Viewer;

impl Viewer {
    /// Cache the dimension-specific sky resources used only for snapshots.
    ///
    /// Snapshot camera state must not share the window camera buffer: queue
    /// writes are observed when submitted commands execute, so two writes to
    /// one buffer before submission can make both sky passes see the last one.
    pub(crate) fn ensure_snapshot_sky_resources(
        &mut self,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        if self
            .sky_snapshot_cache
            .as_ref()
            .is_some_and(|cache| cache.width == width && cache.height == height)
        {
            return Ok(());
        }

        let camera_data = [0.0_f32; 72];
        let camera = tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("viewer.sky.snapshot.camera"),
                contents: bytemuck::cast_slice(&camera_data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let output = tracked_create_texture(
            &self.device,
            &crate::viewer::init::sky_output_descriptor(width, height),
        )?;
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.sky.snapshot.compute_bg"),
            layout: &self.sky_bind_group_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sky_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });
        let camera_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.sky.snapshot.camera_bg"),
            layout: &self.sky_bind_group_layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        let present_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.sky.snapshot.present_bg"),
            layout: &self.sky_present_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sky_present_sampler),
                },
            ],
        });
        self.sky_snapshot_cache = Some(SkySnapshotResources {
            width,
            height,
            camera,
            _output: output,
            output_view,
            compute_bind_group,
            camera_bind_group,
            present_bind_group,
        });
        Ok(())
    }
}
