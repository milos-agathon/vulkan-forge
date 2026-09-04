// src/viewer/render/main_loop.rs
// Main render loop for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

mod finalize;
pub(crate) mod frame_anchor;
mod frame_anchor_stats;
mod frame_setup;
mod geometry;
mod postfx;
mod postfx_cache;
mod secondary;
mod snapshot_sky;

use crate::viewer::Viewer;

#[derive(Clone, Copy)]
pub(super) struct RenderAvailability {
    pub(super) have_gi: bool,
    pub(super) have_pipe: bool,
    pub(super) have_cam: bool,
    pub(super) have_vb: bool,
    pub(super) have_z: bool,
    pub(super) have_bgl: bool,
}

impl RenderAvailability {
    pub(super) fn ready(self) -> bool {
        self.have_gi
            && self.have_pipe
            && self.have_cam
            && self.have_vb
            && self.have_z
            && self.have_bgl
    }
}

impl Viewer {
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.prepare_frame_anchor();
        let (output, view, snapshot_dimensions, mut encoder) = self.prepare_render_frame()?;
        let availability = self.render_geometry_stage(&mut encoder, &view);
        encoder = self.render_secondary_paths(encoder, &view, snapshot_dimensions, availability);
        self.finish_render_frame(encoder, output);
        Ok(())
    }
}
