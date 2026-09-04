// src/viewer/init/mod.rs
// Initialization module for Viewer::new() decomposition
// Split from mod.rs as part of the viewer refactoring

mod composite_init;
mod device_init;
mod fallback_init;
mod fog_init;
mod gbuffer_init;
mod gi_baseline_init;
mod lit_init;
mod sky_init;
mod viewer_new;

pub use device_init::create_device_and_surface;
pub use fallback_init::create_fallback_pipeline;
pub use fog_init::create_fog_resources;
pub use gbuffer_init::create_gbuffer_resources;
pub use gi_baseline_init::create_gi_baseline_resources;
pub use lit_init::create_lit_resources;
pub use sky_init::{create_sky_resources, sky_output_descriptor};
// viewer_new exports Viewer::new() impl directly
