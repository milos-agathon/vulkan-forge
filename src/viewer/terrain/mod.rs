// src/viewer/terrain/mod.rs
// Terrain viewer module - standalone terrain rendering without PyO3 dependencies
// Split from viewer_terrain.rs as part of the viewer refactoring

pub mod denoise;
mod dof;
mod motion_blur;
mod motion_blur_depth;
pub mod overlay;
mod pbr_renderer;
mod post_process;
mod render;
mod scene;
mod shader;
mod shader_pbr;
pub mod vector_overlay;
mod volume_density;
mod volumetrics;

#[allow(unused_imports)]
pub use overlay::{BlendMode, OverlayConfig, OverlayData, OverlayLayer, OverlayStack};
#[allow(unused_imports)]
pub use pbr_renderer::ViewerTerrainPbrConfig;
pub use scene::ViewerTerrainScene;

// Option B: Vector overlay geometry exports
#[allow(unused_imports)]
pub use vector_overlay::{
    OverlayPrimitive, RenderLayerParams, VectorOverlayGpu, VectorOverlayLayer, VectorOverlayStack,
    VectorOverlayUniforms, VectorVertex, VECTOR_OVERLAY_SHADER,
};
