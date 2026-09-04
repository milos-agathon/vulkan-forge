//! Shadow technique orchestration with atlas budgeting and parameter uniforms

mod budget;
mod system;
mod types;

pub use system::ShadowManager;
pub use types::{ShadowManagerConfig, DEFAULT_MEMORY_BUDGET_BYTES};
pub(crate) use types::{
    DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS, DEFAULT_PCSS_FILTER_RADIUS_TEXELS, DEFAULT_PCSS_LIGHT_SIZE,
};
